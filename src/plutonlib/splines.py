

import plutonlib.read_write as prw
import plutonlib.simulations as ps

import numpy as np

import networkx as nx
from skan import Skeleton, summarize
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.interpolate import splprep,splev

import matplotlib.pyplot as plt

#---PRAiSE SB -> skeleton path---#
def skeleton_setup(grid_xy,log_sb,percentile = 50,plot=False):
    """Skeletonise the surface brightness image using a percentile cutoff"""
    increment = 0.5
    is_split = False
    while not is_split:
        binary_mask = np.where(np.isnan(log_sb), False, log_sb > np.nanpercentile(log_sb, percentile))
        mask_labels = label(binary_mask) #labels each mask level
        regions = regionprops(mask_labels)
        sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
        # print([s.area for s in sorted_regions])
        if len(sorted_regions) <2:
            percentile += increment
            continue
            # raise ValueError(f"Only found 1 image region, `skeleton_setup` requires 2 image areas to trace jet, try increasing percentile cut: percentile = {percentile}")

        jet1, jet2 = sorted_regions[0], sorted_regions[1] 
        is_split = (jet2.area / jet1.area >=0.5) and np.sum([jet1.area,jet2.area]) / np.sum([s.area for s in sorted_regions]) >= 0.9 #second jet at least half the size and combined areas = 90% of full area
        percentile += increment
        if percentile > 90:
            raise ValueError("Percentile has incremented past 90%, exiting")

    print(f"Percentile = {percentile}")
    jet_masks = [mask_labels == jet1.label,mask_labels == jet2.label]
    skel_objs = [Skeleton(skeletonize(m)) for m in jet_masks]
    skel_setup = {
        "skel_objs":skel_objs,
        "log_sb":log_sb,
        "mask_labels":mask_labels,
        "sorted_regions": sorted_regions
        }

    if plot:
        plt.figure(figsize=(6,6))
        plt.pcolormesh(grid_xy[0], grid_xy[1], mask_labels, cmap="tab10")
        plt.colorbar(label="component label")
        plt.gca().set_aspect("equal")
        plt.show()

    return skel_setup

def graph_weigh_edges(skel_setup):
    """Setup graph, weigh edges by mean sb along path, use this graph for finding the longest path weighted by sb value"""
    graphs = []
    log_sb = skel_setup["log_sb"]
    for skel_obj in skel_setup["skel_objs"]:
        branch_data = summarize(skel_obj, separator='-')
        # log_sb_clean = np.where(np.isnan(log_sb), 0, log_sb)  # or another fill value below your floor

        G = nx.Graph() #build a graph from the branch data with nodes
        for _, row in branch_data.iterrows():
            G.add_edge(
                row['node-id-src'],
                row['node-id-dst'],
                weight=row['branch-distance'],   # placeholder weight for now
                sb=None                           # fill in once you attach your SB metric per branch
            )

        for u, v, data in G.edges(data=True): #assign a mean sb value to each edge
            # find the matching branch_data row for this edge
            row = branch_data[
                ((branch_data['node-id-src'] == u) & (branch_data['node-id-dst'] == v)) |
                ((branch_data['node-id-src'] == v) & (branch_data['node-id-dst'] == u))
            ].iloc[0]

            # get the actual path index skan assigned this branch, then pull pixel coords
            coords = skel_obj.path_coordinates(row.name).astype(int)
            data['sb'] = log_sb[coords[:, 0], coords[:, 1]].mean() #NOTE was log_sb_clean

        graphs.append(G)

    graph_setup = {
        "graphs":graphs,
        "log_sb":log_sb,
        "skel_objs":skel_setup["skel_objs"],
    }
    return graph_setup

def path_weigh_endpoints(graph_setup,sim,output,grid_xy,grid_dxy):
    """For every path in the graph, find the path with the highest sb along to endpoint, collapsing graph to one path"""
    graphs = graph_setup["graphs"]
    skel_objs = graph_setup["skel_objs"]
    results = []
    for G, skel_obj in zip(graphs, skel_objs):
        branch_data = summarize(skel_obj, separator='-')  # easier to track if generated again

        inj = sim.get_injection_region(output)
        inj_row = (inj[2].value - grid_xy[1][0]) / grid_dxy[1]   # z -> row
        inj_col = (inj[0].value - grid_xy[0][0]) / grid_dxy[0]   # x -> col

        graph_nodes = np.array(list(G.nodes))
        node_coords = skel_obj.coordinates[graph_nodes.astype(int)]
        dists = np.hypot(node_coords[:, 0] - inj_row, node_coords[:, 1] - inj_col)
        start_node = graph_nodes[np.argmin(dists)]
        print(f"Start node = {start_node}")

        endpoints = [n for n in G.nodes if G.degree(n) == 1 and n != start_node]
        best_path, best_sb = None, -np.inf
        for ep in endpoints:
            path = nx.shortest_path(G,start_node,ep)
            total_sb = sum(G[path[i]][path[i+1]]['sb'] for i in range(len(path)-1))
            if total_sb > best_sb:
                best_sb, best_path = total_sb,path

        results.append({ #lists of results containing a dict
            "best_path": best_path, #NOTE removing the last branch as it often veers off
            "best_sb": best_sb,
            "branch_data": branch_data,
            "skel_obj": skel_obj,
        })
    return results

def path_to_coords(path_results, grid_xy, grid_dxy):
    """Convert the best path into useable coordinates"""
    pathpoints_split = []
    for res in path_results:
        best_path = res["best_path"]
        branch_data = res["branch_data"]
        skel_obj = res["skel_obj"]

        x_coords,y_coords = [],[]
        for i in range(len(best_path) - 1):

            u, v = best_path[i], best_path[i + 1]

            row = branch_data[
                ((branch_data['node-id-src'] == u) &
                 (branch_data['node-id-dst'] == v)) |
                ((branch_data['node-id-src'] == v) &
                 (branch_data['node-id-dst'] == u))
            ].iloc[0]

            coords = skel_obj.path_coordinates(row.name).astype(int)

            # Get the coordinates of the graph nodes
            u_coord = skel_obj.coordinates[int(u)]
            v_coord = skel_obj.coordinates[int(v)]

            # Skan stores the branch in its own src -> dst direction.
            # Reverse it if that direction doesn't match u -> v.
            if np.linalg.norm(coords[0] - u_coord) > np.linalg.norm(coords[-1] - u_coord):
                coords = coords[::-1]

            # Remove the point shared with the previous branch
            if i > 0:
                coords = coords[1:]

            x_coords.extend(grid_xy[0][0] + coords[:, 1] * grid_dxy[0])
            y_coords.extend(grid_xy[1][0] + coords[:, 0] * grid_dxy[1])

        pathpoints_split.append(np.column_stack((x_coords, y_coords)))

    pathpoints = np.vstack((pathpoints_split[0][::-1], pathpoints_split[1][1:]))
    path_data = {
        "pathpoints":pathpoints,
        "pathpoints_split":pathpoints_split,
        "grid_xy":grid_xy,
        "grid_dxy":grid_dxy,
    }
    return path_data

def skeleton_splines(path_data,grid_mxy): #NOTE Make this function barebones and move bits to path tracing
    """turn pathpoints into splines using interpolation"""
    pathpoints = path_data["pathpoints"]
    pathpoints_split = path_data["pathpoints_split"]
    grid_dxy = path_data["grid_dxy"]

    arc_length = np.sum(np.sqrt(np.diff(pathpoints[:,0])**2 + np.diff(pathpoints[:,1])**2))
    max_resolution = min([grid_dxy[0],grid_dxy[1]])
    n_points = int(arc_length / max_resolution) #determines the resampling resolution

    # remove duplicate / zero-distance pathpoints
    d = np.sqrt(np.diff(pathpoints[:, 0])**2 + np.diff(pathpoints[:, 1])**2)
    keep = np.concatenate([[True], d > 1e-6])
    pathpoints = pathpoints[keep]
    k = min(3, len(pathpoints) - 1)

    tck, u = splprep([pathpoints[:,0],pathpoints[:,1]],s=n_points * 0.05,k=k) #NOTE weighing by the log_sb 
    u2 = np.linspace(u[0],u[-1],n_points) #use the spline array to generate a higher resolution array
    spline_points = splev(u2,tck)
    spline_points = np.column_stack(spline_points) #resampled pathpoints with higher resolution

    spline_data = { #TODO remove dict
        "spline_points": spline_points,
    }
    return spline_data

def skeletonise_sb(sim,output,freq,redshift,percentile = 50,sbdata=None,angles_alt = None):
    """AIO function"""
    # angles = [0,0,0]
    angles = angles_alt
    p_output = round(sim.simtime_to_part(output))
    if sbdata is None:
        sbdata = prw.load_sb_hdf5(sim=sim,outputs = [p_output],angles = [angles],freqs=[freq],redshift = redshift)
    entry = sbdata[sim][p_output][tuple(angles)]
    obs = entry["obs_properties"]
    log_sb = entry["log_sb"]

    grid_dxy = [obs["grid_x"][1] - obs["grid_x"][0],obs["grid_y"][1] - obs["grid_y"][0]]
    grid_mxy = [obs["grid_mx"],obs["grid_my"]]
    grid_xy = [obs["grid_x"],obs["grid_y"]]

    skel_setup = skeleton_setup(grid_xy=grid_xy,log_sb=log_sb,percentile=percentile,plot=True)
    graph_setup = graph_weigh_edges(skel_setup=skel_setup)
    path_results = path_weigh_endpoints(graph_setup=graph_setup,sim=sim,output=output,grid_xy=grid_xy,grid_dxy=grid_dxy)
    path_data = path_to_coords(path_results=path_results,grid_xy=grid_xy,grid_dxy=grid_dxy)
    spline_data = skeleton_splines(path_data=path_data,grid_mxy=grid_mxy)

    return spline_data

#---Skeleton -> jet path---#
def weigh_skeleton_sb(sbdata_entry,spline_points,search_width):
    """Weighs the skeletonised sb path by averaging x,y points weighted by sb values for each point in a search width"""

    dx = np.gradient(spline_points[:, 0])
    dz = np.gradient(spline_points[:, 1])
    norm = np.sqrt(dx**2 + dz**2)
    unit_vector = np.column_stack([dx, dz]) / norm[:, None]
    unit_vector_90 = np.column_stack([-unit_vector[:, 1], unit_vector[:, 0]])

    log_sb = sbdata_entry['log_sb']
    obs = sbdata_entry['obs_properties']
    X, Z = np.meshgrid(obs["grid_mx"], obs["grid_my"])

    sb_pathpoints = []
    for i,sp in enumerate(spline_points):
        x_cur,z_cur = sp[0],sp[1]
        x_local = (X - x_cur)*unit_vector[:,0][i] + (Z - z_cur)*unit_vector[:,1][i]
        z_local = (X - x_cur)*unit_vector_90[:,0][i] + (Z - z_cur)*unit_vector_90[:,1][i]

        sb_plane = (np.abs(x_local)< 0.2*search_width) & (np.abs(z_local)<search_width) & ~np.isnan(log_sb) #sb plane for current spline bound by upper and lower
        weights = 10**(6*log_sb[sb_plane]) #log vals introduce negatives and throws off the calc
        x_new = np.average(X[sb_plane], weights=weights) #average all xz points (midpoint) in plane weighted by value of tracer
        z_new = np.average(Z[sb_plane], weights=weights)  

        sb_pathpoints.append([x_new, z_new])
    sb_pathpoints = np.array(sb_pathpoints)
    return sb_pathpoints    

def iterate_sb_path(sim,sbdata_entry,spline_points,max_itr = 4):
    """Iterates the path weighing"""
    search_width = sim.jet.radius.value * 2.5
    cutoff = 0.5 #points 0.5kpc together are dupes

    sb_pathpoints = weigh_skeleton_sb(sbdata_entry=sbdata_entry,spline_points=spline_points,search_width=search_width) #weigh skeleton path by sb
    for itr in range(max_itr):
        d = np.sqrt(np.diff(sb_pathpoints[:,0])**2 + np.diff(sb_pathpoints[:,1])**2)
        keep = np.concatenate([[True], d > cutoff])
        sb_pathpoints = weigh_skeleton_sb(sbdata_entry=sbdata_entry,spline_points=sb_pathpoints[keep],search_width=search_width)

    return sb_pathpoints

def interpolate_sb_path(sim,sbdata_entry,spline_points,grid_dxy):
    """re-interpolates the weighed path back into splines"""
    sb_pathpoints = iterate_sb_path(sim=sim,sbdata_entry=sbdata_entry,spline_points=spline_points)
    arc_length = np.sum(np.sqrt(np.diff(sb_pathpoints[:,0])**2 + np.diff(sb_pathpoints[:,1])**2))
    max_resolution = min([grid_dxy[0],grid_dxy[1]])
    n_points = int(arc_length / max_resolution) #determines the resampling resolution
    k = min(3, len(sb_pathpoints) - 1)

    tck, u = splprep([sb_pathpoints[:,0],sb_pathpoints[:,1]],s=n_points * 0.07,k=k) #NOTE weighing by the log_sb 
    u2 = np.linspace(u[0],u[-1],n_points) #use the spline array to generate a higher resolution array
    jet_splines = splev(u2,tck)
    jet_splines = np.column_stack(jet_splines) #resampled pathpoints with higher resolution

    return jet_splines

def get_jet_splines_sb(sim,output,angles,freq,redshift,percentile = 50):
    p_output = round(sim.simtime_to_part(output))
    # angles_norot = [0,0,0]
    angles_norot = angles
    sbdata = prw.load_sb_hdf5(sim=sim,outputs = [p_output],angles = [angles_norot],freqs=[freq],redshift = redshift)
    entry = sbdata[sim][p_output][tuple(angles_norot)]
    obs = entry["obs_properties"]
    # log_sb = entry["log_sb"]
    grid_dxy = [obs["grid_x"][1] - obs["grid_x"][0],obs["grid_y"][1] - obs["grid_y"][0]]
    grid_mxy = [obs["grid_mx"],obs["grid_my"]]
    grid_xy = [obs["grid_x"],obs["grid_y"]]

    skeleton_data = skeletonise_sb(sim=sim,output=output,freq=freq,redshift=redshift,percentile=percentile,sbdata=sbdata,angles_alt = angles)
    skel_splines = skeleton_data['spline_points']
    jet_splines = interpolate_sb_path(sim=sim,sbdata_entry=entry,spline_points=skel_splines,grid_dxy=grid_dxy)

    inj = sim.get_injection_region(output)
    inj_xz = np.array([inj[0].value, inj[2].value])  # x=29.07, z=29.07
    inj_idx = np.argmin(np.linalg.norm(jet_splines - inj_xz, axis=1))

    splines_bot = jet_splines[:inj_idx+1]
    splines_top = jet_splines[inj_idx:]

    x_idx = np.searchsorted(grid_mxy[0], jet_splines[:, 0])
    y_idx = np.searchsorted(grid_mxy[1], jet_splines[:, 1])
    spline_slice_map = (x_idx, y_idx)

    arc_length = [np.sum(np.sqrt(np.diff(splits[:,0])**2 + np.diff(splits[:,1])**2)) for splits in [splines_bot, splines_top]]
    # arc_length = np.sum(np.sqrt(np.diff(jet_splines[:,0])**2 + np.diff(jet_splines[:,1])**2))
    print(f"Arc length per jet side = {arc_length}, full = {np.sum(arc_length):.2f}")

    #NOTE Add interpolation with flag for PLUTO grid

    spline_data = {
        "skel_splines":skel_splines,
        "jet_splines":jet_splines,
        "splines_top":splines_top,
        "splines_bot":splines_bot,
        "arc_length":arc_length,
        "spline_slice_map":spline_slice_map,

    }
    return spline_data
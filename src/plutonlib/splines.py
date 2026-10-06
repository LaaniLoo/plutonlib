

import plutonlib.read_write as prw
import plutonlib.simulations as ps

import numpy as np

import networkx as nx
from skan import Skeleton, summarize
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from scipy.interpolate import splprep,splev
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.ndimage import gaussian_filter

@dataclass
class SplineParams:
    percentile: float = 45
    increment: float= 0.5
    max_itr: int = 0 
    smoothing: float = 0.03
    sb_weight: float = 10

class SkeletonSetupError(Exception):
    def __init__(self,message,debug_figs=None):
        super().__init__(message)
        self.debug_figs = debug_figs

#---PRAiSE SB -> skeleton path---#
def debug_skel_labels(fig_list=None,**skel_setup_args):
    percentile = skel_setup_args.get("percentile")
    grid_xy = skel_setup_args["grid_xy"]
    mask_labels = skel_setup_args["mask_labels"]

    fig = plt.figure(figsize=(6,6))
    plt.pcolormesh(grid_xy[0], grid_xy[1], mask_labels, cmap="tab10")
    plt.colorbar(label="component label")
    plt.title(f"Labels @ percentile = {percentile}")
    plt.gca().set_aspect("equal")

    if fig_list is not None:
        fig_list.append(fig)

    # else:
    #     plt.show()

    return fig

def skeleton_setup(grid_xy,log_sb,params=None):
    """Skeletonise the surface brightness image using a percentile cutoff"""
    is_split = False
    params = params or SplineParams()
    debug_figs = []
    percentile = params.percentile
    while not is_split:
        binary_mask = np.where(np.isnan(log_sb), False, log_sb > np.nanpercentile(log_sb, percentile))
        mask_labels = label(binary_mask) #labels each mask level
        regions = regionprops(mask_labels)
        sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)

        # print(f"percentile = {percentile}:",[s.area for s in sorted_regions])
        # print(f"Jet Area ratio = {jet2.area / jet1.area}, Jet combined area = {np.sum([jet1.area,jet2.area]) / np.sum([s.area for s in sorted_regions])}")
        # debug_skel_labels(fig_list=debug_figs,**locals()) #appends each mask label fig to debug_figs

        if len(sorted_regions) <2:
            percentile += params.increment
            continue

        jet1, jet2 = sorted_regions[0], sorted_regions[1] 
        is_split = (jet2.area / jet1.area >=0.5) and np.sum([jet1.area,jet2.area]) / np.sum([s.area for s in sorted_regions]) >= 0.9 #second jet at least half the size and combined areas = 90% of full area

        percentile += params.increment
        if percentile > 90:
            raise SkeletonSetupError(
                "Percentile has incremented past 90%, exiting.\nCheck exception for list of debug figs",
                debug_figs=debug_figs)

    # print(f"Percentile = {percentile - params.increment}") #NOTE debug 
    jet_masks = [mask_labels == jet1.label,mask_labels == jet2.label]
    skel_objs = [Skeleton(skeletonize(m)) for m in jet_masks]
    skel_setup = {
        "skel_objs":skel_objs,
        "log_sb":log_sb,
        "mask_labels":mask_labels,
        "sorted_regions": sorted_regions,
        "debug_figs": debug_figs,
        }

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

def path_weigh_endpoints(graph_setup,sim,grid_output,grid_xy,grid_dxy):
    """For every path in the graph, find the path with the highest sb along to endpoint, collapsing graph to one path"""
    graphs = graph_setup["graphs"]
    skel_objs = graph_setup["skel_objs"]
    results = []
    for G, skel_obj in zip(graphs, skel_objs):
        branch_data = summarize(skel_obj, separator='-')  # easier to track if generated again

        inj = sim.get_injection_region(grid_output)
        inj_row = (inj[2].value - grid_xy[1][0]) / grid_dxy[1]   # z -> row
        inj_col = (inj[0].value - grid_xy[0][0]) / grid_dxy[0]   # x -> col

        graph_nodes = np.array(list(G.nodes))
        node_coords = skel_obj.coordinates[graph_nodes.astype(int)]
        dists = np.hypot(node_coords[:, 0] - inj_row, node_coords[:, 1] - inj_col)
        start_node = graph_nodes[np.argmin(dists)]

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

    # spline_data = { #TODO remove dict
    #     "spline_points": spline_points,
    # }
    return spline_points

def skeletonise_sb(sim,grid_output,freq,redshift,sbdata=None,angles = None,params=None):
    """AIO function"""
    params = params or SplineParams()
    if sbdata is None:
        sbdata = prw.load_sb_hdf5(sim=sim,grid_outputs=[grid_output],angles=[angles],freqs=[freq],redshift=redshift)
    entry = sbdata[sim][grid_output][tuple(angles)]
    obs = entry["obs_properties"]
    log_sb = entry["log_sb"]

    grid_dxy = [obs["grid_x"][1] - obs["grid_x"][0],obs["grid_y"][1] - obs["grid_y"][0]]
    grid_mxy = [obs["grid_mx"],obs["grid_my"]]
    grid_xy = [obs["grid_x"],obs["grid_y"]]

    skel_setup = skeleton_setup(grid_xy=grid_xy,log_sb=log_sb,params=params)     
    graph_setup = graph_weigh_edges(skel_setup=skel_setup)
    path_results = path_weigh_endpoints(graph_setup=graph_setup,sim=sim,grid_output=grid_output,grid_xy=grid_xy,grid_dxy=grid_dxy)
    path_data = path_to_coords(path_results=path_results,grid_xy=grid_xy,grid_dxy=grid_dxy)
    spline_points = skeleton_splines(path_data=path_data,grid_mxy=grid_mxy)

    return spline_points

#---Skeleton -> jet path---#
def weigh_skeleton_sb(sbdata_entry,spline_points,search_width,params=None):
    """Weighs the skeletonised sb path by averaging x,y points weighted by sb values for each point in a search width"""
    params = params or SplineParams()
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
        
        #use Gaussian filtering to weigh sb path
        log_sb_smooth = gaussian_filter(np.nan_to_num(log_sb, nan=-np.inf), sigma=1)        
        dist2 = x_local[sb_plane]**2 + z_local[sb_plane]**2
        score = log_sb_smooth[sb_plane] - (1/params.sb_weight) * dist2 / search_width**2
        best = np.argmax(score)
        x_new,z_new = X[sb_plane][best],Z[sb_plane][best]

        sb_pathpoints.append([x_new, z_new])
    sb_pathpoints = np.array(sb_pathpoints)
    return sb_pathpoints    

def iterate_sb_path(sim,sbdata_entry,spline_points,params=None):
    """Iterates the path weighing"""
    search_width = sim.jet.radius.value * 2.5
    cutoff = 0.25 #points 0.5kpc together are dupes
    params = params or SplineParams()
    sb_pathpoints = weigh_skeleton_sb(sbdata_entry=sbdata_entry,spline_points=spline_points,search_width=search_width,params=params) #weigh skeleton path by sb
    for itr in range(params.max_itr):
        d = np.sqrt(np.diff(sb_pathpoints[:,0])**2 + np.diff(sb_pathpoints[:,1])**2)
        keep = np.concatenate([[True], d > cutoff])
        sb_pathpoints = weigh_skeleton_sb(sbdata_entry=sbdata_entry,spline_points=sb_pathpoints[keep],search_width=search_width,params=params)

    return sb_pathpoints

def interpolate_sb_path(sim,sbdata_entry,spline_points,grid_dxy,params=None):
    """re-interpolates the weighed path back into splines"""
    params = params or SplineParams()
    sb_pathpoints = iterate_sb_path(sim=sim,sbdata_entry=sbdata_entry,spline_points=spline_points,params=params)
    d = np.hypot(*np.diff(sb_pathpoints, axis=0).T)
    sb_pathpoints = sb_pathpoints[np.r_[True, d > 1e-6]]
    arc_length = np.sum(np.sqrt(np.diff(sb_pathpoints[:,0])**2 + np.diff(sb_pathpoints[:,1])**2))
    max_resolution = min([grid_dxy[0],grid_dxy[1]])
    n_points = int(arc_length / max_resolution) #determines the resampling resolution
    k = min(3, len(sb_pathpoints) - 1)

    tck, u = splprep([sb_pathpoints[:,0],sb_pathpoints[:,1]],s=n_points * params.smoothing,k=k) #NOTE weighing by the log_sb 
    u2 = np.linspace(u[0],u[-1],n_points) #use the spline array to generate a higher resolution array
    jet_splines = splev(u2,tck)
    jet_splines = np.column_stack(jet_splines) #resampled pathpoints with higher resolution

    return jet_splines

#---Jet splines functions---#
def get_jet_splines_tr(sdata,grid_output,tr_stop=0.2):
    """Fits ridgepoints along the jet length by looking at a tracer slice in the jet radius 
    and weighting the x,z coordinates by the maximum tracer, stops when a window of 5 points has 
    reached a an average of some cutoff value e.g. 0.2. Will also stop if the dot product is (-) 
    (chaing directions). Splines are then fitted to these ridgepoints to resample at a higher resolution 
    (based on the grid resolution) using scipy splprep.

    Args:
        sdata (SimulationData): SimulationData object
        output (int): PLUTO file output
        tr_cut (float): tracer cuttoff value to truncate the data #NOTE not sure if required

    Returns:
        dict: 
            spline_points: x,z array e.g. x = spline_points[:,0] of fitted splines 
            spline_slice_map: x,z indecies that match spline points to grid cells
            ridgepoints: x,z array of the ridgepoints found along the jet
            
    """
    sim_time = grid_output
    particle_data = sdata.load_particle_data(grid_output=(sim_time,),tr_cut=None)

    #was int but that causes rounding errorload_particle_data
    def calc_ridgepoints(jet_side,particle_data,sim_time):
        window = [] #windowed averages
        window_size = 5
        step_size = 0.5 #kpc
        tr_peak = 0
        tr_min = 1 
        n_steps = 0 

        inj_array = sdata.get_injection_region(grid_output=sim_time) #gets the location of the injection region -> converts particle time to simtime
        
        xz_start = [inj_array[0].value, inj_array[2].value] # start at injection region coords
        z_current = xz_start[1]
        x_current = xz_start[0]
        r_jet = sdata.jet.radius.value #jet analytic radius
        ridgepoints = [xz_start.copy()] #initial ridgepoint at inj region

        while True:
            n_steps += 1 

            if n_steps > 5000:
                print("Maximum ridgepoint iterations reached")
                break

            if len(ridgepoints) >=2:
                dx = ridgepoints[-1][0] - ridgepoints[-2][0] #difference btwn last two x ridgepoints
                dz = ridgepoints[-1][1] - ridgepoints[-2][1] #difference btwn last two z ridgepoints
                norm = np.sqrt(dx**2 + dz**2)
                x_current = ridgepoints[-1][0] + (dx/norm)*step_size #update xz position vector with some step size
                z_current = ridgepoints[-1][1] + (dz/norm)*step_size

            else:
                if jet_side == "top":
                    z_current += step_size #initial move, move down 1 kpc
                if jet_side == "bot":
                    z_current -= step_size #initial move, move down 1 kpc

            tr_plane = (np.abs(particle_data["x3"] - z_current) < r_jet) & \
                    (np.abs(particle_data["x1"] - x_current) < r_jet) #tracers at the ridgepoint with plane width of jet radius
            
            if tr_plane.sum() == 0: #if no tracers
                # x_current += step_size
                continue
                # break
            
            tr_val = particle_data['tr1'][tr_plane][np.argmax(particle_data['tr1'][tr_plane])] #current max tracer in plane
            tr_peak = max(tr_peak, tr_val) #maximum recorded tracer
            tr_min = min(tr_min,tr_val)
            window.append(tr_val)

            if len(window) > window_size: #shift window values over to fit new value
                window.pop(0)

            if len(window) == window_size and np.mean(window) < tr_stop: #NOTE once window is full and if min tracer drops below some value, stop
                print(f"Jet head detected @ z= {z_current:.3f} kpc")
                break
            
            x_new = np.average(particle_data["x1"][tr_plane], weights=particle_data['tr1'][tr_plane]) #average all xz points (midpoint) in plane weighted by value of tracer
            z_new = np.average(particle_data["x3"][tr_plane], weights=particle_data['tr1'][tr_plane])

            # Prevent following backflow
            tolerance = 0.1*step_size
            if len(ridgepoints) >= 2 and tr_stop > 1e-3:

                dz_motion = z_new - ridgepoints[-1][1]
                if jet_side == "top" and dz_motion < -tolerance:
                    print(f"Backflow detected @ z = {z_current:.3f} kpc")
                    ridgepoints.pop()
                    break

                if jet_side == "bot" and dz_motion > tolerance:
                    print(f"Backflow detected @ z = {z_current:.3f} kpc")
                    ridgepoints.pop()
                    break
            
            elif len(ridgepoints) >= 2 and tr_stop <= 1e-3:

                prev_step = np.array(ridgepoints[-1]) - np.array(ridgepoints[-2])
                new_step  = np.array([x_new, z_new]) - np.array(ridgepoints[-1])

                prev_norm = np.linalg.norm(prev_step)
                new_norm  = np.linalg.norm(new_step)

                if prev_norm > tolerance and new_norm > tolerance:

                    prev_dir = prev_step / prev_norm
                    new_dir  = new_step / new_norm

                    # Only stop if it reverses relative to the local plume direction
                    if np.dot(prev_dir, new_dir) < -0.2:
                        print(f"Plume reversal detected @ z = {z_current:.3f} kpc")
                        ridgepoints.pop()
                        break

            ridgepoints.append([x_new, z_new])
        ridgepoints = np.array(ridgepoints)
        return ridgepoints

    ridgepoints_top = calc_ridgepoints(jet_side = "top",particle_data=particle_data,sim_time=sim_time)
    ridgepoints_bot = calc_ridgepoints(jet_side = "bot",particle_data=particle_data,sim_time=sim_time)
    top_length = np.sum(np.sqrt(np.diff(ridgepoints_top[:,0])**2 + np.diff(ridgepoints_top[:,1])**2))
    bot_length = np.sum(np.sqrt(np.diff(ridgepoints_bot[:,0])**2 + np.diff(ridgepoints_bot[:,1])**2))


    ridgepoints_all = np.vstack((ridgepoints_bot[::-1],ridgepoints_top[1:]))

    arc_length = np.sum(np.sqrt(np.diff(ridgepoints_all[:,0])**2 + np.diff(ridgepoints_all[:,1])**2)) #length of jet spline
    max_resolution = min([sdata.grid_setup['x1-grid']['dx'],sdata.grid_setup['x2-grid']['dx'],sdata.grid_setup['x3-grid']['dx']])
    n_points = int(arc_length / max_resolution) #determines the resampling resolution

    # remove duplicate / zero-distance ridgepoints
    d = np.sqrt(np.diff(ridgepoints_all[:, 0])**2 + np.diff(ridgepoints_all[:, 1])**2)
    keep = np.concatenate([[True], d > 1e-6])
    ridgepoints_all = ridgepoints_all[keep]
    k = min(3, len(ridgepoints_all) - 1)

    tck, u = splprep([ridgepoints_all[:,0],ridgepoints_all[:,1]],s=0,k=k)
    u2 = np.linspace(u[0],u[-1],n_points) #use the spline array to generate a higher resolution array
    spline_points = splev(u2,tck)
    spline_points = np.column_stack(spline_points) #resampled ridgepoints with higher resolution

    u_inj = u[len(ridgepoints_bot) - 1]
    inj_idx_spline = np.argmin(np.abs(u2 - u_inj))

    x1_data = sdata.load_fluid_data(["ccx"], grid_output=sim_time, load_slice=sdata.quick_slice_1D("yz"))["ccx"]
    x3_data = sdata.load_fluid_data(["ccz"], grid_output=sim_time, load_slice=sdata.quick_slice_1D("xy"))["ccz"]
    # spline_slice_map = (np.searchsorted(x1_data, np.sort(spline_points[:, 0])), np.searchsorted(x3_data, np.sort(spline_points[:, 1])))   # direct numpy index tuple

    x_indices = np.searchsorted(x1_data, spline_points[:, 0])
    z_indices = np.searchsorted(x3_data, spline_points[:, 1])

    # Clip to valid array bounds for safety
    x_indices = np.clip(x_indices, 0, len(x1_data) - 1)
    z_indices = np.clip(z_indices, 0, len(x3_data) - 1)

    spline_slice_map = (x_indices, z_indices)

    returns = {
        "spline_points": spline_points,
        "spline_slice_map": spline_slice_map,
        "ridgepoints": ridgepoints_all,
        "jet_length": [top_length,bot_length],
        "inj_idx": inj_idx_spline,
    }
    return returns

def get_jet_splines_sb(sim,grid_output,angles,freq,redshift,params=None):
    """
    Generates splines along the AGN jet and lobe by starting with a SB image which is skeletonised
    and weighed by its brightness values.

    Args:
        sim (_type_): _description_
        grid_output (_type_): _description_
        angles (_type_): _description_
        freq (_type_): _description_
        redshift (_type_): _description_
        params (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    params = params or SplineParams()
    sbdata_flat = prw.load_sb_hdf5(sim=sim,grid_outputs=[grid_output],angles=[[0,0,0]],freqs=[freq],redshift = redshift)
    skel_splines = skeletonise_sb( #get skeleton splines for unrotated data
        sim=sim,
        grid_output=grid_output,
        freq=freq,
        redshift=redshift,
        sbdata=sbdata_flat,
        angles = [0,0,0],
        params=params
        )

    #load the rotated data
    sbdata = prw.load_sb_hdf5(sim=sim,grid_outputs=[grid_output],angles=[angles],freqs=[freq],redshift = redshift)
    entry = sbdata[sim][grid_output][tuple(angles)]
    log_sb = entry["log_sb"]

    obs = entry["obs_properties"]
    grid_dxy = [obs["grid_x"][1] - obs["grid_x"][0],obs["grid_y"][1] - obs["grid_y"][0]]
    grid_mxy = [obs["grid_mx"],obs["grid_my"]]
    grid_xy = [obs["grid_x"],obs["grid_y"]]

    arr_3d = np.column_stack([skel_splines[:, 0], np.zeros(skel_splines.shape[0]), skel_splines[:, 1]])
    rotated = arr_3d @ obs["rot_mat"].T
    rot_splines = rotated[:, [0, 2]]

    jet_splines = interpolate_sb_path( #interpolate and weigh points using rotated splines
        sim=sim,
        sbdata_entry=entry,
        spline_points=rot_splines,
        grid_dxy=grid_dxy,
        params=params
        )

    #magic to remove path retracing
    radius  = 1.5 * min(grid_dxy)   # weighted points snap to pixel centres, so retrace is ~0-1 pixel
    min_gap = 10                    # points; ~10 pixels of path since resampling is ~1 pixel per point

    tree = cKDTree(jet_splines)
    for i, nbrs in enumerate(tree.query_ball_point(jet_splines, r=radius)):
        if any(i - j > min_gap for j in nbrs):
            jet_splines = jet_splines[:i]     # cut where the path starts re-tracing itself
            break        

    inj = sim.get_injection_region(grid_output)
    inj_3d = np.array([inj[0].value, 0, inj[2].value])   
    inj_xz = (inj_3d @ obs["rot_mat"].T)[[0, 2]]         
    inj_idx = np.argmin(np.linalg.norm(jet_splines - inj_xz, axis=1))

    splines_bot = jet_splines[:inj_idx+1]
    splines_top = jet_splines[inj_idx:][::-1] #reverse so both jets go from inj outwards

    x_idx = np.clip(np.searchsorted(grid_mxy[0], jet_splines[:, 0]),0,log_sb.shape[0] -1)
    y_idx = np.clip(np.searchsorted(grid_mxy[1], jet_splines[:, 1]),0,log_sb.shape[1] -1)
    spline_slice_map = (x_idx, y_idx)

    arc_length = [np.sum(np.sqrt(np.diff(splits[:,0])**2 + np.diff(splits[:,1])**2)) for splits in [splines_top, splines_bot]]
    # print(f"Arc length per jet side = {arc_length}, full = {np.sum(arc_length):.2f}") #NOTE debug

    spline_data = {
        "skel_splines":rot_splines, #return the rotated skeleton splines
        "jet_splines":jet_splines,
        "log_sb_spline":log_sb[y_idx,x_idx],
        "splines_top":splines_top,
        "splines_bot":splines_bot,
        "jet_length":arc_length,
        "inj_idx": inj_idx,
        "spline_slice_map":spline_slice_map,
        "grid_mx": grid_mxy[0][x_idx], 
        "grid_my": grid_mxy[1][y_idx],

    }
    return spline_data

def interp_splines_plutogrid(sim,grid_output,freq,redshift):
    """Re-interpolate the sb path to a finer resolution based on the PLUTO grid, used to get fluid vars along sb path"""
    spline_data = get_jet_splines_sb(sim,grid_output,[0,0,0],freq,redshift)
    jet_splines = spline_data["jet_splines"]   # at SB/observing-grid resolution, already physical (x,z) at 0,0,0

    # reinterpolate the SB path onto PLUTO's finer arc-length resolution
    fluid_resolution = min([sim.grid_setup['x1-grid']['dx'],sim.grid_setup['x2-grid']['dx'],sim.grid_setup['x3-grid']['dx']])
    n_points_fluid = int(np.sum(spline_data["jet_length"]) / fluid_resolution)

    k = min(3, len(jet_splines) - 1)
    tck, u = splprep([jet_splines[:,0], jet_splines[:,1]], s=0, k=k)
    u2 = np.linspace(u[0], u[-1], n_points_fluid)
    jet_splines_fluid = np.column_stack(splev(u2, tck))

    u_inj = u[spline_data["inj_idx"]]
    inj_idx = np.argmin(np.abs(u2 - u_inj))

    # build the spline_slice_map against PLUTO's own grid, not the observing grid
    ccx = sim.load_fluid_data(["ccx"], grid_output=grid_output, load_slice=sim.quick_slice_1D("yz"))["ccx"]
    ccz = sim.load_fluid_data(["ccz"], grid_output=grid_output, load_slice=sim.quick_slice_1D("xy"))["ccz"]
    x_idx = np.clip(np.searchsorted(ccx, jet_splines_fluid[:,0]), 0, len(ccx)-1)
    z_idx = np.clip(np.searchsorted(ccz, jet_splines_fluid[:,1]), 0, len(ccz)-1)
    spline_slice_map = (x_idx, z_idx)

    interp_data = {
        "jet_splines":jet_splines_fluid,
        "spline_slice_map":spline_slice_map,
        "inj_idx": inj_idx,
        "jet_length": spline_data["jet_length"]
    }

    return interp_data
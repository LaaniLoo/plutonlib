

import plutonlib.read_write as prw
# import plutonlib.simulations as ps
import plutonlib.analysis as pa

import numpy as np

from scipy.ndimage import label
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep,splev

from skimage.graph import route_through_array
from skimage.filters import threshold_otsu

import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PathParams:
    sb_percentile: float = 50
    smoothing: float = 5e-3
    length_penalty: float = 1e-4 #path finding length penalty
    exponent: float = 2 #path finding exponent, the lower this value is the more costly dim pixels are

    bright_exp: float = 1 #increase value to weight brightness on path score
    inj_exp: float = 1 #increase value to weight distance from injection region

    block_width: float = 1
    angle_cut: float = 35

def path_setup(sim,grid_output,freq,redshift,angle): #setup required data, POS image
    """Setup all of the sb data required to path trace"""
    #new method using sim obj
    sbdata = sim.load_sb_data(grid_output=grid_output,angles=angle,freq=freq,redshift=redshift) 
    
    grid_dxy = [sbdata["grid_x"][1] - sbdata["grid_x"][0],sbdata["grid_y"][1] - sbdata["grid_y"][0]]
    grid_mxy = [sbdata["grid_mx"],sbdata["grid_my"]]
    # grid_xy = [obs["grid_x"],obs["grid_y"]]

    inj_grid = sim.get_injection_region(grid_output)
    inj_3d = np.array([inj_grid[0].value, 0, inj_grid[2].value])
    inj_xz = (inj_3d @ sbdata["rot_mat"].T)[[0, 2]]

    inj_pxl = [pa.nearest_cell_idx(sbdata["grid_my"], inj_xz[1]),   # z -> rows
        pa.nearest_cell_idx(sbdata["grid_mx"], inj_xz[0])]   # x -> cols
    log_sb = sbdata["log_sb"]
    path_data = {
        "grid_dxy":grid_dxy,
        "grid_mxy":grid_mxy,
        "angle": angle,
        "rot_mat":sbdata["rot_mat"],
        "inj_xz": inj_xz,
        "inj_pxl": inj_pxl,
        "log_sb":log_sb,
    }

    return path_data

def path_cost(path_data,path_params):
    """Create a cost map for the sb data using cost = length_penalty + (1-normalised_sb)^exponent.
    This gives the cost to each sb value form inj point"""
    path_params = path_params or PathParams()


    log_sb = path_data['log_sb']
    is_valid = np.isfinite(log_sb)
    valids = log_sb[is_valid]

    ceil = np.percentile(valids, 99) #set the ceiling at the 99th percentile of sb
    t = threshold_otsu(valids) + 0 #pick the floor using skimage? plus some offset to raise the floor
    flr = threshold_otsu(valids[valids > t]) #run it again
    path_data["ceil"] = ceil
    path_data["flr"] = flr

    sb_clipped = np.clip(log_sb, flr, ceil)
    sb_norm = (sb_clipped - flr) / (ceil - flr) #sb values are clipped and normalised to either be 0 below for or 1 above it
    path_data['sb_norm'] = sb_norm

    cost = path_params.length_penalty + (1-sb_norm)**path_params.exponent #cost of each normalised sb value 
    cost = np.where(is_valid,cost,np.inf) #sets the non-valid values to inf
    path_data['path_cost'] = cost
    return path_data

def path_build_with_endpoints(path_data,path_params):
    """
    Build a lowest cost path for each jet, starting from the injection region using route_through_array. 
    
    The first endpoint is selected from the path_cost map, as the furthest bright pixel away from the 
    injection region. The lowest cost path is then drawn from inj to ep1 
    
    The second endpoint is then selected using the same bright-far score, where it must also be 
    separated from ep1 by some angle_cut (reduces early path branching). A second cost map is defined
    not where path1 is (plus some width) with a shared region around the injection zone. The second
    path is then traced from inj though a bright region (further reducing early branching), 
    then from the bright region to the ep2.
    """

    path_params = path_params or PathParams()
    
    if "sb_norm" not in path_data:
        path_data = path_cost(path_data=path_data,path_params=path_params)
        # raise RuntimeError("Run path_cost before path_waypoints")

    log_sb = path_data["log_sb"]
    inj_z, inj_x = path_data["inj_pxl"]

    low_mask = log_sb > path_data["flr"]
    labels, n_blobs = label(low_mask, structure=np.ones((3, 3)))
    path_data['labels'] = labels

    if labels[inj_z, inj_x] == 0:
        raise ValueError("Injection pixel is below the surface brightness floor")

    source = labels == labels[inj_z, inj_x]
    cut_level = np.percentile(log_sb[source], path_params.sb_percentile)
    source_cut = source & (log_sb > cut_level)
    path_data['source'] = source
    path_data["source_cut"] = source_cut

    pixel_z, pixel_x = np.nonzero(source_cut)
    bright = path_data["sb_norm"][pixel_z, pixel_x] ** path_params.bright_exp  #can scale the brightness weight by power
    dist_inj = np.hypot(pixel_z - inj_z, pixel_x - inj_x) ** path_params.inj_exp
    score = dist_inj * bright #far from inj and bright = good endpoint 

    #---First path---#
    end1 = (pixel_z[score.argmax()], pixel_x[score.argmax()]) #furthest pixel from the injection region
    path1, total_cost = route_through_array(
        path_data["path_cost"], start=(inj_z, inj_x), end=end1,
        fully_connected=True, geometric=True)
    path1 = np.array(path1)

    #---Separate path1 from path2 by using an angle cut---#
    k = min(10, len(path1) - 1)
    dz, dx = pixel_z - inj_z, pixel_x - inj_x
    v = np.array(end1) - np.array([inj_z, inj_x])          # chord injection -> end1
    cos_angle = (dz * v[0] + dx * v[1]) / (np.hypot(dz, dx) * np.hypot(*v) + 1e-9)
    separate = cos_angle < np.cos(np.radians(path_params.angle_cut))   # at least angle_cut away from end1
    
    #---second path endpoints---#
    if separate.any(): #
        score2 = np.where(separate, score, -np.inf)
        end2 = (pixel_z[score2.argmax()], pixel_x[score2.argmax()])
    else:
        end2 = (inj_z, inj_x) #one sided jet case 
 
    #---Second path cost---#
    #create a new cost map based on not path1
    cell = min(path_data["grid_dxy"])
    free = np.ones(path_data["path_cost"].shape, bool)
    free[path1[:, 0], path1[:, 1]] = False
    d_to_path1 = distance_transform_edt(free)  # pixels to the nearest path1 pixel

    zz, xx = np.indices(free.shape)
    inj_radius = 0.1 * len(path1) #NOTE hardcoded free-region shared by path 1 and 2 as 10% length of p1 

    near_inj = np.hypot(zz - inj_z, xx - inj_x) < inj_radius   # leave the start region open

    cost2 = np.where((d_to_path1 < (path_params.block_width/cell)) & ~near_inj, np.inf, path_data["path_cost"])
    cell_ok = d_to_path1[pixel_z, pixel_x] > path_params.block_width / cell   # not inside the blocked corridor
    way_ok = separate & cell_ok

    #---Second path has to pass though a bright region first---#
    #inj -> bright_region, bright_region -> end2
    if way_ok.any():
        w = np.where(way_ok, bright, -np.inf).argmax()
        way = (pixel_z[w], pixel_x[w])
        leg1, _ = route_through_array(cost2, (inj_z, inj_x), way, fully_connected=True, geometric=True)
        leg2, _ = route_through_array(cost2, way, end2, fully_connected=True, geometric=True)
        path2 = np.array(leg1 + leg2[1:])           # drop the duplicated waypoint
    else:
        path2, _ = route_through_array(cost2, (inj_z, inj_x), end2, fully_connected=True, geometric=True)
        path2 = np.array(path2)

    if len(path2) < 4:        # end2 sits right next to the injection: treat as one-sided
        path2 = path2[:1]

    path_data["path1"] = path1
    path_data["path2"] = path2
    path_data["end1"] = end1
    path_data["end2"] = end2

    #--angle between path midpoints?---#
    # v1 = path1[len(path1) // 2] - np.array([inj_z, inj_x])
    # v2 = path2[len(path2) // 2] - np.array([inj_z, inj_x])
    # cos_e = np.dot(v1, v2) / (np.hypot(*v1) * np.hypot(*v2) + 1e-9)
    # angle_between_ends = np.degrees(np.arccos(np.clip(cos_e, -1, 1)))
    # print(angle_between_ends)
    return path_data

def interpolate_path(path,max_resolution,smoothing):
    if len(path) < 2:
        return path

    arc_length = pa.calc_path_length(path)
    n_points = int(arc_length / max_resolution) #determines the resampling resolution
    k = min(3, len(path) - 1)
    tck, u = splprep([path[:,0],path[:,1]],s=n_points * smoothing,k=k) 
    u2 = np.linspace(u[0],u[-1],n_points) #use the spline array to generate a higher resolution array
    splines = splev(u2,tck)
    splines = np.column_stack(splines) #resampled splines with higher resolution
    return splines

#---Jet splines functions---#
def get_jet_splines_sb(sim,grid_output,freq,redshift,angle,path_params):
    """
    Generates splines along the AGN jet and lobe by path tracing a cost-weighted SB image 

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
    path_params = path_params or PathParams()
    setup_data = path_setup(sim,grid_output,freq,redshift,angle=angle)
    path_data = path_build_with_endpoints(path_data=setup_data,path_params=path_params)

    mx,my = path_data["grid_mxy"]
    grid_dxy = path_data["grid_dxy"]
    path1 = path_data['path1']
    path2 = path_data['path2']

    #---Setting top/bottom axis---#
    # check the direction each path is heading relative to each other
    chk = min(5, len(path1) - 1)
    if len(path2) < 2:
        path1_is_top = my[path1[chk, 0]] > inj_xz[1]      # inj_xz[1] is z in kpc
    else:
        chk = min(chk, len(path2) - 1)
        path1_is_top = my[path1[chk, 0]] > my[path2[chk, 0]]
    top, bot = (path1, path2) if path1_is_top else (path2, path1)

    top = np.column_stack([mx[top[:, 1]], my[top[:, 0]]]) # x = [:,0], z = [:,1]
    bot = np.column_stack([mx[bot[:, 1]], my[bot[:, 0]]])

    #---Interpolation---#
    max_resolution = min([grid_dxy[0],grid_dxy[1]])
    splines_top = interpolate_path(top,max_resolution,path_params.smoothing)
    splines_bot = interpolate_path(bot,max_resolution,path_params.smoothing)
  
    inj_xz = path_data['inj_xz']
    inj_idx = len(splines_bot) - 1  #back to x,z
    top[0] = inj_xz #set the injection region as the 0th point
    bot[0] = inj_xz

    jet_splines = np.concatenate([splines_bot[::-1], splines_top]) #bot reversed to restore path order
    x_idx = pa.nearest_cell_idx(mx, jet_splines[:, 0])
    y_idx = pa.nearest_cell_idx(my, jet_splines[:, 1])
    spline_slice_map = (x_idx, y_idx)
    arc_length = [pa.calc_path_length(splits) for splits in [splines_top, splines_bot]]

    spline_data = {
        "jet_splines":jet_splines,
        "splines_top":splines_top,
        "splines_bot":splines_bot,
        "log_sb_spline": path_data["log_sb"][y_idx,x_idx],
        "jet_length":np.asarray(arc_length), #NOTE is top -> bottom
        "path_data":path_data,
        "inj_xz": inj_xz,
        "inj_idx": inj_idx,
        "spline_slice_map":spline_slice_map,
        "grid_mx": mx[x_idx], 
        "grid_my": my[y_idx],
    }
    return spline_data

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
    particle_data = sdata.load_particle_data(grid_output=(sim_time,),tr_cut=None,var_choice=['tr1','x1','x3'])

    #was int but that causes rounding errorload_particle_data
    def calc_ridgepoints(jet_side,particle_data,sim_time):
        window = [] #windowed averages
        window_size = 5
        step_size = 0.8 #kpc
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
    
    top_length = pa.calc_path_length(ridgepoints_top)
    bot_length = pa.calc_path_length(ridgepoints_bot)

    ridgepoints_all = np.vstack((ridgepoints_bot[::-1],ridgepoints_top[1:]))

    max_resolution = min([sdata.grid_setup['x1-grid']['dx'],sdata.grid_setup['x2-grid']['dx'],sdata.grid_setup['x3-grid']['dx']])
    n_points = int((top_length + bot_length) / max_resolution) #determines the resampling resolution

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

    x_indices = pa.nearest_cell_idx(x1_data, spline_points[:, 0])
    z_indices = pa.nearest_cell_idx(x3_data, spline_points[:, 1])

    spline_slice_map = (x_indices, z_indices)

    returns = {
        "spline_points": spline_points,
        "spline_slice_map": spline_slice_map,
        "ridgepoints": ridgepoints_all,
        "jet_length": [top_length,bot_length],
        "inj_idx": inj_idx_spline,
    }
    return returns

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
    # x_idx = np.clip(np.searchsorted(ccx, jet_splines_fluid[:,0]), 0, len(ccx)-1)
    # z_idx = np.clip(np.searchsorted(ccz, jet_splines_fluid[:,1]), 0, len(ccz)-1)
    x_idx = pa.nearest_cell_idx(ccx, jet_splines_fluid[:,0])
    z_idx = pa.nearest_cell_idx(ccz, jet_splines_fluid[:,1])

    spline_slice_map = (x_idx, z_idx)

    interp_data = {
        "jet_splines":jet_splines_fluid,
        "spline_slice_map":spline_slice_map,
        "inj_idx": inj_idx,
        "jet_length": spline_data["jet_length"]
    }

    return interp_data

def point_at_arclength(path, arc_target):
    """Find the xz point along the spline data arc"""
    path_len = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
    dists = np.concatenate([[0], np.cumsum(path_len)]) #distance from each previous point

    x = np.interp(arc_target, dists, path[:, 0])
    z = np.interp(arc_target, dists, path[:, 1])
    return np.array([x, z])
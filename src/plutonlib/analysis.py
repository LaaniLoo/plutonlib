import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl
import plutonlib.simulations as ps
import plutonlib.plot as pp
from plutonlib.colours import pcolours 

import plutokore.radio as pk_radio

import numpy as np
import scipy as sp
from scipy import stats
from scipy import constants

from astropy import units as u
from astropy import cosmology as cosmo  # Astropy cosmology
import astropy.constants as astro_const

from scipy.spatial.transform import Rotation
from astropy.convolution import convolve, Gaussian2DKernel  # Astropy convolutions

import matplotlib.pyplot as plt

from collections import defaultdict 
from IPython.display import display, Latex
import inspect

def find_nearest(array, value):
    """Find closes value in array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return {"idx":idx, "value": array[idx]}

def get_grid_idx(value,sdata,coord):
    """
    Method for finding array index of a specific value without loading full PLUTO grid arrays,
    note that this only works if the value and the pluto ini grid are in the same units, and if the grid patch
    containing `value` resides in a uniform patch. 

    value: value to find grid idx
    sdata: SimulationData object
    coord: dimension to find value for e.g. "x1","x2" or "x3"
    """
    coord = pu.map_coord_name(coord) #this lets you put in ncx and x1 etc 
    grid = sdata.grid_setup[f"{coord}-grid"]
    starts = grid["start"]
    ends = grid["end"]
    patch_cells = grid["patch_cells"]

    for patch in range(grid["n_patches"]): #loop across each grid patch to find which one contains origin
        cur_patch = f"[{starts[patch]} {patch_cells[patch]} {grid['type'][patch][0]} {ends[patch]}]"
        start = starts[patch]
        end = ends[patch]
        
        if starts[patch] <= value <= ends[patch]: #find the patch where value is located            
            if grid["type"][patch] == "uniform":
                start_disp = (value - start) #distance from start of patch 
                patch_length = (end-start) #total length of patch 
                patch_idx = (start_disp / patch_length)*patch_cells[patch] #fraction of distance to total patch scaled by number of cells gives the grid cell of value.
                idx = int(sum(patch_cells[:patch])+patch_idx) # since there can be multiple patches, need to add previous patch cells to above calculation 
                return idx

            else:
                raise NotImplementedError(f"{coord} = {value} lies in stretched patch: {cur_patch}, need to numerically solve the grid stretching ratio")
                #NOTE would  
            
    raise ValueError(f"Value {coord} = {value} is outside the grid extent for {cur_patch}")

def calc_var_prof(sdata,sel_coord,value_2D: float = None,value_1D: dict = None,**kwargs):
    """
    Calculates the array profile for two cases e.g. for sel_coord = "x1", value = 0:
        slice_1D : (slice(None, None, None), 600, 733) 
            -> 1D slice along x1 at x2_mid, x3_mid
            -> Shape: (n_x1,)

        slice_2D: (800, slice(None, None, None), slice(None, None, None)) 
            -> 2D plane in x2,x3 sliced in x1 at x1_mid
            -> Shape: (n_x2, n_x3)

    Parameters:
    sdata: 
        SimulationData object
    sel_coord: 
        Selected coordinate to slice in or about
    value_2D (float): 
        Value to slice at for 2D slice, e.g x1 = 20kpc, defaults to value_2D = 0 (midpoint)
    value_1D (dict): 
        Used to make a slice at value for different coord to sel_coord for 1D slice, 
        e.g. slice at x1 = 20kpc and x2 = 0kpc -> value_1D = {"x1":20,"x2":0} 

    """
    sel_coord = pu.map_coord_name(sel_coord) #strips array_type from sel_coord e.g. ncx -> x1
    x,y,z = "x1","x2","x3" 

    idx_map = {x:None, y:None,z:None}   
    for coord in idx_map.keys():
        if value_1D and coord in value_1D:
            idx_map[coord] = get_grid_idx(value=value_1D[coord],sdata=sdata,coord=coord) 
        
        elif value_2D is not None and coord == sel_coord:
            idx_map[coord] = get_grid_idx(value=value_2D,sdata=sdata,coord=coord)
        
        else:
            idx_map[coord] = get_grid_idx(value=0,sdata=sdata,coord=coord)




    # --- Define slicing maps ---
    if sdata.grid_ndim > 2:
        slice_map_1D = {
            x : (slice(None), idx_map[y], idx_map[z]),
            y: (idx_map[x], slice(None), idx_map[z]),
            z: (idx_map[x], idx_map[y], slice(None)),
        }

        slice_map_2D = {
            x: (idx_map[x], slice(None), slice(None)),
            y: (slice(None), idx_map[y], slice(None)),
            z: (slice(None), slice(None), idx_map[z]),
        }

    else:
        slice_map_1D = {
            x: (slice(None), idx_map[y]),
            y: (idx_map[x], slice(None)),
        }
        slice_map_2D = None

    slice_1D = slice_map_1D[sel_coord]
    slice_2D = slice_map_2D[sel_coord] if sdata.grid_ndim > 2 else None

    return {
        "slice_1D": slice_1D,
        "slice_2D": slice_2D,
    }

#---Praise Setup---#
def setup_obs_properties_praise(sdata,redshift,angle,plane="xz"):
    #--------------------------------------------------------#
    #            Set up the observing properties             #  
    #--------------------------------------------------------#

    pixel_size = 1 * u.arcsec
    beam_fwhm = 3 * u.arcsec  # generally this should be ~3x bigger than pixel size

    arcsec2kpc = cosmo.Planck15.kpc_proper_per_arcmin(redshift).to(u.kpc/u.arcsec)
    pixel_size_kpc = (pixel_size * arcsec2kpc).to(u.kpc)
    beam_kpc = (beam_fwhm * arcsec2kpc).to(u.kpc)

    # beam equations
    fwhm_to_sigma = 1 / (8 * np.log(2)) ** 0.5
    beam_sigma = beam_fwhm * fwhm_to_sigma
    omega_beam = 2 * np.pi * beam_sigma ** 2  # Area for a circular 2D gaussian

    # part_ind = part_outputs   

    ## integration grid cell size
    grid_spacing = pixel_size_kpc.value * 1.0 #0th redshift?

    # set a min and max of the grid
    grid_setup = sdata.grid_setup
    plane_grid_map = {
    "xy" : [grid_setup["x1-grid"]["grid_extent"],grid_setup["x2-grid"]["grid_extent"]],
    "xz" : [grid_setup["x1-grid"]["grid_extent"],grid_setup["x3-grid"]["grid_extent"]],
    "yz" : [grid_setup["x2-grid"]["grid_extent"],grid_setup["x3-grid"]["grid_extent"]],
    }
    grid_lim_x = plane_grid_map[plane][0] #e.g if xz-plane 0th element is x limits e.g. (-80,80)
    grid_lim_y = plane_grid_map[plane][1]

    # ray properties
    delta_r = 0.3         
    ray_depth_min = -200
    ray_depth_max = 200

    # specify our grid
    grid_x = np.arange(grid_lim_x[0], grid_lim_x[1], grid_spacing)
    grid_y = np.arange(grid_lim_y[0], grid_lim_y[1], grid_spacing)
    grid_mx = np.diff(grid_x) * 0.5 + grid_x[:-1]
    grid_my = np.diff(grid_y) * 0.5 + grid_y[:-1]

    # grid_mx = test.fluid_data("ccx",load_slice=(0,0,slice(None)))["ccx"]
    # grid_my = test.fluid_data("ccz",load_slice=(0,0,slice(None)))["ccz"]

    rot_mat = Rotation.from_euler("X", angle, degrees=True).as_matrix()
    gaussian_sigma = (beam_kpc.value * fwhm_to_sigma) / grid_spacing
    gaussian_kernel = Gaussian2DKernel(gaussian_sigma)  # create our gaussian convlution kernel

    returns = {
        "grid_x":grid_x,
        "grid_y":grid_y,
        "grid_mx":grid_mx,
        "grid_my":grid_my,
        "delta_r":delta_r,
        "omega_beam":omega_beam,
        "ray_depth_min":ray_depth_min,
        "ray_depth_max":ray_depth_max,
        "rot_mat":rot_mat,
        "gaussian_kernel": gaussian_kernel
    }

    return returns

def calc_surface_brightness_praise(sdata,freqs=[1.4],redshift=0.05,particle_outputs="last",angle=0,plane="xz"):
    """
    Calculates the particle emssion using PRAiSE (pk_radio) under adiabatic, sychrotron and inverse compton losses.
    Surface brightness is then calculated by integrating emissivity with raytracing.    
    :param sdata: Description
    :param freqs: Description
    :param redshift: Description
    :param particle_outputs: Description
    :param angle: Description
    """
    pk_sim = sdata.to_plutokore() #convert SimulationData object to plutokore PlutoSimulation
    particle_outputs = [pl.get_particle_outputs(sdata.wdir)] if particle_outputs == "last" else particle_outputs
    particle_spacing = sdata.part_to_simtime(particle_outputs[0]) / particle_outputs[0]
    s=2.2   # for injection spectral index alpha=-0.55. NOTE: the PRAiSE default is also 2.2

    particle_data = sdata.load_particle_data()
    particle_times = sdata.load_particle_data()["particle_times"]
    particle_emis = pk_radio.praise2.praise(
        sim=pk_sim,
        max_output=particle_outputs[-1],
        emit_outputs=particle_outputs, #calc emission for these outputs
        output_system="particles", #idx in grid or particles
        freqs=(freqs*u.GHz).si.value, #list of GHz freqs to calc for 
        part_data=particle_data, 
        part_times=particle_times,
        particle_spacing=particle_spacing * u.Myr, #particle outputs per Myr 
        redshift=redshift, # 0.05 redshift
        lst_index=2, #last shock time index, lowest to highest str -> 2 = only strong shocks
        losses=4, #idx to include losses: adiab, synch, inverse compton losses -> 4 = include all losses
    )

    # Remove the NaNs for each output 
    # create an empty list of particle coordinates, velocities and the nan masks which will all be different for each simulation output. 
    all_part_coords = []  
    all_vel_vec = []
    all_nan_masks = []
    sb_arr = []
    integ_emis = []
    for i in  range(0, len(particle_outputs), 1):    
        part_ind = particle_outputs[i]
        nan_mask = ~np.isnan(particle_data["id"][:, part_ind]) 
        all_nan_masks.append(nan_mask)
        
        part_coords = np.c_[
            (
                particle_data["x1"][:, part_ind][nan_mask],
                particle_data["x2"][:, part_ind][nan_mask],
                particle_data["x3"][:, part_ind][nan_mask],
            )
        ]
        all_part_coords.append(part_coords)
        
        # set up particle velocities
        vel_vec = np.c_[
            particle_data["vx1"][:, part_ind][nan_mask],
            particle_data["vx2"][:, part_ind][nan_mask],
            particle_data["vx3"][:, part_ind][nan_mask],
        ]
        all_vel_vec.append(vel_vec)

        obs_properties = setup_obs_properties_praise(sdata=sdata,redshift=redshift,angle=angle,plane=plane)
        integrated_emissivity = pk_radio.raytracing.raytrace_particles_multiple_freq(
            grid=(obs_properties["grid_mx"], obs_properties["grid_my"]),
            ray_depth_lim=(obs_properties["ray_depth_min"], obs_properties["ray_depth_max"]),
            rot_mat=obs_properties["rot_mat"],
            s= s,
            delta_r=obs_properties["delta_r"],
            coords=all_part_coords[i],
            dist_upper_bound = 2,      # <-- NOTE the praise default is 20... this will mess up your results. 2 is better
            vel_vec=all_vel_vec[i],
            obs_normal=[0, 1, 0],
            # particle_emissivities=part_emis['full'][i]['emis'][all_nan_masks[i],:,:]  #only the non-nan particles, all frequencies, and the ith snapshot
            particle_emissivities=particle_emis['full'][i]['emis'][all_nan_masks[i],:,0] #only the non-nan particles, all frequencies, and the ith snapshot 
        )
        integ_emis.append(integrated_emissivity)

        # we multiple our integrated emissivity by kpc (to account for integration), and divide by 4pi to account for solid angle
        surface_brightness = (integrated_emissivity * u.kpc / (4 * np.pi)).to(u.mJy / u.beam, equivalencies=u.beam_angular_area(obs_properties["omega_beam"]))
        sb_arr.append(surface_brightness)
    
    for freq_ind in range(len(freqs)):
        sb_arr[i][:,:,freq_ind] = np.nan_to_num(sb_arr[i][:,:,freq_ind], copy=True, nan=0.0, posinf=0.0, neginf=0.0) # get rid of NaNs (replace with zero)

    return sb_arr

def plot_surface_brightness(sdata,freqs=[1.4],redshift=0.05,particle_outputs="last",angle=0,plane="xz",**kwargs):
    # if pdata is None:
    # loop over all outputs and all frequencies
    particle_outputs = [pl.get_particle_outputs(sdata.wdir)] if particle_outputs == "last" else particle_outputs
    pdata = pp.PlotData(var_choice=freqs,plane=plane,load_outputs = particle_outputs,**kwargs)
    pdata.plot_idx = 0
    pdata.axes, pdata.fig = pp.subplot_base(sdata=sdata,pdata=pdata,load_outputs=particle_outputs)
    extras = pp.plot_extras(sdata,pdata)

    obs_properties = setup_obs_properties_praise(sdata=sdata,redshift=redshift,angle=angle,plane=plane)
    sb_arr = calc_surface_brightness_praise(
        sdata=sdata,
        freqs=freqs,
        redshift=redshift,
        particle_outputs=particle_outputs,
        angle=angle,
        plane=plane,
    )
    for i in range(0, len(particle_outputs), 1):
        for freq_ind, freq in enumerate(freqs):
            pdata.output = particle_outputs[i]

            # sb_arr[i][:,:,freq_ind] = np.nan_to_num(sb_arr[i][:,:,freq_ind], copy=True, nan=0.0, posinf=0.0, neginf=0.0) # get rid of NaNs (replace with zero)
            freq_sb = convolve(sb_arr[i][:, :, freq_ind].to(u.mJy / u.beam), obs_properties["gaussian_kernel"], boundary='extend') * (u.mJy / u.beam) #convolve the surface brightness
            freq_sb[freq_sb == 0] = np.nan # replace 0 SB areas with nan
            max_sb_lim = np.log10(np.nanpercentile(freq_sb, 99).value) # calculate sensible limits
            min_sb_lim = max_sb_lim - 1
            sb_contours = np.linspace(min_sb_lim, max_sb_lim, 3) # calculate contours

            # a = np.log10(freq_sb.value.T) # replace 0 SB areas with suitably low value
            # a[np.isnan(a)] = -100
            ax = pdata.axes[pdata.plot_idx]
    
            im = ax.pcolormesh(obs_properties["grid_x"], obs_properties["grid_y"], np.log10(freq_sb.value.T), vmin= min_sb_lim, vmax=max_sb_lim)
            cbar = pdata.fig.colorbar(im,ax=ax,fraction = 0.05)
            cbar.set_label(f"$\log_{{10}}$(SB [$\mathrm{{mJy \; beam^{{-1}}}}$])")

            ax.contour(obs_properties["grid_mx"], obs_properties["grid_my"], np.log10(freq_sb.value.T), levels=sb_contours, colors='white')

            ax.set_aspect("equal")
            ax.set_title(f"Surface Brightness [{freq}$GHz$] ({sdata.run_name})")
            ax.set_xlabel(extras["xy_labels"][pdata.coord_choice[0]])
            ax.set_ylabel(extras["xy_labels"][pdata.coord_choice[1]])
            pdata.plot_idx += 1



            time_str = f"{sdata.part_to_simtime(pdata.output):.1f} Myr"
            ax.annotate(
                f'{time_str}',
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8
                )
            )

            ax.annotate(
                f'angle = ${angle}^\circ$',
                xy=(0.05, 0.90),
                xycoords='axes fraction',
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8
                )
            )


#---Plot Grid structure---#
def plot_xz_grid_structure(sdata,pdata=None, d_file=None, n_lines=10, **kwargs):
    """
    Visualize the stretched grid structure in XZ plane
    """
    if sdata.load_slice is None or sdata.slice_shape == "slice_1D":
        raise ValueError(f"SimulationData load_slice is None or 1D ({sdata.load_slice}), use a 2D slice to plot")
    
    if pdata is None:
        pdata = pp.PlotData()

    if d_file is None:
        d_file = sdata.d_files[0]
    
    # Get coordinate arrays using get_grid_data()
    x1 = sdata.get_grid_data(d_file)["x1"]  # X coordinates
    x2 = sdata.get_grid_data(d_file)["x2"]  # Y coordinates  
    x3 = sdata.get_grid_data(d_file)["x3"]  # Z coordinates
    
    pdata.fig, pdata.axes = plt.subplots(1, 1, figsize=(5, 5))
    # XZ plane view - take middle y-plane
    y_slice = x1.shape[1] // 2  # Middle y-plane
    
    # Plot x-grid lines (constant x1)
    for i in range(0, x1.shape[0], max(1, x1.shape[0] // n_lines)):
        pdata.axes.plot(x1[i, :], x3[i, :], 'k-', alpha=0.6, linewidth=0.8)

    # Plot z-grid lines (constant x3)
    for k in range(0, x1.shape[1], max(1, x1.shape[1] // n_lines)):
        pdata.axes.plot(x1[:, k], x3[:, k], 'k-', alpha=0.6, linewidth=0.8)

    pp.plot_label(sdata,pdata,idx=0,no_title = True)
    pdata.axes.set_title(f'{sdata.run_name} XZ Grid Structure')
    pdata.axes.set_aspect("auto")
    pdata.axes.grid(True, alpha=0.3)
    
    pp.plot_save(sdata,pdata,**kwargs)


#---Peak Finding---#
def peak_findr(sel_coord,sel_var,sdata,**kwargs):
    """
    Calculates the max values of an array and their location, e.g. use to find max values of x2 for vx2 to find jet radius
    """

    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    radius = []
    peak_info = []
    peak_var = []
    locs = []

    var_profile = calc_var_prof(sdata,sel_coord,**kwargs)["slice_1D"]
    for d_file in sdata.d_files:
        var = sdata.get_vars(d_file)


        var_sliced = var[sel_var][var_profile]
        max_loc = np.where(var_sliced == np.max(var_sliced)) #index location of max variable val

        locs.append(max_loc[0])

        coord_array = var[sel_coord][var_profile] if sdata.grid_ndim >2 else var[sel_coord]    

        peak_info.append(f"{d_file} Radius: {coord_array[max_loc][0]:.2e} m, {sel_var}: {var_sliced[max_loc][0]:.2e}")
        peak_var.append(var_sliced[max_loc][0])
        radius.append(coord_array[max_loc][0])

    return {"peak_info": peak_info,"radius": radius, "peak_var": peak_var,"locs": locs } 

def graph_peaks(sel_coord,sel_var,sdata,**kwargs): #TODO Put in peak findr 
    """Follows a similar process to peak_findr() except it uses scipy signal peak finding, good for visual representation"""
    
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    coord_units = (sdata.get_var_info(sel_coord)["usr_uv"]).to_string('latex')
    var_units = (sdata.get_var_info(sel_var)["usr_uv"]).to_string('latex')

    var_peak_idx = defaultdict(list)
    peak_info = []
    peak_vars = []
    peak_coords = []
    
    var_profile = calc_var_prof(sdata,sel_coord)["slice_1D"]
    for d_file in sdata.d_files: #find graphical peaks across all data files

        var = sdata.get_vars(d_file)[sel_var]
        coord =  sdata.get_vars(d_file)[sel_coord]
        var_sliced = var[var_profile]

        var_peak_idx[d_file], _ = sp.signal.find_peaks(var_sliced)

        if np.any(var_peak_idx[d_file]):  # Only print if peaks exist 
            peak_vars.append(var_sliced[var_peak_idx[d_file][-1]])
            peak_coords.append(coord[var_peak_idx[d_file][-1]])

            #NOTE not sure what 
            peak_var = var_sliced[var_peak_idx[d_file][-1]]
            peak_coord = coord[var_peak_idx[d_file][-1]]

            peak_info.append(f"{d_file}: {sel_coord} = {peak_coord:.2e} {coord_units} {sel_var} = {peak_var:.2e} {var_units}")

        else:
            print(f"No peaks found in {d_file}")

    return {"var_peak_idx": var_peak_idx,"var_sliced": var_sliced,"peak_coords": peak_coords, "peak_vars": peak_vars, "peak_info":peak_info}

def all_graph_peaks(sel_coord,sel_var,sdata,**kwargs): #NOTE used for plotting same alg as peak_findr
    """A version of peak_findr() used for plotting, but includes all peak values """

    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    var_peak_idx = defaultdict(list)
    peak_vars = []
    peak_coords = []
    
    var = sdata.get_vars(sdata.d_last)[sel_var]
    coord = sdata.get_vars(sdata.d_last)[sel_coord]

    ndim = sdata.grid_ndim
    var_profile = calc_var_prof(sdata,sel_coord,**kwargs)["slice_1D"]
    var_sliced = var[var_profile]
    coord = coord[var_profile] if sdata.grid_ndim >2 else coord

    var_peak_idx, _ = sp.signal.find_peaks(var_sliced)
    var_trough_idx, _ = sp.signal.find_peaks(-var_sliced) 

    thresh = 1e-10 #threshold for trough finding

    if np.any(var_peak_idx):  # Only print if peaks exist 
        pvars_data = var_sliced[var_peak_idx]
        pcoords_data = coord[var_peak_idx]

        peak_vars = pvars_data[(pvars_data > 0) & (np.abs(pvars_data) > thresh)]
        peak_coords = pcoords_data[(pvars_data > 0) & (np.abs(pvars_data) > thresh)]

    if np.any(var_trough_idx):  # Only print if peaks exist and is <0
        tvars_data = var_sliced[var_trough_idx]
        tcoords_data = coord[var_trough_idx]

        trough_vars = tvars_data[(tvars_data < 0) & (np.abs(tvars_data) > thresh)]
        trough_coords = tcoords_data[(tvars_data < 0) & (np.abs(tvars_data) > thresh)]


    else:
        print(f"No peaks found in {sdata.d_last}")

    returns = {
        "var_sliced": var_sliced,
        "var_peak_idx": var_peak_idx,
        "var_trough_idx": var_trough_idx,
        "var_profile": var_profile,
        "peak_coords": peak_coords,
        "peak_vars": peak_vars,
        "trough_vars": trough_vars,
        "trough_coords": trough_coords,
        }
    
    return returns

def plot_peaks(sel_coord,sel_var,sdata,**kwargs): #TODO doesn't work for stelar wind rho
    """Plots the peaks found by all_graph_peaks()"""

    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    peak_data = all_graph_peaks(sel_coord,sel_var,sdata=sdata,**kwargs) #NOTE all goes wrong with graph_peaks, something to do with d_files
    if sdata.grid_ndim >2: #3D array needs sliced coords
        plot_coords = sdata.get_vars(sdata.d_last)[sel_coord][peak_data["var_profile"]]
    elif sdata.grid_ndim <=2:
        sdata.get_vars(sdata.d_last)[sel_coord]
    var_sliced = peak_data["var_sliced"]
    peak_coords = peak_data["peak_coords"]
    peak_vars = peak_data["peak_vars"]

    is_log = sel_var in ('rho','prs')
    base_plot_data = np.log10(var_sliced) if is_log else var_sliced
    peak_plot_data = np.log10(peak_vars) if is_log else peak_vars

    xlab = f"{sdata.get_var_info(sel_coord)['coord_name']} [{sdata.get_var_info(sel_coord)['usr_uv']}]"
    ylab = f"log10({sdata.get_var_info(sel_var)['var_name']}) [{sdata.get_var_info(sel_var)['usr_uv']}]" if is_log else f"{sdata.get_var_info(sel_var)['var_name']} [{sdata.get_var_info(sel_var)['usr_uv']}]"
    label = f"Peak {ylab}"
    title = f"{sdata.sim_type} Peak {ylab} Across {xlab}"


    f,a = plt.subplots(figsize = (7,7))
    a.plot(plot_coords,base_plot_data) # base plot
    a.plot(peak_coords,peak_plot_data,"x",label= label)
    a.legend()
    a.set_xlabel(xlab)
    a.set_ylabel(ylab)
    a.set_title(title)

    if 'xlim' in kwargs: # xlim kwarg to change x limits
        a.set_xlim(kwargs['xlim']) 

    if 'ylim' in kwargs: # xlim kwarg to change x limits
        a.set_ylim(kwargs['ylim']) 

    pdata = pp.PlotData() # assigning pdata fig object
    pdata.fig = plt.gcf()  #not sure but it works
    pp.plot_save(sdata,pdata,custom=1)

def plot_troughs(sel_coord,sel_var,sdata,**kwargs): #TODO doesn't work for stelar wind rho
    """Plots the peaks found by all_graph_peaks()"""

    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    vars_last = sdata.get_vars(sdata.d_last)
    peak_data = all_graph_peaks(sel_coord,sel_var,sdata=sdata,**kwargs) #NOTE all goes wrong with graph_peaks, something to do with d_files
    var_sliced = peak_data["var_sliced"]
    trough_coords = peak_data["trough_coords"]
    trough_vars = peak_data["trough_vars"]

    is_log = sel_var in ('rho','prs')
    base_plot_data = np.log10(var_sliced) if is_log else var_sliced
    peak_plot_data = np.log10(trough_vars) if is_log else trough_vars

    xlab = f"{sdata.get_var_info(sel_coord)['coord_name']} [{sdata.get_var_info(sel_coord)['usr_uv']}]"
    ylab = f"log10({sdata.get_var_info(sel_var)['var_name']}) [{sdata.get_var_info(sel_var)['usr_uv']}]" if is_log else f"{sdata.get_var_info(sel_var)['var_name']} [{sdata.get_var_info(sel_var)['usr_uv']}]"
    label = f"trough {ylab}"
    title = f"{sdata.sim_type} trough {ylab} Across {xlab}"


    f,a = plt.subplots(figsize = (7,7))
    a.plot(vars_last[sel_coord],base_plot_data) # base plot
    a.plot(trough_coords,peak_plot_data,"x",label= label)
    a.legend()
    a.set_xlabel(xlab)
    a.set_ylabel(ylab)
    a.set_title(title)

    if 'xlim' in kwargs: # xlim kwarg to change x limits
        a.set_xlim(kwargs['xlim']) 

    if 'ylim' in kwargs: # xlim kwarg to change x limits
        a.set_ylim(kwargs['ylim']) 

    pdata = pp.PlotData() # assigning pdata fig object
    pdata.fig = plt.gcf()  #not sure but it works
    pp.plot_save(sdata,pdata,custom=1)


# ---Analysis plots---#
def get_jet_length_dim(sdata):
    """Gets the array with the longest grid size, used for Jet length as safety net"""
    coords = sdata.get_coords() #loads all coords at d_last
    longest = max([("x1", len(coords["x1"])), ("x2", len(coords["x2"])), ("x3", len(coords["x3"]))], key=lambda x: x[1])[0]

    return longest

def tprog_phelper(sel_coord,r,sdata,type,**kwargs):
    """ Helper function for plot_time_prog, handles plotting assignment"""
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)


    var_info = sdata.get_var_info(sel_coord)

    xlab = f"sim_time [{sdata.get_var_info('sim_time')['usr_uv']}]"
    ylab = f"{var_info['coord_name']}-Radius [{var_info['usr_uv']}]"
    title = f"{sdata.sim_type} {ylab} across {xlab}"

    t_values = []
    for data in sdata.d_files:
        t_values.append(sdata.get_vars(data)["sim_time"])

    #Legend assignment based on sim_time
    if sdata.grid_setup["dimensions"] == 2:
        longest_array = get_jet_length_dim(sdata) # used to avoid confusion btwn jet length and width
        measurement = "length" if sel_coord == longest_array else "width"
        legend_base = f"{sdata.sim_type} {var_info['coord_name']}-Radius ({measurement})" #names legend based on width or length

    elif sdata.grid_setup["dimensions"] == 3:
        legend_base = f"{sdata.sim_type} {var_info['coord_name']}-Radius"
    
    if type == "def": #default type plot
        f,a = plt.subplots()

        print(t_values)
        print(r)

        a.plot(t_values, r, color = "darkorchid") # base plot
        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
        a.set_title(title)

        a.legend([legend_base])
        a.text(0.05, 0.8,f"R = {r[-1]:.2e} m", transform=a.transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 

        a.plot(t_values,r,"x", label = sdata.d_files)
        for i, d_file in enumerate(sdata.d_files):
            a.annotate(d_file.strip("data_"), (t_values[i], r[i]), textcoords="offset points", xytext=(1,1), ha='right')

        pdata = pp.PlotData()
        pdata.fig = plt.gcf()  
        pp.plot_save(sdata,pdata,custom=1,**kwargs)
    
        return None
    
    elif type == "log": #log-log type plot
        d_files = sdata.d_files[1:]
        t_values = np.log10(t_values[1:])
        r = np.log10(r[1:])

        slope, intercept, r_value, p_value, std_err = stats.linregress(t_values, r)
        eqn = f'$R_{var_info["coord_name"]} \\propto t^{{{slope:.2f}}} \\pm {std_err:.2f} \\; [m]$'
        display(Latex(eqn))

        f,a = plt.subplots()
        a.plot(t_values, r, color = "orange") # base plot

        # r_ideal = (t_values ** 0.6)
        # a.plot(t_values, r_ideal, color="hotpink")

        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
        a.set_title(title)

        a.legend([legend_base,r'Ideal: $t^{0.6}$'])
        a.text(0.05, 0.8, eqn, transform=a.transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 
        # a.text(slope)
        a.plot(t_values,r,"x", label = d_files)
        for i, d_file in enumerate(d_files):
            a.annotate(d_file.strip("data_"), (t_values[i], r[i]), textcoords="offset points", xytext=(1,1), ha='right')

        pdata = pp.PlotData()
        pdata.fig = plt.gcf()  
        pp.plot_save(sdata,pdata,custom=1,**kwargs)

        return slope

def plot_time_prog(sel_coord,sdata,type="def",**kwargs): #NOTE removed sel_var as it shouldn't matter unless stellar wind
    """
    Plots the calculated radius of a sim, e.g. jet radius across sim_time
    * Jet: graph_peaks() for plotting, calculates the peak dens at d_file -> end of jet -> assigns radius
    * Stellar_Wind: peak_finder() for plotting
    """
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)


    r = []


    if sdata.grid_setup["dimensions"] == 2:
        sel_var = "rho" #graph peaks doesn't care which var is used, a peak is a peak?
        peak_data = graph_peaks(sel_coord,sel_var,sdata) 
        var_peak_idx = peak_data["var_peak_idx"]


        for d_file in sdata.d_files:
            coord = sdata.get_vars(d_file)[sel_coord]

            if np.any(var_peak_idx[d_file]):
                r.append(coord[var_peak_idx[d_file]][-1])
            else:
                r.append(0)

    elif sdata.grid_setup["dimensions"] == 3:
        #TODO fix below assignment -> fix peak_findr 
        coord_dim = sel_coord.strip("x")
        sel_var = "vx" + coord_dim #NOTE peak_findr DOES care which var is used, set to vel?
        print(f"{pcolours.WARNING}Note: stellar wind only works for velocity components setting sel_var = {sel_var}")

        peak_data = peak_findr(sel_coord,sel_var,sdata=sdata) 
        r = peak_data["radius"]


    slope = tprog_phelper(sel_coord,r,sdata,type,**kwargs)

def plot_ram_pressure(sdata,load_outputs= None,**kwargs):

    pdata = pp.PlotData(var_choice=["vx3"],**kwargs)
    pdata.arr_type = "cc"
    pdata.var_name = "vx3"

    load_outputs = sdata.load_outputs if load_outputs is None else load_outputs
    axes, fig = pp.subplot_base(sdata,pdata,load_outputs=pdata.load_outputs,**kwargs)
    plot_idx = 0  # Keep track of which subplot index we are using

    for output in load_outputs:
        pdata.output = output
        extras_data = pp.plot_extras(sdata,pdata)
        xy_labels = extras_data["xy_labels"]
        var_label = sdata.get_var_info("prs").var_name
        var_units = sdata.get_var_info("prs").usr_uv
        coord_units = sdata.get_var_info("z").usr_uv

        inj_loc = sdata.get_injection_region(output=output)[0].value #location of the injection region for moving injection regions
        value_dict = {"x1":inj_loc} #dict of coord then location of inj region
        slices = calc_var_prof(sdata,"x3",value_1D=value_dict)
        fluid_data_1d = sdata.load_fluid_data(var_choice=["rho","vx3","ccz"],output=output,load_slice=slices['slice_1D'])

        ram_prs = fluid_data_1d['rho']*np.pow(fluid_data_1d["vx3"],2)
        inv_z = 1/fluid_data_1d["ccz"]**2
        scale_factor = ram_prs[np.argmax(ram_prs)] / inv_z[np.argmax(ram_prs)]

        ax = axes[plot_idx]
        ax.set_yscale("log")
        ax.plot(fluid_data_1d["ccz"],ram_prs)
        ax.plot(fluid_data_1d["ccz"],inv_z * scale_factor,color = "g",linestyle = ":")
        ax.plot(fluid_data_1d["ccz"],inv_z * -1*scale_factor,color = "g",linestyle = ":")
        
        ax.set_xlabel(f"{xy_labels['ccz']}")
        ax.set_ylabel(f"{var_label} [{var_units}]")

        legend_coord, = value_dict.keys() #get the first key from value_dict, NOTE: works assuming that only 1 slice arg is used 
        value = f"{(value_dict.get(legend_coord)):.2f}" #scaling factor makes it easier to read
        legend_str = f"$P_{{\\rm{{ram}}}}$ @ {sdata.get_var_info(legend_coord).coord_name} = {value} {coord_units}"
        fit_str = '$1/z^{{2}}$'

        ax.legend([legend_str,fit_str])
        ax.set_title(f"Plot of $P_{{\\rm{{ram}}}}$ along jet axis ($z$) with $1/z^{{2}}$ fit")
        pp.plot_axlim(ax,kwargs)
        plot_idx += 1
    pp.plot_save(sdata,pdata,**kwargs)

def jet_kinetic_power(radius,rho,vel):
    eqn = 0.5*4*np.pi*(radius**2)*rho*(vel**3)
    return eqn.si

def EOS(rho=None,prs=None,T=None,mu = 0.60364):
    """
    Simple Equation of state calculator to get Temp for a given density and pressure etc...
    """
    m_H = constants.m_p
    kb = constants.k
    
    if not T:
        unit = (u.Kelvin)
        T = (prs*mu*m_H)/(rho*kb)*unit
        return T 
    
    if not prs:
        unit = (u.pascal)
        prs = (kb*rho*T)/(mu*m_H)*unit
        return prs 
    
    if not rho:
        unit = (u.kg)/(u.m**3)
        rho = (prs*mu*m_H)/(T*kb)*unit
        return rho 

def calc_sound_speed(rho_0,T):
   prs_0 = EOS(rho =rho_0,T = T).value
   nonrel_gamma = 5/3
   unit = u.m / u.s
   return (np.sqrt((nonrel_gamma * prs_0) / (rho_0)))*unit

def calc_inlet_speed(rho_0,T,wind_vxx):
    """
    Gets speed in kpc/Myr for a jet with moving injection region

    Args:
        rho_0 (float): environment density in kg/m^3
        T (float): environment temperature in K
        wind_vxx (list): wind speed as multiple of environment sound speed e.g. WIND_VX1,WIND_VX2,WIND_VX3 = [2,0,0] -> 2*c_s in x

    Returns:
        inlet_vxx (list): List of inlet speeds in kpc/Myr per wind_vx component
    """

    inlet_vxx = []
    for vx in wind_vxx:
        inlet_vxx.append((vx*calc_sound_speed(rho_0=rho_0,T=T)).to(u.kpc / u.Myr))
    return inlet_vxx

def locate_injection_region(rho_0,T,wind_vxx,sim_time):
    """
    Gives location of jet injection region (in kpc) for a given timestep

    Args:
        rho_0 (float): environment density in kg/m^3
        T (float): environment temperature in K
        wind_vxx (list): wind speed as multiple of environment sound speed e.g. WIND_VX1,WIND_VX2,WIND_VX3 = [2,0,0] -> 2*c_s in x
        sim_time (float): simulation time in Myr e.g 35

    Returns:
        inj_xyz (list): list with 4 elements, x,y,z location of injection region, then timestep
    """
    sim_time = sim_time * u.Myr
    inlet_vxx = calc_inlet_speed(rho_0=rho_0,T=T,wind_vxx=wind_vxx)
    inj_xyz = []
    for vx in inlet_vxx:
        inj_xyz.append(-vx*sim_time)
    inj_xyz.append(sim_time)
    return inj_xyz

def calc_length_scales(Q,rho,v_jet,theta,T,v_wind = 0):
    """
    Calculates the length scales from Krause (2012). 
    L1: Length at which the jet density becomes compariable to the external density
    L1a: Jet recollimation, sideways ram pressure = ambient pressure
    L1b: cocoon formation
    L1c: terminal shock
    L2: buoyancy scale

    Args:
        Q (float): Jet kinetic power [W]
        rho (float): Environment density [kgm^-3]
        v_jet (float): Jet injection velocity [c]
        theta (float): Half opening angle of jet in degreees
        T (float): Environment temperature [K]
        v_wind (float): Velocity of environment cross wind [ms^-1], defaults to 0 (no wind)

    Returns:
        lscale_dict (dict): dictionary with L1x as keys (length scales in kpc)
    """
    Q, rho, v_jet, theta, T,v_wind = [v.value if isinstance(v, u.Quantity) else v for v in [Q, rho, v_jet, np.deg2rad(theta), T,v_wind]]
    
    to_kpc = astro_const.kpc.value / u.kpc
    # to_kpc = 1

    gamma = 5/3
    Omega = 2*np.pi*(1-np.cos(theta))
    c_s = calc_sound_speed(rho,T).value

    M_jet = v_jet/c_s
    M_wind = v_wind/c_s

    L1 = (2 * np.sqrt(2) * np.sqrt(Q / (rho * v_jet ** 3)))
    L1a = (np.sqrt(((gamma)/(4*Omega)) * M_jet**2 * np.sin(theta)**2 * L1**2)) 
    L1b = (np.sqrt((1/(4*Omega)) * L1**2)) 
    L1c = (np.sqrt((gamma/(4*Omega)) * M_jet**2 * L1**2))
    L2 = (np.sqrt(Q /(rho * c_s**3)))

    r_jet = L1a * np.tan(theta)
    L_bend = 2*r_jet * (M_jet/M_wind)**2 

    eta = (L1b/L1a)**2

    lscale_dict = {
            "L1": L1 / to_kpc,
            "L1a": L1a / to_kpc,
            "L1b": L1b / to_kpc,
            "L1c": L1c / to_kpc,
            "L2": L2 / to_kpc,
            "L_bend": L_bend / to_kpc,
            "r_jet": r_jet / to_kpc,
            "eta": eta,

        }

    return lscale_dict

#---chi parameter---#
def l2s(lorentz):
    return np.sqrt(astro_const.c**2 * (1 - (1 / lorentz) ** 2))

def s2l(speed):
    return (1 / np.sqrt(1 - (speed / astro_const.c) ** 2)).value

def calc_jet_area(theta, radius):
    theta = np.deg2rad(theta)
    return (2 * np.pi * (1 - np.cos(theta))) * (radius**2)

def calc_jet_density(Q_jet, v_jet, theta, r_jet):
    adiab_ind = 5.0 / 3.0
    area = calc_jet_area(theta, r_jet)

    return 2 * Q_jet / (v_jet**3 * area)

def calc_jet_density_rel(
    power, speed, half_opening_angle, radius, adiab_ind=5.0 / 3.0, prs=None, chi=None
):
    area = calc_jet_area(half_opening_angle, radius)
    lorentz = s2l(speed)

    if chi is None:
        return (1 / (lorentz * (lorentz - 1) * astro_const.c**2)) * (
            (power / (speed * area)) - lorentz**2 * (adiab_ind) / (adiab_ind - 1) * prs
        )

    elif prs is None:
        return (power) / (
            (speed * area * astro_const.c**2)
            * (lorentz * (lorentz - 1) + (lorentz**2) / (chi))
        )

def calc_chi(density, pressure, adiabatic_ind):
    return ((adiabatic_ind - 1) / (adiabatic_ind)) * (density * astro_const.c**2) / (pressure)

# ---Jet angle---#
def binned_mean_tracer_mask(bin_size, x, y, tr_array, tr_cut=None, side=None, **kwargs):
    """
    Calculates a mean value of x/y across some number of bins for a given tracer cutoff.
    """
    x_mean = [0]
    y_mean = [0]
    bin_counter = 0

    for i in range(0, len(x), bin_size):
        x_slice = x[i:i+bin_size]
        y_slice = y[i:i+bin_size]
        tr_slice = tr_array[i:i+bin_size]

        # simple mask
        if side == "top":
            mask = (
                (y_slice > 0) &
                (tr_slice >= tr_cut)
            )
        elif side == "bot":
            mask = (
                (y_slice < 0) &
                (tr_slice >= tr_cut)
            )
        else:
            raise ValueError("side must be 'top' or 'bot'")

        x_cut = x_slice[mask]
        y_cut = y_slice[mask]

        if len(y_cut) == 0:
            continue

        x_next_mean = np.mean(x_cut)
        y_next_mean = np.mean(y_cut)
  
        last_y = y_mean[-1]
        last_x = x_mean[-1]


        if (np.abs(y_next_mean) / np.abs(last_y) > 0.95):
        # if x_next_mean/last_x > 0.33:
            x_mean.append(x_next_mean)
            y_mean.append(y_next_mean)


        bin_counter += 1

    return x_mean, y_mean

def dx_mean_tracer_mask(nslices,x, y, tr_array, tr_cut=None, side=None, **kwargs):
    """
    Calculates a mean value of x/y across some number of bins for a given tracer cutoff.
    """
    x_mean = [0]
    y_mean = [0]

    grid_slices = np.linspace(0,x[-1],nslices)
    for grid_start in grid_slices:
        dx = grid_slices[1] - grid_slices[0]

        # simple mask
        if side == "top":
            mask = (
                (y >= 0) &
                (tr_array >= tr_cut) &
                (x >= grid_start) &
                (x <= grid_start + dx)
            )
        elif side == "bot":
            mask = (
                (y <= 0) &
                (tr_array >= tr_cut) &
                (x >= grid_start) &
                (x <= grid_start + dx)
            )
        else:
            raise ValueError("side must be 'top' or 'bot'")

        x_cut = x[mask]
        y_cut = y[mask]

        if len(y_cut) == 0:
            continue

        # y_next_mean = np.max(y_cut) if side == "top" else np.min(y_cut)
        if side == "top":
            x_next_mean = np.mean(x_cut)
            # y_next_mean = np.percentile(y_cut,80)
            y_next_mean = np.max(y_cut)
            
        if side == "bot":
            x_next_mean = np.mean(x_cut)
            y_next_mean = np.min(y_cut)

            # y_next_mean = np.percentile(y_cut,20)

        if len(x_mean) > 0:
            last_y = y_mean[-1]
            last_x = x_mean[-1]

            x_mean.append(x_next_mean)
            y_mean.append(y_next_mean)

        else:
            # First valid data point
            x_mean.append(x_next_mean)
            y_mean.append(y_next_mean)


    return x_mean, y_mean

def jet_angle_linegress(sdata,load_outputs,bin_size,tr_cut = None,**kwargs):
    """
    Calculates the jet angle from the binned mean tracer cutoff using a linear regression
    """
    jet_angle = []
    angles_top, angles_bot = [],[]
    # sdata.load_particles(load_outputs)
    part_output = sdata.particle_data
    part_files = sdata.particle_files

    for part_file in part_files:
        particle_data = part_output[part_file]
        x_bmt, y_bmt = binned_mean_tracer_mask(bin_size,particle_data["x1"],particle_data["x3"],particle_data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = binned_mean_tracer_mask(bin_size,particle_data["x1"],particle_data["x3"],particle_data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)
        slope_top = stats.linregress(x_bmt,y_bmt).slope
        slope_bot = stats.linregress(x_bmb,y_bmb).slope

        angle_top = np.abs(np.rad2deg(np.atan(slope_top)))
        angle_bot = np.abs(np.rad2deg(np.atan(slope_bot)))

        # print("Angle of top lobe WRT horizontal:",f"{angle_top:.2f} deg")
        # print("Angle of bottom lobe WRT horizontal:",f"{angle_bot:.2f} deg")

        angles_top.append(angle_top)
        angles_bot.append(angle_bot)
        jet_angle.append(angle_top+angle_bot)
        # print("Angle of jet:",f"{angle_top+angle_bot:.2f} deg")
        print("Top angle:",f"{angle_top:.2f}","Bottom angle:",f"{angle_bot:.2f}")

        mask_vars = ["x1", "x2", "x3","tracer"]
        for mask_var in mask_vars:
            tracer_mask = particle_data["tracer"] > tr_cut # change to tr_cut to actually represent

            particle_data[mask_var] = particle_data[mask_var][tracer_mask]


    returns = {
        "part_output":part_output,"part_files":part_files,"angles_top":angles_top,"angles_bot":angles_bot,"jet_angle":jet_angle,"tr_cut":tr_cut
    }
    return returns

def jet_angle_vector_old(sdata,load_outputs,nslices,tr_cut = None,**kwargs):
    """
    Calculates the jet angle from the binned mean tracer cutoff using a linear regression
    """
    jet_angle = []
    angles_top, angles_bot = [],[]
    sdata.load_particles(load_outputs)
    part_output = sdata.particle_data
    part_files = sdata.particle_files

    for part_file in part_files:
        particle_data = part_output[part_file]
        x_bmt, y_bmt = dx_mean_tracer_mask(nslices=nslices,x=particle_data["x1"],y=particle_data["x3"],tr_array=particle_data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = dx_mean_tracer_mask(nslices=nslices,x=particle_data["x1"],y=particle_data["x3"],tr_array=particle_data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)

        A = np.array([x_bmt, y_bmt]) 
        B = np.array([x_bmb, y_bmb]) 

        # min_len = min(A.shape[1], B.shape[1])
        # A_trim = A[:, :min_len]  # shape (2, min_len)
        # B_trim = B[:, :min_len]

        y_max_idx_A = np.argmax(A[1,:])
        y_min_idx_B = np.argmin(B[1,:])

        start_A = A[:,0]
        start_B = B[:,0]
        end_A = A[:,y_max_idx_A]
        end_B = B[:,y_min_idx_B]

        vec_A = end_A - start_A
        vec_B = end_B - start_B

        # Dot product & angle
        dot_product = np.dot(vec_A, vec_B)
        magnitude_A = np.linalg.norm(vec_A)
        magnitude_B = np.linalg.norm(vec_B)
        angle_radians = np.arccos(dot_product / (magnitude_A * magnitude_B))
        angle_degrees = np.degrees(angle_radians)
        jet_angle.append(angle_degrees)
        # print(f"Angle between top and bottom curves: {angle_degrees:.2f} degrees")


        mask_vars = ["x1", "x2", "x3","tracer"]
        for mask_var in mask_vars:
            tracer_mask = particle_data["tracer"] > tr_cut # change to tr_cut to actually represent

            particle_data[mask_var] = particle_data[mask_var][tracer_mask]


    returns = {
        "part_output":part_output,"part_files":part_files,"jet_angle":jet_angle,"tr_cut":tr_cut
    }
    return returns

def jet_angle_vector(sdata,load_outputs,nslices,tr_cut = None,**kwargs):
    """
    Calculates the jet angle from the binned mean tracer cutoff using a linear regression
    """
    jet_angle = []
    angles_top, angles_bot = [],[]
    sdata.load_particles(load_outputs)
    part_output = sdata.particle_data
    part_files = sdata.particle_files

    for part_file in part_files:
        particle_data = part_output[part_file]
        x_bmt, y_bmt = dx_mean_tracer_mask(nslices=nslices,x=particle_data["x1"],y=particle_data["x3"],tr_array=particle_data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = dx_mean_tracer_mask(nslices=nslices,x=particle_data["x1"],y=particle_data["x3"],tr_array=particle_data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)

        A = np.array([x_bmt, y_bmt]) 
        B = np.array([x_bmb, y_bmb]) 

        # min_len = min(A.shape[1], B.shape[1])
        # A_trim = A[:, :min_len]  # shape (2, min_len)
        # B_trim = B[:, :min_len]

        y_max_idx_A = np.argmax(A[1,:])
        y_min_idx_B = np.argmin(B[1,:])

        # y_max_idx_A = np.argmin(np.abs(A[1,:] - np.median(A[1,:])))
        # y_min_idx_B = np.argmin(np.abs(B[1,:] - np.median(B[1,:])))
        start_A = A[:,0]
        start_B = B[:,0]
        end_A = A[:,y_max_idx_A]
        end_B = B[:,y_min_idx_B]

        vec_A = end_A - start_A
        vec_B = end_B - start_B

        # Dot product & angle
        dot_product = np.dot(vec_A, vec_B)
        magnitude_A = np.linalg.norm(vec_A)
        magnitude_B = np.linalg.norm(vec_B)
        angle_radians = np.arccos(dot_product / (magnitude_A * magnitude_B))
        angle_degrees = np.degrees(angle_radians)
        jet_angle.append(angle_degrees)
        # print(f"Angle between top and bottom curves: {angle_degrees:.2f} degrees")


        mask_vars = ["x1", "x2", "x3","tracer"]
        for mask_var in mask_vars:
            tracer_mask = particle_data["tracer"] > tr_cut # change to tr_cut to actually represent

            particle_data[mask_var] = particle_data[mask_var][tracer_mask]


    returns = {
        "part_output":part_output,"part_files":part_files,"jet_angle":jet_angle,"tr_cut":tr_cut
    }
    return returns

def plot_jet_angle_particles(sdata,outputs,tr_cut,bin_size,show_means=True,**kwargs):
    """
    Plots the jet angle across simulation outputs using the two helper functions:
    `binned_mean_tracer_mask`, `jet_angle_particles`
    """
    jet_angle_data = jet_angle_linegress(sdata,outputs,bin_size=bin_size,tr_cut=tr_cut,**kwargs) #smaller cuttoff  is better for later data 
    part_files = jet_angle_data["part_files"]
    part_output = jet_angle_data["part_output"]
    jet_angles = jet_angle_data["jet_angle"]

    pdata = pp.PlotData(**kwargs)
    pdata.axes,pdata.fig = pp.subplot_base(sdata,fig_resize = 5)

    for ax, part_file,angle in zip(pdata.axes, part_files,jet_angles):
        data = part_output[part_file]

        # # scatter plot using pre masked data
        im = ax.scatter(
            data["x1"], data["x3"],
            c=data["tracer"],
            s=2.5
        )

        ax.text(
                0.05, 0.98, f"Angle ={angle:.2f}°",
                transform=ax.transAxes,   # relative to axis (0–1 coords)
                fontsize=8,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
            )
        
        # median points and regression of median points
        x_bmt, y_bmt = binned_mean_tracer_mask(bin_size,data["x1"],data["x3"],data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = binned_mean_tracer_mask(bin_size,data["x1"],data["x3"],data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)
        
        z1 = np.polyfit(x_bmt, y_bmt, 1)
        p1 = np.poly1d(z1)
        ax.plot(x_bmt, p1(x_bmt), "r--")

        z2 = np.polyfit(x_bmb, y_bmb, 1)
        p2 = np.poly1d(z2)
        ax.plot(x_bmb, p2(x_bmb), "r--")

        #additional scatter of means
        if show_means:
            ax.scatter(x_bmt, y_bmt, color="r", s=2.5) 
            ax.scatter(x_bmb, y_bmb, color="r", s=2.5)

        #title etc
        # ax.set_aspect("equal")
        ax.set_xlabel("X / kpc")
        ax.set_ylabel("Z / kpc")
        ax.set_title(f"{part_file}")

    # plt.tight_layout()
    pdata.fig.colorbar(im, ax=pdata.axes, label="Tracer")#,fraction = 0.05)
    plt.show()

def plot_jet_angle_particles_vector(sdata,outputs,tr_cut,nslices,show_means=True,**kwargs):
    jet_angle_data = jet_angle_vector(sdata,outputs,nslices=nslices,tr_cut=tr_cut,**kwargs) #smaller cuttoff  is better for later data 
    part_files = jet_angle_data["part_files"]
    part_output = jet_angle_data["part_output"]
    jet_angles = jet_angle_data["jet_angle"]

    pdata = pp.PlotData(**kwargs)
    pdata.axes,pdata.fig = pp.subplot_base(sdata,fig_resize = 5)

    for ax, part_file,angle in zip(pdata.axes, part_files,jet_angles):
        data = part_output[part_file]

        # # scatter plot using pre masked data
        im = ax.scatter(
            data["x1"], data["x3"],
            c=data["tracer"],
            s=2.5
        )

        ax.text(
                0.05, 0.98, f"Angle ={angle:.2f}°",
                transform=ax.transAxes,   # relative to axis (0–1 coords)
                fontsize=8,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
            )
        
        # median points and regression of median points
        x_bmt, y_bmt = dx_mean_tracer_mask(nslices=nslices,x=data["x1"],y=data["x3"],tr_array=data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = dx_mean_tracer_mask(nslices=nslices,x=data["x1"],y=data["x3"],tr_array=data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)
        
        A = np.array([x_bmt, y_bmt]) 
        B = np.array([x_bmb, y_bmb]) 

        y_max_idx_A = np.argmax(A[1,:])
        y_min_idx_B = np.argmin(B[1,:])

        # y_max_idx_A = np.argmin(np.abs(A[1,:] - np.median(A[1,:])))
        # y_min_idx_B = np.argmin(np.abs(B[1,:] - np.median(B[1,:])))

        start_A = A[:,0]
        start_B = B[:,0]
        end_A = A[:,y_max_idx_A]
        end_B = B[:,y_min_idx_B]
        vec_A = end_A - start_A
        vec_B = end_B - start_B

        origin = A[0]
        ax.quiver(*start_A, *vec_A, angles='xy', scale_units='xy', scale=1, color="black",width = 1.5e-2)
        ax.quiver(*start_B, *vec_B, angles='xy', scale_units='xy', scale=1, color="black",width = 1.5e-2)

        #additional scatter of means
        if show_means:
            ax.scatter(A[0,:], A[1,:], color="r", s=2.0,alpha = 0.4)
            ax.scatter(B[0,:], B[1,:], color="r", s=2.0,alpha = 0.4)

        #title etc
        ax.set_aspect("equal")
        ax.set_xlabel("X / kpc")
        ax.set_ylabel("Z / kpc")
        ax.set_title(f"{part_file}")

    # plt.tight_layout()
    pdata.fig.colorbar(im, ax=pdata.axes[-1], label="Tracer",pad = 0.0001)#,fraction = 0.05)

    plt.show()
    pp.plot_save(sdata,pdata,**kwargs) 

def jet_angle_tprog_linegress(sdata,outputs,tr_cut,bin_size):
    """
    Plots the calculated jet angle across simulation outputs
    """
    jet_angle_data = jet_angle_linegress(sdata,outputs,tr_cut=tr_cut,bin_size=bin_size) #smaller cuttoff  is better for later data 
    part_files = jet_angle_data["part_files"]
    part_output = jet_angle_data["part_output"]
    jet_angles = jet_angle_data["jet_angle"]

    times = []
    for output in outputs:
        times.append(output / 10)

    slope, intercept, r_value, p_value, std_err = stats.linregress(times, jet_angles)
    eqn = f'$\\text{{slope}} = {{{slope:.2f}}} \\pm {std_err:.2f}\\;\\;[\\mathrm{{deg/Myr}}]$'
    plt.title("Jet angle vs Time ")
    plt.ylabel("Angle [deg]")
    plt.xlabel("Time [Myr]")
    plt.text(0.05, 0.05, eqn, transform=plt.gca().transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 

    plt.plot(times,jet_angles)   

def jet_angle_tprog_vector(sdata,outputs,tr_cut,nslices):
    """
    Plots the calculated jet angle across simulation outputs
    """
    jet_angle_data = jet_angle_vector(sdata,outputs,tr_cut=tr_cut,nslices=nslices) #smaller cuttoff  is better for later data 
    part_files = jet_angle_data["part_files"]
    part_output = jet_angle_data["part_output"]
    jet_angles = jet_angle_data["jet_angle"]

    times = []
    for output in outputs:
        times.append(output / 10)

    slope, intercept, r_value, p_value, std_err = stats.linregress(times, jet_angles)
    eqn = f'$\\text{{slope}} = {{{slope:.2f}}} \\pm {std_err:.2f}\\;\\;[\\mathrm{{deg/Myr}}]$'
    plt.title("Jet angle vs Time ")
    plt.ylabel("Angle [deg]")
    plt.xlabel("Time [Myr]")
    plt.text(0.05, 0.05, eqn, transform=plt.gca().transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 

    plt.plot(times,jet_angles) 


import plutonlib.utils as pu
import plutonlib.read_write as prw
# import plutonlib.simulations as ps

import plutokore.radio as pk_radio

import numpy as np

from astropy import units as u
from astropy import cosmology as cosmo  # Astropy cosmology

from scipy.spatial.transform import Rotation
from astropy.convolution import convolve, Gaussian2DKernel  # Astropy convolutions

import os
import h5py
import multiprocessing
import resource


def setup_obs_properties_praise(sdata,redshift,angles = [0,0,0],plane="xz"):
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

    # rot_mat = Rotation.from_euler("X", angle, degrees=True).as_matrix()
    rot_mat = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()

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

def calc_surface_brightness_praise(sdata,freqs=[1.4],redshift=0.05,part_outputs="last",angles=[0,0,0],plane="xz"):
    """
    Calculates the particle emssion using PRAiSE (plutokore: pk_radio) under adiabatic, sychrotron and inverse compton losses.
    Surface brightness is then calculated by integrating emissivity with raytracing.    
    :param sdata: Description
    :param freqs: Description
    :param redshift: Description
    :param particle_outputs: Description
    :param angle: Description
    """
    pk_sim = sdata.to_plutokore() #convert SimulationData object to plutokore PlutoSimulation
    part_outputs = [prw.get_particle_outputs(sdata.wdir)] if part_outputs == "last" else part_outputs
    particle_spacing = sdata.part_to_simtime(part_outputs[0]) / part_outputs[0]
    s=2.2   # for injection spectral index alpha=-0.55. NOTE: the PRAiSE default is also 2.2

    #load all the available particle files from hdf5
    particle_data = sdata.load_particle_data(part_output=part_outputs[-1],force_check=False) #NOTE force_check turned off here to allow faster calcs with no checks
    alias_map = {"rho": "density", "prs": "pressure", "tr1": "tracer"} #NOTE use to convert to praise format
    for short_key, long_key in alias_map.items():
        if long_key not in particle_data and short_key in particle_data:
            particle_data[long_key] = particle_data[short_key]

    particle_times = particle_data["particle_times"]
    particle_emis = pk_radio.praise2.praise(
        sim=pk_sim,
        max_output=part_outputs[-1],
        emit_outputs=part_outputs, #calc emission for these outputs
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
    for i in  range(0, len(part_outputs), 1):    
        part_ind = part_outputs[i]
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

        obs_properties = setup_obs_properties_praise(sdata=sdata,redshift=redshift,angles=angles,plane=plane)
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
            particle_emissivities=particle_emis['full'][i]['emis'][all_nan_masks[i],:,0] #only the non-nan particles, all frequencies, and the ith snapshot 
        )
        integ_emis.append(integrated_emissivity)

        # we multiple our integrated emissivity by kpc (to account for integration), and divide by 4pi to account for solid angle
        surface_brightness = (integrated_emissivity * u.kpc / (4 * np.pi)).to(u.mJy / u.beam, equivalencies=u.beam_angular_area(obs_properties["omega_beam"]))
        sb_arr.append(surface_brightness)
    
    for freq_ind in range(len(freqs)):
        sb_arr[i][:,:,freq_ind] = np.nan_to_num(sb_arr[i][:,:,freq_ind], copy=True, nan=0.0, posinf=0.0, neginf=0.0) # get rid of NaNs (replace with zero)

    return sb_arr

def _compute_sb_task(sim,grid_output,angles,freqs,redshift,plane='xz'):
    """Runs the computation needed for save_sb_hdf5 to run in parallel"""
    print(f"Computing SB data for {sim.run_name} grid output={grid_output} angle={angles}°...")
    obs_properties = setup_obs_properties_praise(
        sdata=sim, redshift=redshift, angles=angles, plane=plane   # angles= not angle=
    )

    part_output = sim.simtime_to_part(grid_output,round_val=True) #convert to particle output for calculation
    sb_arr = calc_surface_brightness_praise(
        sdata=sim,
        freqs=freqs,
        redshift=redshift,
        part_outputs=[part_output],
        angles=angles,
        plane=plane,
    )

    #mutli-freq calculation
    freq_sb = np.stack([
        convolve(sb_arr[0][:, :, k].to(u.mJy / u.beam),
        obs_properties["gaussian_kernel"], boundary='extend')
        for k in range(len(freqs))
    ], axis=-1) * (u.mJy / u.beam)          # shape (nx, ny, n_freqs)

    freq_sb[freq_sb == 0] = np.nan

    log_sb = np.log10(freq_sb.value.T)      # shape (n_freqs, ny, nx)

    contour_levels = np.stack([
        np.linspace( 
            np.log10(np.nanpercentile(freq_sb[:, :, k], 99).value) - 1,
            np.log10(np.nanpercentile(freq_sb[:, :, k], 99).value),
            3
        ) for k in range(len(freqs))
    ])   # shape (n_freqs, 3)

    n_pairs = len(freqs) // 2
    if len(freqs) % 2 != 0:
        print(f"Note: {len(freqs)} freqs given, freqs[{len(freqs)-1}]={freqs[-1]} GHz has no pair, skipping alpha for it")

    alpha = None
    if n_pairs > 0: #only create and calculate alpha dataset if there are freq pairs
        alpha = np.stack([
            (np.log10(freq_sb[:, :, 2*p].value.T) - np.log10(freq_sb[:, :, 2*p + 1].value.T)) /
            (np.log10(freqs[2*p]) - np.log10(freqs[2*p + 1]))
            for p in range(n_pairs)
        ])   # shape (n_pairs, ny, nx)

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)  # KB -> GB
    print(f"Memory used: {peak_gb:.2f} GB")  

    task_data = {
        "run_name": sim.run_name,
        # "output": part_output,
        "grid_output": grid_output,     # the identifier you key by
        "part_output": part_output,     # only used internally to fetch/compute
        "angles": angles,
        "freq_sb": freq_sb,
        "log_sb": log_sb,
        "alpha": alpha,
        "contour_levels": contour_levels,
        "obs_properties": obs_properties,
        "peak_gb": peak_gb,
        "redshift": redshift
    }
  
    return task_data

def save_sb_hdf5(sim,grid_outputs,angles, freqs, redshift=0.05, plane='xz',memory=None,task_req_mem=30):    
    # part_outputs = sim.simtime_to_part(grid_outputs,round_val=True)
    file_path = os.path.join(sim.wdir, f"sbdata.h5") 
    fmode = "a" if os.path.exists(file_path) else "w"

    task_args = []
    with h5py.File(file_path, fmode) as h5f:
        for grid_output in grid_outputs:
            for angle_set in angles: #calculate sb per output per angle set 
                if f'{redshift}/{grid_output}/{angle_set}' in h5f:
                    existing = set(h5f[f'{redshift}/{grid_output}/{angle_set}'].attrs.get('freqs', []))
                    if set(freqs) <= existing:
                        print(f"Found angles '{angle_set}' with freqs {freqs} in '{file_path}/{redshift}/{grid_output}', skipping calculation...")
                        continue
                task_args.append((sim, grid_output, angle_set, freqs, redshift, plane))


    n_workers = pu.setup_workers(n_tasks=len(task_args),task_req_mem=task_req_mem,memory=memory) #NOTE helper to find number of workers
    # context = multiprocessing.get_context("spawn") #NOTE use if you get errors with mp
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.starmap(_compute_sb_task, task_args, chunksize=1)

    with h5py.File(file_path, "a") as h5f: #reopen file to assign data
        h5f.attrs['run_name'] = sim.run_name #store run_name and freq as base attrs
        for task_data in results:
            run_data = h5f.require_group(f"{redshift}")
            run_data.attrs['redshift'] = redshift
            existing_freqs = set(run_data.attrs.get('freqs', []))
            run_data.attrs['freqs'] = sorted(existing_freqs | set(freqs))

            metadata = run_data.require_group("metadata") #all share a metadata group?
            grid = run_data.require_group("grid").require_group(plane) #store the grid data per run and per plane
            grid_written = "grid_x" in grid 
            
            timestep = run_data.require_group(f"{task_data['grid_output']}") #NOTE this is a string of the particle output number -> maybe tstr?
            angle = timestep.require_group(f"{task_data['angles']}") # /timestep/angle e.g. 500/[0,0,0] as str

            for name in ('sb', 'log_sb', 'rot_mat', 'contour_levels', 'alpha'): #NOTE this line runs when new freqs are found -> delete old and recompute 
                if name in angle:
                    del angle[name]

            angle.attrs['freqs'] = freqs #NOTE to keep track of freqs btwn file and whats been previously calcualted
            angle.create_dataset('sb',data=task_data["freq_sb"]) #sb data for run at timestep at angle
            angle.create_dataset('log_sb', data=task_data["log_sb"]) 
            angle.create_dataset('rot_mat', data=task_data["obs_properties"]['rot_mat'])
            angle.create_dataset('contour_levels', data=task_data["contour_levels"])

            if task_data["alpha"] is not None:
                angle.create_dataset('alpha', data=task_data["alpha"])
            
            if not grid_written:
                grid.create_dataset('grid_x', data=task_data["obs_properties"]['grid_x'])
                grid.create_dataset('grid_y', data=task_data["obs_properties"]['grid_y'])
                grid.create_dataset('grid_mx', data=task_data["obs_properties"]['grid_mx'])
                grid.create_dataset('grid_my', data=task_data["obs_properties"]['grid_my'])
                grid_written = True

            #these values should all be constant across calculations -> only set once
            if "gaussian_kernel" not in metadata:
                metadata.create_dataset("gaussian_kernel",data=task_data["obs_properties"]["gaussian_kernel"].array)
                metadata.attrs['delta_r'] = task_data["obs_properties"]['delta_r']
                metadata.attrs['ray_depth_min'] = task_data["obs_properties"]['ray_depth_min']
                metadata.attrs['ray_depth_max'] = task_data["obs_properties"]['ray_depth_max']
                metadata.attrs['omega_beam'] = task_data["obs_properties"]['omega_beam']
                metadata.attrs['plane'] = plane

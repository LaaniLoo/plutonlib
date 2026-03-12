
import plutokore as pk
from plutokore.jet import UnitValues

import numpy as np
import scipy
import scipy.constants as const
from astropy import units as u  # Astropy units

import h5py
import os
import glob 
import logging

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit


src_path = os.path.join(os.path.expanduser('~'),'plutonlib/src/plutonlib')
main_path = os.path.join(os.path.expanduser('~'),'plutonlib/')

# diagnositc_path = './x-ray_diagnostics'
# os.makedirs(diagnositc_path, exist_ok=True)

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

def percent_err(measured,expected):
    return f"Percent Error from {expected:.2e}: {((measured-expected)/expected)*100:.1f}%"

def timestep_to_output(timestep,sim_dir):
    sim = pk.PlutoSimulation(
        simulation_name="",  # a descriptive simulation name
        simulation_directory=sim_dir,  # the path to our simulation
        simulation_description="",  # a short description
        datatype="float",  # or 'float', change to suit your data files
        dimensions=3,  # or 2 if 2D
    )

    if isinstance(timestep,str):
        timestep = float(timestep.strip(" Myr"))
    else:
        timestep = timestep 

    out_per_myr = sim.total_outputs / round(sim.load_output(sim.total_outputs).sim_time * 0.0032615638,2) #get outputs/myr

    return int((out_per_myr * timestep))

def get_tabulated_env(env_type = 'cluster'):
    
    """Units in SI
    tabulated_cosmo_002003 is cluster env
    """
    if env_type == 'cluster':
        file_path = os.path.join(main_path,"xraise_utils/tabulated_cosmo_002003.csv") 
    elif env_type == "group":
        file_path = os.path.join(main_path,"xraise_utils/tabulated_cosmo_002031.csv")
    else:
        raise ValueError(f"env_type = {env_type}, must be either 'cluster' or 'group'")

    env_file = np.loadtxt(file_path)

    radius = env_file[:, 0]
    rho = env_file[:, 1]
    prs = env_file[:, 2]
    temp = env_file[:, 3]
    g = env_file[:, 4]
    # col_5 = env_file[:, 5]

    env_dict = {
        "radius": radius,
        "rho": rho,
        "prs": prs,
        "temp": temp,
        "g": g,
        # "col_5": col_5,
    }

    return env_dict

def get_hdf5_paths_xray(config,redshift,file_path):
    
    """
    Get the HDF5 file paths for config and redshift
        :param config: config for xraise calculation see emission_params.yml e.g. "basic_xray_all"

        :param redshift: redshift e.g. 0.1

        :param file_path: file path (str) to hdf5 file

        :param losses
            0=none (doesn't really work), 
            1=adiabatic, 
            2=adiabatic+synch, 
            3=adiabatic+ic, 
            4=full
            5=synch, 
            6=IC,
            7=radiative
            
        :param angle
            defulats to observing angle of 0

        :return paths
        dict of HDF5 file paths with timestep as the key
            e.g for config = "basic_xray_all", redshift = 0.1
            {'106.00 Myr': 'basic_xray_all/0.1/241799050.4024/4/0.0/106.00 Myr',
            '152.00 Myr': 'basic_xray_all/0.1/241799050.4024/4/0.0/152.00 Myr',
            '36.78 Myr': 'basic_xray_all/0.1/241799050.4024/4/0.0/36.78 Myr',
            '96.00 Myr': 'basic_xray_all/0.1/241799050.4024/4/0.0/96.00 Myr'}
    """
    # paths = {}
    with h5py.File(file_path, "r") as file:
            hdf5_redshifts = [(redshifts) for redshifts in list(file[config].keys())]
            # print('Redshifts:', hdf5_redshifts)

            redshift = str(redshift) if not isinstance(redshift,str) else redshift

            if redshift not in hdf5_redshifts:
                raise KeyError(f"Redshift = {redshift} is not in HDF5 values: {hdf5_redshifts}")

            freq = list(file[config][redshift].keys())[0]
            losses = list(file[config][redshift][freq].keys())[0]
            angles = list(file[config][redshift][freq][losses].keys())[0]
            tsteps = list(file[config][redshift][freq][losses][angles].keys())

            paths = {}
            for tstep in tsteps:
                paths[tstep] = (f"{config}/{redshift}/{freq}/{losses}/{angles}/{tstep}")
                    
    return paths

def get_hdf5_data_xray(var_choice,config,redshift,timestep,file_path):
    
    """
    Gets the hdf5 data arrays from the xraise file paths (`get_hdf5_paths_xray`)  

    :param var_choice: list of variables required to load e.g. ["flux","mx_kpc","x_kpc"]

    :param config: config for xraise calculation see emission_params.yml e.g. "basic_xray_all"

    :param redshift: redshift e.g. 0.1

    :param timestep: timestep str as seen in the hdf5 file paths e.g. "38.00 Myr"  

    :param file_path: file path (str) to hdf5 file

    :return hdf5_data (dict): dict of variable arrays
        {'timestep': '96.00 Myr',
        'flux': array([[0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                ...,
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.],
                [0., 0., 0., ..., 0., 0., 0.]], shape=(681, 681))}
    """
    grid_values = ["mx_kpc","my_kpc","x_kpc","y_kpc"]
    hdf5_data = {}
    hdf5_data["timestep"] = timestep
    
    with h5py.File(file_path, "a") as file:
        hdf5_paths = get_hdf5_paths_xray(config=config,redshift=redshift,file_path=file_path)

        if timestep not in list(hdf5_paths.keys()):
            raise ValueError(f"Timestep = {timestep} is not in available timesteps: {list(hdf5_paths.keys())}")
        
        for var in var_choice:
            if var in grid_values:
                hdf5_data[var] = file['grid'][f'{config}'][var][f'{timestep}'][:]
            else:
                hdf5_data["flux"] = file[hdf5_paths[timestep]][:]
    
    return hdf5_data

def get_timesteps_all_files(config,redshift,data_dir):
    file_paths = glob.glob(os.path.join(data_dir,"*.hdf5"))

    file_tsteps = {}
    for f in file_paths:    
        file_tsteps[f] = list(get_hdf5_paths_xray(config,redshift,f).keys())
        file_tsteps[f] = sorted(file_tsteps[f],key=lambda x: float(x.split()[0]))
    return file_paths,file_tsteps

def calc_cavity_volume_hd(sim_dir,output = "last",tr_cut = 1e-4):

    """
    Calculates the volume of the jet cavity using a tracer cuttof, summing across all hydrodynamic 
    grid cell volumes

    :param output (int): output number 
    :param sim_dir (str): simulation directory containing hdf5 files
    :param tr_cut(flt): tracer cuttoff, e.g 1e-4 
    """
    sim = pk.PlutoSimulation(
        simulation_name="",  # a descriptive simulation name
        simulation_directory=sim_dir,  # the path to our simulation
        simulation_description="",  # a short description
        datatype="float",  # or 'float', change to suit your data files
        dimensions=3,  # or 2 if 2D
    )

    # output = timestep_to_output(timestep,sim_dir)
    output = sim.total_outputs if output == "last" else output
    data = sim.load_output(output)
    all_cell_volumes = pk.simulations.calculate_cell_volume_fast(data) 
    tr_mask = np.where(data.tr1[:] >= tr_cut)

    cavity_volume_kpc3 = np.sum(all_cell_volumes[tr_mask])

    return cavity_volume_kpc3

def calc_ellipse_radii_old(config,redshift,timestep,file_path,plot=False):
    
    """
    Calculates an ellipse with semi-major/minor axis a,b based on inflection points or FWHM of the surface brightness distribution.

    :param config: config for xraise calculation see emission_params.yml e.g. "basic_xray_all"

    :param redshift: redshift e.g. 0.1

    :param timestep: timestep str as seen in the hdf5 file paths e.g. "38.00 Myr"  

    :param file_path: file path (str) to hdf5 file

    :param plot (bool): defaults to False, displays a diagnostic plot showing turning points on SB dist when plot=True

    :return tuple(a,b): tuple containing semi-major/minor axes a,b 
    """

    hdf5_data = get_hdf5_data_xray(["flux","mx_kpc","my_kpc"],config=config,redshift=redshift,timestep=timestep,file_path=file_path)
    fig, ax = plt.subplots(figsize=(8, 8)) if plot else (None,None)

    percentile = 66
    coords = ["mx","my"]
    r_mean = []
    poi_success = True
    for i,coord in enumerate(coords):
        linestyle = '--' if coord == 'mx'  else ':'
        j = 0
        coord_data = hdf5_data[f"{coord}_kpc"]
        midpoint = len(coord_data)//2
        flux = hdf5_data["flux"][midpoint,:] if coord == "mx" else hdf5_data["flux"][:,midpoint]
        # flux = np.where(flux >= np.percentile(flux, percentile), flux, 0)

        ddx2 = np.gradient(np.gradient(flux,coord_data),coord_data)
        peak_idx,_ = scipy.signal.find_peaks(flux)
        poi_idx = np.where(np.diff(np.sign(ddx2)))[0]
        poi_peaks = poi_idx[np.where((poi_idx >= peak_idx[0]) & (poi_idx <= peak_idx[-1]))]
        poi_after_max = poi_peaks[poi_peaks > np.argmax(flux)]

        if len(poi_after_max) > 0:
            poi_peaks = np.array([poi_after_max[0], poi_peaks[-1]])
        elif len(poi_peaks) != 0:
            poi_peaks = poi_peaks[[0, -1]]
        else:
            poi_success=False #failed to use POI method

        if poi_success:
            r_mean.append(np.mean(np.abs([coord_data[poi_peaks[0]], coord_data[poi_peaks[1]]]))) 

        # if coord == "my":
        else: #try fitting a Gaussian and using FWHM
            print(f"Cannot resolve cavity using inflection points, atempting FWHM of Gaussian")
            popt, _ = curve_fit(gaussian, coord_data, flux)
            flux_fit = gaussian(coord_data, *popt)
            fwhm = 2.3548*(popt[2])
            fwhm_points = np.where(flux>=(popt[0]/2))[0][[0,-1]]
            r_mean.append(fwhm/2)

        if plot:
            poi_label = "Points of inflection" if i == 0 else None  # Only label first
            cavity_label = "Cavity region" if i == 0 else None
            gauss_label = f"FWHM of Gaussian" if j == 0 else None
            
            if poi_success:
                #outer region plots
                ax.plot(coord_data[:poi_peaks[0]],flux[:poi_peaks[0]],linestyle = linestyle,label = f"{coord}-axis",color = 'black')
                ax.plot(coord_data[poi_peaks[-1]:],flux[poi_peaks[-1]:],linestyle = linestyle,color = 'black')

                #inner cavity region plots
                ax.scatter(coord_data[poi_peaks],flux[poi_peaks],label = poi_label,color = "orange")
                ax.plot(coord_data[poi_peaks[0]:poi_peaks[-1]],flux[poi_peaks[0]:poi_peaks[-1]],label = cavity_label, color = "red")

            # if coord == "my":
            else: #if point of inflection method fails
                ax.plot(coord_data, flux_fit, linestyle=linestyle, label=f"{coord} Gaussian fit", color='blue')

                #outer region plots
                ax.plot(coord_data[:fwhm_points[0]],flux[:fwhm_points[0]],linestyle = linestyle,label = f"{coord}-axis",color = 'black')
                ax.plot(coord_data[fwhm_points[-1]:],flux[fwhm_points[-1]:],linestyle = linestyle,color = 'black')

                #inner cavity region plots
                ax.scatter(coord_data[fwhm_points],flux[fwhm_points],label = gauss_label,color = "green")
                ax.plot(coord_data[fwhm_points[0]:fwhm_points[-1]],flux[fwhm_points[0]:fwhm_points[-1]],color = "red")

            plt.title(f"Surface Brightness vs axis with POI @ {timestep}")
            ax.set_xlabel("x/y-axis [kpc]")
            ax.set_ylabel("Surface Brightness [mJy]")
            ax.legend()

            # sim_name = os.path.basename(file_path).split(".")[0]
            # os.makedirs(diagnositc_path, exist_ok=True)
            # os.makedirs(f"{diagnositc_path}/{sim_name}/", exist_ok=True)
            # plt.savefig(f"{diagnositc_path}/{sim_name}/{timestep.split(' ')[0]}_{timestep.split(' ')[-1]}_ellipsoid.png")

        j =+1

    a = max(r_mean)
    b = min(r_mean)

    return (a,b,fig)

def calc_ellipse_radii(config,redshift,timestep,file_path,plot=False):
    
    """
    Calculates an ellipse with semi-major/minor axis a,b based on inflection points or FWHM of the surface brightness distribution.

    :param config: config for xraise calculation see emission_params.yml e.g. "basic_xray_all"

    :param redshift: redshift e.g. 0.1

    :param timestep: timestep str as seen in the hdf5 file paths e.g. "38.00 Myr"  

    :param file_path: file path (str) to hdf5 file

    :param plot (bool): defaults to False, displays a diagnostic plot showing turning points on SB dist when plot=True

    :return tuple(a,b): tuple containing semi-major/minor axes a,b 
    """

    hdf5_data = get_hdf5_data_xray(["flux","mx_kpc","my_kpc"],config=config,redshift=redshift,timestep=timestep,file_path=file_path)
    fig, ax = plt.subplots(figsize=(8, 8)) if plot else (None,None)

    percentile = 76
    coords = ["mx","my"]
    r_mean = []
    poi_success = True
    for i,coord in enumerate(coords):
        linestyle = '--' if coord == 'mx'  else ':'
        j = 0
        coord_data = hdf5_data[f"{coord}_kpc"]
        midpoint = len(coord_data)//2
        flux = hdf5_data["flux"][midpoint,:] if coord == "mx" else hdf5_data["flux"][:,midpoint]
        cut_flux = np.where(flux >= np.percentile(flux, percentile), flux, 0)

        threshold = 0.06 * np.max(flux)
        peak_idx,_ = scipy.signal.find_peaks(flux, prominence = threshold)#width=(None,150)
        
        if len(peak_idx) == 2:
            ddx2 = np.gradient(np.gradient(flux,coord_data),coord_data)
            poi_idx = np.where(np.diff(np.sign(ddx2)))[0]
            poi_peaks = poi_idx[np.where((poi_idx >= peak_idx[0]) & (poi_idx <= peak_idx[-1]))]
            poi_after_max = poi_peaks[poi_peaks > np.argmax(flux)]

            if len(poi_after_max) > 0:
                poi_peaks = np.array([poi_after_max[0], poi_peaks[-1]])
            elif len(poi_peaks) != 0:
                poi_peaks = poi_peaks[[0, -1]]
            else:
                poi_success=False #failed to use POI method
        else:
            poi_success=False

        if poi_success:
            r_mean.append(np.max(np.abs([coord_data[poi_peaks[0]], coord_data[poi_peaks[1]]]))) 
            # r_min_peak = np.min(np.abs([coord_data[peak_idx[0]], coord_data[peak_idx[-1]]])) 
            # r_mean.append(coord_data[coord_data < r_min_peak][-1])

        # if coord == "my":
        else: #try fitting a Gaussian and using FWHM
            # print(f"Cannot resolve cavity using inflection points, atempting FWHM of Gaussian")
            popt, _ = curve_fit(gaussian, coord_data, flux)
            flux_fit = gaussian(coord_data, *popt)
            fwhm = 2.3548*(popt[2])
            fwhm_points = np.where(flux>=(popt[0]/2))[0][[0,-1]]
            
            # r_mean.append(fwhm/2)
            r_mean.append(np.max(np.abs([coord_data[fwhm_points[0]],coord_data[fwhm_points[-1]]])))
            
        if plot:
            poi_label = "Points of inflection" if i == 0 else None  # Only label first
            cavity_label = "Cavity region" if i == 0 else None
            gauss_label = f"FWHM of Gaussian" if j == 0 else None
            
            if poi_success:
                #outer region plots
                ax.plot(coord_data[:poi_peaks[0]],flux[:poi_peaks[0]],linestyle = linestyle,label = f"{coord}-axis",color = 'black')
                ax.plot(coord_data[poi_peaks[-1]:],flux[poi_peaks[-1]:],linestyle = linestyle,color = 'black')

                #inner cavity region plots
                ax.scatter(coord_data[poi_peaks],flux[poi_peaks],label = poi_label,color = "orange")
                ax.plot(coord_data[poi_peaks[0]:poi_peaks[-1]],flux[poi_peaks[0]:poi_peaks[-1]],label = cavity_label, color = "red")

            else:
            # else: #if point of inflection method fails
                # ax.plot(coord_data, flux_fit, linestyle=linestyle, label=f"{coord} Gaussian fit", color='blue')

                #outer region plots
                ax.plot(coord_data[:fwhm_points[0]],flux[:fwhm_points[0]],linestyle = linestyle,label = f"{coord}-axis",color = 'black')
                ax.plot(coord_data[fwhm_points[-1]:],flux[fwhm_points[-1]:],linestyle = linestyle,color = 'black')

                #inner cavity region plots
                ax.scatter(coord_data[fwhm_points],flux[fwhm_points],label = gauss_label,color = "green")
                ax.plot(coord_data[fwhm_points[0]:fwhm_points[-1]],flux[fwhm_points[0]:fwhm_points[-1]],color = "red")

            plt.title(f"Surface Brightness vs axis with POI @ {timestep}")
            ax.set_xlabel("x/y-axis [kpc]")
            ax.set_ylabel("Surface Brightness [mJy]")
            ax.legend()

        j =+1

    a = max(r_mean)
    b = min(r_mean)

    return (a,b,fig)

def fit_ellipsoid(a,b,mpl_ax=None,e_type="prolate"):
    
    """
    Adds a matplotlib Ellipse to input axis bases on a prolate/oblate ellipse with semi-major/minor axes a,b

    :param a: semi-major axis

    :param b: semi-minor axis

    :param mpl_ax: maplotlib axes e.g. 
        fig, ax = plt.subplots(figsize=(8, 8))
        fit_ellipsoid(...,mpl_ax=ax)

    :param e_type: ellipsoid type, either prolate or oblate

    :return ellipsoid_values (dict): dict of ellipsoid values such as a,b and volume
    """

    if e_type == "prolate":
        volume = (4/3) * np.pi * a * b**2 #assuming prolate ellipsoid
    
    elif e_type == "oblate":
        volume = (4/3) * np.pi * a**2 * b #assuming oblate ellipsoid

    else:
        raise ValueError(f"Ellipsoid type is '{e_type}', can only be 'oblate' or 'prolate'")
        
    if mpl_ax:
        ellipse = Ellipse(xy=(0, 0), width=b*2, height=a*2, 
                        edgecolor='r', facecolor='none', linewidth=2)
        mpl_ax.add_patch(ellipse)

    ellipsoid_values = {
        "a":a,
        "b":b,
        "type":e_type,
        "vol":volume
    }
    return ellipsoid_values

def calc_jet_power_cavity(a,b,env_type='cluster'):
    
    """
    Calculates jet power by fiting an ellipsoid, calculating work and power to expand the bubble of x volume

    :param a: ellipsoid semi-major axis

    :param b: ellipsoid semi-minor axis

    :param timestep: timestep str as seen in the hdf5 file paths e.g. "38.00 Myr"  

    :param env_file: tabulated environment csv file, provides a tabulated list of radii and pressures for a cluster env

    :return tuple(power,t_av): tuple containing jet power and cavity timescale
    """
    env = get_tabulated_env(env_type=env_type)
    kpc_to_m = 3.086e+19
    myr_to_sec = 3.15576e+13
    v_cavity = fit_ellipsoid(a,b,None,"prolate")["vol"] * kpc_to_m**3 #volume of caivty in L
    
    # dr = 13.752002592352559
    dr = 0
    radial_mask = env["radius"] <= (a + dr) * kpc_to_m  #find all radii up to top of ellipsoid
    R_proj = np.sqrt(a*b) * kpc_to_m #projected distance from cluster center to the center of the  cavity (Mendygral et al. 2011)
    # R_proj = np.sqrt(a**2 + b**2) * kpc_to_m #projected distance from cluster center to the center of the  cavity (Mendygral et al. 2011)

    R_proj_idx = np.where(env["radius"]>=R_proj)[0][0]
    
    t_refill = 2*np.sqrt(R_proj/-env["g"][R_proj_idx]) #refill time from cluster gravity 
    t_sound = R_proj*np.sqrt((0.60364*const.m_p)/(5/3*const.k*env["temp"][R_proj_idx]))
    t_buoy = R_proj * np.sqrt((0.75* np.pi * (b*kpc_to_m)**2)/(2*-env["g"][R_proj_idx]*v_cavity))
    t_av = np.mean([t_refill,t_sound,t_buoy])


    work_av = np.mean(v_cavity*env["prs"][radial_mask])
    power_av = (work_av/t_av)/2 #dividing by two as the cavity encapsulates both jets

    work2 = v_cavity * env["prs"][R_proj_idx]
    power2 = ((work2/t_av)/2)
 
    # print(f"a = {a}, Radii (tab) : {env['radius'][radial_mask]/kpc_to_m}")
    # print(f"R_proj = {R_proj / kpc_to_m}, R_proj (tab): {env['radius'][R_proj_idx]/kpc_to_m}")
    # print(f"Power average: {power_av}, Power 2: {power2}")
    returns ={
        "power": power_av,
        "t_refill":t_refill,
        "t_sound": t_sound,
        "t_buoy": t_buoy,
        "t_av": t_av
    }
    return returns


def plot_cavity_raw(config,redshift,timestep,file_path):
    
    """
    Plots the raw data of the xray cavity 
-
    :param config: config for xraise calculation see emission_params.yml e.g. "basic_xray_all"

    :param redshift: redshift e.g. 0.1

    :param timestep: timestep str as seen in the hdf5 file paths e.g. "38.00 Myr"  

    :param file_path: file path (str) to hdf5 file

    :return tuple(fig,ax): tuple of matplotlib figure and axes variables
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    hdf5_data = get_hdf5_data_xray(["flux","mx_kpc","my_kpc"],config=config,redshift=redshift,timestep=timestep,file_path=file_path)

    im = plt.pcolormesh(hdf5_data["mx_kpc"], hdf5_data["my_kpc"], 
                        np.log10(hdf5_data["flux"]), 
                        cmap='bone', 
                        rasterized=True,
                        vmin=np.percentile(np.log10(hdf5_data["flux"]),75)
                        # vmax=-7
                    )
        
    plt.ylabel('y-axis [kpc]')
    plt.xlabel('x-axis [kpc]')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    cbar = fig.colorbar(im, shrink = 0.83, pad=0.0)
    cbar.set_label(f"$\log_{{10}}$(Surface brightness [mJY])",fontsize = 12)

    # nans = np.sum(hdf5_data["flux"][np.isinf(hdf5_data["flux"])])
    # print(nans)

    return fig,ax
    
def plot_cavity_ellipsoid(config,redshift,timestep,file_path,env_type = 'cluster'):
   
    """
    Plots the raw data of the xray cavity with a fitted ellipse

    :param config: config for xraise calculation see emission_params.yml e.g. "basic_xray_all"

    :param redshift: redshift e.g. 0.1

    :param timestep: timestep str as seen in the hdf5 file paths e.g. "38.00 Myr"  

    :param file_path: file path (str) to hdf5 file
    """
    fig,ax = plot_cavity_raw(config,redshift,timestep,file_path)

    a,b, _ = calc_ellipse_radii(config=config,redshift=redshift,timestep=timestep,file_path=file_path,plot=False)
    ellipsoid = fit_ellipsoid(a=a,b=b,mpl_ax=ax)

    cavity_data = calc_jet_power_cavity(a=a,b=b,env_type=env_type)

    sim_name = os.path.basename(file_path).split(".")[0]
    ax.set_title(f"Cavity with ellipsoid fit: {sim_name} @ {timestep}")
    ax.annotate(
        f'Q = {cavity_data["power"]:.3e} [$\mathrm{{W}}$]',
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
        f'a: {a:.2f} [$\mathrm{{kpc}}$], b: {b:.2f} [$\mathrm{{kpc}}$]',
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
    ax.annotate(
        rf'Volume: {ellipsoid["vol"]:.2e} [$\mathrm{{kpc}}^3$]',
        xy=(0.05, 0.85),
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
        f'Cavity timescale: {cavity_data["t_av"]/3.15576e+13:.2f} [$\mathrm{{Myr}}$]',
        xy=(0.05, 0.80),
        xycoords='axes fraction',
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8
        )
    )

    return fig
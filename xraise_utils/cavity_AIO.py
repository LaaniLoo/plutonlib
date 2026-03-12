import plutonlib.xray.cavity_analysis as cavity_analysis

import plutokore as pk

import numpy as np
import scipy
import scipy.constants as const

import h5py
import os
import yaml

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit

import logging

#---Script config---#
cluster = False #need to load data files from cluster to find cavity_volume_hd
config = "basic_xray_all"
redshift = 0.05
data_dir = "./test"
diagnositc_path = os.path.join(data_dir,"cavity_diagnostics_better_peaks")

file_paths,file_tsteps = cavity_analysis.get_timesteps_all_files(config=config,redshift=redshift,data_dir=data_dir)

#loop across all available hdf5 files and timesteps
for file_path in file_paths:
    #---Files and paths---#
    src_path = os.path.join(os.path.expanduser('~'),'plutonlib/src/plutonlib')
    main_path = os.path.join(os.path.expanduser('~'),'plutonlib/')

    sim_name = os.path.basename(file_path).split(".hdf5")[0]
    with open (os.path.join(main_path,'xraise_utils/sim_directories.yml'),'r') as file:
        sim_dir_yml = yaml.safe_load(file)
    sim_dir = sim_dir_yml[sim_name]['sim_dir']

    os.makedirs(diagnositc_path, exist_ok=True)
    os.makedirs(f"{diagnositc_path}/{sim_name}/", exist_ok=True)

    # for timestep in sorted(file_tsteps[file_path],key=lambda x: float(x.split()[0])):
    for timestep in file_tsteps[file_path]:
        #---other---#
        output = cavity_analysis.timestep_to_output(timestep,sim_dir) if cluster else "None"
        env_type = 'group' if sim_name.split("_")[-1] == 'G' else 'cluster' #automatically change env to group if file ends with _G
        q_expect = 10**int(sim_name.split('_')[0].strip("Q")) #expected jet power from sim name e.g Q38 = 1e38

        #---Cavity Calculations---#
        kpc_to_m = 3.086e+19
        myr_to_sec = 3.15576e+13

        if cluster: #calculate cavity volume from grid cells 
            # cavity_volume_hd = cavity_analysis.calc_cavity_volume_hd(sim_dir,timestep)
            cavity_volume_hd = cavity_analysis.calc_cavity_volume_hd(sim_dir,output)


        a,b,flux_plt = cavity_analysis.calc_ellipse_radii(config=config,redshift=redshift,timestep=timestep,file_path=file_path,plot=True)
        v_cavity = cavity_analysis.fit_ellipsoid(a,b,None,"prolate")["vol"] * kpc_to_m**3 #volume of caivty in L
        cavity_data = cavity_analysis.calc_jet_power_cavity(a=a,b=b,env_type=env_type)

        #---Plotting---#
        cavity_plt = cavity_analysis.plot_cavity_ellipsoid(config=config,redshift=redshift,timestep=timestep,file_path=file_path,env_type=env_type)
        cavity_plt.savefig(f"{diagnositc_path}/{sim_name}/{timestep.split(' ')[0]}_{timestep.split(' ')[-1]}_cavity.png")
        flux_plt.savefig(f"{diagnositc_path}/{sim_name}/{timestep.split(' ')[0]}_{timestep.split(' ')[-1]}_ellipsoid_fit.png")

        #---Logging---#
        logging.basicConfig(
        filename=os.path.join(diagnositc_path,sim_name,'cavity_properties.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%H:%M:%S',
        force=True
        )

        def log_raw(text):
            logger = logging.getLogger()  # get root logger
            for handler in logger.handlers:
                for line in text.splitlines():   # split at \n
                    handler.stream.write(line + '\n')
                    handler.flush()

        header = (
        f"\n{'='*80}"
        f"\n CAVITY PROPERTIES: {os.path.basename(file_path)}"
        f"\n Timestep: {timestep.split(' ')[0]} [Myr]"
        f"\n Output: {output}"
        f"\n Environment type: {env_type}"
        f"\n Path: {file_path}"
        f"\n Data Path: {sim_dir}"
        f"\n{'='*80}")
        log_raw(header)

        log_raw(f'ELLIPSOID PROPERTIES: \nsemi-major axis a = {a:.2f} [kpc], semi-minor axis b = {b:.2f} [kpc]')
        if cluster:
            log_raw(f'Simulated (grid-cell) cavity volume = {cavity_volume_hd:.3e} [kpc^3]')
            log_raw(f"Estimated cavity volume = {v_cavity/kpc_to_m**3:.3e} [kpc^3]")
            log_raw(cavity_analysis.percent_err(v_cavity/kpc_to_m**3,cavity_volume_hd))
        else:
            log_raw(f"Estimated cavity volume = {v_cavity/kpc_to_m**3:.3e} [kpc^3]")

        log_raw(f'\nTIMESCALES [Myr]: \nrefill = {cavity_data["t_refill"]/myr_to_sec:.2f}, sound = {cavity_data["t_sound"]/myr_to_sec:.2f}, buoyant = {cavity_data["t_buoy"]/myr_to_sec:.2f}, average = {cavity_data["t_av"]/myr_to_sec:.2f}\n')

        log_raw(f'\nPOWER CALCULATION:\nJet power = {cavity_data["power"]:.3e} [W]')
        log_raw(cavity_analysis.percent_err(cavity_data["power"],q_expect))

import plutonlib.utils as pu
from plutonlib.colours import pcolours


import os
from astropy import units as u
import numpy as np
import configparser
from pathlib import Path 

# start_dir = r"/mnt/g/My Drive/Honours S4E (2025)/Notebooks/" #starting directory, used to save files starting in this dir

#TODO env_var or config file?
try: #Checks if PLUTON_START_DIR is an env var
    start_dir = os.environ["PLUTONLIB_START_DIR"]
except KeyError: #if not env var creates a plutonlib_output folder in current wd
    new_dir = os.path.join(os.getcwd(),"plutonlib_output")
    is_dir = os.path.isdir(new_dir)

    start_dir = new_dir if is_dir is True else os.makedirs(new_dir)

    print("\n")
    print(f"{pcolours.WARNING}environment variable PLUTONLIB_START_DIR not found, setting save location as {start_dir}")
    print(f"Creating plutonlib_output folder in {os.getcwd()}") if is_dir else None
    # print("\n")

src_path = os.path.join(os.path.expanduser('~'),'plutonlib/src/plutonlib')

try: #Checks for PLUTO_DIR env var
    plutodir = os.environ["PLUTO_DIR"]
except KeyError:
    print(f'{pcolours.WARNING}PLUTO_DIR env var not found, please set the location of the PLUTO code')


if os.path.isdir(os.path.join(plutodir, "Simulations")): #if simulation dir doesn't exist 
    sim_dir = os.path.join(plutodir, "Simulations")
else:
    raise FileNotFoundError(f"{pcolours.WARNING} Simulation directory not found, needs to be in PLUTO_DIR ({plutodir}), see sim_save.sh")

profiles = {
    "all": ["x1", "x2", "x3", "rho", "prs", "vx1", "vx2", "vx3", "SimTime"],
    
    "xy_rho_prs": ["x1", "x2", "rho", "prs"],
    "xz_rho_prs": ["x1", "x3", "rho", "prs"],
    "yz_rho_prs": ["x2", "x3", "rho", "prs"],

    "xy_vel": ["x1", "x2", "vx1", "vx2"],
    "xz_vel": ['x1','x3','vx1','vx3'],
    "yz_vel": ["x2", "x3", 'vx2','vx3'],
}

coord_systems = {
    "CYLINDRICAL": ['r','z','theta'],
    "CARTESIAN": ['x','y','z'],
    "SPHERICAL": ['r','theta','phi']  
}

# Read norm values from ini file
cur_dir = Path(__file__).resolve().parent          # directory containing this .py
# ini_path_default = cur_dir / "pluto_units_default.ini" 
ini_path_default = cur_dir / "pluto_units.ini" 


def get_pluto_units(sim_coord,d_files,ini_path=ini_path_default):
    """
    gets the values required to normalise PLUTO "code-units" to CGS, then can converted to SI
    """
    sel_coords = coord_systems[sim_coord] #gets the coord vars for the specific coord sys

    config = configparser.ConfigParser()
    config.optionxform = str
    # print("Loading:",ini_path)
    config.read(ini_path)
    # print(f"Read {ini_path}")
    norm_values = {k: float(v) for k, v in config["normalisations"].items()}

    pluto_units = {
    "x1": {"norm": norm_values["x1"], "cgs": u.cm, "si": u.m, "var_name": "x1", "coord_name": f"{sel_coords[0]}"},
    "x2": {"norm": norm_values["x2"], "cgs": u.cm, "si": u.m, "var_name": "x2", "coord_name": f"{sel_coords[1]}"},
    "x3": {"norm": norm_values["x3"], "cgs": u.cm, "si": u.m, "var_name": "x3", "coord_name": f"{sel_coords[2]}"},
    "rho": {"norm": norm_values["rho"], "cgs": u.g / u.cm**3, "si": u.kg / u.m**3, "var_name": "Density"},
    "prs": {"norm": norm_values["prs"], "cgs": u.dyn / u.cm**2, "si": u.Pa, "var_name": "Pressure"},
    "vx1": {"norm": norm_values["vx1"], "cgs": u.cm / u.s, "si": u.m / u.s, "var_name": f"{sel_coords[0]}_Velocity"},
    "vx2": {"norm": norm_values["vx2"], "cgs": u.cm / u.s, "si": u.m / u.s, "var_name": f"{sel_coords[1]}_Velocity"},
    "vx3": {"norm": norm_values["vx3"], "cgs": u.cm / u.s, "si": u.m / u.s, "var_name": f"{sel_coords[2]}_Velocity"},
    "T": {"norm": norm_values["T"], "cgs": u.K, "si": u.K, "var_name": "Temperature"},
    "SimTime_s": {"norm": np.linspace(0, norm_values["SimTime_s"], len(d_files)), "cgs": u.s, "si": u.s, "var_name": "Time (seconds)"},
    "SimTime": {"norm": np.linspace(0, norm_values["SimTime"], len(d_files)), "cgs": u.yr, "si": u.s, "var_name": "Time"},
    }
    
    return pluto_units 

def value_norm_conv(var_name,d_files,raw_data = None, self = 0):
    """
    gets value from get_pluto_units to convert to SI or CGS
    """

    pluto_units = get_pluto_units("CARTESIAN",d_files) #NOTE I don't think it needs sim_coord so left as CARTESIAN

    cgs_unit =  pluto_units[var_name]["cgs"]
    si_unit = pluto_units[var_name]["si"]
    norm = pluto_units[var_name]["norm"]

    np.asarray(raw_data) if np.any(raw_data) and not isinstance(raw_data,np.ndarray) else raw_data #calc only works if raw_data is numpy array 

    if self: #used to convert norm values from pluto_units into si or cgs
        conv_si = (norm*cgs_unit).si.value #converts the units as well as normalize 
        conv_cgs = (norm*cgs_unit).value


    else: #convert raw_data
        conv_si = (raw_data * norm *cgs_unit).si.value #converts the units as well as normalize 
        conv_cgs = (raw_data * norm *cgs_unit).value

    returns = {"cgs": conv_cgs,
               "si": conv_si

    }

    return returns

#---OLD CGS UNITS---#
# CGS_code_units = {
#     "x1": [1.496e13, (u.cm), u.m, "x1", f"{sel_coord[0]}"],
#     "x2": [1.496e13, (u.cm), u.m, "x2", f"{sel_coord[1]}"],
#     "x3": [1.496e13, (u.cm), u.m, "x3", f"{sel_coord[2]}"],
#     "rho": [1.673e-24, (u.gram / u.cm**3), u.kg / u.m**3, "Density"],
#     "prs": [1.673e-14, (u.dyn / u.cm**2), u.Pa, "Pressure"],
#     "vx1": [1.000e05, (u.cm / u.s), u.m / u.s, f"{sel_coord[0]}_Velocity"],
#     "vx2": [1.000e05, (u.cm / u.s), u.m / u.s, f"{sel_coord[1]}_Velocity"],
#     "vx3": [1.000e05, (u.cm / u.s), u.m / u.s, f"{sel_coord[2]}_Velocity"],
#     "T": [1.203e02, (u.K), u.K, "Temperature"],
#     "t_s": [1.496e08, (u.s),u.s, "Time"],
#     "t_yr": [4.744e00, (u.yr), u.s, "Time "], 
# }

#---Sim data tree structure---#
# In pluto-master
# Simulations
# ├── Jet
# │   ├── Assn_hllc
# │   │   ├── data.0000.dbl
# │   │   ├── data.0001.dbl
# │   │   ├── data.0002.dbl
# │   │   ├── data.0003.dbl
# │   │   ├── data.0004.dbl
# │   │   ├── data.0005.dbl
# │   │   ├── data.0006.dbl
# │   │   ├── dbl.out
# │   │   ├── grid.out
# │   │   └── restart.out

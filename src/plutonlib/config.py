# import plutonlib.utils as pu
# import plutonlib.plot as pp
from plutonlib.colours import pcolours


import os

from astropy import units as u
from astropy import constants as const 

import numpy as np

import configparser
import glob 

from pathlib import Path 
import time
from functools import lru_cache


# start_dir = r"/mnt/g/My Drive/Honours S4E (2025)/Notebooks/" #starting directory, used to save files starting in this dir

src_path = os.path.join(os.path.expanduser('~'),'plutonlib/src/plutonlib')

try: #Checks for PLUTO_DIR env var
    plutodir = os.environ["PLUTO_DIR"]
except KeyError:
    print(f'{pcolours.WARNING}PLUTO_DIR env var not found, please set the location of the PLUTO code')


if os.path.isdir(os.path.join(plutodir, "Simulations")): #if simulation dir doesn't exist 
    sim_dir = os.path.join(plutodir, "Simulations")
else:
    raise FileNotFoundError(f"{pcolours.WARNING} Simulation directory not found, needs to be in PLUTO_DIR ({plutodir}), see sim_save.sh")

arr_type_key = {
    "e": "1D cell edge coordinate arrays [x, y, z]",
    "m": "1D cell midpoint coordinate arrays [x, y, z]",
    "d": "1D cell delta arrays [x, y, z]",
    "nc": "3D cell edge arrays [x, y, z]",
    "cc": "3D cell midpoint arrays [x, y, z]"
    }

def profiles(arr_type=None):

    coord_prefixes = {
        "e":  ["ex",  "ey",  "ez"],
        "m":  ["mx",  "my",  "mz"],
        "d":  ["dx",  "dy",  "dz"],
        # "x":  ["x1",  "x2",  "x3"],
        "nc": ["ncx", "ncy", "ncz"],
        "cc": ["ccx", "ccy", "ccz"],
        None: ["x1",  "x2",  "x3"],  # default if None
    }

    # Get the correct coordinate labels, defaults to mx ...
    coords = coord_prefixes.get(arr_type, ["x1", "x2", "x3"])
    x, y, z = coords

    profiles = {
        "all": [x, y, z, "rho", "prs", "vx1", "vx2", "vx3",'tr1', "sim_time"],

        "grid_time": [x,y,z,"sim_time"],
        "grid_tracer": [x,y,z,"tr1"],
        "grid": [x,y,z],

        "xy_rho_prs": [x, y, "rho", "prs"],
        "xz_rho_prs": [x, z, "rho", "prs"],
        "yz_rho_prs": [y, z, "rho", "prs"],

        "xy_vel":     [x, y, "vx1", "vx2"],
        "xz_vel":     [x, z, "vx1", "vx3"],
        "yz_vel":     [y, z, "vx2", "vx3"],
    }
    returns = {"profiles":profiles,"coord_prefixes":coord_prefixes}
    return returns

#TODO env_var or config file?
def get_start_dir(origin_dir,**kwargs):
    warnings = []
    try: #Checks if PLUTON_START_DIR is an env var
        start_dir = os.environ["PLUTONLIB_START_DIR"]
        warnings.append(f"{pcolours.WARNING}{start_dir} (PLUTONLIB_START_DIR)")

    except KeyError: #if not env var creates a plutonlib_output folder at origin_dir
        if 'run_name' in kwargs: #make plutonlib_output folder more descriptive if run_name is passed
            run_name = kwargs['run_name']
            new_dir = os.path.join(origin_dir,f"{run_name}_plutonlib_output")
        else:
            new_dir = os.path.join(origin_dir,"plutonlib_output")

        is_dir = os.path.isdir(new_dir)
        if not is_dir:
            os.makedirs(new_dir)
            print(f"{pcolours.WARNING}Creating plutonlib_output folder in {origin_dir}") 
            print("\n")

        start_dir = new_dir
        warnings.append(f"{pcolours.WARNING}{start_dir} (Missing environment variable PLUTONLIB_START_DIR)")
    return {"start_dir":start_dir,"warnings":warnings}

coord_systems = {
    "CYLINDRICAL": ['r','z','theta'],
    "CARTESIAN": ['x','y','z'],
    "SPHERICAL": ['r','theta','phi']  
}

# Read norm values from ini file
def get_ini_file(ini_file = None):
    """
    Gets pluto_units.ini file from plutonlib src directory,
    should follow naming convention: name_units.ini 
    """
    if ini_file is None:
        ini_path = os.path.join(src_path,"pluto_units" + ".ini")
    
    else:
        ini_path = os.path.join(src_path,ini_file + ".ini")
    
    is_file = os.path.isfile(ini_path)
    if not is_file:
        raise FileNotFoundError(f"{pcolours.WARNING}{ini_path} Not found")
    
    return ini_path 

def pluto_ini_info(sim_dir):
    job_info_dir = os.path.join(sim_dir,"job_info")
    job_dir_files = glob.glob(f"{job_info_dir}/*.ini") #gets all ini files in job_info_dir 
    latest_ini = max(job_dir_files,key=os.path.getctime)
    ini_name = latest_ini.split("/")[-1]

    config = configparser.ConfigParser(allow_no_value=True)
    config.read(latest_ini)

    grid_setup = config.options("Grid")
    raw_usr_params = config.options("Parameters")

    usr_params = {
        k: float(v.split(";",1)[0].strip())
        for line in raw_usr_params if " " in line
        for k,v in [line.split(None,1)]
    }

    key_params = {key: usr_params[key] for key in ['jet_pwr','jet_spd','jet_chi','env_rho_0','env_temp','wind_vx1','wind_vx2','wind_vx3']}

    returns = {"grid_setup": grid_setup,"usr_params":usr_params,"key_params":key_params,"ini_name":ini_name}

    return returns

def get_pluto_units(sim_coord,ini_file):
    """
    gets the code and user unit values e.g. x1 = 1*kpc from the specified ini file, 
    where multiplying an array by the code unit value normalizes it to that unit.
    """
    if ini_file is None: #gets raise error safer than null assignment
        ini_path = get_ini_file(ini_file=None)
        raise ValueError(f"{pcolours.WARNING}ini_file is None, please load defaults from {ini_path}")
    
    ini_path = get_ini_file(ini_file=ini_file)

    sel_coords = coord_systems[sim_coord] #gets the coord vars for the specific coord sys

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(ini_path)

    c_unit = u.def_unit("c",const.c) #adds 1c (speed of light) as constant
    u.add_enabled_units([c_unit])
    code_unit_values = {k: u.Unit(v) for k, v in config["code_unit_values"].items()}
    usr_unit_values = {k: u.Unit(v) for k, v in config["usr_unit_values"].items()}

    pluto_units = {
        "x1": {"code_uv": code_unit_values["x1"], "usr_uv": usr_unit_values["x1"], "var_name": "x1", "coord_name": f"{sel_coords[0]}"},
        "x2": {"code_uv": code_unit_values["x2"], "usr_uv": usr_unit_values["x2"], "var_name": "x2", "coord_name": f"{sel_coords[1]}"},
        "x3": {"code_uv": code_unit_values["x3"], "usr_uv": usr_unit_values["x3"], "var_name": "x3", "coord_name": f"{sel_coords[2]}"},
        "rho": {"code_uv": code_unit_values["rho"], "usr_uv": usr_unit_values["rho"], "var_name": "Density"},
        "prs": {"code_uv": code_unit_values["prs"], "usr_uv": usr_unit_values["prs"], "var_name": "Pressure"},
        "vx1": {"code_uv": code_unit_values["vx1"], "usr_uv": usr_unit_values["vx1"], "var_name": f"{sel_coords[0]}_Velocity"},
        "vx2": {"code_uv": code_unit_values["vx2"], "usr_uv": usr_unit_values["vx2"], "var_name": f"{sel_coords[1]}_Velocity"},
        "vx3": {"code_uv": code_unit_values["vx3"], "usr_uv": usr_unit_values["vx3"], "var_name": f"{sel_coords[2]}_Velocity"},
        "tr1" : {"code_uv": None, "usr_uv": None, "var_name": "Tracer_1"},
        "sim_time": {"code_uv": code_unit_values["sim_time"], "usr_uv": usr_unit_values["sim_time"], "var_name": "Time"},
        "ini_file": ini_file,
    }
    
    return pluto_units 

def code_to_usr_units(var_name,raw_data = None, self = 0,ini_file = None):
    """
    gets unit value from get_pluto_units to convert from code units to the user specified units in the ini file.
    """
    pluto_units = get_pluto_units("CARTESIAN",ini_file=ini_file) #NOTE I don't think it needs sim_coord so left as CARTESIAN

    code_uv =  pluto_units[var_name]["code_uv"]
    usr_uv = pluto_units[var_name]["usr_uv"]
    # norm = pluto_units[var_name]["norm"]

    np.asarray(raw_data) if np.any(raw_data) and not isinstance(raw_data,np.ndarray) else raw_data #calc only works if raw_data is numpy array 

    #convert raw_data
    if code_uv is None or usr_uv is None:  #skips if it doesn't need converting
        conv_to_code_units = raw_data
        conv_data_cuv = raw_data  #equiv to arrays in cgs units
        conv_data_uuv = raw_data
        uv_usr = None
    else:
        conv_to_code_units = (raw_data*code_uv)  
        conv_data_cuv = conv_to_code_units.value  #equiv to arrays in cgs units
        conv_data_uuv = conv_to_code_units.to(usr_uv).value #equiv to arrays in si units
        uv_usr = (1*code_uv).to(usr_uv).value

    returns = {
        "uv_usr":uv_usr, #like a scale factor??
        "conv_data_cuv":conv_data_cuv, 
        "conv_data_uuv":conv_data_uuv 
    }

    return returns

#---Sim data tree structure---#
def plutonlib_tree_helper():
    tree = """pluto_master/
└── Simulations/
    └── sim_type/
        └── run_name/
            ├── data.0000.dbl.h5
            ├── data.0000.dbl.xmf
            ├── dbl.h5.out
            ├── grid.out
            ├── restart.out
            ├── job_info
            │   └── pluto_template.ini
            ├── log
            │   ├── pluto.0.log
            │   ├── pluto.1.log
            │   ├── pluto.2.log
            │   ├── pluto.3.log
            │   ├── pluto.4.log
            │   └── pluto.5.log
            └── run_name_plutonlib_output
                └── Jet_wind_test_temp_xz_vel_plot.png"""
    print(tree)


import os
from astropy import units as u
import numpy as np

start_dir = r"/mnt/g/My Drive/Honours S4E (2025)/Notebooks/" #starting directory, used to save files starting in this dir

plutodir = os.environ["PLUTO_DIR"]

profiles = {
    "all": ["x1", "x2", "x3", "rho", "prs", "vx1", "vx2", "vx3", "SimTime"],
    "2d_rho_prs": ["x1", "x2", "rho", "prs"],
    "2d_vel": ["x1", "x2", "vx1", "vx2"],

    "yz_rho_prs": ["x2", "x3", "rho", "prs"],
    "yz_vel": ["x2", "x3", 'vx2','vx3'],

    "xz_rho_prs": ['x1','x3','rho','prs'],
    "xz_vel": ['x1','x3','vx1','vx3'],
}

coord_systems = {
    "CYLINDRICAL": ['r','z','theta'],
    "CARTESIAN": ['x','y','z'],
    "SPHERICAL": ['r','theta','phi']  
}

def get_pluto_units(sim_coord,d_files):
    """
    gets the values required to normalise PLUTO "code-units" to CGS, then can converted to SI
    """
    sel_coords = coord_systems[sim_coord] #gets the coord vars for the specific coord sys

    #TODO assign from config etc?
    #key: norm, CGS, SI, var_name?, formatted coord in sys
    pluto_units = {
    "x1": {"norm": 1.496e13, "cgs": u.cm, "si": u.m, "var_name": "x1", "coord_name": f"{sel_coords[0]}"},
    "x2": {"norm": 1.496e13, "cgs": u.cm, "si": u.m, "var_name": "x2", "coord_name": f"{sel_coords[1]}"},
    "x3": {"norm": 1.496e13, "cgs": u.cm, "si": u.m, "var_name": "x3", "coord_name": f"{sel_coords[2]}"},
    "rho": {"norm": 1.673e-24, "cgs": (u.gram / u.cm**3), "si": (u.kg / u.m**3), "var_name": "Density"},
    "prs": {"norm": 1.673e-14, "cgs": (u.dyn / u.cm**2), "si": u.Pa, "var_name": "Pressure"},
    "vx1": {"norm": 1.000e05, "cgs": (u.cm / u.s), "si": (u.m / u.s), "var_name": f"{sel_coords[0]}_Velocity"},
    "vx2": {"norm": 1.000e05, "cgs": (u.cm / u.s), "si": (u.m / u.s), "var_name": f"{sel_coords[1]}_Velocity"},
    "vx3": {"norm": 1.000e05, "cgs": (u.cm / u.s), "si": (u.m / u.s), "var_name": f"{sel_coords[2]}_Velocity"},
    "T": {"norm": 1.203e02, "cgs": u.K, "si": u.K, "var_name": "Temperature"},

    "SimTime_s": {"norm": np.linspace(0,1.496e08,len(d_files)), "cgs": u.s, "si": u.s, "var_name": "Time"}, #NOTE not needed as below can be converted to si for seconds
    "SimTime": {"norm": np.linspace(0,4.744e00,len(d_files)), "cgs": u.yr, "si": u.s, "var_name": "Time"}, 
    }
    
    return pluto_units 

def value_norm_conv(sim_coord,var_name,d_files,raw_data = None,self = 0):
    """
    gets value from get_pluto_units to convert to SI or CGS
    """

    pluto_units = get_pluto_units(sim_coord,d_files)
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
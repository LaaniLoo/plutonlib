import plutonlib.utils as pu
import plutonlib.config as pc
import sys

profiles = pc.profiles
coord_systems = pc.coord_systems
plutodir = pc.plutodir

import os
import numpy as np

import plutokore.io as pk_io
from plutokore.simulations import get_output_count as pk_sim_count

from astropy import units as u
from collections import defaultdict 

import time
from concurrent.futures import ThreadPoolExecutor

def pluto_loader(sim_type, run_name, profile_choice,max_workers = None):
    """
    Loads simulation data from a specified Pluto simulation.

    Parameters:
    -----------
    sim_type : str
        Type of simulation to load (e.g., "Jet", "Stellar_Wind") #NOTE see config for saving structure.
    run_name : str
        Name of the specific simulation file to load e.g. "default".
    profile_choice : str
        Index selecting a profile from predefined variable lists (#NOTE found in config.py):
        - "2d_rho_prs": ["x1", "x2", "rho", "prs"]

    Returns:
    --------
    dict
        Dictionary containing:
        - vars: dictionary of order vars[d_file][var_name] e.g. vars["data_0"]["x1"]
        - var_choice: List of variable names corresponding to the selected profile.
        - vars_extra: contains the geometry of the sim
        - d_files: contains a list of the available data files for the sim
    """
    vars = defaultdict(list) # Stores variables for each D_file
    vars_extra = []

    var_choice = profiles[profile_choice]
    # print("Var Choice:", var_choice)
    wdir = os.path.join(plutodir, "Simulations", sim_type, run_name)

    #NOTE USE FOR LAST OUTPUT ONLY
    # nlinf = pk_io.nlast_info(w_dir=wdir) #info dict about PLUTO outputs

    n_outputs = pk_sim_count(wdir) # grabs number of data output files, might need datatype
    d_files = [f"data_{i}" for i in range(n_outputs + 1)]

    data_0 = pk_io.pload(0,wdir)
    geometry = data_0.geometry #gets the geometry of the first file = fast

    loaded_vars = [v for v in var_choice if hasattr(data_0, v)]
    # print("Loaded Vars:", loaded_vars)
    non_vars = set(var_choice) - set(loaded_vars)

    if non_vars:
        print(f"Simulation {run_name} doesn't contain: {', '.join(non_vars)}")
        print("\n")


    def load_file(output_num):
        data = pk_io.pload(output_num, w_dir=wdir)
        return output_num, {v: getattr(data, v) for v in loaded_vars}

    # Parallel load all files
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(load_file, range(n_outputs + 1))
        
        # Process results in completion order
        for output_num, file_data in results:
            vars[f"data_{output_num}"] = file_data
    

    var_choice = [v for v in var_choice if v not in non_vars]
    vars_extra.append(geometry) # gets the geo of the sim, always loads first file

    return {"vars": vars, "var_choice": var_choice,"vars_extra": vars_extra,"d_files": d_files} #"nlinf": nlinf

def pluto_conv(sim_type, run_name, profile_choice,**kwargs):
    """
    Converts Pluto simulation variables from code units to CGS and SI units.

    Parameters:
    -----------
    sim_type : str
        Type of simulation to load (e.g., "Jet", "Stellar_Wind") #NOTE see config for saving structure.
    run_name : str
        Name of the specific simulation file to load e.g. "default".
    profile_choice : str
        Index selecting a profile from predefined variable lists (#NOTE found in config.py):
        - "2d_rho_prs": ["x1", "x2", "rho", "prs"]

    Returns:
    --------
    dict
        Dictionary containing:
        - vars_si: dictionary of order vars[d_file][var_name] e.g. vars["data_0"]["x1"]
        - var_choice: List of variable names corresponding to the selected profile.
        - d_files: contains a list of the available data files for the sim
    """
    loaded_data = pluto_loader(sim_type, run_name, profile_choice)
    d_files = loaded_data["d_files"]
    vars_dict = loaded_data["vars"]
    var_choice = loaded_data["var_choice"] # chosen vars at the chosen profile
    sim_coord = loaded_data["vars_extra"][0] #gets the coordinate sys of the current sim
    
    
    vars_si = defaultdict(dict)


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


    # Process each file and variable

    for d_file in d_files:
        for var_name in var_choice:
            # if var_name not in vars_dict[d_file]: 
            #     continue  # Skip missing variables

            raw_data =  vars_dict[d_file][var_name]

            conv_vals = pc.value_norm_conv(sim_coord,var_name,d_files,raw_data) #converts the raw pluto array
            vars_si[d_file][var_name] = conv_vals["cgs"] if var_name == "SimTime" else conv_vals["si"] #NOTE keep SimTime as yrs for now
    



    return {"vars_si": vars_si, "var_choice": var_choice,"d_files": d_files,"sim_coord": sim_coord}

def get_profiles(sim_type,run,profiles):
    """
    Prints available profiles for a specific simulation
    """
    data = pluto_loader(sim_type,run,"all") #NOTE pl should be faster than pc
    var_choice = data["var_choice"]
    vars = data["vars"]["data_0"]

    for var in var_choice[:-1]: # doesn't include SimTime as it has no size
        if vars[var].size == 1:
            avail_vars = var_choice
            avail_vars.remove(var) # removes e.g. x3 in "Jet" if its only len 1 so x3 profiles aren't included

    else:
        avail_vars = var_choice

    keys = list(profiles.keys())

    print("Available profiles:")
    for i, prof in enumerate(keys):
        vars_set = set(avail_vars)
        prof_set = set(profiles[prof])
        common = vars_set & prof_set

        if len(common) >=4: #since the profiles are usually 4 elements make sure at least 4 match
            print(f"{i}: {prof}, {profiles[prof]}")
            sys.stdout.flush()

    return keys

def select_profile(sim_type,run,profiles):
    keys = get_profiles(sim_type,run,profiles)

    while True:
        choice = input("Enter the number of the profile you want to select (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            print("Selection cancelled.")
            return None  # Return None if the user quits

        if choice.isdigit():
            choice = int(choice)

            if 0 <= int(choice) < len(profiles):
                return keys[choice]
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(profiles) - 1}.")
        else:
            print("Invalid input. Please enter a valid number or 'q' to quit.")

def pluto_load_profile(sim_type,sel_runs,sel_prof,all = 0):

    sel = 0 if sel_runs is None else 1

    #TODO could be in config to load following save tree?
    run_dirs = os.path.join(plutodir, "Simulations", sim_type)
    all_runs = [
        d for d in os.listdir(run_dirs) if os.path.isdir(os.path.join(run_dirs, d))
    ]

    # used to selected if plotting all subdirs or select runs
    if sel == 0:
        run_names = all_runs
        print(f"Subdirectories:, {run_names}")

    elif sel == 1:
        run_names = sel_runs

    # Assign a profile number for each run (supports duplicates)
    profile_choices = defaultdict(list)  # Change from a single variable to defaultdict(list)

    if sel_prof is None: #use if usr input multiple profiles, diff per run
        if not all:
            for run in run_names:
                profile_choice = select_profile(sim_type,run,profiles)
                profile_choices[run].append(profile_choice)  # Appends profile to list
                print(f"Selected profile {profile_choice} for run {run}: {profiles[profile_choice]}")

        elif all:
                profile_choice = "all"
                print(f"Selected profile {profiles[profile_choice]} for all runs")
                print("\n")
    
    if sel_prof is not None: #use if want one profile across all runs
        for run in run_names:
            get_profiles(sim_type,run,profiles)
            profile_choice = sel_prof
            profile_choices[run].append(profile_choice)  # Appends profile to list

            try:
                print(f"Selected profile {profile_choice} for run {run}: {profiles[profile_choice]}")
                print("\n")

            except KeyError:
                raise KeyError(f"{profile_choice} is not an available profile")



    return {'run_names':run_names,'profile_choices':profile_choices}

def pluto_sim_info(sim_type,sel_runs = None): #TODO Stellar wind is symm so indexing wont work
    run_data = pluto_load_profile(sim_type, sel_runs,all = 1)
    run_names = run_data['run_names'] #loads the run names and selected profiles for runs
    coord_shape = defaultdict(list)
    # pluto_load_data = pluto_loader(sim_type,run_names[0],profile_choices[run_names[0]][0])
    # d_files = pluto_load_data["d_files"]


    for run in run_names:
        var_shape = []
        coord_join = []
        data = pluto_conv(sim_type,run,"all")
        d_files = data["d_files"]
        var_dict = data["vars_si"]
        var_choice = data["var_choice"]

        geo = data["sim_coord"]
        
        coords = var_choice[:3]
        vars = var_choice[3:]

        title_string = f"Run {run}: geometry = {geo} "
        print(title_string)
        print("-"*len(title_string))
        print(f"Available data files for {run}: {d_files}")
        d = var_dict[d_files[0]]

        for i, var in enumerate(vars):

            var_shape.append(d[var].shape)

            if i < len(coords):
                coord_shape[run].append(d[coords[i]].shape)
                

            if coord_shape[run][-1:][0][0] == 1: #removes last if small
                coord_shape[run] = coord_shape[run][:-1]
                coords = coords[:-1]



        coord_join.append(tuple(cs[0] for cs in coord_shape[run]))
        # coord_join = [(coord_shape[run][0][0],coord_shape[run][1][0])]
        print(f"{coords} shape: {coord_join}")


        for i, shp in enumerate(var_shape):
            if i < len(vars) and shp == coord_join[0]:
                print(f"{vars[i]} is indexed {coords}, with shape: {shp}")
            elif i < len(vars) and shp != coord_join[0]:
                print(f"{vars[i]} has shape {shp}")
        print("\n")

    return {"coord_shape": coord_shape}




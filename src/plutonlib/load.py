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

def pluto_loader(sim_type, run_name, profile_choice):
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
    non_vars = []
    vars_extra = []

    var_choice = profiles[profile_choice]

    wdir = os.path.join(plutodir, "Simulations", sim_type, run_name)

    #NOTE USE FOR LAST OUTPUT ONLY
    nlinf = pk_io.nlast_info(w_dir=wdir) #info dict about PLUTO outputs
    n_outputs = pk_sim_count(wdir) # grabs number of data output files, might need datatype
    # D = pk.io.pload(nlinf["nlast"], w_dir=wdir)

    # Load all available data files, change d_all from 0,1 for 0th output NOTE excludes 0th
    d_all = {f"data_{output}": pk_io.pload(output, w_dir=wdir) for output in range(0,n_outputs + 1)} # Loads all available data files
    d_files = list(d_all.keys()) # list of data files as keys
    # print("Loaded files:",d_files) 

    for d_file in d_files:
        vars[d_file] = {} # not defaultdict(list) as would double list

        # Validate variable names and store by var_name
        for var_name in var_choice:
            if hasattr(d_all[d_file], var_name):  # Check first file, store by name NOTE
                vars[d_file][var_name] = (getattr(d_all[d_file], var_name))

            elif var_name not in non_vars: #if sim doesnt have var, elif for first file otherwise print lots, checks flagged vars
                if not hasattr(d_all[d_files[0]], var_name): 
                    print(f"Simulation {run_name} Doesn't Contain", var_name)
                    print("\n")
                    non_vars.append(var_name)

    var_choice = [x for x in var_choice if x not in non_vars] #removes vars not in sim
    vars_extra.append(d_all[d_files[0]].geometry) # gets the geo of the sim, always loads first file

    return {"vars": vars, "var_choice": var_choice,"vars_extra": vars_extra,"d_files": d_files, "nlinf": nlinf}

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
        - CGS_code_units: dictionary of the normalization value, units in CGS/SI and other important labels
        - var_choice: List of variable names corresponding to the selected profile.
        - d_files: contains a list of the available data files for the sim
    """
    loaded_data = pluto_loader(sim_type, run_name, profile_choice)
    d_files = loaded_data["d_files"]
    vars_dict = loaded_data["vars"]
    var_choice = loaded_data["var_choice"] # chosen vars at the chosen profile
    sim_coord = loaded_data["vars_extra"][0] #gets the coordinate sys of the current sim
    vars
    
    vars_si = defaultdict(dict)

    # coord_shape = vars_dict[d_files[0]]["x2"].shape[0] #size of x2 dimension, not sure why x2, used for time linspace
    t_array = []
    sel_coord = coord_systems[sim_coord]

    CGS_code_units = {
        "x1": [1.496e13, (u.cm), u.m, "x1", f"{sel_coord[0]}"],
        "x2": [1.496e13, (u.cm), u.m, "x2", f"{sel_coord[1]}"],
        "x3": [1.496e13, (u.cm), u.m, "x3", f"{sel_coord[2]}"],
        "rho": [1.673e-24, (u.gram / u.cm**3), u.kg / u.m**3, "Density"],
        "prs": [1.673e-14, (u.dyn / u.cm**2), u.Pa, "Pressure"],
        "vx1": [1.000e05, (u.cm / u.s), u.m / u.s, f"{sel_coord[0]}_Velocity"],
        "vx2": [1.000e05, (u.cm / u.s), u.m / u.s, f"{sel_coord[1]}_Velocity"],
        "vx3": [1.000e05, (u.cm / u.s), u.m / u.s, f"{sel_coord[2]}_Velocity"],
        "T": [1.203e02, (u.K), u.K, "Temperature"],
        "t_s": [1.496e08, (u.s),u.s, "Time"],
        "t_yr": [4.744e00, (u.yr), u.s, "Time "], 
    }

    # Process each file and variable

    for d_file in d_files:
        for var_name in var_choice:
            # if var_name not in vars_dict[d_file]: 
            #     continue  # Skip missing variables
                
            raw_data = vars_dict[d_file][var_name]
            
            # Finding simtime
            if var_name == "SimTime":
                t_var = "t_yr"  #NOTE use "t_s" for seconds
                norm = CGS_code_units[t_var][0] # Normalize by the value given by pluto
                t_array.append(raw_data * norm)
                conv_val = np.asarray(t_array) #TODO fix this later, make as array earlier

            elif var_name:
                # Standard unit conversion
                norm = CGS_code_units[var_name][0]
                conv_val = (raw_data * norm * CGS_code_units[var_name][1]).si.value #converts the units as well as normalize 
            
            vars_si[d_file][var_name] = conv_val
    



    return {"vars_si": vars_si, "CGS_code_units": CGS_code_units, "var_choice": var_choice,"d_files": d_files,"sim_coord": sim_coord}

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






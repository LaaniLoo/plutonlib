import plutonlib.p_utils as pu
# save_dir = pu.setup_dir()
save_dir = pu.save_dir
import os
import numpy as np
import matplotlib.pyplot as plt

import plutokore.io as pk_io
from plutokore.simulations import get_output_count as pk_sim_count

import matplotlib as mpl
from astropy import units as u
from collections import defaultdict 

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
 
def pluto_loader(sim_type, run_name, profile_choice):
    """
    Loads simulation data from a specified Pluto simulation.

    Parameters:
    -----------
    sim_type : str
        Type of simulation to load.
    run_name : str
        Name of the specific simulation run.
    profile_choice : int
        Index selecting a profile from predefined variable lists:
        - 0: ["x1", "x2", "rho", "prs"]
        - 1: ["x1", "x2", "vx1", "vx2"]

    Returns:
    --------
    dict
        Dictionary containing:
        - vars: List of selected variables from the simulation data.
        - var_choice: List of variable names corresponding to the selected profile.
        - nlinf: Dictionary containing metadata about the latest simulation output.
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
        Type of simulation to load (e.g., "hydro", "mhd").
    run_name : str
        Name of the specific simulation run.
    profile_choice : int
        Index selecting a profile from predefined variable lists.

    Returns:
    --------
    dict
        Dictionary containing:
        - vars_si: List of variables converted to SI units.
        - CGS_code_units: Dictionary of CGS code units used for conversion.
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
        # "t_s": [np.linspace(0,1.496e08,coord_shape), (u.s), u.s, "Time (Seconds)"],
        "t_yr": [4.744e00, (u.yr), u.s, "Time"], #NOTE makes array of shape x2, possibly why its x2-like ha! "t_yr": [np.linspace(0,4.744e00,coord_shape), (u.yr), u.s, "Time"],
        # "t_yr": [np.linspace(0,4.744e00,coord_shape), (u.yr), u.s, "Time"], 
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

def pluto_load_profile(sim_type,sel_runs,all = 0):

    sel = 0 if sel_runs is None else 1

    # profile_choice = select_profile(profiles)
    # if profile_choice:
    #     print(f"Selected profile {profile_choice}:", profiles[profile_choice])

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

    if not all:
        for run in run_names:
            profile_choice = pu.select_profile(profiles)
            profile_choices[run].append(profile_choice)  # Appends profile to list
            print(f"Selected profile {profile_choice} for run {run}: {profiles[profile_choice]}")
    elif all:
            profile_choice = "all"
            print(f"Selected profile {profiles[profile_choice]} for all runs")
            print("\n")

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

def plot_extras(profile_choice, sim_type, run_name, t=0, **kwargs):
    """
    Adds extra plotting functions for plotting Pluto simulations.

    Parameters:
    -----------
    profile_choice : int
        Index selecting a profile from predefined variable lists.
    sim_type : str
        Type of simulation to load (e.g., "hydro", "mhd").
    run_name : str
        Name of the specific simulation run.
    t : int, optional
        Flag to determine if the plot should be grouped. Default is 0 (ungrouped).
    **kwargs : dict
        Additional keyword arguments for future extensibility (not currently used).

    Returns:
    --------
    dict
        Dictionary containing:
        - f: matplotlib.figure.Figure or None
        - a: matplotlib.axes.Axes or None
        - c_maps: list of matplotlib.colors.Colormap
        - cbar_labels: list of str
        - labels: list of str
    """
    cbar_labels = []
    labels = []
    coord_labels = []
    xy_labels = []
    title_other = []

    # loaded_data = pluto_loader(sim_type, run_name, profile_choice)
    # var_choice = loaded_data["var_choice"]

    # nlinf = loaded_data["nlinf"]
    # print("Last timestep info:", nlinf)

    conv_data = pluto_conv(sim_type, run_name, profile_choice)
    vars = conv_data["vars_si"]
    var_choice = conv_data["var_choice"]
    CGS_code_units = conv_data["CGS_code_units"]
    c_map_names = []

    for var_name in var_choice[0:2]: #assigning x,y,z etc labels
        coord_labels.append(CGS_code_units[var_name][4])
        xy_labels.append(f"{CGS_code_units[var_name][4]} [{CGS_code_units[var_name][2]}]")  

    for var_name in var_choice[2:4]: #assigning cbar and title labs from rho prs etc
        cbar_labels.append(CGS_code_units[var_name][3]+ " " + f"[{(CGS_code_units[var_name][2]).to_string('latex')}]")
        labels.append(CGS_code_units[var_name][3])

    title_other.append(f"{sim_type} {labels[1]}/{labels[0]} Across {coord_labels[0]}/{coord_labels[1]} ({run_name})")
    
    if t == 0:
        f, a = plt.subplots(figsize=(7, 7))
        a.set_aspect("equal")
        a.set_title(title_other[0])
        a.set_xlabel(f"{CGS_code_units[var_choice[0]][4]} [{CGS_code_units[var_choice[0]][2]}]")
        a.set_ylabel(f"{CGS_code_units[var_choice[1]][4]} [{CGS_code_units[var_choice[1]][2]}]")
    elif t == 1:
        f, a = None, None


        if "vel" in profile_choice.lower(): #velocity profiles have different colour maps if profile_choice % 2 == 0:
            # c_map_names = ['inferno','viridis']
            c_map_names = ["inferno", "hot"]

        elif "rho" in profile_choice.lower(): #dens/prs profiles have different colour maps
            # c_map_names = ["inferno", "hot"]
            c_map_names = ['inferno','viridis']


    #assigning colour maps
    c_maps = []
    for i in range(len(c_map_names)):
        c_maps.append(mpl.colormaps[c_map_names[i]]) #https://matplotlib.org/stable/users/explain/colors/colormaps.html

    return {"f": f, "a": a, "c_maps": c_maps, "cbar_labels": cbar_labels, "labels": labels, 
            "coord_labels": coord_labels, "xy_labels": xy_labels, "title_other": title_other}

def plot_cmap_d_file(sim_type,d_file,sel_runs = None, grouped=1, **kwargs): #TODO might also not need grouped as only handles one file?
    """
    Plots color maps of selected variables from Pluto simulations.
    Can plot either grouped subplots or individual plots based on the `grouped` parameter.

    Parameters:
    -----------
    profile_choice : int
        Index selecting a profile from predefined variable lists.
    sim_type : str
        Type of simulation to load (e.g., "hydro", "mhd").
    sel : int, optional
        Flag to select which runs to plot. Default is 0 (plots all runs).
    sel_runs : list of str, optional
        List of selected run names to plot. Used only if `sel` is 1.
    grouped : int, optional
        If 1, plots all runs in a grouped subplot layout. If 0, plots individually.
    **kwargs : dict
        Additional keyword arguments passed to the `plot_extras` function.

    Returns:
    --------
    None
    """
    run_data = pluto_load_profile(sim_type, sel_runs)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs

    if grouped: #setting parameters if plotting grouped 
        n_runs = len(run_names)
        cols = 3
        rows = (n_runs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        axes = axes.flatten()

    for idx,run in enumerate(run_names):  # Iterating over runs and profile choices
        
        profile_choice = profile_choices[run][0]

        loaded_data = pluto_loader(sim_type, run, profile_choice)

        var_choice = loaded_data["var_choice"]  # TODO: Add var_choice to conv so no need to call bofa

        conv_data = pluto_conv(sim_type, run, profile_choice)

        vars = conv_data["vars_si"][d_file]  # List which data file to plot
        CGS_code_units = conv_data["CGS_code_units"]

        if grouped:

            extras_data = plot_extras(profile_choice, sim_type, run, t=1, **kwargs)
            c_maps = extras_data["c_maps"]
            cbar_labels = extras_data["cbar_labels"]
            labels = extras_data["labels"] 
            coord_labels = extras_data["coord_labels"]
            xy_labels = extras_data["xy_labels"]
            title = extras_data["title_other"][0]

            ax = axes[idx]
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel(xy_labels[0])
            ax.set_ylabel(xy_labels[1])

        if not grouped: # When not grouped, plot details are assigned by extras
            extras_data = plot_extras(profile_choice, sim_type, run, t=0, **kwargs)
            fig = extras_data["f"]
            ax = extras_data["a"]
            c_maps = extras_data["c_maps"]
            cbar_labels = extras_data["cbar_labels"]
            labels = extras_data["labels"]

        # Get just the variables to plot (skip coordinates x1 and x2)
        plot_vars = var_choice[2:]

        for i, var_name in enumerate(plot_vars):
            if var_name not in vars:
                print(f"Warning: Variable {var_name} not found in data, skipping")
                continue
                
            # Apply log scale if density or pressure
            is_log = var_name in ('rho', 'prs')
            vars_data = np.log10(vars[var_name].T) if is_log else vars[var_name].T
            
            # Determine plot side and colormap
            if i % 2 == 0:  # Even index vars on right
                im = ax.pcolormesh(vars['x1'], vars['x2'], vars_data, cmap=c_maps[i])
            else:           # Odd index vars on left (flipped)
                im = ax.pcolormesh(-1 * vars['x1'], vars['x2'], vars_data, cmap=c_maps[i])
            
            # Add colorbar with appropriate label
            cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.25 if grouped else 0.17)
            cbar.set_label(
                f"Log10({cbar_labels[i]})" if is_log else cbar_labels[i],
                fontsize=14
            )

        if not grouped:
            save = input(f"Save {run}? 1 = Yes, 0 = No")
            if save == "1":
                plt.savefig(f"{save_dir}/{run}_Prof_{profile_choice}.png")
                print(f"Saved to {save_dir}/{run}_Prof_{profile_choice}.png")
            plt.show()



    #indent stops
    if grouped:
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])
        save = input("Save grouped plot? 1 = Yes, 0 = No")
        if save == "1":
            plt.savefig(f"{save_dir}/{sim_type}_Grouped_Prof_{profile_choice}.png")
            print(f"Saved to {save_dir}/{sim_type}_Grouped_Prof_{profile_choice}.png")
        plt.show()

    print(f"Selected Profile {profile_choice}: ", var_choice)

def plot_cmap_all_data(sim_type,sel_runs = None, grouped=1, **kwargs): #TODO Fix grouped, might remove as done by _all_data #plots progession of data files as grouped cmap
    """
    Plots color maps of selected variables from Pluto simulations.
    Can plot either grouped subplots or individual plots based on the `grouped` parameter.

    Parameters:
    -----------
    profile_choice : int
        Index selecting a profile from predefined variable lists.
    sim_type : str
        Type of simulation to load (e.g., "hydro", "mhd").
    sel : int, optional
        Flag to select which runs to plot. Default is 0 (plots all runs).
    sel_runs : list of str, optional
        List of selected run names to plot. Used only if `sel` is 1.
    grouped : int, optional
        If 1, plots all runs in a grouped subplot layout. If 0, plots individually.
    **kwargs : dict
        Additional keyword arguments passed to the `plot_extras` function.

    Returns:
    --------
    None
    """
    run_data = pluto_load_profile(sim_type, sel_runs)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs

    for run in run_names:  # Loop over each run
        profile_choice = profile_choices[run][0] #NOTE sus of 0 index
        loaded_data = pluto_loader(sim_type, run, profile_choice)
        var_choice = loaded_data["var_choice"]
        d_files = loaded_data['d_files']

        if grouped:  # Set parameters if plotting grouped
            n_files = len(d_files)
            cols = 3
            rows = (n_files + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
            fig.subplots_adjust(wspace=0.3, hspace=0.3)
            axes = axes.flatten()

        for idx, d_file in enumerate(d_files):  # Loop over each data file
            conv_data = pluto_conv(sim_type, run, profile_choice)
            vars = conv_data["vars_si"][d_file]  # List which data file to plot
            CGS_code_units = conv_data["CGS_code_units"]

            if grouped:
                extras_data = plot_extras(profile_choice, sim_type, run, t=1, **kwargs)
                c_maps = extras_data["c_maps"]
                cbar_labels = extras_data["cbar_labels"]
                labels = extras_data["labels"]
                xy_labels = extras_data["xy_labels"]
                title = extras_data["title_other"][0]

                ax = axes[idx]
                ax.set_aspect("equal")

                ax.set_title(f"{title}, ({d_file})")

                ax.set_xlabel(xy_labels[0])
                ax.set_ylabel(xy_labels[1])

            if not grouped:  # When not grouped, plot details are assigned by extras
                extras_data = plot_extras(profile_choice, sim_type, run, t=0, **kwargs)
                fig = extras_data["f"]
                ax = extras_data["a"]
                c_maps = extras_data["c_maps"]
                cbar_labels = extras_data["cbar_labels"]
                labels = extras_data["labels"]

            # Get just the variables to plot (skip coordinates x1 and x2)
            plot_vars = var_choice[2:]
            for i, var_name in enumerate(plot_vars):
                if var_name not in vars:
                    print(f"Warning: Variable {var_name} not found in data, skipping")
                    continue
                    
                # Apply log scale if density or pressure
                is_log = var_name in ('rho', 'prs')
                vars_data = np.log10(vars[var_name].T) if is_log else vars[var_name].T
                
                # Determine plot side and colormap
                if i % 2 == 0:  # Even index vars on right
                    im = ax.pcolormesh(vars[var_choice[0]], vars[var_choice[1]], vars_data, cmap=c_maps[i])
                else:           # Odd index vars on left (flipped)
                    im = ax.pcolormesh(-1 * vars[var_choice[0]], vars[var_choice[1]], vars_data, cmap=c_maps[i])
                
                # Add colorbar with appropriate label
                cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.25 if grouped else 0.17)
                cbar.set_label(
                    f"Log10({cbar_labels[i]})" if is_log else cbar_labels[i],
                    fontsize=14
                )


            if not grouped:
                save = input(f"Save {run}_{d_file}? 1 = Yes, 0 = No")
                if save == "1":
                    plt.savefig(f"{save_dir}/{run}_{d_file}_Prof_{profile_choice}.png")
                    print(f"Saved to {save_dir}/{run}_{d_file}_Prof_{profile_choice}.png")
                plt.show()

        if grouped:
            for j in range(idx + 1, len(axes)):
                fig.delaxes(axes[j])
            save = input(f"Save grouped plot for {run}? 1 = Yes, 0 = No")
            if save == "1":
                plt.savefig(f"{save_dir}/{sim_type}_{run}_Grouped_Prof_{profile_choice}.png")
                print(f"Saved to {save_dir}/{sim_type}_{run}_Grouped_Prof_{profile_choice}.png")
            plt.show()

def plot_cmap_3d(sim_type,d_files,sel_runs = None, **kwargs): #NOTE takes multiple d_files

    run_data = pluto_load_profile(sim_type, sel_runs)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs
    
    num_vars = len(pluto_conv(sim_type, run_names[0], profile_choices[run_names[0]][0])["vars_si"][d_files[0]]) - 2 #loads a default config to find the number of vars
    # n_plots = len(run_names) * num_vars  # Total number of subplots
    n_plots = len(run_names) * len(d_files) * num_vars  # Total number of subplots

    cols = 3
    rows = (n_plots + cols - 1) // cols  # Compute rows needed

    # Create figure and axes for grouped plots
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    axes = axes.flatten()  # Flatten in case of multi-dimensional array

    plot_idx = 0  # Keep track of which subplot index we are using

    for run in run_names:
        for d_file in d_files:
            profile_choice = profile_choices[run][0]

            loaded_data = pluto_loader(sim_type, run, profile_choice)
            var_choice = loaded_data["var_choice"]

            conv_data = pluto_conv(sim_type, run, profile_choice)
            vars = conv_data["vars_si"][d_file]
            CGS_code_units = conv_data["CGS_code_units"]

            extras_data = plot_extras(profile_choice, sim_type, run, t=1, **kwargs)
            c_maps = extras_data["c_maps"]
            cbar_labels = extras_data["cbar_labels"]
            labels = extras_data["labels"]
            coord_labels = extras_data["coord_labels"]
            xy_labels = extras_data["xy_labels"]

            # Get just the variables to plot (skip coordinates x1 and x2)
            plot_vars = var_choice[2:]
            # num_vars = len(plot_vars)

            for i, var_name in enumerate(plot_vars):
                if plot_idx >= len(axes):  # Avoid out-of-bounds errors
                    print(f"Breaking early at plot_idx={plot_idx} out of {len(axes)}")
                    break
                    
                ax = axes[plot_idx]
                plot_idx += 1

                ax.set_aspect("equal")
                ax.set_title(f"{sim_type} {labels[i]} ({run}, {d_file})")
                ax.set_xlabel(xy_labels[0])
                ax.set_ylabel(xy_labels[1])

                # Handle 3D data by taking first slice
                dim = vars[var_name].shape
                slice = dim[2]//2
                if vars[var_name].ndim == 3:
                    vars[var_name] = vars[var_name][:,:,slice]  # gives a 2D array

                # Apply log scale if variable is density or pressure
                is_log = var_name in ('rho', 'prs')
                vars_data = np.log10(vars[var_name].T) if is_log else vars[var_name].T

                # Plot on right (even index) or left (odd index)
                if i % 2 == 0:
                    im = ax.pcolormesh(vars[var_choice[0]], vars[var_choice[1]], vars_data, cmap=c_maps[i])
                    if 'xlim' in kwargs: # xlim kwarg to change x limits
                        ax.set_xlim(kwargs['xlim'])
                    if 'ylim' in kwargs: # xlim kwarg to change x limits
                        ax.set_ylim(kwargs['ylim']) 
                else:
                    im = ax.pcolormesh(-1 * vars[var_choice[0]], vars[var_choice[1]], vars_data, cmap=c_maps[i])
                    if 'xlim' in kwargs: # xlim kwarg to change x limits
                        ax.set_xlim(kwargs['xlim']) 
                    if 'ylim' in kwargs: # xlim kwarg to change x limits
                        ax.set_ylim(kwargs['ylim'])


                # Add colorbar with appropriate label
                cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.1)
                cbar.set_label(
                    f"Log10({cbar_labels[i]})" if is_log else cbar_labels[i],
                    fontsize=14
                )
    # Remove unused axes
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])

    save = input("Save grouped plot? 1 = Yes, 0 = No")
    if save == "1":
        plt.savefig(f"{save_dir}/{sim_type}_Grouped_Prof_{profile_choice}.png")
        print(f"Saved to {save_dir}/{sim_type}_Grouped_Prof_{profile_choice}.png")

    plt.show()

def plotter(sim_type, run_name, coords, sel_vars,sel_d_file = None,**kwargs):#NOTE use over plot_lines
    """
    Plots 1D slices of selected variables from Pluto simulations.

    Parameters:
    -----------
    sim_type : str
        Type of simulation to load (e.g., "hydro", "mhd").
    run_name : str
        Name of the specific simulation run.
    coords : list or str
        List of coordinates to plot against.
    sel_vars : list or str
        List of variables to plot.

    Returns:
    --------
    None
    """
    conv_data = pluto_conv(sim_type, run_name, "all")
    CGS_code_units = conv_data["CGS_code_units"]

    loaded_data = pluto_loader(sim_type, run_name, "all")
    # var_choice = loaded_data["var_choice"]


    d_files = loaded_data["d_files"] if sel_d_file is None else sel_d_file

    

    # num_vars = len(pluto_conv(sim_type, run_name, 0)["vars_si"][d_files[0]]) - 2 #loads a default config to find the number of vars
    # n_plots = len(run_names) * num_vars  # Total number of subplots
    n_plots = len(d_files)  # Total number of subplots

    cols = 3
    rows = (n_plots + cols - 1) // cols  # Compute rows needed

    # Create figure and axes for grouped plots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.subplots_adjust(wspace=0.4, hspace=0.3)
    axes = axes.flatten()  # Flatten in case of multi-dimensional array

    plot_idx = 0  # Keep track of which subplot index we are using


    for d_file in d_files: # plot across all files
        vars = conv_data["vars_si"][d_file]

        coords_dict = {"x1": vars["x1"], "x2": vars["x2"], "x3": vars["x3"], "t_yr": vars["SimTime"]}
        vars_dict = {"rho": vars["rho"], "prs": vars["prs"], "vx1": vars["vx1"], "vx2": vars["vx2"]}

        extras_data = plot_extras("all", sim_type, run_name, t=1, **kwargs)
        # c_maps = extras_data["c_maps"]
        cbar_labels = extras_data["cbar_labels"]
        labels = extras_data["labels"] 
        coord_labels = extras_data["coord_labels"]
        xy_labels = extras_data["xy_labels"]

        title = extras_data["title_other"][0]
        for coord in coords:
            for var_name in sel_vars:
                
                title_str = f"{sim_type} {CGS_code_units[var_name][3]}"

                sel_var = vars_dict[var_name]
                sel_coord = coords_dict[coord]

                # fig, ax = plt.subplots(figsize=(5, 5))

                ax = axes[plot_idx]

                if 'xlim' in kwargs: # xlim kwarg to change x limits
                    ax.set_xlim(kwargs['xlim']) 


            
                if coord in ("x1", "x2", "x3"): # if selected any x coordinate label x
                    ax.set_title(
                        f"{title_str} vs {CGS_code_units[coord][4]} ({run_name}, {d_file})"
                    )
                    ax.set_xlabel(f"{xy_labels[1]}")


                else: # if selected time
                    ax.set_title(f"{title_str} vs {coord} ({run_name},{d_file})")

                    ax.set_xlabel(
                        f"{CGS_code_units[coord][3]} [{CGS_code_units[coord][1]}]"
                    )

                #Clunky Try, Except to index plotted variable correctly
                try:
                    if var_name in ("vx1", "vx2"):
                        ax.set_ylabel(
                            f"{CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}]"
                        )
                        ax.plot(sel_coord, sel_var[0, :])

                    else:
                        ax.set_ylabel(
                            f"log₁₀({CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}])"
                        )
                        ax.plot(sel_coord, np.log10(sel_var[0, :]))

                except ValueError:
                    if var_name in ("vx1", "vx2"):
                        ax.set_ylabel(
                            f"{CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}]"
                        )
                        ax.plot(sel_coord, sel_var[:, 0])

                    else:
                        ax.set_ylabel(
                            f"log₁₀({CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}])"
                        )
                        ax.plot(sel_coord, np.log10(sel_var[:, 0]))

                plot_idx += 1
    
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])                

    save_name = f"1D_Slice_{sim_type}_{var_name}_{run_name}"
    save = input(f"Save {save_name}? 1 = Yes, 0 = No")

    if save == "1":
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{save_name}.png")
        print(f"{save_dir}/{save_name}.png")

def plot_lines(sim_type, run_name, coords, sel_vars, **kwargs): #TODO add d_file #TODO fix var dict
    """
    Plots 1D slices of selected variables from Pluto simulations with multiple y-axes.

    Parameters:
    -----------
    sim_type : str
        Type of simulation to load (e.g., "hydro", "mhd").
    run_name : str
        Name of the specific simulation run.
    coords : list or str
        List of coordinates to plot against.
    sel_vars : list or str
        List of variables to plot.

    Returns:
    --------
    None
    """
    conv_data = pluto_conv(sim_type, run_name, 0)
    vars = conv_data["vars_si"]
    CGS_code_units = conv_data["CGS_code_units"]

    loaded_data = pluto_loader(sim_type, run_name, 0)
    var_choice = loaded_data["var_choice"]

    coords_dict = {"x1": vars[0], "x2": vars[1], "x3": vars[2], "t_yr": vars[7], "t_s": vars[7]}
    vars_dict = {"rho": vars[3], "prs": vars[4], "vx1": vars[5], "vx2": vars[6]}

    extras_data = plot_extras(0, sim_type, run_name, t=1, **kwargs)
    labels = extras_data["labels"]
    coord_labels = extras_data["coord_labels"]
    xy_labels = extras_data["xy_labels"]

    for j, coord in enumerate(coords):
        save_name = f"1D_Slice_{sim_type}_{'_'.join(sel_vars)}_{run_name}"
        save = input(f"Save {save_name}? 1 = Yes, 0 = No: ")

        sel_coord = coords_dict[coord]
        #TODO FIX XLAB
        fig, ax1 = plt.subplots(figsize=(7, 5))
        
        for i, var_name in enumerate(sel_vars):
            if coord in ("x1", "x2", "x3"): # if selected any x coordinate label x
                ax1.set_title(
                    f"{sim_type} {CGS_code_units[sel_vars[0]][3]},{CGS_code_units[sel_vars[1]][3]} vs {CGS_code_units[coord][4]} ({run_name})"
                )
                ax1.set_xlabel(
                    f"{CGS_code_units[coord][4]} [{CGS_code_units[coord][2]}]"
                )

            else: # if selected time
                ax1.set_title(f"{sim_type} {CGS_code_units[sel_vars[0]][3]},{CGS_code_units[sel_vars[1]][3]} vs {coord} ({run_name})")

                ax1.set_xlabel(
                    f"{CGS_code_units[coord][3]} [{CGS_code_units[coord][1]}]"
                )

            axes = [ax1]  # Store axes for different variables
            colors = ['firebrick','darkorange','goldenrod', 'palegreen', 'm', 'y']  # Color choices

 
            if var_name not in vars_dict:
                print(f"Warning: Variable '{var_name}' not found in vars_dict. Skipping...")
                continue

            sel_var = vars_dict[var_name]

            if i == 0:
                ax = ax1  # Use primary axis for the first variable
            else:
                ax = ax1.twinx()  # Create a twin y-axis for other variables
                axes.append(ax)

            color = colors[i % len(colors)]
            # ax.tick_params(axis='y', labelcolor=color)
            # ax.spines['right'].set_position(('outward', 60 * (i - 1)))  # Offset multiple axes
            
            if sel_var.ndim >2:
                if var_name in ("vx1", "vx2"): #color=color
                    ax.set_ylabel(f"{CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}]" )
                    ax.plot(sel_coord, sel_var[0,0,:], color=color, label=var_name)
                else:
                    ax.set_ylabel(f"log₁₀({CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}])")
                    ax.plot(sel_coord, np.log10(sel_var[0,0,:]), color=color, label=var_name)

            else:
                try: #trying just first column of array e.g x2
                    if var_name in ("vx1", "vx2"): #color=color
                        ax.set_ylabel(f"{CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}]" )
                        ax.plot(sel_coord, sel_var[0, :], color=color, label=var_name)
                    else:
                        ax.set_ylabel(f"log₁₀({CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}])")
                        ax.plot(sel_coord, np.log10(sel_var[0, :]), color=color, label=var_name)

                except ValueError: 
                    if var_name in ("vx1", "vx2"): #color=color:
                        ax.set_ylabel(f"{CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}]" )
                        ax.plot(sel_coord, sel_var[:, 0], color=color, label=var_name)
                    else:
                        ax.set_ylabel(f"log₁₀({CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}])")
                        ax.plot(sel_coord, np.log10(sel_var[:, 0]), color=color, label=var_name)
                    


        # Add a single legend for all axes
        lines, labels = [], []
        for ax in axes:
            line, label = ax.get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
        
        if 'xlim' in kwargs: # xlim kwarg to change x limits
            ax.set_xlim(kwargs['xlim']) 

        fig.legend(lines, labels, loc='upper right')

        plt.show()

        if save == "1":
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{save_name}.png")
            print(f"Saved: {save_dir}/{save_name}.png")

def peak_findr(sim_type, run_name, coords, sel_vars,**kwargs): #TODO FIX USE SCIPI FIND PEAKS
    loaded_data = pluto_loader(sim_type, run_name, "all")
    d_files = loaded_data["d_files"]

    coord_peaks = [] 
    var_peaks = []
    peak_inds = []

    for d_file in d_files:
        vars = pluto_conv(sim_type, run_name,"all")["vars_si"][d_file]

        coords_dict = {"x1": vars["x1"], "x2": vars["x2"], "x3": vars["x3"], "t_yr": vars["SimTime"]}
        vars_dict = {"rho": vars["rho"], "prs": vars["prs"], "vx1": vars["vx1"], "vx2": vars["vx2"]}
 
        
        for coord in coords:
            for var_name in sel_vars:
                sel_var = vars_dict[var_name]
                sel_coord = coords_dict[coord]

                #might only work in 2D for now
                if sel_var.ndim <= 2:
                    var_cut = sel_var[0,:] # cut in the shape of x2
                elif sel_var.ndim > 2:
                    var_cut = sel_var[:,:,0] # x1,x2 cut 

                var_peak = np.max(var_cut)
                peak_index = np.where(var_cut == np.max(var_cut))

                coord_peaks.append(sel_coord[peak_index][0])
                var_peaks.append(var_peak)
                peak_inds.append(peak_index)

                #TODO Needs units
                print(f"{d_file} peak value:", f'{var_name} = {var_peak:.2e}',",", f'{coord} = {sel_coord[peak_index][0]:.2e}')

                # coord_val = sel_coord[peak_index]

    return {"coord_peaks": coord_peaks, "var_peaks": var_peaks, "peak_inds": peak_inds}

# def grouped_init(grouped):
#     # Setting parameters if plotting grouped
#     if grouped:
#         n_runs = len(run_names)
#         cols = 3
#         rows = (n_runs + cols - 1) // cols
#         fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
#         fig.subplots_adjust(wspace=0.3, hspace=0.3)
#         axes = axes.flatten()




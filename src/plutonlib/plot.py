import matplotlib as mpl
import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl

import numpy as np
import matplotlib.pyplot as plt

save_dir = pu.setup_dir(pc.start_dir) #set the save dir using the setup function and start location found in config


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

    conv_data = pl.pluto_conv(sim_type, run_name, profile_choice)
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
    run_data = pl.pluto_load_profile(sim_type, sel_runs)
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

        loaded_data = pl.pluto_loader(sim_type, run, profile_choice)

        var_choice = loaded_data["var_choice"]  # TODO: Add var_choice to conv so no need to call bofa

        conv_data = pl.pluto_conv(sim_type, run, profile_choice)

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
    run_data = pl.pluto_load_profile(sim_type, sel_runs)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs

    for run in run_names:  # Loop over each run
        profile_choice = profile_choices[run][0] #NOTE sus of 0 index
        loaded_data = pl.pluto_loader(sim_type, run, profile_choice)
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
            conv_data = pl.pluto_conv(sim_type, run, profile_choice)
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

    run_data = pl.pluto_load_profile(sim_type, sel_runs)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs
    
    num_vars = len(pl.pluto_conv(sim_type, run_names[0], profile_choices[run_names[0]][0])["vars_si"][d_files[0]]) - 2 #TODO This is a hard watch, loads a default config to find the number of vars
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

            loaded_data = pl.pluto_loader(sim_type, run, profile_choice)
            var_choice = loaded_data["var_choice"]

            conv_data = pl.pluto_conv(sim_type, run, profile_choice)
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
    conv_data = pl.pluto_conv(sim_type, run_name, "all")
    CGS_code_units = conv_data["CGS_code_units"]

    loaded_data = pl.pluto_loader(sim_type, run_name, "all")
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
    conv_data = pl.pluto_conv(sim_type, run_name, 0)
    vars = conv_data["vars_si"]
    CGS_code_units = conv_data["CGS_code_units"]

    loaded_data = pl.pluto_loader(sim_type, run_name, 0)
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
    loaded_data = pl.pluto_loader(sim_type, run_name, "all")
    d_files = loaded_data["d_files"]

    coord_peaks = [] 
    var_peaks = []
    peak_inds = []

    for d_file in d_files:
        vars = pl.pluto_conv(sim_type, run_name,"all")["vars_si"][d_file]

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
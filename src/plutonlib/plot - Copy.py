import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl

import numpy as np
import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt

import inspect 

from collections import defaultdict 


save_dir = pu.setup_dir(pc.start_dir) #set the save dir using the setup function and start location found in config

class PlotData:
    """First attempt at using a class to load and access all plot data"""
    def __init__(self, sim_type = None, run = None, profile_choice = None, **kwargs):
        self.sim_type = sim_type
        self.run = run
        self.profile_choice = profile_choice

        self.d_files = None
        self.vars = None
        self.var_choice = None 

        self.fig = None
        self.axes = None

        self.extras = None #storing plot_extras() data
        self.conv_data = None #storing pluto_conv() data 
        self.__dict__.update(kwargs)

def subplot_base(d_files = None, pdata = None): #sets base subplots determined by number of data_files
    called_func = inspect.stack()[1].function #finds which function is calling c_map base


    if pdata is None:
        pdata = PlotData()

    pdata.d_files = d_files = d_files if d_files is not None else pdata.d_files

    # Validate we have files to plot
    if not pdata.d_files:
        raise ValueError("No data files provided (d_files is empty)")

    plot_vars = pdata.var_choice[2:]
    n_plots = len(pdata.d_files) if called_func == "plot_jet_profile" else len(pdata.d_files)*len(plot_vars)
    cols = 3 
    rows = max(1, (n_plots + cols - 1) // cols)  # Ensure at least 1 row

    figsize_width = min(7 * cols, 21)  # Cap maximum width
    figsize_height = 7 * rows

    pdata.fig, axes = plt.subplots(rows, cols, figsize=(figsize_width, figsize_height),constrained_layout = True) 
    pdata.axes = axes.flatten() #note that axes is assigned to pdata when flattened

    # Hide unused subplots
    for i in range(n_plots, len(pdata.axes)):  
        pdata.fig.delaxes(pdata.axes[i])  

    return pdata.axes, pdata.fig

def plot_extras(pdata = None, **kwargs):
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

    if pdata is None:
        pdata = PlotData(**kwargs)

    if pdata.extras is not None:
        return pdata.extras

    

    cbar_labels = []
    c_map_names = []
    c_maps = []
    labels = []
    coord_labels = []
    xy_labels = []
    title_other = []

    # loaded_data = pluto_loader(sim_type, run_name, profile_choice)
    # var_choice = loaded_data["var_choice"]

    # nlinf = loaded_data["nlinf"]
    # print("Last timestep info:", nlinf)

    conv_data = pl.pluto_conv(pdata.sim_type, pdata.run,pdata.profile_choice ) 
    vars = conv_data["vars_si"]
    pdata.var_choice = conv_data["var_choice"]
    CGS_code_units = conv_data["CGS_code_units"]

    for var_name in pdata.var_choice[0:2]: #assigning x,y,z etc labels
        coord_labels.append(CGS_code_units[var_name][4])
        xy_labels.append(f"{CGS_code_units[var_name][4]} [{CGS_code_units[var_name][2]}]")  

    for var_name in pdata.var_choice[2:4]: #assigning cbar and title labs from rho prs etc
        cbar_labels.append(CGS_code_units[var_name][3]+ " " + f"[{(CGS_code_units[var_name][2]).to_string('latex')}]")
        labels.append(CGS_code_units[var_name][3])

    title_other.append(f"{pdata.sim_type} {labels[1]}/{labels[0]} Across {coord_labels[0]}/{coord_labels[1]} ({pdata.run})")
    

    if "vel" in pdata.profile_choice.lower(): #velocity profiles have different colour maps if profile_choice % 2 == 0:
        # c_map_names = ['inferno','viridis']
        c_map_names = ["inferno", "hot"]

    elif "rho" in pdata.profile_choice.lower(): #dens/prs profiles have different colour maps
        # c_map_names = ["inferno", "hot"]
        c_map_names = ['inferno','viridis']


    #assigning colour maps
    for i in range(len(c_map_names)):
        c_maps.append(mpl.colormaps[c_map_names[i]]) #https://matplotlib.org/stable/users/explain/colors/colormaps.html

    pdata.extras = {"c_maps": c_maps, "cbar_labels": cbar_labels, "labels": labels, "coord_labels": coord_labels, "xy_labels": xy_labels, "title_other": title_other}
        
    return pdata.extras

def cmap_base(pdata = None, **kwargs):
    called_func = inspect.stack()[1].function #finds which function is calling c_map base

    if pdata is None:
        pdata = PlotData(**kwargs)
    
    extras = plot_extras(pdata=pdata)
    idx = kwargs.get('ax_idx',0) #gets the plot index as a kwarg
    var_name = kwargs.get('var_name')
    ax = pdata.axes[idx] # sets the axis as an index

    plot_vars = pdata.var_choice[2:]
    
    #plotting in 3D
    if called_func == "plot_stellar_wind":
        var_idx = pdata.var_choice[2:].index(var_name)

        if pdata.vars[var_name].ndim == 3:
            dim = pdata.vars[var_name].shape
            slice = dim[2]//2
            vars_profile = pdata.vars[var_name][:,:,slice]  #TODO gives a 2D array in Z add profile slice on var


            is_log = var_name in ('rho', 'prs')
            vars_data = np.log10(vars_profile.T) if is_log else vars_profile.T #NOTE Why transpose?

            
            c_map = extras["c_maps"][var_idx]
            cbar_label = extras["cbar_labels"][var_idx]

            im = ax.pcolormesh(pdata.vars[pdata.var_choice[0]], pdata.vars[pdata.var_choice[1]], vars_data, cmap=c_map)

            cbar = pdata.fig.colorbar(im, ax=ax,fraction = 0.05) #, fraction=0.050, pad=0.25
            cbar.set_label(f"Log10({cbar_label})" if is_log else cbar_label, fontsize=14)

    for i, var_name in enumerate(plot_vars): #NOTE might need to move above jet_profile
        if var_name not in pdata.vars: #TODO Change to an error
            print(f"Warning: Variable {var_name} not found in data, skipping")
            continue

        #used for plotting jet
        if called_func == "plot_jet_profile":
            # Apply log scale if density or pressure
            is_log = var_name in ('rho', 'prs')
            vars_data = np.log10(pdata.vars[var_name].T) if is_log else pdata.vars[var_name].T
            
            # Determine plot side and colormap
            if i % 2 == 0:  # Even index vars on right
                im = ax.pcolormesh(pdata.vars[pdata.var_choice[0]], pdata.vars[pdata.var_choice[1]], vars_data, cmap=extras["c_maps"][i])
            else:           # Odd index vars on left (flipped)
                im = ax.pcolormesh(-1 * pdata.vars[pdata.var_choice[0]], pdata.vars[pdata.var_choice[1]], vars_data, cmap=extras["c_maps"][i])
            
            # Add colorbar with appropriate label
            cbar = pdata.fig.colorbar(im, ax=ax, fraction=0.1) #, pad=0.25
            cbar.set_label(
                f"Log10({extras["cbar_labels"][i]})" if is_log else extras["cbar_labels"][i],
                fontsize=14
            )


#NOTE 2 versions, ver2 used for plot_sim 
def cmap_base2(pdata = None, **kwargs):
    if pdata is None:
        pdata = PlotData(**kwargs)
    
    extras = plot_extras(pdata=pdata)
    idx = kwargs.get('ax_idx',0) #gets the plot index as a kwarg
    var_name = kwargs.get('var_name')
    ax = pdata.axes[idx] # sets the axis as an index

    plot_vars = pdata.var_choice[2:]
    sim_type = pdata.sim_type

    #plotting in 3D
    if sim_type in ("Stellar_Wind"):
        var_idx = pdata.var_choice[2:].index(var_name)

        if pdata.vars[var_name].ndim == 3:
            dim = pdata.vars[var_name].shape
            slice = dim[2]//2
            vars_profile = pdata.vars[var_name][:,:,slice]  #TODO gives a 2D array in Z add profile slice on var


            is_log = var_name in ('rho', 'prs')
            vars_data = np.log10(vars_profile.T) if is_log else vars_profile.T #NOTE Why transpose?

            
            c_map = extras["c_maps"][var_idx]
            cbar_label = extras["cbar_labels"][var_idx]

            im = ax.pcolormesh(pdata.vars[pdata.var_choice[0]], pdata.vars[pdata.var_choice[1]], vars_data, cmap=c_map)

            cbar = pdata.fig.colorbar(im, ax=ax,fraction = 0.05) #, fraction=0.050, pad=0.25
            cbar.set_label(f"Log10({cbar_label})" if is_log else cbar_label, fontsize=14)


    for i, var_name in enumerate(plot_vars): #NOTE might need to move above jet_profile
        if var_name not in pdata.vars: #TODO Change to an error
            print(f"Warning: Variable {var_name} not found in data, skipping")
            continue

        #used for plotting jet
        if sim_type in ("Jet"):
            # Apply log scale if density or pressure
            is_log = var_name in ('rho', 'prs')
            vars_data = np.log10(pdata.vars[var_name].T) if is_log else pdata.vars[var_name].T
            
            # Determine plot side and colormap
            if i % 2 == 0:  # Even index vars on right
                im = ax.pcolormesh(pdata.vars[pdata.var_choice[0]], pdata.vars[pdata.var_choice[1]], vars_data, cmap=extras["c_maps"][i])
            else:           # Odd index vars on left (flipped)
                im = ax.pcolormesh(-1 * pdata.vars[pdata.var_choice[0]], pdata.vars[pdata.var_choice[1]], vars_data, cmap=extras["c_maps"][i])
            
            # Add colorbar with appropriate label
            cbar = pdata.fig.colorbar(im, ax=ax, fraction=0.1) #, pad=0.25
            cbar.set_label(
                f"Log10({extras["cbar_labels"][i]})" if is_log else extras["cbar_labels"][i],
                fontsize=14
            )
def subplot_base2(d_files = None, pdata = None): #sets base subplots determined by number of data_files
    if pdata is None:
        pdata = PlotData()

    pdata.d_files = d_files = d_files if d_files is not None else pdata.d_files
    sim_type = pdata.sim_type
    # Validate we have files to plot
    if not pdata.d_files:
        raise ValueError("No data files provided (d_files is empty)")

    plot_vars = pdata.var_choice[2:]
    n_plots = len(pdata.d_files) if sim_type in ("Jet") else len(pdata.d_files)*len(plot_vars)
    cols = 3 
    rows = max(1, (n_plots + cols - 1) // cols)  # Ensure at least 1 row

    figsize_width = min(7 * cols, 21)  # Cap maximum width
    figsize_height = 7 * rows

    pdata.fig, axes = plt.subplots(rows, cols, figsize=(figsize_width, figsize_height),constrained_layout = True) 
    pdata.axes = axes.flatten() #note that axes is assigned to pdata when flattened

    # Hide unused subplots
    for i in range(n_plots, len(pdata.axes)):  
        pdata.fig.delaxes(pdata.axes[i])  

    return pdata.axes, pdata.fig


def plot_save(pdata=None, **kwargs):
    if pdata is None:
        pdata = PlotData(**kwargs)
    
    if not pdata.fig:
        raise ValueError("No figure to save")

    save = input(f"Save plot for {pdata.run}? [1/0]: ")
    if save == "1":
        filename = f"{save_dir}/{pdata.sim_type}_{pdata.run}_plot.png"
        pdata.fig.savefig(filename, bbox_inches='tight')
        print(f"Saved to {filename}")

def plot_label(pdata=None,idx= 0,d_file = None,**kwargs):
    if pdata is None:
        pdata = PlotData(**kwargs)

    extras_data = plot_extras(pdata=pdata)

    # labels = extras_data["labels"]
    xy_labels = extras_data["xy_labels"]
    title = extras_data["title_other"][0]

    ax = pdata.axes[idx]
    ax.set_aspect("equal")
    ax.set_title(f"{title}, ({d_file})")

    ax.set_xlabel(xy_labels[0])
    ax.set_ylabel(xy_labels[1])   

    return ax



def plot_jet_profile(sel_runs = None,sel_d_files=None,pdata = None, **kwargs):
    """
    Plots colour maps of selected variables from Pluto simulations.
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
    if pdata is None:
        pdata = PlotData(sim_type="Jet",**kwargs)

    sel_d_files = [sel_d_files] if sel_d_files and not isinstance(sel_d_files, list) else sel_d_files

    run_data = pl.pluto_load_profile(pdata.sim_type, sel_runs)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs


    for run in run_names:  # Loop over each run

        pdata.run = run
        pdata.profile_choice = profile_choices[run][0]

        loaded_data = pl.pluto_loader(pdata.sim_type, run, pdata.profile_choice)
        pdata.var_choice = loaded_data["var_choice"]

        pdata.d_files = loaded_data['d_files'] if sel_d_files is None else sel_d_files #load all or specific d_file

        pdata.axes, pdata.fig = subplot_base(pdata=pdata)

        for idx, d_file in enumerate(pdata.d_files):  # Loop over each data file

            conv_data = pl.pluto_conv(pdata.sim_type, pdata.run, pdata.profile_choice)
            pdata.vars = conv_data["vars_si"][d_file]  # List which data file to plot

            plot_label(pdata,idx,d_file)

            cmap_base(pdata, ax_idx = idx) #puts current plot axis into camp_base

        plot_save(pdata) # make sure is under run_names so that it saves multiple runs

def plot_stellar_wind(sim_type,sel_d_files = None,sel_runs = None,pdata = None,**kwargs): #NOTE takes multiple d_files
    if pdata is None:
        pdata = PlotData(sim_type=sim_type,**kwargs)

    sel_d_files = [sel_d_files] if sel_d_files and not isinstance(sel_d_files, list) else sel_d_files

    run_data = pl.pluto_load_profile(sim_type, sel_runs)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs

    for run in run_names:
        
        pdata.run = run
        pdata.profile_choice = profile_choices[run][0]

        loaded_data = pl.pluto_loader(pdata.sim_type, run, pdata.profile_choice)
        pdata.var_choice = loaded_data["var_choice"]
        pdata.d_files = loaded_data['d_files'] if sel_d_files is None else sel_d_files #load all or specific d_file

        pdata.axes, pdata.fig = subplot_base(pdata=pdata)


        plot_vars = pdata.var_choice[2:]
        plot_idx = 0
        for d_file in pdata.d_files:
            conv_data = pl.pluto_conv(pdata.sim_type, pdata.run, pdata.profile_choice)
            pdata.vars = conv_data["vars_si"][d_file]
            
            for var_name in plot_vars:
                if plot_idx >= len(pdata.axes):
                    break
                    
                # Plot each variable in its own subplot
                cmap_base(pdata, ax_idx=plot_idx, var_name=var_name)
                plot_label(pdata, plot_idx, d_file)
                plot_idx += 1

        plot_save(pdata) # make sure is under run_names so that it saves multiple runs

def plot_sim(sim_type,sel_d_files = None,sel_runs = None,pdata = None,**kwargs):
    if pdata is None:
        pdata = PlotData(sim_type=sim_type,**kwargs)

    sel_runs = [sel_runs] if sel_runs and not isinstance(sel_runs,list) else sel_runs
    sel_d_files = [sel_d_files] if sel_d_files and not isinstance(sel_d_files, list) else sel_d_files

    run_data = pl.pluto_load_profile(sim_type, sel_runs)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs

    for run in run_names:
        
        pdata.run = run
        pdata.profile_choice = profile_choices[run][0]

        loaded_data = pl.pluto_loader(pdata.sim_type, run, pdata.profile_choice)
        pdata.var_choice = loaded_data["var_choice"]
        pdata.d_files = loaded_data['d_files'] if sel_d_files is None else sel_d_files #load all or specific d_file

        pdata.axes, pdata.fig = subplot_base2(pdata=pdata)

        if sim_type in ("Jet"):
            for idx, d_file in enumerate(pdata.d_files):  # Loop over each data file

                conv_data = pl.pluto_conv(pdata.sim_type, pdata.run, pdata.profile_choice)
                pdata.vars = conv_data["vars_si"][d_file]  # List which data file to plot

                plot_label(pdata,idx,d_file)

                cmap_base2(pdata, ax_idx = idx) #puts current plot axis into camp_base

        if sim_type in ("Stellar_Wind"):
            plot_vars = pdata.var_choice[2:]
            plot_idx = 0

            for d_file in pdata.d_files:
                conv_data = pl.pluto_conv(pdata.sim_type, pdata.run, pdata.profile_choice)
                pdata.vars = conv_data["vars_si"][d_file]
                
                for var_name in plot_vars:
                    if plot_idx >= len(pdata.axes):
                        break
                        
                    # Plot each variable in its own subplot
                    cmap_base2(pdata, ax_idx=plot_idx, var_name=var_name)
                    plot_label(pdata, plot_idx, d_file)
                    plot_idx += 1
        
        
        plot_save(pdata) # make sure is under run_names so that it saves multiple runs



def plotter(sim_type = None, run_name = None , coords = None, sel_vars = None,sel_d_file = None,pdata = None,**kwargs):
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
    if pdata is None:
        pdata = PlotData(sim_type=sim_type,run=run_name,profile_choice="all",**kwargs)

    #load in data
    conv_data = pl.pluto_conv(pdata.sim_type, pdata.run, pdata.profile_choice)
    CGS_code_units = conv_data["CGS_code_units"]
    loaded_data = pl.pluto_loader(pdata.sim_type, pdata.run, pdata.profile_choice)
    pdata.d_files = loaded_data["d_files"] if sel_d_file is None else sel_d_file


    axes, fig = subplot_base(pdata=pdata)

    plot_idx = 0  # Keep track of which subplot index we are using


    for d_file in pdata.d_files: # plot across all files
        pdata.vars = conv_data["vars_si"][d_file]
        vars = pdata.vars

        coords_dict = {"x1": vars["x1"], "x2": vars["x2"], "x3": vars["x3"], "t_yr": vars["SimTime"]}
        vars_dict = {"rho": vars["rho"], "prs": vars["prs"], "vx1": vars["vx1"], "vx2": vars["vx2"]}
        

        extras_data = plot_extras(pdata=pdata)
        xy_labels = extras_data["xy_labels"]

        title = extras_data["title_other"][0]
        for coord in coords:
            for var_name in sel_vars:

                if vars[var_name].ndim > 2:
                    x_mid = len(vars["x1"])//2
                    y_mid = len(vars["x2"])//2
                    z_mid = len(vars["x3"])//2
                    var_prof = {"x1":vars[var_name][:,y_mid,z_mid],"x2":vars[var_name][x_mid,:,z_mid],"x3":vars[var_name][x_mid,y_mid,:]} #TODO use this for other funcs that require profile slices

                elif vars[var_name].ndim == 2:
                    var_prof = {"x1":vars[var_name][:, 0],"x2":vars[var_name][0, :]} 
                
                title_str = f"{sim_type} {CGS_code_units[var_name][3]}"

                sel_var = vars_dict[var_name]
                sel_coord = coords_dict[coord]

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


                if var_name in ("vx1", "vx2"):
                    ax.set_ylabel(
                        f"{CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}]"
                    )
                    # ax.plot(sel_coord, sel_var[0, :])
                    ax.plot(sel_coord, var_prof[coord])

                else:
                    ax.set_ylabel(
                        f"log₁₀({CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}])"
                    )
                    # ax.plot(sel_coord, np.log10(sel_var[0, :]))
                    ax.plot(sel_coord, np.log10(var_prof[coord]))


                # except ValueError:
                #     if var_name in ("vx1", "vx2"):
                #         ax.set_ylabel(
                #             f"{CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}]"
                #         )
                #         ax.plot(sel_coord, sel_var[:, 0])

                #     else:
                #         ax.set_ylabel(
                #             f"log₁₀({CGS_code_units[var_name][3]} [{CGS_code_units[var_name][2]}])"
                #         )
                #         ax.plot(sel_coord, np.log10(sel_var[:, 0]))

                plot_idx += 1
    
    plot_save(pdata)



# def peak_findr_old(sim_type, run_name, coords, sel_vars,**kwargs):
#     loaded_data = pl.pluto_loader(sim_type, run_name, "all")
#     d_files = loaded_data["d_files"]

#     coord_peaks = [] 
#     var_peaks = []
#     peak_inds = []

#     for d_file in d_files:
#         vars = pl.pluto_conv(sim_type, run_name,"all")["vars_si"][d_file]

#         coords_dict = {"x1": vars["x1"], "x2": vars["x2"], "x3": vars["x3"], "t_yr": vars["SimTime"]}
#         vars_dict = {"rho": vars["rho"], "prs": vars["prs"], "vx1": vars["vx1"], "vx2": vars["vx2"]}
 
        
#         for coord in coords:
#             for var_name in sel_vars:
#                 sel_var = vars_dict[var_name]
#                 sel_coord = coords_dict[coord]

#                 #might only work in 2D for now
#                 if sel_var.ndim <= 2:
#                     var_cut = sel_var[0,:] if np.any(sel_coord == "x2") else sel_var[:,0] # cut in the shape of x2 #NOTE MID SLICE?

#                 if sel_var.ndim > 2: #TODO 3D CASE NEEDS FIXING
#                     print("3D CASE NEEDS FIXING")
#                     dim = sel_var.shape
#                     slice = dim[2]//2
#                     var_cut = sel_var[:,:,slice] # x1,x2 cut 

#                 var_peak = np.max(var_cut)
#                 peak_index = np.where(var_cut == np.max(var_cut))

#                 coord_peaks.append(sel_coord[peak_index][0])
#                 var_peaks.append(var_peak)
#                 peak_inds.append(peak_index)

#                 #TODO Needs units
#                 print(f"{d_file} peak value:", f'{var_name} = {var_peak:.2e}',",", f'{coord} = {sel_coord[peak_index][0]:.2e}')

#     return {"coord_peaks": coord_peaks, "var_peaks": var_peaks, "peak_inds": peak_inds}

def peak_findr(sim_type, run_name,sel_coord,sel_var):
    conv_data = pl.pluto_conv(sim_type,run_name,"all")
    vars = conv_data["vars_si"]
    d_files = conv_data["d_files"]
    radius = []
    peak_info = []
    peak_var = []
    locs = []

    for d_file in d_files:
        var = vars[d_file]

        x_mid = len(var["x1"])//2
        y_mid = len(var["x2"])//2
        z_mid = len(var["x3"])//2
        t_yr = var["SimTime"]
    
        array_profiles = {"x1":var[sel_var][:,y_mid,z_mid],"x2":var[sel_var][x_mid,:,z_mid],"x3":var[sel_var][x_mid,y_mid,:]} #TODO use this for other funcs that require profile slices
        
        var_prof = (array_profiles[sel_coord])
        max_loc = np.where(var_prof == np.max(var_prof)) #index location of max variable val

        locs.append(max_loc[0])

        coord_array = var[sel_coord]   
        var_array = var[sel_var] 

        peak_info.append(f"{d_file} Radius: {coord_array[max_loc][0]:.2e} m, {sel_var}: {var_prof[max_loc][0]:.2e}")
        peak_var.append(var_prof[max_loc][0])
        radius.append(coord_array[max_loc][0])

    return {"peak_info": peak_info,"radius": radius, "peak_var": peak_var,"locs": locs } 

def graph_peaks(sim_type,run_name,sel_coord,sel_var):

    #load data
    conv_data = pl.pluto_conv(sim_type, run_name,"all")
    d_files = conv_data["d_files"]
    var_dict = conv_data["vars_si"]
    coord_units = conv_data["CGS_code_units"][sel_coord][2]
    var_units = conv_data["CGS_code_units"][sel_var][2]

    var_peak_ind = defaultdict(list)
    peak_info = []
    peak_vars = []
    peak_coords = []
    for d_file in d_files: #find graphical peaks across all data files

        var = var_dict[d_file][sel_var]
        coord = var_dict[d_file][sel_coord]

        # if sel_coord == "x2":
        #     var_prof = var[0,:]  
        # else: 
        #     raise ValueError("not finding x2 peaks")  #x2 shape

        if var.ndim > 2:
            x_mid = len(var_dict[d_file]["x1"])//2
            y_mid = len(var_dict[d_file]["x2"])//2
            z_mid = len(var_dict[d_file]["x3"])//2
            var_prof = {"x1":var[:,y_mid,z_mid],"x2":var[x_mid,:,z_mid],"x3":var[x_mid,y_mid,:]} #TODO use this for other funcs that require profile slices

        elif var.ndim == 2:
            var_prof = {"x1":var[:, 0],"x2":var[0, :]} 
                
        var_prof = var_prof[sel_coord]
        var_peak_ind[d_file], _ = sp.signal.find_peaks(var_prof)

        if np.any(var_peak_ind[d_file]):  # Only print if peaks exist 
            peak_var = var_prof[var_peak_ind[d_file][-1]]
            # peak_vars = var_prof[var_peak_ind[d_file]]
            peak_vars.append(var_prof[var_peak_ind[d_file][-1]])
            peak_coord = coord[var_peak_ind[d_file][-1]]
            # peak_coords = coord[var_peak_ind[d_file]]
            peak_coords.append(coord[var_peak_ind[d_file][-1]])


            peak_info.append(f"{d_file}: {sel_coord} = {peak_coord:.2e} {coord_units} {sel_var} = {peak_var:.2e} {var_units}")

        else:
            print(f"No peaks found in {d_file}")

    return {"var_peak_ind": var_peak_ind,"var_prof": var_prof,"peak_coords": peak_coords, "peak_vars": peak_vars, "peak_info":peak_info}

def plot_peaks(sim_type,run_name,sel_coord,sel_var):
    #plot showing all peaks found by scipy
    conv_data = pl.pluto_conv(sim_type, run_name,"all")
    units = conv_data["CGS_code_units"]
    d_files = conv_data["d_files"]
    d_file = d_files[-1]
    vars = conv_data["vars_si"][d_file]

    peak_data = graph_peaks(sim_type,run_name,sel_coord,sel_var)
    var_prof = peak_data["var_prof"]
    peak_coords = peak_data["peak_coords"]
    peak_vars = peak_data["peak_vars"]

    # print("var_prof:", type(var_prof), "peak_vars:", type(peak_vars))
    is_log = sel_var in ('rho','prs')
    base_plot_data = np.log10(var_prof) if is_log else var_prof
    peak_plot_data = np.log10(peak_vars) if is_log else peak_vars

    xlab = f"{units[sel_coord][4]} [{units[sel_coord][2]}]"
    ylab = f"log10({units[sel_var][3]}) [{units[sel_var][2]}]" if is_log else f"{units[sel_var][3]} [{units[sel_var][2]}]"
    label = f"Peak {ylab}"
    title = f"{sim_type} Peak {ylab} Across {xlab}"


    f,a = plt.subplots()
    a.plot(vars[sel_coord],base_plot_data) # base plot
    a.plot(peak_coords,peak_plot_data,"x",label= label)
    a.legend()
    a.set_xlabel(xlab)
    a.set_ylabel(ylab)
    a.set_title(title)

def plot_time_prog(sim_type,run_name,sel_coord,sel_var):
    #plot showing all peaks found by scipy
    conv_data = pl.pluto_conv(sim_type, run_name,"all")
    units = conv_data["CGS_code_units"]
    d_files = conv_data["d_files"]
    d_file_last = d_files[-1]
    vars = conv_data["vars_si"]

    xlab = f"SimTime [{units["t_yr"][1]}]"
    ylab = f"{units[sel_coord][4]}-Radius [{units[sel_coord][2]}]"
    title = f"{sim_type} {ylab} across {xlab}"

    t_yr = vars[d_file_last]["SimTime"]
    r = []


    if sim_type == "Jet":
        peak_data = graph_peaks(sim_type,run_name,sel_coord,sel_var) #NOTE uses sel_var to find peak x, not sure if this changes the actual radius
        var_peak_ind = peak_data["var_peak_ind"]


        for d_file in d_files:
            coord = vars[d_file][sel_coord]

            if np.any(var_peak_ind[d_file]):
                r.append(coord[var_peak_ind[d_file]][-1])
            else:
                r.append(0)

    elif sim_type == "Stellar_Wind":
        peak_data = peak_findr(sim_type,run_name,sel_coord,sel_var) #NOTE uses sel_var to find peak x, not sure if this changes the actual radius
        r = peak_data["radius"]

    f,a = plt.subplots()
    a.plot(t_yr, r, color = "darkorchid") # base plot
    a.set_xlabel(xlab)
    a.set_ylabel(ylab)
    a.set_title(title)

    a.legend(["Radius"])
    a.plot(t_yr,r,"x", label = d_files)
    for i, d_file in enumerate(d_files):
        a.annotate(i, (t_yr[i], r[i]), textcoords="offset points", xytext=(1,1), ha='right')

import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl
import plutonlib.analysis as pa
from plutonlib.colours import pcolours

import numpy as np
import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt

import inspect
import time
import logging

from collections import defaultdict 

class PlotData:
    """
    Class used to store data that needs to be accessed btwn multiple plotting functions e.g. 
    * matplotlib figures: fig
    * matplotlib axes: axes
    * plot_extras returns: extras
    * data files from SimulationData: d_files
    * function args for vars and coords: sel_var, sel_coord
    """
    def __init__(self,d_file = None, **kwargs):
        self.d_file = d_file 

        self.sel_coord = None
        self.sel_var = None

        self.d_files = None #used for sel d_files in plots
        self.vars = None

        self.fig = None
        self.axes = None

        self.extras = None #storing plot_extras() data
        self.__dict__.update(kwargs)

#---Plot Helper Functions---#
def sim_type_match(sdata):
    is_jet_2d = sdata.sim_type.split("_")[0] in ("Jet") and sdata.grid_ndim == 2
    is_jet_3d = sdata.sim_type.split("_")[0] in ("Jet") and sdata.grid_ndim == 3
    is_stellar_wind = "_".join(sdata.sim_type.split("_",2)[:2]) in ("Stellar_Wind")
    
    returns = {
        "is_jet_2d":is_jet_2d,
        "is_jet_3d":is_jet_3d,
        "is_stellar_wind":is_stellar_wind,
    }

    return returns

def subplot_base(sdata, pdata = None,d_files = None,**kwargs): #sets base subplots determined by number of data_files
    """
    Sets up and calculates the number of required subplots

    Returns:
    -------
    tuple
        pdata.axes, pdata.fig

    """
    if pdata is None:
        pdata = PlotData()


    # pdata.d_files = d_files if d_files is not None else sdata.d_files #TODO this line is useful but breaks for sdata
    pdata.d_files = d_files if d_files is not None else getattr(sdata, 'd_files', [])

    #DEBUG
    # print(f"in subplot_base (called by {inspect.currentframe().f_back.f_code.co_name}):", pdata.d_files)


    sim_type = sdata.sim_type
    # Validate we have files to plot
    if not sdata.d_files:
        raise ValueError("No data files provided (d_files is empty)")

    try: #only some funcs use var_choice hence try except
        plot_vars = sdata.var_choice[2:]
    except TypeError: #e.g. plotter()
        print("No var_choice, setting plot_vars to None")
        print("\n")
        plot_vars = None

    #NOTE plot sim has two types of plot sizes, two var per subp or one var per subp
    ndim = sdata.get_var_info("rho")["ndim"] #gets rho ndim info for nplots 
    called_func = inspect.stack()[1].function
    if called_func == "plot_sim": 
        n_plots = len(pdata.d_files) if ndim == 2 else len(pdata.d_files)*len(plot_vars) #NOTE because Jet has two vars per plot
    
    else: #all other functions only need d_file sized plot
         n_plots = len(pdata.d_files)
    
    cols = 3 
    rows = max(1, (n_plots + cols - 1) // cols)  # Ensure at least 1 row

    # overwrite base fig size of 7 with kwarg
    if 'fig_resize' in kwargs:
        base_size = kwargs['fig_resize']
    else:
        base_size = 7

    figsize_width = min(base_size * cols, 21)  # Cap maximum width
    figsize_height = base_size * rows

    pdata.fig, axes = plt.subplots(rows, cols, figsize=(figsize_width, figsize_height),constrained_layout = True) 
    pdata.axes = axes.flatten() #note that axes is assigned to pdata when flattened

    # Hide unused subplots
    for i in range(n_plots, len(pdata.axes)):  
        pdata.fig.delaxes(pdata.axes[i])  

    return pdata.axes, pdata.fig

def plot_extras(sdata,pdata = None, **kwargs):
    """
    Assigns labels, data and information about desired plot
    Parameters:
    -----------
    c_maps: list of colour maps
    cbar_labels: list of colour bar labels
    labels: important strings used for title etc
    coord_labels: dict of coord and their labels for title etc
    xy_labels: x,y,z labels used to label x axis etc, different from coord_labels??
    title_other: list containing title string
    _last_d_file: used for storing last used data file #NOTE not sure if still req

    Returns:
    --------
    dict
        Dictionary containing all above parameters

    """

    if pdata is None:
        pdata = PlotData(**kwargs)


    if pdata.extras and pdata.extras.get("_last_d_file") == pdata.d_file: #TODO fix whatever this is
        return pdata.extras
    

    cbar_labels = []
    c_map_names = []
    c_maps = []
    labels = []
    coord_labels = []
    xy_labels = {}
    title_other = []


    #Gets last timestep if req
    # nlinf = loaded_data["nlinf"]
    # print("Last timestep info:", nlinf)


    #assigning x,y,z etc labels
    for var_name in sdata.var_choice:
        if var_name in ("x1","x2","x3"): 
            coord_label = sdata.get_var_info(var_name)["coord_name"]
            coord_units = (sdata.get_var_info(var_name)["si"]).to_string('latex')

            coord_labels.append(coord_label)
            xy_labels[var_name] = (f"{coord_label} [{coord_units}]")  

        #TODO add rest of labels,
        # else:
        #     print("else:", var_name)

    #assigning cbar and title labs from rho prs etc
    for var_name in sdata.var_choice[2:4]:
        var_label = sdata.get_var_info(var_name)["var_name"]
        var_units = (sdata.get_var_info(var_name)["si"]).to_string('latex')

        cbar_labels.append(var_label + " " + f"[{var_units}]")
        labels.append(var_label)

    #assigning title if jet: two vars per subplot
    if sim_type_match(sdata)["is_jet_2d"]:
        title = f"{sdata.sim_type} {labels[1]}/{labels[0]} Across {coord_labels[0]}/{coord_labels[1]} ({sdata.run_name}, {pdata.d_file})"
        title_other.append(title)

    #assigning title if other: one var per subplot
    if sim_type_match(sdata)["is_stellar_wind"] or sim_type_match(sdata)["is_jet_3d"]:
        title_L = f"{sdata.sim_type} {labels[0]} Across {coord_labels[0]}/{coord_labels[1]} ({sdata.run_name}, {pdata.d_file})"
        title_R = f"{sdata.sim_type} {labels[1]} Across {coord_labels[0]}/{coord_labels[1]} ({sdata.run_name}, {pdata.d_file})"
        title_other.append([title_L,title_R])




    if "vel" in sdata.profile_choice.lower(): #velocity profiles have different colour maps if profile_choice % 2 == 0:
        # c_map_names = ['inferno','viridis']
        c_map_names = ["inferno", "hot"]

    elif "rho" in sdata.profile_choice.lower(): #dens/prs profiles have different colour maps
        # c_map_names = ["inferno", "hot"]
        c_map_names = ['inferno','viridis']


    #assigning colour maps
    for i in range(len(c_map_names)):
        c_maps.append(mpl.colormaps[c_map_names[i]]) #https://matplotlib.org/stable/users/explain/colors/colormaps.html

    pdata.extras = {
        "c_maps": c_maps, 
        "cbar_labels": cbar_labels, 
        "labels": labels, 
        "coord_labels": coord_labels, 
        "xy_labels": xy_labels, 
        "title_other": title_other,
        "_last_d_file": pdata.d_file #saves last data file, used to regenerate pdata.extras if changes
        }
        
    return pdata.extras

def pcmesh_3d(sdata,pdata = None, **kwargs):    
    """
    Assigns the pcolormesh data for 3D data array e.g. for a 3D jet simulation or stellar wind. 
    Also assigns colour bar and label
    """ 
    var_name = kwargs.get('var_name')
    extras = kwargs.get('extras')
    ax = kwargs.get('ax')

    if var_name is None or extras is None or ax is None:
        raise ValueError("Missing one of required kwargs: 'var_name', 'extras', 'ax'")
    
    var_idx = sdata.var_choice[2:].index(var_name)

    slice_var = (set(sdata.coord_names) - set(sdata.var_choice[:2])).pop()
    slice = pa.calc_var_prof(sdata,slice_var)["var_profile_single"]

    is_log = var_name in ('rho', 'prs')
    vars_data = np.log10(pdata.vars[var_name][slice]).T if is_log else pdata.vars[var_name][slice].T

    c_map = extras["c_maps"][var_idx]
    cbar_label = extras["cbar_labels"][var_idx]

    im = ax.pcolormesh(pdata.vars[sdata.var_choice[0]], pdata.vars[sdata.var_choice[1]], vars_data, cmap=c_map)


    cbar = pdata.fig.colorbar(im, ax=ax,fraction = 0.05) #, fraction=0.050, pad=0.25
    cbar.set_label(f"Log10({cbar_label})" if is_log else cbar_label, fontsize=14)

def pcmesh_3d_nc(sdata,pdata = None, **kwargs):    
    """
    Assigns the pcolormesh data for 3D data array e.g. for a 3D jet simulation or stellar wind. 
    Also assigns colour bar and label
    """ 
    var_name = kwargs.get('var_name')
    extras = kwargs.get('extras')
    ax = kwargs.get('ax')

    if var_name is None or extras is None or ax is None:
        raise ValueError("Missing one of required kwargs: 'var_name', 'extras', 'ax'")
    
    var_idx = sdata.var_choice[2:].index(var_name)

    slice_var = (set(sdata.coord_names) - set(sdata.var_choice[:2])).pop()
    slice = pa.calc_var_prof(sdata,slice_var)["var_profile_single"]

    is_log = var_name in ('rho', 'prs')
    vars_data = np.log10(pdata.vars[var_name][slice]) if is_log else pdata.vars[var_name][slice]

    c_map = extras["c_maps"][var_idx]
    cbar_label = extras["cbar_labels"][var_idx]

    # print(pdata.vars[sdata.var_choice[0]].shape)
    im = ax.pcolormesh(pdata.vars[sdata.var_choice[0]][slice], pdata.vars[sdata.var_choice[1]][slice], vars_data, cmap=c_map)


    cbar = pdata.fig.colorbar(im, ax=ax,fraction = 0.05) #, fraction=0.050, pad=0.25
    cbar.set_label(f"Log10({cbar_label})" if is_log else cbar_label, fontsize=14)

def pcmesh_2d(sdata,pdata = None, **kwargs):   
    """
    Assigns the pcolormesh data for 2D data array e.g. for a 2D jet simulation. 
    Also assigns colour bar and label
    """ 
    # var_name = kwargs.get('var_name')
    extras = kwargs.get('extras')
    ax = kwargs.get('ax')

    if extras is None or ax is None:
        raise ValueError("Missing one of required kwargs: 'var_name', 'extras', 'ax'")
    
    plot_vars = sdata.var_choice[2:]
    for i, var_name in enumerate(plot_vars):
        if var_name not in sdata.get_vars(sdata.d_files[-1]): #TODO Change to an error
            print(f"Warning: Variable {var_name} not found in data, skipping")
            continue

        # Apply log scale if density or pressure
        is_log = var_name in ('rho', 'prs')
        is_vel = var_name in ('vx1','vx2')
        
        vars_data = np.log10(pdata.vars[var_name].T) if is_log else pdata.vars[var_name].T
        v_min_max =  [-2500,2500] if is_vel else [None,None] #TODO programmatically assign values, sets cbar min max    
        # norm=mpl.colors.SymLogNorm(linthresh=0.03, linscale=0.01,
        #                                   vmin=-5000, vmax=5000.0, base=10)


        # Determine plot side and colormap
        if i % 2 == 0:  # Even index vars on right
            #,vmin = -5000, vmax = 5000
            im = ax.pcolormesh(
                pdata.vars[sdata.var_choice[0]], 
                pdata.vars[sdata.var_choice[1]], 
                vars_data, 
                cmap=extras["c_maps"][i],
                # norm = norm
                vmin = v_min_max[0],
                vmax =  v_min_max[1]
                )
        else:           # Odd index vars on left (flipped)
            im = ax.pcolormesh(
                -1 * pdata.vars[sdata.var_choice[0]], 
                pdata.vars[sdata.var_choice[1]], 
                vars_data, 
                cmap=extras["c_maps"][i],
                # norm = norm
                vmin =  v_min_max[0],
                vmax =  v_min_max[1]
                )
            
        # Add colorbar with appropriate label
        cbar = pdata.fig.colorbar(im, ax=ax, fraction=0.1) #, pad=0.25
        cbar.set_label(
            f"Log10({extras['cbar_labels'][i]})" if is_log else extras["cbar_labels"][i],
            fontsize=14
        )

def cmap_base(sdata,pdata = None, **kwargs):
    """
    Assigns the colour map data based on var e.g. 
    if rho -> log space, also changes based on simulation, e.g. jet has two colour maps per plot
    * Note needs to be looped over d_files and var_name for all info to be loaded (see plot_sim)
    """
    if pdata is None:
        pdata = PlotData(**kwargs)
    
    extras = plot_extras(sdata,pdata)
    idx = kwargs.get('ax_idx',0) #gets the plot index as a kwarg
    var_name = kwargs.get('var_name') #NOTE not sure why this is a kwarg maybe for plotter to insert var
    
    #Simple error in case of wrong profile
    if len(sdata.var_choice) >4:
        raise TypeError(f"{pcolours.WARNING}sdata.profile_choice is set to '{sdata.profile_choice}' with vars: {sdata.var_choice} only 4 vars can be handled try sel_prof")

    #error if selected wrong profile and in 2D
    y_shape = sdata.get_vars(sdata.d_last)[sdata.var_choice[1]].shape
    if y_shape == (1,):
        raise TypeError(f"Cannot plot current var {sdata.var_choice[1]} due to its shape {y_shape}")

    # If being called by self:
    if pdata.axes is None:
        logging.warning("pdata.axes is None, calling subplot_base to assign")
        pdata.axes, pdata.fig = subplot_base(sdata,**kwargs)
        for d_file in sdata.d_files:
            pdata.vars = sdata.get_vars(d_file)

    ax = pdata.axes[idx] # sets the axis as an index

    #plotting in 3D for Stellar Wind and 3D jet
    if sim_type_match(sdata)["is_stellar_wind"]:
        pcmesh_3d(sdata, pdata=pdata, var_name=var_name, extras=extras, ax=ax)

    if sim_type_match(sdata)["is_jet_3d"]:
        pcmesh_3d_nc(sdata, pdata=pdata, var_name=var_name, extras=extras, ax=ax)

    # used for plotting jet,
    if sim_type_match(sdata)["is_jet_2d"]:
        pcmesh_2d(sdata, pdata=pdata, extras=extras, ax=ax)


def plot_label(sdata,pdata=None,idx= 0,**kwargs):
    """
    Generates titles, x/y labels for given plot/s 
    * Note needs to be looped over d_files and var_name for all info to be loaded (see plot_sim)

    """
    if pdata is None:
        pdata = PlotData(**kwargs)

    extras_data = plot_extras(sdata,pdata)

    # If being called by self:
    if pdata.axes is None:
        logging.warning("pdata.axes is None, calling subplot_base to assign")
        pdata.axes, pdata.fig = subplot_base(sdata,**kwargs)
        for d_file in sdata.d_files:
            pdata.vars = sdata.get_vars(d_file)

    xy_labels = extras_data["xy_labels"]
    title = extras_data["title_other"][0]

    ax = pdata.axes[idx] #get ax from PlotData class
    ax.set_aspect("equal")

    ax.set_xlabel(xy_labels[sdata.var_choice[0]])
    ax.set_ylabel(xy_labels[sdata.var_choice[1]])   

    if sim_type_match(sdata)["is_stellar_wind"] or sim_type_match(sdata)["is_jet_3d"]:
        ax.set_title(f"{title[0]}") if idx % 2 == 0 else ax.set_title(f"{title[1]}")

    elif sim_type_match(sdata)["is_jet_2d"]:
        ax.set_title(f"{title}")

def plot_axlim(ax,kwargs):
    if 'xlim' in kwargs: # xlim kwarg to change x limits
        ax.set_xlim(kwargs['xlim']) 

    if 'ylim' in kwargs: # xlim kwarg to change x limits
        ax.set_ylim(kwargs['ylim']) 

def plot_save(sdata,pdata=None,custom=0,**kwargs):
    """
    Saves generated figure as png default
    * if custom = 1: used to label the file by the function it was called rather than sim_type/run
    """
    if pdata is None:
        pdata = PlotData(**kwargs)
    
    if not pdata.fig:
        raise ValueError("No figure to save: Check sdata/pdata")

    file_type = kwargs["file_type"] if 'file_type' in kwargs else "png"
    print("Note: saving as pdf takes a while...") if file_type == "pdf" else None

    if 'save_ovr' in kwargs:
        save = kwargs['save_ovr'] #overwrite value to skip loop
    
    else:
        save = input(f"Save plot for {sdata.run_name}? [1 = Yes, 0 = No, 2 = Custom label]:")

    if save == "1":
        if custom:
            caller_frame = inspect.currentframe().f_back
            current_func_name = caller_frame.f_code.co_name
            filename =f"{sdata.save_dir}/{sdata.sim_type}_{current_func_name}_plot.{file_type}"

        else:
            filename = f"{sdata.save_dir}/{sdata.sim_type}_{sdata.run_name}_{sdata.profile_choice}_plot.{file_type}"

        pdata.fig.savefig(filename, bbox_inches='tight')
        print(f"Saved to {filename}")

    elif save == "2":
        custom_marker = input(f"input custom marker for end of file name")
        filename = f"{sdata.save_dir}/{sdata.sim_type}_{sdata.run_name}_{custom_marker}_plot.{file_type}"

        pdata.fig.savefig(filename, bbox_inches='tight')
        print(f"Saved to {filename}")
    
    else:
        print("Exiting plot_save")

#---Plotting Functions---#
def plot_sim(sdata,sel_d_files = None,sel_runs = None,sel_prof = None, pdata = None,**kwargs):
    """
    Plots the current simulation as either a L-R symmetrical colour map for rho/prs or vx1/vx2 (e.g. for jet) 
    or as separate colour map plots for each var 
    * Both plot across all d_files by default, can be changed with sel_d_files
    * multiple plots can be generated by specifying sel_runs, also will override sdata.run_name
    """
    if pdata is None:
        pdata = PlotData(**kwargs)

    # Ensure sel_runs is either None or a single string (not a list)
    if isinstance(sel_runs, list):
        if len(sel_runs) == 1:
            sel_runs = sel_runs[0]  # Unwrap single-element lists
        else:
            raise ValueError("sel_runs must be a single run name or None")

    #TODO make a class or function or something to streamline this
    # sel_runs = [sel_runs] if sel_runs and not isinstance(sel_runs,list) else sel_runs
    sel_d_files = [sel_d_files] if sel_d_files and not isinstance(sel_d_files, list) else sel_d_files
    # sdata.run_name = sel_runs if sel_runs else [sdata.run_name]
    sdata.run_name = sel_runs if sel_runs else sdata.run_name
    sel_prof = sdata.profile_choice if sel_prof is None else sel_prof 

    # print("load state:", sdata._is_loaded)

    run_data = pl.pluto_load_profile(sdata.sim_type,sdata.run_name,sel_prof)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs

    for run in run_names:
        sdata.run_name = run
        sdata.profile_choice = profile_choices[run][0]
        loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
        arr_type = kwargs.get('arr_type', sdata.arr_type)
        sdata = pl.SimulationData(
            sdata.sim_type,
            sdata.run_name,
            sdata.profile_choice,
            sdata.subdir_name,
            load_outputs=loaded_outputs,
            arr_type=arr_type
            )
        pdata.d_files = sdata.d_files if sel_d_files is None else sel_d_files #load all or specific d_file

        # Handle list selection
        if isinstance(loaded_outputs, list):
            pdata.d_files = [f"data_{i}" for i in loaded_outputs if f"data_{i}" in sdata.d_files]
        else:
            pdata.d_files = sdata.d_files if sel_d_files is None else sel_d_files

            

        pdata.axes, pdata.fig = subplot_base(sdata,pdata,d_files=pdata.d_files,**kwargs)

        # Jet only needs to iterate over d_file
        if sim_type_match(sdata)["is_jet_2d"]:
            for idx, d_file in enumerate(pdata.d_files):  # Loop over each data file
                pdata.d_file = d_file
                pdata.vars = sdata.get_vars(d_file)

                plot_label(sdata,pdata,idx)
                cmap_base(sdata = sdata,ax_idx = idx,pdata = pdata) #puts current plot axis into camp_base
                plot_axlim(pdata.axes[idx],kwargs)


        # Stellar_Wind needs to iterate  over d_file and var name 
        if sim_type_match(sdata)["is_stellar_wind"] or sim_type_match(sdata)["is_jet_3d"]:
        # if sdata.get_var_info("rho")["ndim"] == 3:
            plot_vars = sdata.var_choice[2:]
            plot_idx = 0 #only way to index plot per var 

            for d_file in pdata.d_files:
                pdata.d_file = d_file
                pdata.vars = sdata.get_vars(d_file)
                for var_name in plot_vars:
                    if plot_idx >= len(pdata.axes):
                        break
                        
                    # Plot each variable in its own subplot
                    cmap_base(sdata,pdata, ax_idx=plot_idx, var_name=var_name)
                    plot_label(sdata,pdata,plot_idx)
                    plot_axlim(pdata.axes[plot_idx],kwargs)

                    plot_idx += 1
        
        plot_save(sdata,pdata,**kwargs) # make sure is indent under run_names so that it saves multiple runs

def plotter(sel_coord,sel_var,sdata,sel_d_files = None,**kwargs):
    """
    Plots 1D slices of selected variables from Pluto simulations.
    """
    pdata = PlotData(**kwargs)
    # sdata = pl.SimulationData(sim_type=sdata.sim_type,run_name=sdata.run_name,profile_choice="all",subdir_name=sdata.subdir_name)

    # sel_coords = [sel_coords] if sel_coords and not isinstance(sel_coords,list) else sel_coords
    # sel_vars = [sel_vars] if sel_vars and not isinstance(sel_vars,list) else sel_vars
    sel_d_files = [sel_d_files] if sel_d_files and not isinstance(sel_d_files, list) else sel_d_files

    if sdata.load_outputs is not None:
        pdata.d_files = sdata.d_files[:sdata.load_outputs] #truncate d_files if loading specific
    else:
        pdata.d_files = sdata.d_files if sel_d_files is None else sel_d_files #load all or specific d_file

    axes, fig = subplot_base(sdata,pdata,d_files=pdata.d_files,**kwargs) #,d_files=pdata.d_files
    plot_idx = 0  # Keep track of which subplot index we are using

    for d_file in pdata.d_files: # plot across all files
        pdata.vars = sdata.get_vars(d_file)
        extras_data = plot_extras(sdata,pdata)
        xy_labels = extras_data["xy_labels"]
        title = extras_data["title_other"][0]


        # for coord, var_name in zip(sel_coord, sel_var): #NOTE 
        var_array = pdata.vars[sel_var]
        coord_array = pdata.vars[sel_coord]

        pdata.sel_coord = sel_coord
        pdata.sel_var = sel_var

        calc_prof_data = pa.calc_var_prof(sdata,sel_coord,**kwargs)
        var_profile = calc_prof_data["var_profile"]
        coord_sliced = calc_prof_data["coord_sliced"]
        var_sliced = var_array[var_profile]

        coord_label = sdata.get_var_info(sel_coord)["coord_name"]
        coord_units = sdata.get_var_info(sel_coord)["si"]
        var_label = sdata.get_var_info(sel_var)["var_name"]
        var_units = (sdata.get_var_info(sel_var)["si"]).to_string('latex')


        title_str = f"{sdata.sim_type} {var_label}"
        ax = axes[plot_idx]

        plot_axlim(ax,kwargs)
            
        ax.set_title(
            f"{title_str} vs {coord_label} ({sdata.run_name}, {d_file})"
        )
        ax.set_xlabel(f"{xy_labels[sel_coord]}")


        if sel_var in ("vx1", "vx2"):
            ax.set_ylabel(
                f"{var_label} [{var_units}]"
            )
            ax.plot(coord_array, var_sliced,color = "orange")

        else: #pressure or dens is logspace
            ax.set_ylabel(
                f"log₁₀({var_label} [{var_units}])"
            )
            ax.plot(coord_array, np.log10(var_sliced),color = "mediumslateblue")

        #Assigning legend
        legend_coord = sdata.get_var_info({"x2": "x1", "x1": "x2"}.get(sel_coord, sel_coord))["coord_name"]
        value = f"{(coord_sliced / 1e+12):.4f}" #scaling factor makes it easier to read
        legend_str = f"{title_str} @ {legend_coord} = {value}$\\times 10^{{12}}$ {coord_units}"
        ax.legend([legend_str])

        plot_idx += 1
    plot_save(sdata,pdata,**kwargs)



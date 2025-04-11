import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl

import numpy as np
import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt

import inspect
import time

from collections import defaultdict 


save_dir = pu.setup_dir(pc.start_dir) #set the save dir using the setup function and start location found in config

class PlotData:
    """First attempt at using a class to load and access all plot data"""
    def __init__(self, sim_type = None, run = None, profile_choice = None,d_file = None, **kwargs):
        self.sim_type = sim_type
        self.run = run
        self.profile_choice = profile_choice
        self.d_file = d_file 

        self.d_files = None
        self.vars = None
        self.var_choice = None 
        self.sel_coord = None
        self.sel_var = None

        self.fig = None
        self.axes = None

        self.extras = None #storing plot_extras() data
        self.conv_data = None #storing pluto_conv() data 
        self.__dict__.update(kwargs)

def subplot_base(pdata = None,d_files = None): #sets base subplots determined by number of data_files
    if pdata is None:
        pdata = PlotData()

    pdata.d_files = d_files = d_files if d_files is not None else pdata.d_files
    sim_type = pdata.sim_type
    # Validate we have files to plot
    if not pdata.d_files:
        raise ValueError("No data files provided (d_files is empty)")

    try: #only some funcs use var_choice hence try except
        plot_vars = pdata.var_choice[2:]
    except TypeError: #e.g. plotter()
        print("No var_choice, setting plot_vars to None")
        print("\n")
        plot_vars = None

    called_func = inspect.stack()[1].function
    if called_func == "plot_sim": #plot sim has two types of plot sizes
        n_plots = len(pdata.d_files) if sim_type in ("Jet") else len(pdata.d_files)*len(plot_vars) #NOTE because Jet has two vars per plot
    
    else: #all other functions only need d_file sized plot
         n_plots = len(pdata.d_files)
    
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

def cmap_base(pdata = None, **kwargs):
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

    if pdata.extras and pdata.extras.get("_last_d_file") == pdata.d_file:
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

    conv_data = pl.pluto_conv(pdata.sim_type, pdata.run,pdata.profile_choice ) 
    pdata.var_choice = conv_data["var_choice"]

    pluto_units = pc.get_pluto_units(conv_data["sim_coord"],pdata.d_files) #units dict

    #assigning x,y,z etc labels
    for var_name in ["x1","x2","x3"]: 
        coord_label = pluto_units[var_name]["coord_name"]
        coord_units = (pluto_units[var_name]["si"]).to_string('latex')

        coord_labels.append(coord_label)
        xy_labels[var_name] = (f"{coord_label} [{coord_units}]")  

    #assigning cbar and title labs from rho prs etc
    for var_name in pdata.var_choice[2:4]:
        var_label = pluto_units[var_name]["var_name"]
        var_units = (pluto_units[var_name]["si"]).to_string('latex')

        cbar_labels.append(var_label + " " + f"[{var_units}]")
        labels.append(var_label)

    #assigning title if jet: two vars per subplot
    if pdata.sim_type in ("Jet"):
        title = f"{pdata.sim_type} {labels[1]}/{labels[0]} Across {coord_labels[0]}/{coord_labels[1]} ({pdata.run}, {pdata.d_file})"
        title_other.append(title)

    #assigning title if other: one var per subplot
    if pdata.sim_type in ("Stellar_Wind"):
        title_L = f"{pdata.sim_type} {labels[0]} Across {coord_labels[0]}/{coord_labels[1]} ({pdata.run}, {pdata.d_file})"
        title_R = f"{pdata.sim_type} {labels[1]} Across {coord_labels[0]}/{coord_labels[1]} ({pdata.run}, {pdata.d_file})"
        title_other.append([title_L,title_R])

    if "vel" in pdata.profile_choice.lower(): #velocity profiles have different colour maps if profile_choice % 2 == 0:
        # c_map_names = ['inferno','viridis']
        c_map_names = ["inferno", "hot"]

    elif "rho" in pdata.profile_choice.lower(): #dens/prs profiles have different colour maps
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


def plot_label(pdata=None,idx= 0,**kwargs):
    if pdata is None:
        pdata = PlotData(**kwargs)

    extras_data = plot_extras(pdata)

    # labels = extras_data["labels"]
    xy_labels = extras_data["xy_labels"]
    title = extras_data["title_other"][0]

    # Plot suptitle, not sure if req
    # fig = pdata.fig
    # st = fig.suptitle("suptitle", fontsize="x-large")

    ax = pdata.axes[idx] #get ax from PlotData class
    ax.set_aspect("equal")

    ax.set_xlabel(xy_labels[pdata.var_choice[0]])
    ax.set_ylabel(xy_labels[pdata.var_choice[1]])   

    if pdata.sim_type in ("Stellar_Wind"):
        ax.set_title(f"{title[0]}") if idx % 2 == 0 else ax.set_title(f"{title[1]}")

    else:
        ax.set_title(f"{title}")

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



def plot_sim(sim_type=None,sel_d_files = None,sel_runs = None,sel_prof = None, pdata = None,**kwargs):

    if pdata is None:
        pdata = PlotData(sim_type=sim_type,**kwargs)

    sel_runs = [sel_runs] if sel_runs and not isinstance(sel_runs,list) else sel_runs
    sel_d_files = [sel_d_files] if sel_d_files and not isinstance(sel_d_files, list) else sel_d_files

    pdata.run = sel_runs if sel_runs else [pdata.run]
    run_data = pl.pluto_load_profile(pdata.sim_type,pdata.run,sel_prof)
    # run_data = pl.pluto_load_profile(sim_type, sel_runs,sel_prof)
    run_names, profile_choices = run_data['run_names'], run_data['profile_choices'] #loads the run names and selected profiles for runs



    for run in run_names:
        pdata.run = run
        pdata.profile_choice = profile_choices[run][0]

        loaded_data = pl.pluto_loader(pdata.sim_type, run, pdata.profile_choice)
        pdata.var_choice = loaded_data["var_choice"]
        pdata.d_files = loaded_data['d_files'] if sel_d_files is None else sel_d_files #load all or specific d_file

        pdata.axes, pdata.fig = subplot_base(pdata=pdata)

        # Jet only needs to iterate  over d_file
        if pdata.sim_type in ("Jet"):
            for idx, d_file in enumerate(pdata.d_files):  # Loop over each data file

                pdata.d_file = d_file

                conv_data = pl.pluto_conv(pdata.sim_type, pdata.run, pdata.profile_choice)
                pdata.vars = conv_data["vars_si"][pdata.d_file]  # List which data file to plot

                plot_label(pdata,idx)
                cmap_base(ax_idx = idx,pdata = pdata) #puts current plot axis into camp_base


        # Stellar_Wind needs to iterate  over d_file and var name 
        if pdata.sim_type in ("Stellar_Wind"):
            plot_vars = pdata.var_choice[2:]
            plot_idx = 0 #only way to index plot per var 

            for d_file in pdata.d_files:
                pdata.d_file = d_file
                conv_data = pl.pluto_conv(pdata.sim_type, pdata.run, pdata.profile_choice)
                pdata.vars = conv_data["vars_si"][pdata.d_file]
                
                for var_name in plot_vars:
                    if plot_idx >= len(pdata.axes):
                        break
                        
                    # Plot each variable in its own subplot
                    cmap_base(pdata, ax_idx=plot_idx, var_name=var_name)
                    plot_label(pdata,plot_idx)
                    plot_idx += 1
        

        plot_save(pdata) # make sure is indent under run_names so that it saves multiple runs

def plotter(sel_coords,sel_vars,sim_type = None,run_name = None,sel_d_file = None,pdata = None,**kwargs):
    """
    Plots 1D slices of selected variables from Pluto simulations.

    Parameters:
    -----------
    sim_type : str
        Type of simulation to load (e.g., "hydro", "mhd").
    run_name : str
        Name of the specific simulation run.
    sel_coords : list or str
        List of coordinates to plot against.
    sel_vars : list or str
        List of variables to plot.

    Returns:
    --------
    None
    """
    if pdata is None:
        pdata = PlotData(sim_type=sim_type,run=run_name,profile_choice="all",**kwargs)

    sel_coords = [sel_coords] if sel_coords and not isinstance(sel_coords,list) else sel_coords
    sel_vars = [sel_vars] if sel_vars and not isinstance(sel_vars,list) else sel_vars

    #load in data
    conv_data = pl.pluto_conv(pdata.sim_type, pdata.run, "all")
    pluto_units = pc.get_pluto_units(conv_data["sim_coord"],pdata.d_files) #units dict

    pdata.d_files = conv_data["d_files"] if sel_d_file is None else sel_d_file
    pdata.var_choice = conv_data["var_choice"]
    pdata.vars = conv_data["vars_si"]

    axes, fig = subplot_base(pdata=pdata) #,d_files=pdata.d_files
    plot_idx = 0  # Keep track of which subplot index we are using

    for d_file in pdata.d_files: # plot across all files
        extras_data = plot_extras(pdata=pdata)
        xy_labels = extras_data["xy_labels"]
        title = extras_data["title_other"][0]

        for coord in sel_coords: #TODO possibly use zip? 
            for var_name in sel_vars:
                sel_var = pdata.vars[d_file][var_name]
                sel_coord = pdata.vars[d_file][coord]

                pdata.sel_coord = coord
                pdata.sel_var = var_name
                var_profile = calc_var_prof(pdata)
                var_sliced = sel_var[var_profile]

                coord_label = pluto_units[var_name]["coord_name"]
                coord_units = pluto_units[var_name]["si"]
                var_label = pluto_units[var_name]["var_name"]
                var_units = (pluto_units[var_name]["si"]).to_string('latex')


                title_str = f"{pdata.sim_type} {var_label}"
                ax = axes[plot_idx]

                if 'xlim' in kwargs: # xlim kwarg to change x limits
                    ax.set_xlim(kwargs['xlim']) 
            
                ax.set_title(
                    f"{title_str} vs {coord_label} ({pdata.run}, {d_file})"
                )
                ax.set_xlabel(f"{xy_labels[coord]}")


                if var_name in ("vx1", "vx2"):
                    ax.set_ylabel(
                        f"{var_label} [{var_units}]"
                    )
                    ax.plot(sel_coord, var_sliced)

                else: #pressure or dens is logspace
                    ax.set_ylabel(
                        f"log₁₀({var_label} [{var_units}])"
                    )
                    ax.plot(sel_coord, np.log10(var_sliced))


                plot_idx += 1
    plot_save(pdata)


#TODO Maybe make an analysis.py
def calc_var_prof(pdata):
    vars_last = pdata.vars[pdata.d_files[-1]]

    ndim = vars_last[pdata.sel_var].ndim
    if ndim >2:
        try:
            x_mid = len(vars_last["x1"])//2 
            y_mid = len(vars_last["x2"])//2 
            z_mid = len(vars_last["x3"])//2 
        except KeyError:
            raise ValueError("all coord data was not loaded, make sure profile_choice = 'all'")
        
        slice_map = { #slices in shape of coord
        "x1": (slice(None), y_mid, z_mid),
        "x2": (x_mid, slice(None), z_mid),
        "x3": (x_mid, y_mid, slice(None))
        }

    else:
        slice_map = { #slices in shape of coord
        "x1": (slice(None), 0),
        "x2": (0, slice(None)),
        }

    var_profile = slice_map[pdata.sel_coord]

    return var_profile

def peak_findr(sel_coord,sel_var,sim_type = None, run_name = None,pdata = None):
    if pdata is None:
        pdata = PlotData(sim_type=sim_type,run=run_name,profile_choice="all") #**kwargs

    conv_data = pl.pluto_conv(pdata.sim_type, pdata.run, "all")
    pdata.vars = conv_data["vars_si"]
    pdata.d_files = conv_data["d_files"]
    pdata.sel_coord = sel_coord
    pdata.sel_var = sel_var

    radius = []
    peak_info = []
    peak_var = []
    locs = []

    var_profile = calc_var_prof(pdata)
    for d_file in pdata.d_files:
        var = pdata.vars[d_file]


        var_sliced = var[sel_var][var_profile]
        max_loc = np.where(var_sliced == np.max(var_sliced)) #index location of max variable val

        locs.append(max_loc[0])

        coord_array = var[sel_coord]   
        # var_array = var[sel_var] 

        peak_info.append(f"{d_file} Radius: {coord_array[max_loc][0]:.2e} m, {sel_var}: {var_sliced[max_loc][0]:.2e}")
        peak_var.append(var_sliced[max_loc][0])
        radius.append(coord_array[max_loc][0])

    return {"peak_info": peak_info,"radius": radius, "peak_var": peak_var,"locs": locs } 

def graph_peaks(sel_coord,sel_var,sim_type = None, run_name = None,pdata = None): #TODO Put in peak findr 
    if pdata is None:
        pdata = PlotData(sim_type=sim_type,run=run_name,profile_choice="all") #**kwargs

    #load data
    conv_data = pl.pluto_conv(pdata.sim_type, pdata.run,"all")
    pdata.d_files = conv_data["d_files"]
    pdata.vars = conv_data["vars_si"]
    pdata.sel_coord = sel_coord
    pdata.sel_var = sel_var

    pluto_units = pc.get_pluto_units(conv_data["sim_coord"],pdata.d_files) #units dict
    coord_units = pluto_units[sel_coord]["si"]
    var_units = (pluto_units[sel_var]["si"]).to_string('latex')

    # coord_units = conv_data["CGS_code_units"][sel_coord][2]
    # var_units = conv_data["CGS_code_units"][sel_var][2]


    var_peak_ind = defaultdict(list)
    peak_info = []
    peak_vars = []
    peak_coords = []
    
    var_profile = calc_var_prof(pdata)
    for d_file in pdata.d_files: #find graphical peaks across all data files

        var = pdata.vars[d_file][sel_var]
        coord = pdata.vars[d_file][sel_coord]
        var_sliced = var[var_profile]

        var_peak_ind[d_file], _ = sp.signal.find_peaks(var_sliced)

        if np.any(var_peak_ind[d_file]):  # Only print if peaks exist 
            peak_vars.append(var_sliced[var_peak_ind[d_file][-1]])
            peak_coords.append(coord[var_peak_ind[d_file][-1]])

            #NOTE not sure what 
            peak_var = var_sliced[var_peak_ind[d_file][-1]]
            peak_coord = coord[var_peak_ind[d_file][-1]]

            peak_info.append(f"{d_file}: {sel_coord} = {peak_coord:.2e} {coord_units} {sel_var} = {peak_var:.2e} {var_units}")
            # peak_info.append(f"{d_file}: {sel_coord} = {peak_coords[i]:.2e} {coord_units} {sel_var} = {peak_vars[i]:.2e} {var_units}")

        else:
            print(f"No peaks found in {d_file}")

    return {"var_peak_ind": var_peak_ind,"var_sliced": var_sliced,"peak_coords": peak_coords, "peak_vars": peak_vars, "peak_info":peak_info}

def all_graph_peaks(sel_coord,sel_var,sim_type = None, run_name = None,pdata = None): #NOTE used for plotting
    if pdata is None:
        pdata = PlotData(sim_type=sim_type,run=run_name,profile_choice="all") #**kwargs

    #load data
    conv_data = pl.pluto_conv(pdata.sim_type, pdata.run,"all")
    pdata.d_files = conv_data["d_files"]
    pdata.vars = conv_data["vars_si"]
    pdata.sel_coord = sel_coord
    pdata.sel_var = sel_var

    var_peak_ind = defaultdict(list)
    peak_vars = []
    peak_coords = []
    
    # for d_file in d_files: #find graphical peaks across all data files
    d_file = pdata.d_files[-1]
    var = pdata.vars[d_file][sel_var]
    coord = pdata.vars[d_file][sel_coord]

    var_profile = calc_var_prof(pdata)  
    var_sliced = var[var_profile]

    var_peak_ind, _ = sp.signal.find_peaks(var_sliced)

    if np.any(var_peak_ind):  # Only print if peaks exist 
        peak_vars = var_sliced[var_peak_ind]
        peak_coords = coord[var_peak_ind]


    else:
        print(f"No peaks found in {d_file}")

    return {"var_peak_ind": var_peak_ind,"var_sliced": var_sliced,"peak_coords": peak_coords, "peak_vars": peak_vars}

def plot_peaks(sel_coord,sel_var,sim_type = None, run_name = None,pdata = None): #TODO doesn't work for stelar wind rho
    if pdata is None:
        pdata = PlotData(sim_type=sim_type,run=run_name,profile_choice="all") #**kwargs

    #load data
    conv_data = pl.pluto_conv(pdata.sim_type, pdata.run,"all")
    pdata.d_files = conv_data["d_files"]
    pdata.vars = conv_data["vars_si"]

    units = pc.get_pluto_units(conv_data["sim_coord"],pdata.d_files)["pluto_units"] #units dict


    d_file = pdata.d_files[-1]
    vars_last = pdata.vars[d_file]

    peak_data = all_graph_peaks(sel_coord,sel_var,pdata=pdata)
    var_sliced = peak_data["var_sliced"]
    peak_coords = peak_data["peak_coords"]
    peak_vars = peak_data["peak_vars"]

    # print("var_prof:", type(var_prof), "peak_vars:", type(peak_vars))
    is_log = sel_var in ('rho','prs')
    base_plot_data = np.log10(var_sliced) if is_log else var_sliced
    peak_plot_data = np.log10(peak_vars) if is_log else peak_vars

    xlab = f"{units[sel_coord][4]} [{units[sel_coord][2]}]"
    ylab = f"log10({units[sel_var][3]}) [{units[sel_var][2]}]" if is_log else f"{units[sel_var][3]} [{units[sel_var][2]}]"
    label = f"Peak {ylab}"
    title = f"{sim_type} Peak {ylab} Across {xlab}"


    f,a = plt.subplots()
    a.plot(vars_last[sel_coord],base_plot_data) # base plot
    a.plot(peak_coords,peak_plot_data,"x",label= label)
    a.legend()
    a.set_xlabel(xlab)
    a.set_ylabel(ylab)
    a.set_title(title)

def plot_time_prog(sel_coord,sel_var,sim_type = None, run_name = None,pdata = None):
    if pdata is None:
        pdata = PlotData(sim_type=sim_type,run=run_name,profile_choice="all") #**kwargs

    #load data
    conv_data = pl.pluto_conv(pdata.sim_type, pdata.run,"all")
    pdata.d_files = conv_data["d_files"]
    pdata.vars = conv_data["vars_si"]

    #plot showing all peaks found by scipy
    units = pc.get_pluto_units(conv_data["sim_coord"],pdata.d_files)["pluto_units"] #units dict
    d_file_last = pdata.d_files[-1]

    xlab = f"SimTime [{units["t_yr"][1]}]"
    ylab = f"{units[sel_coord][4]}-Radius [{units[sel_coord][2]}]"
    title = f"{pdata.sim_type} {ylab} across {xlab}"

    t_yr = pdata.vars[d_file_last]["SimTime"]
    r = []


    if pdata.sim_type == "Jet":
        peak_data = graph_peaks(sel_coord,sel_var,pdata=pdata) 
        var_peak_ind = peak_data["var_peak_ind"]


        for d_file in pdata.d_files:
            coord = pdata.vars[d_file][sel_coord]

            if np.any(var_peak_ind[d_file]):
                r.append(coord[var_peak_ind[d_file]][-1])
            else:
                r.append(0)

    elif pdata.sim_type == "Stellar_Wind":
        peak_data = peak_findr(sel_coord,sel_var,pdata=pdata) 
        r = peak_data["radius"]

    f,a = plt.subplots()
    a.plot(t_yr, r, color = "darkorchid") # base plot
    a.set_xlabel(xlab)
    a.set_ylabel(ylab)
    a.set_title(title)

    a.legend(["Radius"])
    a.plot(t_yr,r,"x", label = pdata.d_files)
    for i, d_file in enumerate(pdata.d_files):
        a.annotate(i, (t_yr[i], r[i]), textcoords="offset points", xytext=(1,1), ha='right')

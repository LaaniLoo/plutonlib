import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl
import plutonlib.simulations as ps
import plutonlib.analysis as pa
from plutonlib.colours import pcolours

import numpy as np
import scipy as sp
from astropy import units as u
from astropy.convolution import convolve, Gaussian2DKernel  # Astropy convolutions

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import matplotlib.animation as animation
from matplotlib.patches import Circle

import inspect
import time
import logging

import os

from collections import defaultdict 

class PlotData:
    """
    Class used to store data that needs to be accessed btwn multiple plotting functions e.g. 
    * matplotlib figures: fig
    * matplotlib axes: axes
    * plot_extras returns: extras
    * data files from SimulationData: load_outputs
    * function args for vars and coords: sel_var, sel_coord
    """
    def __init__(self,arr_type = "nc",plane = "xz",var_choice = None,output = None,show_cbar=True, **kwargs):
        self.output = output 

        self.sel_coord = None
        self.sel_var = None #NOTE not sure what this is for might be a duplicate of below
        self.var_name = None #keep track of var_name in loops across var_choice 

        self.load_outputs = None #used for sel load_outputs in plots

        self.fig = None
        self.axes = None
        self.plot_idx = 0
        self.extras = None #storing plot_extras() data
        self.value = 0
        self.__dict__.update(kwargs)

        self.var_choice = var_choice
        self.arr_type = arr_type
        self.plane = plane
        self.show_cbar = show_cbar
        self.cbar_label = None
        
        self.cmap_dict = {
            "vx1" : "hot",
            "vx2" : "hot",
            "vx3" : "hot",
            "rho": "inferno",
            "prs": "viridis"
        }

    def get_colourmap(self,var_name):  

        if var_name not in self.cmap_dict.keys():
            return mpl.colormaps["PuRd"]

        return mpl.colormaps[self.cmap_dict[var_name]]
        
    @property
    def spare_coord(self):
        '''e.g. returns y if profile is xz etc...'''
        un_mapped_coords = [pu.map_coord_name(c) for c in self.coord_choice]
        spare_coord = ({'x1','x2','x3'} - set(un_mapped_coords)).pop()
        
        return pu.unmap_coord_name(spare_coord)
    
    @property
    def coord_choice(self):
        x,y,z = pu.get_coord_names(arr_type=self.arr_type)
        plane_map = {
            "xy": [x,y],
            "xz": [x,z],
            "yz": [y,z],
        }

        if self.plane not in plane_map:
            raise KeyError(f"{self.plane} not recognised plane, see {plane_map}")

        return plane_map[self.plane]

# ---Plot Helper Functions---#
def subplot_base(sdata, pdata = None,load_outputs = None,**kwargs): #sets base subplots determined by number of data_files
    """
    Sets up and calculates the number of required subplots

    Returns:
    -------
    tuple
        pdata.axes, pdata.fig

    """
    called_func = inspect.stack()[1].function
    if pdata is None:
        pdata = PlotData(**kwargs)

    # if "particles" not in called_func: #use load_outputs if not particles plotter else use part_files
    # sel_d_files = load_outputs if load_outputs is not None else getattr(sdata, 'load_outputs', [])
    pdata.load_outputs = sdata.load_outputs if load_outputs is None else load_outputs
    # Validate we have files to plot
    if not sdata.load_outputs:
        raise ValueError("No data files provided (load_outputs is empty)")
        
    #NOTE plot sim has two types of plot sizes, two var per subp or one var per subp
    # print("called function:", called_func)
    if "fluid" in called_func:
        ndim = sdata.grid_ndim #gets gird ndim for plots
        n_plots = len(pdata.load_outputs) if ndim == 2 else len(pdata.load_outputs)*len(pdata.var_choice) #NOTE because Jet has two vars per plot
    else: #all other functions only need output sized plot
        #  n_plots = len(sel_d_files)
        n_plots = len(pdata.load_outputs)*len(pdata.var_choice)
    
    cols = 3 
    rows = max(1, (n_plots + cols - 1) // cols)  # Ensure at least 1 row

    # overwrite base fig size of 7 with kwarg
    if 'fig_resize' in kwargs:
        base_size = kwargs['fig_resize']
    else:
        base_size = 7

    max_width = 15 # Cap maximum width @ 9 for particles and 21 for other
    figsize_width = min(base_size * cols, max_width)  
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


    if pdata.extras and pdata.extras.get("output") == pdata.output: #Basically loads the extras from cache
        return pdata.extras
    
    cbar_labels = []
    coord_labels = []
    xy_labels = {}
    titles = {}

    #assigning x,y,z etc labels
    for coord_name in pdata.coord_choice:
        if pu.is_coord(coord_name): 
            # coord_label = sdata.get_var_info(coord_name)["coord_name"]
            coord_label = getattr(sdata.get_var_info(coord_name),"coord_name")
            coord_units = (getattr(sdata.get_var_info(coord_name),"usr_uv")).to_string('latex')

            coord_labels.append(coord_label)
            xy_labels[coord_name] = (f"{coord_label} [{coord_units}]")  
    
    for var_name in pdata.var_choice:  #assigning cbar and title labs from rho prs etc
        
        try:
            var_label = getattr(sdata.get_var_info(var_name),"var_name")
            var_units = getattr(sdata.get_var_info(var_name),"usr_uv").to_string('latex')
        except AttributeError:
            # print(f"skipping {var_name} units due to no available unit values...")
            var_units = "None"
            var_label = "None"
        except TypeError:
            print("pdata.var_choice elements are not of type(str), skipping units/labels")
            var_units = "None"
            var_label = "None"

        if var_units == '$\\mathrm{}$': #if no units
            cbar_labels.append(var_label)
        else:
            cbar_labels.append(var_label + " " + f"[{var_units}]")


        title_str = f"{var_label} Across {coord_labels[0]}/{coord_labels[1]} ({sdata.run_name})"
        titles[var_name] = title_str

    pdata.extras = {
        "cbar_labels": cbar_labels, 
        "xy_labels": xy_labels, 
        "titles": titles,
        "output": pdata.output #saves last data file, used to regenerate pdata.extras if changes
        }
    
    return pdata.extras

def pcmesh_3d_fluid(sdata,pdata = None, **kwargs):    
    """
    Assigns the pcolormesh data for 3D data array e.g. for a 3D jet simulation or stellar wind. 
    Also assigns colour bar and label
    """
    if pdata is None:
        pdata = PlotData(**kwargs) 

    ax = pdata.axes[pdata.plot_idx]
    extras = plot_extras(sdata,pdata)
    pdata.value = kwargs.get('value') if 'value' in kwargs else 0

    var_idx = pdata.var_choice.index(pdata.var_name) if pdata.var_name in pdata.var_choice else 0
    slice_var = pdata.spare_coord #e.g. if plot xz -> profile in y
    slice_to_load = pa.calc_var_prof(sdata,slice_var,value_2D=pdata.value)["slice_2D"]
    # sdata.arr_type = "nc" #set to nc if not already to make sure for 3D 
    fluid_data = sdata.load_fluid_data(
        output=pdata.output,
        var_choice=pdata.coord_choice + [pdata.var_name],
        load_slice=slice_to_load,
    )
    is_log = pdata.var_name in ('rho', 'prs')
    vars_data = (
        np.log10(fluid_data[pdata.var_name])
        if is_log
        else fluid_data[pdata.var_name]
    )

    X = fluid_data[pdata.coord_choice[0]]
    Y = fluid_data[pdata.coord_choice[1]]
    cbar_label = extras["cbar_labels"][var_idx]

    im = ax.pcolormesh(
        X,
        Y, 
        vars_data, 
        cmap=pdata.get_colourmap(var_name = pdata.var_name)
        )
    
    pdata.cbar_label = f"$\log_{{10}}$({cbar_label})" if is_log else cbar_label
    if pdata.show_cbar:
        cbar = pdata.fig.colorbar(im, ax=ax,fraction = 0.05) #, fraction=0.050, pad=0.25
        cbar.set_label(pdata.cbar_label, fontsize=14)

def scatter_3d_particles(sdata,tr_cut=None,pdata = None, **kwargs):    
    """
    Assigns the scatterplot data for 3D data array e.g. for a 3D jet simulation or stellar wind. 
    Also assigns colour bar and label
    """
    if pdata is None:
        pdata = PlotData(**kwargs) 

    plane_map = {"xy": ["x1","x2"], "xz": ["x1","x3"], "yz": ["x2","x3"]}
    var_map = {"tr1":"tracer","rho":"density","prs":"pressure"}

    ax = pdata.axes[pdata.plot_idx]
    extras = plot_extras(sdata,pdata)

    var_idx = pdata.var_choice.index(pdata.var_name) if pdata.var_name in pdata.var_choice else 0
    particle_var = var_map.get(pdata.var_name,pdata.var_name)

    particle_data = sdata.load_particle_data(output=(pdata.output,),tr_cut=tr_cut)
    is_log = pdata.var_name in ('rho', 'prs')
    vars_data = (
        np.log10(particle_data[particle_var])
        if is_log
        else particle_data[particle_var]
    )

    coord_choice = plane_map[pdata.plane]
    X = particle_data[coord_choice[0]]
    Y = particle_data[coord_choice[1]]
    cbar_label = extras["cbar_labels"][var_idx]

    im = ax.scatter(
        X,
        Y, 
        c=vars_data, 
        s=2.5,
        cmap = pdata.get_colourmap(var_name = pdata.var_name)

        )
    if pdata.show_cbar:
        cbar = pdata.fig.colorbar(im, ax=ax,fraction = 0.05) #, fraction=0.050, pad=0.25
        cbar.set_label(f"$\log_{{10}}$({cbar_label})" if is_log else cbar_label, fontsize=14)

def plot_label(sdata,pdata=None,**kwargs):
    """
    Generates titles, x/y labels for given plot/s 
    * Note needs to be looped over load_outputs and var_name for all info to be loaded (see plot_sim)
    """
    
    if pdata is None:
        pdata = PlotData(**kwargs)

    extras_data = plot_extras(sdata,pdata)

    called_func = inspect.stack()[1].function
    # if "particles" in called_func:
    if any(keword in called_func for keword in ("particles","brightness")):
        time_str = f"{sdata.part_to_simtime(pdata.output):.1f} Myr"
    else:
        time_str = sdata.metadata[pdata.output].time_str
    # If being called by self:
    if pdata.axes is None:
        logging.warning("pdata.axes is None, calling subplot_base to assign")
        pdata.axes, pdata.fig = subplot_base(sdata,**kwargs)

    xy_labels = extras_data["xy_labels"]
    title = extras_data["titles"][pdata.var_name]

    try:
        ax = pdata.axes[pdata.plot_idx] #get ax from PlotData class
    except TypeError as e:
        print("skipping axes list:",e)
        ax = pdata.axes

    ax.set_aspect("equal")

    ax.set_xlabel(xy_labels[pdata.coord_choice[0]])
    ax.set_ylabel(xy_labels[pdata.coord_choice[1]])   
    ax.annotate(
        f'{time_str}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.8
        )
    )

    value = True
    if value:
        value_info = sdata.get_var_info(pdata.spare_coord)
        value_str = f"{getattr(value_info,'coord_name')}$ = {pdata.value} \\; [{getattr(value_info,'usr_uv')}]$"
        ax.annotate(
            value_str,
            xy=(0.05, 0.90),
            xycoords='axes fraction',
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.8
            )
        )

    if "no_title" in kwargs:
        ax.set_title(" ")
    else:
        if sdata.grid_setup["dimensions"] == 3:
            # ax.set_title(f"{title[0]}") if idx % 2 == 0 else ax.set_title(f"{title[1]}")
            ax.set_title(f"{title}")

        elif sdata.grid_setup["dimensions"] == 2:
            ax.set_title(f"{title}")

def plot_axlim(ax,kwargs):
    # ax = pdata.axes[pdata.plot_idx]
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

    if 'save' in kwargs:
        save = str(kwargs['save']) #overwrite value to skip loop or to instantly save 
    
    else:
        save = input(f"Save plot for {sdata.run_name}? [0 = No, 1 = Yes, 2 = Custom label]:")

    if save == "1": #generic save into plutonlib_output dir (sim.save_dir)
        if custom: #NOTE not sure if needed
            caller_frame = inspect.currentframe().f_back
            current_func_name = caller_frame.f_code.co_name
            filename =f"{sdata.save_dir}/{sdata.sim_type}_{current_func_name}_plot.{file_type}"

        else:
            filename = f"{sdata.save_dir}/{sdata.sim_type}_{sdata.run_name}_{pdata.var_name}_plot.{file_type}"

        pdata.fig.savefig(filename, bbox_inches='tight')
        print(f"Saved to {filename}")

    elif save == "2": #save with custom file marker
        custom_marker = input(f"input custom marker for end of file name")
        filename = f"{sdata.save_dir}/{sdata.sim_type}_{sdata.run_name}_{custom_marker}_plot.{file_type}"

        pdata.fig.savefig(filename, bbox_inches='tight')
        print(f"Saved to {filename}")

    elif save == "3": #used to save in different dir
        if 'save_dir' in kwargs:
            save_dir = str(kwargs['save_dir'])
        else:
            raise KeyError(f"save_dir not found in kwargs, please use save_dir for save = 3")

        caller_frame = inspect.currentframe().f_back
        current_func_name = caller_frame.f_code.co_name
        filename = f"{save_dir}/{current_func_name}_{sdata.run_name}_{pdata.var_name}.{file_type}"
        pdata.fig.savefig(filename, bbox_inches='tight')
        print(f"Saved to {filename}")

    else:
        print("Exiting plot_save")
    
# ---Plotting Functions---#
def plot_sim_fluid(sdata, pdata = None,load_outputs = None,save=0,show=True,**kwargs):
    """
    pcolormesh plot of PLUTO simulation grid outputs across multiple variables and outputs

    :param sdata: SimulationData object
    :param load_outputs: load a specific particle output
    :param pdata: PlotData object 
        contains plot settigns such as plot `plane`, `var_choice`, etc
        not required as these settings can be passed as kwargs
    """
    if pdata is None:
        pdata = PlotData(**kwargs)

    if not kwargs.get("var_choice") and not pdata.var_choice:
        raise ValueError("var_choice is None, assign pdata.var_choice or use var_choice kwarg")
    

    load_outputs = sdata.load_outputs if load_outputs is None else load_outputs
    sdata.ini_file = kwargs.get('ini_file', sdata.ini_file)

    if pdata is not None:
        pdata.plot_idx = 0
    pdata.axes, pdata.fig = subplot_base(sdata,pdata,load_outputs=load_outputs,**kwargs)

    # Jet only needs to iterate over output
    if sdata.grid_setup["dimensions"] == 2:
        for idx, output in enumerate(load_outputs):  # Loop over each data file
            pdata.output = output

            plot_label(sdata,pdata,idx)
            cmap_base(sdata = sdata,ax_idx = idx,pdata = pdata) #puts current plot axis into camp_base
            plot_axlim(pdata.axes[idx],kwargs)


    # Stellar_Wind needs to iterate  over output and var name 
    if sdata.grid_setup["dimensions"] == 3:
        for output in load_outputs:
            pdata.output = output
            for var_name in pdata.var_choice:
                if pdata.plot_idx >= len(pdata.axes):
                    break
                    
                # Plot each variable in its own subplot
                pdata.var_name = var_name
                pdata.value = kwargs.get('value') if 'value' in kwargs else 0 #value for custom slice
                pcmesh_3d_fluid(sdata,pdata,value = pdata.value)
                plot_label(sdata,pdata,**kwargs)
                plot_axlim(pdata.axes[pdata.plot_idx],kwargs)

                pdata.plot_idx += 1
    
    plot_save(sdata=sdata,pdata=pdata,save=save,**kwargs) # make sure is indent under run_names so that it saves multiple runs

    if show:
        plt.show()

    return pdata

def plot_sim_particles(sdata,tr_cut=None,load_outputs=None, pdata = None,save=0,show=True,**kwargs):
    """
    Scatter plot of particles from a PLUTO simulation, across multiple variables and outputs

    :param sdata: SimulationData object
    :param load_outputs: load a specific particle output
    :param pdata: PlotData object 
        contains plot settigns such as plot `plane`, `var_choice`, etc
        not required as these settings can be passed as kwargs
    """
    if pdata is None:
        pdata = PlotData(**kwargs)

    if not kwargs.get("var_choice") and not pdata.var_choice:
        raise ValueError("var_choice is None, assign pdata.var_choice or use var_choice kwarg")

    load_outputs = sdata.load_outputs if load_outputs is None else load_outputs

    sdata.ini_file = kwargs.get('ini_file', sdata.ini_file)

    if pdata is not None:
        pdata.plot_idx = 0
    pdata.axes, pdata.fig = subplot_base(sdata,pdata,load_outputs=load_outputs,**kwargs)

    if sdata.grid_setup["dimensions"] == 3:
        for output in load_outputs:
            pdata.output = output
            for var_name in pdata.var_choice:
                if pdata.plot_idx >= len(pdata.axes):
                    break
                    
                # Plot each variable in its own subplot
                pdata.var_name = var_name
                pdata.value = kwargs.get('value') if 'value' in kwargs else 0 #value for custom slice
                scatter_3d_particles(sdata=sdata,tr_cut=tr_cut,pdata=pdata,value = pdata.value)
                plot_label(sdata,pdata,**kwargs)
                plot_axlim(pdata,kwargs)
                pdata.plot_idx += 1

    if show:
        plt.show()

    plot_save(sdata=sdata,pdata=pdata,save=save,**kwargs) # make sure is indent under run_names so that it saves multiple runs

    return pdata

def plot_1D_slice(sel_coord,sel_var,sdata,value_dict,load_outputs = None,pdata=None,save=0,show=True,**kwargs):
    """
    Plots 1D slices of selected variables from Pluto simulations.
    
    :param sel_coord:  selected coordinate to plot e.g. "x1"
    :param sel_var: selected variable to plot e.g "vx1"
    :param sdata: SimulationData object
    :param value_dict: used to plot a 1D slice at a specific point  
        e.g. slice at x1 = 20kpc and x2 = 0kpc -> value_1D = {"x1":20,"x2":0} 
    :param load_outputs: load a specific simulation output
    :param pdata: PlotData object (not required)
    """
    if not isinstance(value_dict,dict):
        raise TypeError(f"value_dict = {value_dict}, must be type dict e.g. x = 10kpc -> {{'x1':10}}")

    if pdata is None:
        pdata = PlotData(var_choice=[sel_var],**kwargs)

    pdata.arr_type = "cc"
    pdata.var_name = sel_var

    load_outputs = sdata.load_outputs if load_outputs is None else load_outputs
    axes, fig = subplot_base(sdata,pdata,load_outputs=load_outputs,**kwargs)
    plot_idx = 0  # Keep track of which subplot index we are using
    sel_coord = pu.get_coord_names(arr_type="cc",coord=sel_coord) #converts x1 to ncx etc req for loading, labels
    slice_to_load = pa.calc_var_prof(sdata,sel_coord,value_1D=value_dict)["slice_1D"]
    
    for output in load_outputs: # plot across all files
        pdata.output = output
        extras_data = plot_extras(sdata,pdata)
        xy_labels = extras_data["xy_labels"]
        var_info   = sdata.get_var_info(sel_var)
        coord_info = sdata.get_var_info(sel_coord)

        var_label   = var_info.var_name
        var_units   = var_info.usr_uv
        coord_label = coord_info.coord_name
        coord_units = coord_info.usr_uv

        fluid_data = sdata.load_fluid_data(
            output=pdata.output,
            var_choice=[sel_coord,sel_var],
            load_slice=slice_to_load,
        )        
        
        title_str = f"{sdata.sim_type} {var_label}"
        ax = axes[plot_idx]
            
        ax.set_title(
            f"{sdata.sim_type} {var_label} vs {coord_label} ({sdata.run_name}, {output})"
        )
        ax.set_xlabel(f"{xy_labels[sel_coord]}")

        if sel_var in ("vx1", "vx2","vx3",'tr1'):
            ax.set_ylabel(
                f"{var_label} [{var_units}]"
            )
            ax.plot(fluid_data[sel_coord], fluid_data[sel_var],color = "orange")

        else: #pressure or dens is logspace
            ax.set_ylabel(
                f"log₁₀({var_label} [{var_units}])"
            )
            ax.plot(fluid_data[sel_coord], np.log10(fluid_data[sel_var]),color = "mediumslateblue")

        #Assigning legend
        legend_coord, = value_dict.keys() #get the first key from value_dict, NOTE: works assuming that only 1 slice arg is used 
        value = f"{(value_dict.get(legend_coord)):.2f}" #scaling factor makes it easier to read
        legend_str = f"{title_str} @ {getattr(sdata.get_var_info(legend_coord),'coord_name')} = {value} {coord_units}"

        ax.legend([legend_str])
        plot_axlim(ax,kwargs)

        plot_idx += 1

        if show:
            plt.show()

    plot_save(sdata=sdata,pdata=pdata,save=save,**kwargs) # make sure is indent under run_names so that it saves multiple runs

    return pdata

def plot_inj_region(sdata,load_outputs=None,save=0,**kwargs):
    """
    Plots a close up of the jet injection region, used to diagnose simulation resolution
    """
    load_outputs = sdata.load_outputs if load_outputs is None else load_outputs

    for output in load_outputs:
        inj_loc = sdata.get_injection_region(output=output)[0].value #location of the injection region for moving injection regions
        inj_size = sdata.usr_params['jet_injection_height'] + 1.5 #arbitrary limits
        ylim = (-inj_size,inj_size)
        xlim = (-0.5*inj_size + inj_loc, 0.5*inj_size + inj_loc) #offset for moving injection region

        pdata = plot_sim_fluid(sdata,load_outputs=(output,),var_choice=["vx3"],ylim = ylim,xlim = xlim,no_title = True)
        
        pdata.axes[0].set_title(f"Plot of injection region for {sdata.run_name}")
    pdata.fig.show()

    plot_save(sdata,pdata,save=save,**kwargs)

def plot_jet_splines(sdata,var,output,tr_cut,query_points=None,roc = None,**kwargs):
    """Plots the jet splines fitted to a particle distribution of the simulation

    Args:
        sdata (SimulationData): SimulationData object
        var (str): str of variable to plot on e.g. tracer 'tr1'
        output (int): PLUTO simulation output
        tr_cut (float): tracer cuttoff value for entire particle dataset
        query_points (dict, optional): Used to plot POI's on the scatterplot e.g. {"$L_{{1a}}$": sim3.jet.L1a.value}
        roc (flaot, optional): radius of curvature to plot a circle over. Defaults to None.
    """
    colours = ['darkblue','blueviolet','mediumvioletred','thistle']
    col_idx = 0

    sim_time = output
    part_time = round(sdata.simtime_to_part(sim_time))
    pdata = plot_sim_particles(sdata, var_choice=[var], load_outputs=(part_time,), tr_cut=tr_cut, show=False)
    
    inj_pos = sdata.get_injection_region(sim_time)
    inj_x, inj_z = inj_pos[0].value, inj_pos[2].value
    
    ax = pdata.fig.axes[0]
    ax.set_aspect('equal')   

    spline_data = pa.get_jet_splines(sdata = sdata,output = sim_time,tr_cut = tr_cut)
    spline_points = spline_data["spline_points"]
    ridgepoints = spline_data["ridgepoints"]

    ax.plot(spline_points[:, 0], spline_points[:, 1], color='k',linestyle = '--')
    ax.scatter(ridgepoints[:, 0], ridgepoints[:, 1], color='r', s=5)
    
    if query_points is not None:
        dx = np.diff(spline_points[:, 0])
        dz = np.diff(spline_points[:, 1])
        arc_length = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dz**2))])

        dist = np.sqrt((spline_points[:, 0] - inj_x)**2 + (spline_points[:, 1] - inj_z)**2)
        inj_idx = np.argmin(dist)
        from_inj = arc_length - arc_length[inj_idx]

        ax.scatter(spline_points[inj_idx, 0], spline_points[inj_idx, 1],
                   color='k', marker='x', s=20, zorder=5, label="Injection region")

        for label, offset in query_points.items():
            idx_plus  = np.argmin(np.abs(from_inj - offset))
            idx_minus = np.argmin(np.abs(from_inj + offset))
            idxs = [idx_minus, idx_plus]
            ax.scatter(spline_points[idxs, 0], spline_points[idxs, 1],
                       s=12, zorder=5, label=f"{label} = {offset:.2f} kpc", color=colours[col_idx])
            col_idx += 1
    
    if roc is not None:
        # Fit circle — adjust center to match your jet head position
        circle = Circle(xy=(inj_x+roc, 0), radius=roc, 
                        fill=False, edgecolor='g', linewidth=1.5, linestyle='-')
        ax.add_patch(circle)

    plot_axlim(ax,kwargs)
    ax.legend(fontsize=8)

    plt.savefig(f"./jet_splines_plot.png",bbox_inches='tight',dpi = 1200)
    plt.show()

def plot_1D_slice_splines(sdata,var,output,query_points = None,tick_axis = 'x',fname = None,**kwargs):
    """Plots a 1D slice along the jet arc length for a given variable.

    Args:
        sdata (SimulationData): SimulationData object
        var (str): str of variable to plot on e.g. tracer 'tr1'
        output (int): PLUTO simulation output
        query_points (dict, optional): Used to plot POI's on the plot e.g. {"$L_{{1a}}$": sim3.jet.L1a.value}
        roc (flaot, optional): radius of curvature to plot a circle over. Defaults to None.
        tick_axis (str, optional): which axis will show ticks for e.g if 'x' ticks for x axis values will show on top. Defaults to 'x'.
        fname (str, optional): file name to save to. Defaults to None.
    """
    colours = ['darkblue','blueviolet','mediumvioletred','thistle']
    col_idx = 0

    fig, ax = plt.subplots(figsize=(10, 4))
    pdata = PlotData(var_choice=[var], plane="xz", show_cbar=False)
    pdata.fig = fig
    pdata.axes = ax
    # pdata.plot_idx = plot_idx
    pdata.extras = plot_extras(sdata,pdata)
    pdata.output = output
    pdata.var_name = var

    spline_data = sdata.load_jet_spline_data(["ccx", "ccz", var], output=output)
    dx,dz = np.diff(spline_data["ccx"]), np.diff(spline_data["ccz"]) 
    arc_length = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dz**2))]) #create arc length from jet spline data 

    inj_pos = sdata.get_injection_region(output)
    inj_x, inj_z = inj_pos[0].value, inj_pos[2].value

    dist = np.sqrt((spline_data['ccx'] - inj_x)**2 + (spline_data['ccz'] - inj_z)**2)
    inj_idx = np.argmin(dist)
    from_inj = arc_length - arc_length[inj_idx] #make inj region the 0 point

    is_log = var in ('rho', 'prs')
    var_data = (
        np.log10(spline_data[var])
        if is_log
        else spline_data[var]
    )

    #---Plot the data---#
    ax.plot(from_inj, var_data,color = 'darkcyan') #darkcyan
    ax.set_xlabel("Arc length along jet [kpc]")
    ax.set_ylabel(pdata.extras['cbar_labels'][0])

    # --- Top axis: (x, z) coords at evenly spaced arc-length ticks ---#
    ax2 = ax.twiny()
    n_ticks = 8
    tick_idx = np.linspace(0, len(from_inj) - 1, n_ticks, dtype=int)
    tick_pos = from_inj[tick_idx]
    tick_labels = [
        f"{spline_data[f"cc{tick_axis}"][i]:.1f}"
        for i in tick_idx
    ]

    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, fontsize=7)
    ax2.set_xlabel(pdata.extras['xy_labels'][f"nc{tick_axis}"], labelpad=10)

    #---Injection region point---#
    ax.scatter(from_inj[inj_idx], var_data[inj_idx], label=f"Injection region", zorder=5,s=15,marker='x',color = 'k')

    if query_points is not None:
        for label, offset in query_points.items():
            idx_plus  = np.argmin(np.abs(from_inj - offset))
            idx_minus = np.argmin(np.abs(from_inj + offset))
            idxs = [idx_minus, idx_plus]
            ax.scatter(from_inj[idxs], var_data[idxs], label=f"{label} = {offset:.2f} kpc", zorder=5,s=12,color = colours[col_idx])
            col_idx += 1

    plot_axlim(ax,kwargs)
    ax.legend(fontsize = 8)

    plt.tight_layout()
    if fname is not None:
        for artist in ax.lines + ax.collections:
            artist.set_rasterized(True)
        plt.savefig(f"./{fname}.pdf", bbox_inches='tight', dpi=300)
    plt.show()

def animation_fluid(sdata, var_name, load_outputs=None, pdata=None, cb_lims=[-5, 5], cmap='inferno', textcolour='black', fps=20, **kwargs):
    def _output_exists(sdata, output):
        try:
            sdata.get_metadata(output)
            return True
        except:
            return False
    
    start = time.time()
    print(f"({var_name}): {(time.time() - start):.2f}s - init")

    if pdata is None:
        pdata = PlotData(var_choice=[var_name], **kwargs)
        print(f"({var_name}): {(time.time() - start):.2f}s - PlotData initialized")

    load_outputs = sdata.load_outputs if load_outputs is None else load_outputs
    load_outputs = [o for o in load_outputs if _output_exists(sdata, o)]
    total_outputs = len(load_outputs)
    print(f"({var_name}): {(time.time() - start):.2f}s - {total_outputs} valid outputs found")

    plt.rcParams.update({'font.size': 15})

    with plt.style.context("classic"):
        pdata.fig = plt.figure(figsize=(10, 10))
        pdata.fig.tight_layout()
        pdata.axes = ImageGrid(
            pdata.fig, 111,
            nrows_ncols=(1, 1),
            label_mode="1",
            axes_pad=(0, 0.2),
            cbar_pad=0.05,
            cbar_mode="edge",
            cbar_size=0.1,
            cbar_location="right",
            share_all=True,
        )
        ax = pdata.axes[0]
        im = None

        def update_img(n, cb_lims, cmap, textcolour):
            nonlocal im
            pdata.output = load_outputs[n]

            try:
                ax.clear()

                extras_data = plot_extras(sdata, pdata)
                slice_var = pdata.spare_coord
                profile = pa.calc_var_prof(sdata, slice_var)["slice_2D"]

                fluid_data = sdata.load_fluid_data(
                    output=pdata.output,
                    var_choice=pdata.coord_choice + [var_name],
                    load_slice=profile,
                )

                is_log = var_name in ('rho', 'prs')
                vars_data = np.log10(fluid_data[var_name]) if is_log else fluid_data[var_name]

                X = fluid_data[pdata.coord_choice[0]]
                Y = fluid_data[pdata.coord_choice[1]]

                var_idx = pdata.var_choice.index(var_name)
                c_map = pdata.get_colourmap(var_name)
                cbar_label = extras_data["cbar_labels"][var_idx]

                im = ax.pcolormesh(X, Y, vars_data, cmap=c_map, shading='auto')

                cax = ax.cax
                cax.cla()
                cb = pdata.fig.colorbar(im, cax=cax)
                cb.set_label(f"$\\log_{{10}}$({cbar_label})" if is_log else cbar_label, fontsize=14)
                cb.ax.tick_params(labelsize='large')

                time_str = sdata.metadata[pdata.output].time_str
                ax.text(0.05, 0.95, time_str,
                        horizontalalignment="left", verticalalignment="center",
                        transform=ax.transAxes, fontsize=20, color=textcolour)

                ax.set_xlabel(extras_data["xy_labels"][pdata.coord_choice[0]])
                ax.set_ylabel(extras_data["xy_labels"][pdata.coord_choice[1]])
                ax.tick_params(axis='both')
                ax.set_xlim(np.min(X), np.max(X))
                ax.set_ylim(np.min(Y), np.max(Y))
                plot_axlim(ax, kwargs)

                print(f"({var_name}): {(time.time() - start):.2f}s - frame {n} done")

            except (OSError, KeyError, FileNotFoundError) as e:
                print(f"({var_name}): skipping output {pdata.output} — {type(e).__name__}: {e}")

            return im

        ani = animation.FuncAnimation(
            pdata.fig, update_img,
            frames=total_outputs,
            interval=30,
            save_count=total_outputs,
            fargs=(cb_lims, cmap, textcolour),
        )

        writergif = animation.PillowWriter(fps=fps)
        ani.save(f"{sdata.save_dir}/{sdata.run_name}.gif", writer=writergif, dpi=200)
        print(f"({var_name}): {(time.time() - start):.2f}s - animation saved")

    return ani
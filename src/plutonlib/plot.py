import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl
import plutonlib.simulations as ps
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

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import matplotlib.animation as animation

class PlotData:
    """
    Class used to store data that needs to be accessed btwn multiple plotting functions e.g. 
    * matplotlib figures: fig
    * matplotlib axes: axes
    * plot_extras returns: extras
    * data files from SimulationData: load_outputs
    * function args for vars and coords: sel_var, sel_coord
    """
    def __init__(self,arr_type = "nc",plane = "xz",var_choice = None,output = None, **kwargs):
        self.output = output 

        self.sel_coord = None
        self.sel_var = None
        self._var_name = None #keep track of var_name in loops across var_choice 

        self.load_outputs = None #used for sel load_outputs in plots
        self.vars = None

        self.fig = None
        self.axes = None
        self.plot_idx = 0
        self.extras = None #storing plot_extras() data
        self.value = 0
        self.__dict__.update(kwargs)

        self.var_choice = var_choice
        self.arr_type = arr_type
        self.plane = plane

        
    def get_colourmap(self,var_name):  
        cmap_dict = {
            "vx1" : "hot",
            "vx2" : "hot",
            "vx3" : "hot",
            "rho": "inferno",
            "prs": "viridis"
        }

        if var_name not in cmap_dict.keys():
            return mpl.colormaps["cividis"]

        return mpl.colormaps[cmap_dict[var_name]]
        
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

    if "particles" not in called_func: #use load_outputs if not particles plotter else use part_files
        sel_d_files = load_outputs if load_outputs is not None else getattr(sdata, 'load_outputs', [])

        # Validate we have files to plot
        if not sdata.load_outputs:
            raise ValueError("No data files provided (load_outputs is empty)")
        
    elif "particles" in called_func: # access particle data files for n plots
        sel_part_files = getattr(sdata,'particle_files',[])

    #NOTE plot sim has two types of plot sizes, two var per subp or one var per subp
    print("called function:", called_func)
    if "fluid" in called_func:
        ndim = sdata.grid_ndim #gets gird ndim for plots
        n_plots = len(sel_d_files) if ndim == 2 else len(sel_d_files)*len(pdata.var_choice) #NOTE because Jet has two vars per plot
    
    elif "particles" in called_func:
        n_plots = len(sel_part_files)

    else: #all other functions only need output sized plot
         n_plots = len(sel_d_files)
    
    cols = 3 
    rows = max(1, (n_plots + cols - 1) // cols)  # Ensure at least 1 row

    # overwrite base fig size of 7 with kwarg
    if 'fig_resize' in kwargs:
        base_size = kwargs['fig_resize']
    else:
        base_size = 7

    max_width = 9 if "particles" in called_func else 15 # Cap maximum width @ 9 for particles and 21 for other
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
            coord_label = sdata.get_var_info(coord_name)["coord_name"]
            coord_units = (sdata.get_var_info(coord_name)["usr_uv"]).to_string('latex')
            coord_labels.append(coord_label)
            xy_labels[coord_name] = (f"{coord_label} [{coord_units}]")  
    
    for var_name in pdata.var_choice:  #assigning cbar and title labs from rho prs etc
        var_label = sdata.get_var_info(var_name)["var_name"]
        try:
            var_units = (sdata.get_var_info(var_name)["usr_uv"]).to_string('latex')
        except AttributeError:
            print(f"skipping {var_name} units due to no available unit values...")
            var_units = "None"

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

def pcmesh_3d(sdata,pdata = None, **kwargs):    
    """
    Assigns the pcolormesh data for 3D data array e.g. for a 3D jet simulation or stellar wind. 
    Also assigns colour bar and label
    """
    if pdata is None:
        pdata = PlotData(**kwargs) 

    ax = pdata.axes[pdata.plot_idx]
    extras = plot_extras(sdata,pdata)
    pdata.value = kwargs.get('value') if 'value' in kwargs else 0

    var_idx = pdata.var_choice.index(pdata._var_name) if pdata._var_name in pdata.var_choice else 0
    slice_var = pdata.spare_coord #e.g. if plot xz -> profile in y
    slice_to_load = pa.calc_var_prof(sdata,slice_var,value_2D=pdata.value)["slice_2D"]
    # sdata.arr_type = "nc" #set to nc if not already to make sure for 3D 
    fluid_data = sdata.fluid_data(
        output=pdata.output,
        var_choice=pdata.coord_choice + [pdata._var_name],
        load_slice=slice_to_load,
    )
    is_log = pdata._var_name in ('rho', 'prs')
    vars_data = (
        np.log10(fluid_data[pdata._var_name])
        if is_log
        else fluid_data[pdata._var_name]
    )

    X = fluid_data[pdata.coord_choice[0]]
    Y = fluid_data[pdata.coord_choice[1]]
    c_map = pdata.get_colourmap(var_name = pdata._var_name)
    cbar_label = extras["cbar_labels"][var_idx]

    im = ax.pcolormesh(
        X,
        Y, 
        vars_data, 
        cmap=c_map
        )
    cbar = pdata.fig.colorbar(im, ax=ax,fraction = 0.05) #, fraction=0.050, pad=0.25
    cbar.set_label(f"$Log_{{10}}$({cbar_label})" if is_log else cbar_label, fontsize=14)
    # sdata.clear_fluid_data_cache() #clear the cache after plotting -> can change value etc

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
    
    plot_vars = pdata.var_choice
    for i, var_name in enumerate(plot_vars):
        if var_name not in sdata.get_vars(sdata.load_outputs[-1]): #TODO Change to an error
            print(f"Warning: Variable {var_name} not found in data, skipping")
            continue

        # Apply log scale if density or pressure
        is_log = var_name in ('rho', 'prs')
        is_vel = var_name in ('vx1','vx2')
        
        vars_data = np.log10(sdata.get_vars(pdata.output)[var_name].T) if is_log else sdata.get_vars(pdata.output)[var_name].T
        v_min_max =  [-2500,2500] if is_vel else [None,None] #TODO programmatically assign values, sets cbar min max    
        # norm=mpl.colors.SymLogNorm(linthresh=0.03, linscale=0.01,
        #                                   vmin=-5000, vmax=5000.0, base=10)


        # Determine plot side and colormap
        if i % 2 == 0:  # Even index vars on right
            #,vmin = -5000, vmax = 5000
            im = ax.pcolormesh(
                sdata.get_vars(pdata.output)[pdata.var_choice[0]], 
                sdata.get_vars(pdata.output)[pdata.var_choice[1]], 
                vars_data, 
                cmap=extras["c_maps"][i],
                # norm = norm
                vmin = v_min_max[0],
                vmax =  v_min_max[1]
                )
        else:           # Odd index vars on left (flipped)
            im = ax.pcolormesh(
                -1 * sdata.get_vars(pdata.output)[pdata.var_choice[0]], 
                sdata.get_vars(pdata.output)[pdata.var_choice[1]], 
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

def plot_label(sdata,pdata=None,**kwargs):
    """
    Generates titles, x/y labels for given plot/s 
    * Note needs to be looped over load_outputs and var_name for all info to be loaded (see plot_sim)
    """
    
    if pdata is None:
        pdata = PlotData(**kwargs)

    extras_data = plot_extras(sdata,pdata)
    time_str = sdata.metadata[pdata.output].time_str
    # If being called by self:
    if pdata.axes is None:
        logging.warning("pdata.axes is None, calling subplot_base to assign")
        pdata.axes, pdata.fig = subplot_base(sdata,**kwargs)

    xy_labels = extras_data["xy_labels"]
    title = extras_data["titles"][pdata._var_name]

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
        value_str = f"{value_info['coord_name']}$ = {pdata.value} \\; [{value_info['usr_uv']}]$"
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

def plot_axlim(pdata,kwargs):
    ax = pdata.axes[pdata.plot_idx]
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
        save = kwargs['save'] #overwrite value to skip loop or to instantly save 
    
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

# ---Plotting Functions---#
def plot_sim_fluid(sdata, pdata = None,**kwargs):
    """
    Plots the current simulation as either a L-R symmetrical colour map for rho/prs or vx1/vx2 (e.g. for jet) 
    or as separate colour map plots for each var 
    * Both plot across all load_outputs by default, can be changed with sel_d_files
    * multiple plots can be generated by specifying sel_runs, also will override sdata.run_name
    """
    if pdata is None:
        pdata = PlotData(**kwargs)

    if not kwargs.get("var_choice") and not pdata.var_choice:
        raise ValueError("var_choice is None, assign pdata.var_choice or use var_choice kwarg")
    

    sdata.load_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    sdata.ini_file = kwargs.get('ini_file', sdata.ini_file)

    if pdata is not None:
        pdata.plot_idx = 0
    pdata.axes, pdata.fig = subplot_base(sdata,pdata,load_outputs=sdata.load_outputs,**kwargs)

    # Jet only needs to iterate over output
    if sdata.grid_setup["dimensions"] == 2:
        for idx, output in enumerate(sdata.load_outputs):  # Loop over each data file
            pdata.output = output

            plot_label(sdata,pdata,idx)
            cmap_base(sdata = sdata,ax_idx = idx,pdata = pdata) #puts current plot axis into camp_base
            plot_axlim(pdata.axes[idx],kwargs)


    # Stellar_Wind needs to iterate  over output and var name 
    if sdata.grid_setup["dimensions"] == 3:
        for output in sdata.load_outputs:
            pdata.output = output
            for var_name in pdata.var_choice:
                if pdata.plot_idx >= len(pdata.axes):
                    break
                    
                # Plot each variable in its own subplot
                pdata._var_name = var_name
                pdata.value = kwargs.get('value') if 'value' in kwargs else 0 #value for custom slice
                pcmesh_3d(sdata,pdata,value = pdata.value)
                plot_label(sdata,pdata,**kwargs)
                plot_axlim(pdata,kwargs)

                pdata.plot_idx += 1
    
    plot_save(sdata,pdata,**kwargs) # make sure is indent under run_names so that it saves multiple runs

def plot_1D_slice(sel_coord,sel_var,sdata,value_dict,pdata=None,sel_d_files = None,**kwargs):
    """
    Plots 1D slices of selected variables from Pluto simulations.
    """
    # if sdata.load_slice is None or sdata.slice_shape == "slice_2D":
        # raise ValueError(f"SimulationData load_slice is None or 2D ({sdata.load_slice}), use a 1D slice to plot")
    
    # value = kwargs.pop("value", 0)
    # target = kwargs['value'] if 'value' in kwargs and kwargs['value'] is not None else 0

    if pdata is None:
        pdata = PlotData(var_choice=[sel_var],**kwargs)

    pdata.arr_type = "cc"
    # pdata.value = kwargs.get('value') if 'value' in kwargs else pdata.value #value for custom slice
    sel_d_files = [sel_d_files] if sel_d_files and not isinstance(sel_d_files, list) else sel_d_files
    sel_d_files = sdata.load_outputs if sel_d_files is None else sel_d_files #load all or specific output

    axes, fig = subplot_base(sdata,pdata,load_outputs=sel_d_files,**kwargs)
    plot_idx = 0  # Keep track of which subplot index we are using

    sel_coord = pu.get_coord_names(arr_type="cc",coord=sel_coord) #converts x1 to ncx etc req for loading, labels
    slice_to_load = pa.calc_var_prof(sdata,sel_coord,value_1D=value_dict)["slice_1D"]
    for output in sel_d_files: # plot across all files
        pdata.output = output
        extras_data = plot_extras(sdata,pdata)
        xy_labels = extras_data["xy_labels"]
        var_label = sdata.get_var_info(sel_var)["var_name"]
        var_units = sdata.get_var_info(sel_var)["usr_uv"]
        coord_label = sdata.get_var_info(sel_coord)["coord_name"]
        coord_units = sdata.get_var_info(sel_coord)["usr_uv"]

        fluid_data = sdata.fluid_data(
            output=pdata.output,
            var_choice=[sel_coord,sel_var],
            load_slice=slice_to_load,
        )        
        
        title_str = f"{sdata.sim_type} {var_label}"
        ax = axes[plot_idx]

        # plot_axlim(ax,kwargs)
            
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
        legend_str = f"{title_str} @ {sdata.get_var_info(legend_coord)['coord_name']} = {value} {coord_units}"

        ax.legend([legend_str])

        plot_idx += 1
    plot_save(sdata,pdata,**kwargs)

def frame_density(sdata, var_name, sel_d_files=None, pdata=None, cb_lims=[-5, 5], cmap='inferno', textcolour='black', fps=2):
    start = time.time()
    print(f"({var_name}): {(time.time() - start):.2f}s - init")

    if pdata is None:
        pdata = PlotData()
        print(f"({var_name}): {(time.time() - start):.2f}s - PlotData initialized")

    sel_d_files = sdata.load_outputs if sel_d_files is None else sel_d_files
    total_outputs = len(sel_d_files)
    print(f"({var_name}): {(time.time() - start):.2f}s - data files loaded")

    latex_text_width = 418.25368
    paper_plot_kwargs = {"dpi": 600, "bbox_inches": "tight"}
    plt.rcParams.update({'font.size': 15})
    print(f"({var_name}): {(time.time() - start):.2f}s - rcParams updated")

    with plt.style.context("classic"):
        pdata.fig = plt.figure(figsize=(10, 10))
        pdata.fig.tight_layout()
        pdata.axes = ImageGrid(
            pdata.fig,
            111,
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
        print(f"({var_name}): {(time.time() - start):.2f}s - figure and ImageGrid created")

        def update_img(n, cb_lims, cmap, textcolour):
            ax.clear()
            current_d_file = sel_d_files[n]
            pdata.output = current_d_file
            print(f"({var_name}): {(time.time() - start):.2f}s - frame {n} start")

            extras_data = plot_extras(sdata, pdata)
            print(f"({var_name}): {(time.time() - start):.2f}s - frame {n} extras computed")

            ax.text(0.05, 0.95, f'{pdata.output.split("_")[-1]}',
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    fontsize=20,
                    color=textcolour)

            slice_var = pdata.spare_coord
            profile = pa.calc_var_prof(sdata, slice_var)["slice_2D"]
            print(f"({var_name}): {(time.time() - start):.2f}s - frame {n} profile calculated")

            is_log = var_name in ('rho', 'prs')
            vars_data = np.log10(sdata.get_vars(pdata.output)[var_name][profile]) if is_log else sdata.get_vars(pdata.output)[var_name][profile]
            print(f"({var_name}): {(time.time() - start):.2f}s - frame {n} vars_data fetched")

            var_idx = pdata.var_choice.index(var_name)
            c_map = extras_data["c_maps"][var_idx]
            cbar_label = extras_data["cbar_labels"][var_idx]

            im = ax.pcolormesh(
                sdata.get_vars(pdata.output)[pdata.var_choice[0]][profile],
                sdata.get_vars(pdata.output)[pdata.var_choice[1]][profile],
                vars_data,
                cmap=cmap,
                shading='auto',
            )
            print(f"({var_name}): {(time.time() - start):.2f}s - frame {n} pcolormesh done")

            cax = ax.cax
            cax.cla()
            cb = pdata.fig.colorbar(im, cax=cax)
            cb.set_label(cbar_label, fontsize=14)
            cb.ax.tick_params(labelsize='large')
            print(f"({var_name}): {(time.time() - start):.2f}s - frame {n} colorbar updated")

            ax.set_xlabel(extras_data["xy_labels"][pdata.var_choice[0]])
            ax.set_ylabel(extras_data["xy_labels"][pdata.var_choice[1]])
            ax.tick_params(axis='both')

            xlim = (np.min(sdata.get_vars()[pdata.var_choice[0]]), np.max(sdata.get_vars()[pdata.var_choice[0]]))
            ylim = (np.min(sdata.get_vars()[pdata.var_choice[1]]), np.max(sdata.get_vars()[pdata.var_choice[1]]))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            print(f"({var_name}): {(time.time() - start):.2f}s - frame {n} axis limits set")

            return im

        ani = animation.FuncAnimation(
            pdata.fig,
            update_img,
            frames=total_outputs,
            interval=30,
            save_count=total_outputs,
            fargs=(cb_lims, cmap, textcolour),
        )
        print(f"({var_name}): {(time.time() - start):.2f}s - FuncAnimation created")

        writergif = animation.PillowWriter(fps=fps)
        dpi = 200

        ani.save(f"{sdata.save_dir}/{sdata.sim_type}.gif", writer=writergif, dpi=dpi)
        print(f"({var_name}): {(time.time() - start):.2f}s - animation saved")

    return ani

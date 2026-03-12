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
    def __init__(self,arr_type = "nc",plane = "xz",var_choice = None,output = None, **kwargs):
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

    # if "particles" not in called_func: #use load_outputs if not particles plotter else use part_files
    # sel_d_files = load_outputs if load_outputs is not None else getattr(sdata, 'load_outputs', [])
    pdata.load_outputs = sdata.load_outputs if load_outputs is None else load_outputs
    # Validate we have files to plot
    if not sdata.load_outputs:
        raise ValueError("No data files provided (load_outputs is empty)")
        
    elif "particles" in called_func: # access particle data files for n plots
        sel_part_files = getattr(sdata,'particle_files',[])

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
            print(f"skipping {var_name} units due to no available unit values...")
            var_units = "None"
            var_label = "None"
        except TypeError:
            print("pdata.var_choice elements are not of type(str), skipping units/labels")
            var_units = "None"
            var_label = "None"

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
    cbar = pdata.fig.colorbar(im, ax=ax,fraction = 0.05) #, fraction=0.050, pad=0.25
    cbar.set_label(f"$\log_{{10}}$({cbar_label})" if is_log else cbar_label, fontsize=14)

def scatter_3d_particles(sdata,pdata = None, **kwargs):    
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
    particle_data = sdata.particle_data(output=(pdata.output,))
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
def plot_sim_fluid(sdata, pdata = None,load_outputs = None,save=0,**kwargs):
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
    
    plot_save(sdata,pdata,save=save,**kwargs) # make sure is indent under run_names so that it saves multiple runs
    return pdata

def plot_sim_particles(sdata,load_outputs=None, pdata = None,**kwargs):
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
                scatter_3d_particles(sdata,pdata,value = pdata.value)
                plot_label(sdata,pdata,**kwargs)
                plot_axlim(pdata,kwargs)
                pdata.plot_idx += 1
    
    plt.show()
    plot_save(sdata,pdata,**kwargs) # make sure is indent under run_names so that it saves multiple runs

def plot_1D_slice(sel_coord,sel_var,sdata,value_dict,load_outputs = None,pdata=None,**kwargs):
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
        legend_str = f"{title_str} @ {getattr(sdata.get_var_info(legend_coord),'coord_name')} = {value} {coord_units}"

        ax.legend([legend_str])
        plot_axlim(ax,kwargs)

        plot_idx += 1
    plot_save(sdata,pdata,**kwargs)

def plot_surface_brightness(sdata,freqs=[1.4],redshift=0.05,particle_outputs="last",angle=0,plane="xz",**kwargs):
    # if pdata is None:
    # loop over all outputs and all frequencies
    particle_outputs = [pl.get_particle_outputs(sdata.wdir)] if particle_outputs == "last" else particle_outputs
    pdata = PlotData(var_choice=freqs,plane=plane,load_outputs = particle_outputs,**kwargs)
    pdata.plot_idx = 0
    pdata.axes, pdata.fig = subplot_base(sdata=sdata,pdata=pdata,load_outputs=particle_outputs)
    extras = plot_extras(sdata,pdata)

    obs_properties = pa.setup_obs_properties_praise(sdata=sdata,redshift=redshift,angle=angle,plane=plane)
    sb_arr = pa.calc_surface_brightness_praise(
        sdata=sdata,
        freqs=freqs,
        redshift=redshift,
        particle_outputs=particle_outputs,
        angle=angle,
        plane=plane,
    )
    for i in range(0, len(particle_outputs), 1):
        for freq_ind, freq in enumerate(freqs):
            pdata.output = particle_outputs[i]

            # sb_arr[i][:,:,freq_ind] = np.nan_to_num(sb_arr[i][:,:,freq_ind], copy=True, nan=0.0, posinf=0.0, neginf=0.0) # get rid of NaNs (replace with zero)
            freq_sb = convolve(sb_arr[i][:, :, freq_ind].to(u.mJy / u.beam), obs_properties["gaussian_kernel"], boundary='extend') * (u.mJy / u.beam) #convolve the surface brightness
            freq_sb[freq_sb == 0] = np.nan # replace 0 SB areas with nan
            max_sb_lim = np.log10(np.nanpercentile(freq_sb, 99).value) # calculate sensible limits
            min_sb_lim = max_sb_lim - 1
            sb_contours = np.linspace(min_sb_lim, max_sb_lim, 3) # calculate contours

            # a = np.log10(freq_sb.value.T) # replace 0 SB areas with suitably low value
            # a[np.isnan(a)] = -100
            ax = pdata.axes[pdata.plot_idx]
    
            im = ax.pcolormesh(obs_properties["grid_x"], obs_properties["grid_y"], np.log10(freq_sb.value.T), vmin= min_sb_lim, vmax=max_sb_lim)
            cbar = pdata.fig.colorbar(im,ax=ax,fraction = 0.05)
            cbar.set_label(f"$\log_{{10}}$(SB [$\mathrm{{mJy \; beam^{{-1}}}}$])")

            ax.contour(obs_properties["grid_mx"], obs_properties["grid_my"], np.log10(freq_sb.value.T), levels=sb_contours, colors='white')

            ax.set_aspect("equal")
            ax.set_title(f"Surface Brightness [{freq}$GHz$] ({sdata.run_name})")
            ax.set_xlabel(extras["xy_labels"][pdata.coord_choice[0]])
            ax.set_ylabel(extras["xy_labels"][pdata.coord_choice[1]])
            pdata.plot_idx += 1



            time_str = f"{sdata.part_to_simtime(pdata.output):.1f} Myr"
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

            ax.annotate(
                f'angle = ${angle}^\circ$',
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

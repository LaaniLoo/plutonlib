import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl
import plutonlib.simulations as ps
import plutonlib.plot as pp
from plutonlib.colours import pcolours 

import numpy as np
import scipy as sp
from scipy import stats
from scipy import constants
from astropy import units as u

import matplotlib.pyplot as plt
from collections import defaultdict 
from IPython.display import display, Latex
import inspect

def find_nearest(array, value):
    """Find closes value in array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return {"idx":idx, "value": array[idx]}

def get_grid_idx(value,sdata,coord):
    """
    Method for finding array index of a specific value without loading full PLUTO grid arrays,
    note that this only works if the value and the pluto ini grid are in the same units, and if the grid patch
    containing `value` resides in a uniform patch. 

    value: value to find grid idx
    sdata: SimulationData object
    coord: dimension to find value for e.g. "x1","x2" or "x3"
    """
    coord = pu.map_coord_name(coord) #this lets you put in ncx and x1 etc 
    grid = sdata.grid_setup[f"{coord}-grid"]
    starts = grid["start"]
    ends = grid["end"]
    patch_cells = grid["patch_cells"]

    for patch in range(grid["n_patches"]): #loop across each grid patch to find which one contains origin
        cur_patch = f"[{starts[patch]} {patch_cells[patch]} {grid['type'][patch][0]} {ends[patch]}]"
        start = starts[patch]
        end = ends[patch]
        
        if starts[patch] <= value <= ends[patch]: #find the patch where value is located            
            if grid["type"][patch] == "uniform":
                start_disp = (value - start) #distance from start of patch 
                patch_length = (end-start) #total length of patch 
                patch_idx = (start_disp / patch_length)*patch_cells[patch] #fraction of distance to total patch scaled by number of cells gives the grid cell of value.
                idx = int(sum(patch_cells[:patch])+patch_idx) # since there can be multiple patches, need to add previous patch cells to above calculation 
                return idx

            else:
                raise NotImplementedError(f"{coord} = {value} lies in stretched patch: {cur_patch}, need to numerically solve the grid stretching ratio")
                #NOTE would  
            
    raise ValueError(f"Value {coord} = {value} is outside the grid extent for {cur_patch}")

def calc_var_prof(sdata,sel_coord,value_2D: float = None,value_1D: dict = None,**kwargs):
    """
    Calculates the array profile for two cases e.g. for sel_coord = "x1", value = 0:
        slice_1D : (slice(None, None, None), 600, 733) 
            -> 1D slice along x1 at x2_mid, x3_mid
            -> Shape: (n_x1,)

        slice_2D: (800, slice(None, None, None), slice(None, None, None)) 
            -> 2D plane in x2,x3 sliced in x1 at x1_mid
            -> Shape: (n_x2, n_x3)

    Parameters:
    sdata: 
        SimulationData object
    sel_coord: 
        Selected coordinate to slice in or about
    value_2D: 
        Value to slice at for 2D slice, e.g x1 = 20kpc, defaults to value_2D = 0 (midpoint)
    value_1D: 
        Used to make a slice at value for different coord to sel_coord for 1D slice, 
        e.g. slice at x1 = 20kpc and x2 = 0kpc 

    """
    sel_coord = pu.map_coord_name(sel_coord) #strips array_type from sel_coord e.g. ncx -> x1
    x,y,z = "x1","x2","x3" 

    idx_map = {x:None, y:None,z:None}   
    for coord in idx_map.keys():
        if value_1D and coord in value_1D:
            idx_map[coord] = get_grid_idx(value=value_1D[coord],sdata=sdata,coord=coord) 
        
        elif value_2D is not None and coord == sel_coord:
            idx_map[coord] = get_grid_idx(value=value_2D,sdata=sdata,coord=coord)
        
        else:
            idx_map[coord] = get_grid_idx(value=0,sdata=sdata,coord=coord)




    # --- Define slicing maps ---
    if sdata.grid_ndim > 2:
        slice_map_1D = {
            x : (slice(None), idx_map[y], idx_map[z]),
            y: (idx_map[x], slice(None), idx_map[z]),
            z: (idx_map[x], idx_map[y], slice(None)),
        }

        slice_map_2D = {
            x: (idx_map[x], slice(None), slice(None)),
            y: (slice(None), idx_map[y], slice(None)),
            z: (slice(None), slice(None), idx_map[z]),
        }

    else:
        slice_map_1D = {
            x: (slice(None), idx_map[y]),
            y: (idx_map[x], slice(None)),
        }
        slice_map_2D = None

    slice_1D = slice_map_1D[sel_coord]
    slice_2D = slice_map_2D[sel_coord] if sdata.grid_ndim > 2 else None

    return {
        "slice_1D": slice_1D,
        "slice_2D": slice_2D,
    }

#---Plot Grid structure---#
def plot_xz_grid_structure(sdata,pdata=None, d_file=None, n_lines=10, **kwargs):
    """
    Visualize the stretched grid structure in XZ plane
    """
    if sdata.load_slice is None or sdata.slice_shape == "slice_1D":
        raise ValueError(f"SimulationData load_slice is None or 1D ({sdata.load_slice}), use a 2D slice to plot")
    
    if pdata is None:
        pdata = pp.PlotData()

    if d_file is None:
        d_file = sdata.d_files[0]
    
    # Get coordinate arrays using get_grid_data()
    x1 = sdata.get_grid_data(d_file)["x1"]  # X coordinates
    x2 = sdata.get_grid_data(d_file)["x2"]  # Y coordinates  
    x3 = sdata.get_grid_data(d_file)["x3"]  # Z coordinates
    
    pdata.fig, pdata.axes = plt.subplots(1, 1, figsize=(5, 5))
    # XZ plane view - take middle y-plane
    y_slice = x1.shape[1] // 2  # Middle y-plane
    
    # Plot x-grid lines (constant x1)
    for i in range(0, x1.shape[0], max(1, x1.shape[0] // n_lines)):
        pdata.axes.plot(x1[i, :], x3[i, :], 'k-', alpha=0.6, linewidth=0.8)

    # Plot z-grid lines (constant x3)
    for k in range(0, x1.shape[1], max(1, x1.shape[1] // n_lines)):
        pdata.axes.plot(x1[:, k], x3[:, k], 'k-', alpha=0.6, linewidth=0.8)

    pp.plot_label(sdata,pdata,idx=0,no_title = True)
    pdata.axes.set_title(f'{sdata.run_name} XZ Grid Structure')
    pdata.axes.set_aspect("auto")
    pdata.axes.grid(True, alpha=0.3)
    
    pp.plot_save(sdata,pdata,**kwargs)

#---Peak Finding---#
def peak_findr(sel_coord,sel_var,sdata,**kwargs):
    """
    Calculates the max values of an array and their location, e.g. use to find max values of x2 for vx2 to find jet radius
    """

    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    radius = []
    peak_info = []
    peak_var = []
    locs = []

    var_profile = calc_var_prof(sdata,sel_coord,**kwargs)["slice_1D"]
    for d_file in sdata.d_files:
        var = sdata.get_vars(d_file)


        var_sliced = var[sel_var][var_profile]
        max_loc = np.where(var_sliced == np.max(var_sliced)) #index location of max variable val

        locs.append(max_loc[0])

        coord_array = var[sel_coord][var_profile] if sdata.grid_ndim >2 else var[sel_coord]    

        peak_info.append(f"{d_file} Radius: {coord_array[max_loc][0]:.2e} m, {sel_var}: {var_sliced[max_loc][0]:.2e}")
        peak_var.append(var_sliced[max_loc][0])
        radius.append(coord_array[max_loc][0])

    return {"peak_info": peak_info,"radius": radius, "peak_var": peak_var,"locs": locs } 

def graph_peaks(sel_coord,sel_var,sdata,**kwargs): #TODO Put in peak findr 
    """Follows a similar process to peak_findr() except it uses scipy signal peak finding, good for visual representation"""
    
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    coord_units = (sdata.get_var_info(sel_coord)["usr_uv"]).to_string('latex')
    var_units = (sdata.get_var_info(sel_var)["usr_uv"]).to_string('latex')

    var_peak_idx = defaultdict(list)
    peak_info = []
    peak_vars = []
    peak_coords = []
    
    var_profile = calc_var_prof(sdata,sel_coord)["slice_1D"]
    for d_file in sdata.d_files: #find graphical peaks across all data files

        var = sdata.get_vars(d_file)[sel_var]
        coord =  sdata.get_vars(d_file)[sel_coord]
        var_sliced = var[var_profile]

        var_peak_idx[d_file], _ = sp.signal.find_peaks(var_sliced)

        if np.any(var_peak_idx[d_file]):  # Only print if peaks exist 
            peak_vars.append(var_sliced[var_peak_idx[d_file][-1]])
            peak_coords.append(coord[var_peak_idx[d_file][-1]])

            #NOTE not sure what 
            peak_var = var_sliced[var_peak_idx[d_file][-1]]
            peak_coord = coord[var_peak_idx[d_file][-1]]

            peak_info.append(f"{d_file}: {sel_coord} = {peak_coord:.2e} {coord_units} {sel_var} = {peak_var:.2e} {var_units}")

        else:
            print(f"No peaks found in {d_file}")

    return {"var_peak_idx": var_peak_idx,"var_sliced": var_sliced,"peak_coords": peak_coords, "peak_vars": peak_vars, "peak_info":peak_info}

def all_graph_peaks(sel_coord,sel_var,sdata,**kwargs): #NOTE used for plotting same alg as peak_findr
    """A version of peak_findr() used for plotting, but includes all peak values """

    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    var_peak_idx = defaultdict(list)
    peak_vars = []
    peak_coords = []
    
    var = sdata.get_vars(sdata.d_last)[sel_var]
    coord = sdata.get_vars(sdata.d_last)[sel_coord]

    ndim = sdata.grid_ndim
    var_profile = calc_var_prof(sdata,sel_coord,**kwargs)["slice_1D"]
    var_sliced = var[var_profile]
    coord = coord[var_profile] if sdata.grid_ndim >2 else coord

    var_peak_idx, _ = sp.signal.find_peaks(var_sliced)
    var_trough_idx, _ = sp.signal.find_peaks(-var_sliced) 

    thresh = 1e-10 #threshold for trough finding

    if np.any(var_peak_idx):  # Only print if peaks exist 
        pvars_data = var_sliced[var_peak_idx]
        pcoords_data = coord[var_peak_idx]

        peak_vars = pvars_data[(pvars_data > 0) & (np.abs(pvars_data) > thresh)]
        peak_coords = pcoords_data[(pvars_data > 0) & (np.abs(pvars_data) > thresh)]

    if np.any(var_trough_idx):  # Only print if peaks exist and is <0
        tvars_data = var_sliced[var_trough_idx]
        tcoords_data = coord[var_trough_idx]

        trough_vars = tvars_data[(tvars_data < 0) & (np.abs(tvars_data) > thresh)]
        trough_coords = tcoords_data[(tvars_data < 0) & (np.abs(tvars_data) > thresh)]


    else:
        print(f"No peaks found in {sdata.d_last}")

    returns = {
        "var_sliced": var_sliced,
        "var_peak_idx": var_peak_idx,
        "var_trough_idx": var_trough_idx,
        "var_profile": var_profile,
        "peak_coords": peak_coords,
        "peak_vars": peak_vars,
        "trough_vars": trough_vars,
        "trough_coords": trough_coords,
        }
    
    return returns

def plot_peaks(sel_coord,sel_var,sdata,**kwargs): #TODO doesn't work for stelar wind rho
    """Plots the peaks found by all_graph_peaks()"""

    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    peak_data = all_graph_peaks(sel_coord,sel_var,sdata=sdata,**kwargs) #NOTE all goes wrong with graph_peaks, something to do with d_files
    if sdata.grid_ndim >2: #3D array needs sliced coords
        plot_coords = sdata.get_vars(sdata.d_last)[sel_coord][peak_data["var_profile"]]
    elif sdata.grid_ndim <=2:
        sdata.get_vars(sdata.d_last)[sel_coord]
    var_sliced = peak_data["var_sliced"]
    peak_coords = peak_data["peak_coords"]
    peak_vars = peak_data["peak_vars"]

    is_log = sel_var in ('rho','prs')
    base_plot_data = np.log10(var_sliced) if is_log else var_sliced
    peak_plot_data = np.log10(peak_vars) if is_log else peak_vars

    xlab = f"{sdata.get_var_info(sel_coord)['coord_name']} [{sdata.get_var_info(sel_coord)['usr_uv']}]"
    ylab = f"log10({sdata.get_var_info(sel_var)['var_name']}) [{sdata.get_var_info(sel_var)['usr_uv']}]" if is_log else f"{sdata.get_var_info(sel_var)['var_name']} [{sdata.get_var_info(sel_var)['usr_uv']}]"
    label = f"Peak {ylab}"
    title = f"{sdata.sim_type} Peak {ylab} Across {xlab}"


    f,a = plt.subplots(figsize = (7,7))
    a.plot(plot_coords,base_plot_data) # base plot
    a.plot(peak_coords,peak_plot_data,"x",label= label)
    a.legend()
    a.set_xlabel(xlab)
    a.set_ylabel(ylab)
    a.set_title(title)

    if 'xlim' in kwargs: # xlim kwarg to change x limits
        a.set_xlim(kwargs['xlim']) 

    if 'ylim' in kwargs: # xlim kwarg to change x limits
        a.set_ylim(kwargs['ylim']) 

    pdata = pp.PlotData() # assigning pdata fig object
    pdata.fig = plt.gcf()  #not sure but it works
    pp.plot_save(sdata,pdata,custom=1)

def plot_troughs(sel_coord,sel_var,sdata,**kwargs): #TODO doesn't work for stelar wind rho
    """Plots the peaks found by all_graph_peaks()"""

    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    vars_last = sdata.get_vars(sdata.d_last)
    peak_data = all_graph_peaks(sel_coord,sel_var,sdata=sdata,**kwargs) #NOTE all goes wrong with graph_peaks, something to do with d_files
    var_sliced = peak_data["var_sliced"]
    trough_coords = peak_data["trough_coords"]
    trough_vars = peak_data["trough_vars"]

    is_log = sel_var in ('rho','prs')
    base_plot_data = np.log10(var_sliced) if is_log else var_sliced
    peak_plot_data = np.log10(trough_vars) if is_log else trough_vars

    xlab = f"{sdata.get_var_info(sel_coord)['coord_name']} [{sdata.get_var_info(sel_coord)['usr_uv']}]"
    ylab = f"log10({sdata.get_var_info(sel_var)['var_name']}) [{sdata.get_var_info(sel_var)['usr_uv']}]" if is_log else f"{sdata.get_var_info(sel_var)['var_name']} [{sdata.get_var_info(sel_var)['usr_uv']}]"
    label = f"trough {ylab}"
    title = f"{sdata.sim_type} trough {ylab} Across {xlab}"


    f,a = plt.subplots(figsize = (7,7))
    a.plot(vars_last[sel_coord],base_plot_data) # base plot
    a.plot(trough_coords,peak_plot_data,"x",label= label)
    a.legend()
    a.set_xlabel(xlab)
    a.set_ylabel(ylab)
    a.set_title(title)

    if 'xlim' in kwargs: # xlim kwarg to change x limits
        a.set_xlim(kwargs['xlim']) 

    if 'ylim' in kwargs: # xlim kwarg to change x limits
        a.set_ylim(kwargs['ylim']) 

    pdata = pp.PlotData() # assigning pdata fig object
    pdata.fig = plt.gcf()  #not sure but it works
    pp.plot_save(sdata,pdata,custom=1)



#---Plot length/radius across sim_time---#
def get_jet_length_dim(sdata):
    """Gets the array with the longest grid size, used for Jet length as safety net"""
    coords = sdata.get_coords() #loads all coords at d_last
    longest = max([("x1", len(coords["x1"])), ("x2", len(coords["x2"])), ("x3", len(coords["x3"]))], key=lambda x: x[1])[0]

    return longest

def tprog_phelper(sel_coord,r,sdata,type,**kwargs):
    """ Helper function for plot_time_prog, handles plotting assignment"""
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)


    var_info = sdata.get_var_info(sel_coord)

    xlab = f"sim_time [{sdata.get_var_info('sim_time')['usr_uv']}]"
    ylab = f"{var_info['coord_name']}-Radius [{var_info['usr_uv']}]"
    title = f"{sdata.sim_type} {ylab} across {xlab}"

    t_values = []
    for data in sdata.d_files:
        t_values.append(sdata.get_vars(data)["sim_time"])

    #Legend assignment based on sim_time
    if sdata.grid_setup["dimensions"] == 2:
        longest_array = get_jet_length_dim(sdata) # used to avoid confusion btwn jet length and width
        measurement = "length" if sel_coord == longest_array else "width"
        legend_base = f"{sdata.sim_type} {var_info['coord_name']}-Radius ({measurement})" #names legend based on width or length

    elif sdata.grid_setup["dimensions"] == 3:
        legend_base = f"{sdata.sim_type} {var_info['coord_name']}-Radius"
    
    if type == "def": #default type plot
        f,a = plt.subplots()

        print(t_values)
        print(r)

        a.plot(t_values, r, color = "darkorchid") # base plot
        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
        a.set_title(title)

        a.legend([legend_base])
        a.text(0.05, 0.8,f"R = {r[-1]:.2e} m", transform=a.transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 

        a.plot(t_values,r,"x", label = sdata.d_files)
        for i, d_file in enumerate(sdata.d_files):
            a.annotate(d_file.strip("data_"), (t_values[i], r[i]), textcoords="offset points", xytext=(1,1), ha='right')

        pdata = pp.PlotData()
        pdata.fig = plt.gcf()  
        pp.plot_save(sdata,pdata,custom=1,**kwargs)
    
        return None
    
    elif type == "log": #log-log type plot
        d_files = sdata.d_files[1:]
        t_values = np.log10(t_values[1:])
        r = np.log10(r[1:])

        slope, intercept, r_value, p_value, std_err = stats.linregress(t_values, r)
        eqn = f'$R_{var_info["coord_name"]} \\propto t^{{{slope:.2f}}} \\pm {std_err:.2f} \\; [m]$'
        display(Latex(eqn))

        f,a = plt.subplots()
        a.plot(t_values, r, color = "orange") # base plot

        # r_ideal = (t_values ** 0.6)
        # a.plot(t_values, r_ideal, color="hotpink")

        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
        a.set_title(title)

        a.legend([legend_base,r'Ideal: $t^{0.6}$'])
        a.text(0.05, 0.8, eqn, transform=a.transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 
        # a.text(slope)
        a.plot(t_values,r,"x", label = d_files)
        for i, d_file in enumerate(d_files):
            a.annotate(d_file.strip("data_"), (t_values[i], r[i]), textcoords="offset points", xytext=(1,1), ha='right')

        pdata = pp.PlotData()
        pdata.fig = plt.gcf()  
        pp.plot_save(sdata,pdata,custom=1,**kwargs)

        return slope

def plot_time_prog(sel_coord,sdata,type="def",**kwargs): #NOTE removed sel_var as it shouldn't matter unless stellar wind
    """
    Plots the calculated radius of a sim, e.g. jet radius across sim_time
    * Jet: graph_peaks() for plotting, calculates the peak dens at d_file -> end of jet -> assigns radius
    * Stellar_Wind: peak_finder() for plotting
    """
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)


    r = []


    if sdata.grid_setup["dimensions"] == 2:
        sel_var = "rho" #graph peaks doesn't care which var is used, a peak is a peak?
        peak_data = graph_peaks(sel_coord,sel_var,sdata) 
        var_peak_idx = peak_data["var_peak_idx"]


        for d_file in sdata.d_files:
            coord = sdata.get_vars(d_file)[sel_coord]

            if np.any(var_peak_idx[d_file]):
                r.append(coord[var_peak_idx[d_file]][-1])
            else:
                r.append(0)

    elif sdata.grid_setup["dimensions"] == 3:
        #TODO fix below assignment -> fix peak_findr 
        coord_dim = sel_coord.strip("x")
        sel_var = "vx" + coord_dim #NOTE peak_findr DOES care which var is used, set to vel?
        print(f"{pcolours.WARNING}Note: stellar wind only works for velocity components setting sel_var = {sel_var}")

        peak_data = peak_findr(sel_coord,sel_var,sdata=sdata) 
        r = peak_data["radius"]


    slope = tprog_phelper(sel_coord,r,sdata,type,**kwargs)

     
#---Energy, Density, Radius Calculations---#
def calc_energy(sdata,sel_coord = "x2",type = "sim",plot=0,**kwargs):
    """
    Calculates the Q value for a simulation given its density and velocity
    * type = "sim": calculates the observed value using simulation values
    * type = "calc": calculates the theoretical value using calculated density from calc_density()
    """
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    peak_data = peak_findr("x2","vx2",sdata=sdata) #NOTE USE OF PREDET VARS
    radius = peak_data["radius"] # calculated shell/jet radii at sim_time from max vx2
    vel = peak_data["peak_var"] # corresponding velocity at the above radii 
    locs = peak_data["locs"] #index location where max occurs

    q_jet = []

    if type == "sim": #calculates using simulated values
        rho = []
        profile = calc_var_prof(sdata,sel_coord)["slice_1D"]

        for loc, d_file in zip(locs,sdata.d_files):

            rho_slice = sdata.get_vars(d_file)["rho"][profile]
            rho.append(rho_slice[loc])

        for i in range(len(sdata.d_files)):
            value = 0.5*4*np.pi*(radius[i]**2)*rho[i]*(vel[i]**3)
            
            q_jet.append(value[0])

    if type == "calc": #calculates using theoretical values of rho

        rho_calc = calc_density(sdata)["rho_calc"]

        for i in range(len(sdata.d_files)):
            value = 0.5*4*np.pi*(radius[i]**2)*rho_calc[i]*(vel[i]**3)

            q_jet.append(value)
        

    if plot:
        t = sdata.get_vars(sdata.d_last)["sim_time"]

        eqn = '$Q_{jet} = \\frac{1}{2}4\\pi r_s^2 \\rho(r_s) V^3_s$'
        display(Latex(eqn))

        plt.figure()
        plt.title("Plot of Energy vs sim_time")
        plt.plot(t,q_jet,label = "$Q_{jet}$")
        plt.ylabel("Jet Energy [J]")
        plt.xlabel("sim_time [yr]")
        plt.legend()

        pdata = pp.PlotData()
        pdata.fig = plt.gcf()  
        pp.plot_save(sdata,pdata,custom=1)

    else:
        return q_jet

def calc_radius(sdata,plot =0,**kwargs):
    """
    Calculates the radius as a function of time using calc_energy()
    * has both simulated and calculated values 
    """
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    rt_sim, rt_calc = [], []
    rho_0 = 1 * pc.code_to_usr_units("rho",sdata.d_files)["uv_usr"]

    t = sdata.get_vars(sdata.d_last)["sim_time"]

    q_jet_sim = calc_energy(sdata=sdata,type="sim")
    q_jet_calc = calc_energy(sdata=sdata,type="calc")

    for i in range(len(t)):
        sim = ((q_jet_sim[i]/rho_0)**(1/5))*(t[i]**(3/5)) #simulated values
        calc = ((q_jet_calc[i]/rho_0)**(1/5))*(t[i]**(3/5)) #calculated values
        rt_sim.append(sim)
        rt_calc.append(calc)
    
    returns = {
        "rt_sim": rt_sim,
        "rt_calc": rt_calc,
        "t": t
    }

    if plot:
        eqn = '$R(t) = K({\\frac{Q_{jet}}{\\rho_0}})^{1/5}\\cdot t^{3/5}$'
        display(Latex(eqn))

        plt.figure()
        plt.title("Plot of calculated/simulated R(t) vs sim_time")
        plt.plot(t,rt_calc,label = "calc r(t)")
        plt.plot(t,rt_sim,label = "r(t)")
        plt.ylabel("radius [m]")
        plt.xlabel("sim_time [yr]")
        plt.legend()

        pdata = pp.PlotData()
        pdata.fig = plt.gcf()  
        pp.plot_save(sdata,pdata,custom=1)

    else:
        return returns

def calc_radial_vel(sdata,plot = 0):
    """
    Calculates the radial velocity using its own calculated value of radius, r_0 and v_wind
    """


    r_0 = 1 * pc.code_to_usr_units("x1",sdata.d_files)["uv_usr"]
    v_wind = 1 * pc.code_to_usr_units("vx1",sdata.d_files)["uv_usr"]
    rho_norm = 1 * pc.code_to_usr_units("rho",sdata.d_files)["uv_usr"]

    v_r = []

    all_vars = sdata.get_grid_data()

    if sdata.sim_type == "Jet": #Jet radius is just Z coord?
        r = all_vars["x2"]
    else:
        r = np.sqrt((all_vars["x1"]**2) + (all_vars["x2"]**2) + (all_vars["x3"]**2))


    for i in range(len(r)):
            v_r = np.append(v_r,np.tanh((r[i]/r_0/0.1))*v_wind)

    returns = {
            "r": r,
            "v_r": v_r 
    }

    if plot:
        plt.figure()
        plt.title("Plot of v(r) vs r")
        plt.xlabel("r vector (m)")
        plt.ylabel("v(r)")
        plt.plot(r,v_r)

        pdata = pp.PlotData()
        pdata.fig = plt.gcf()  
        pp.plot_save(sdata,pdata,custom=1)

    else:
        return returns
    
def calc_density(sdata,sel_coord = "x2",plot = 0,**kwargs):
    """
    Calculates the density values using the radial velocity from calc_radial_vel() as well as r_0 and v_wind
    """
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = ps.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)


    r_0 = 1 * pc.code_to_usr_units("x1",sdata.d_files)["uv_usr"]
    v_wind = 1 * pc.code_to_usr_units("vx1",sdata.d_files)["uv_usr"]
    rho_norm = 1 * pc.code_to_usr_units("rho",sdata.d_files)["uv_usr"]

    v_r_data =calc_radial_vel(sdata,plot=0)
    v_r = v_r_data["v_r"]
    r = v_r_data["r"]
    
    calc = (v_wind*(r_0**2))/(v_r*(r**2))
    rho_calc = calc*rho_norm
    rho_sim = sdata.get_vars(sdata.d_last)["rho"]
    profile = calc_var_prof(sdata,sel_coord)["slice_1D"]

    returns = {
        "rho_calc": rho_calc,
        "rho_sim": rho_sim[profile],
        "r": r
    }
    if plot:
        eqn = '$\\rho = \\frac{V_{wind} r_0^2}{v(r) r^2}$'
        display(Latex(eqn))

        plt.figure()
        plt.title("calculated vs simulated densities")
        plt.plot(r,np.log10(rho_sim[profile]), label = "rho")
        plt.plot(r,np.log10(rho_calc), label = "rho_calc")
        plt.ylabel("log_10(density [kgm^-3])")
        plt.xlabel("radius [m]")
        plt.legend()

        pdata = pp.PlotData()
        pdata.fig = plt.gcf()  
        pp.plot_save(sdata,pdata,custom=1)

    else:    
        return returns

def jet_kinetic_power(radius,rho,vel):
    eqn = 0.5*4*np.pi*(radius**2)*rho*(vel**3)
    return eqn.si
# rkpc = 4.5 * u.kpc
# rho = (1e-2*(5/3 * 1e-28)) * (u.gram / u.cm**3)
# jet_kinetic_power(rkpc.to(u.m),rho.si,2.5e7 * (u.meter / u.second))

def EOS(rho=None,prs=None,T=None,mu = 0.60364):
    """
    Simple Equation of state calculator to get Temp for a given density and pressure etc...
    """
    m_H = constants.m_p
    kb = constants.k
    
    if not T:
        unit = (u.Kelvin)
        T = (prs*mu*m_H)/(rho*kb)*unit
        return T 
    
    if not prs:
        unit = (u.pascal)
        prs = (kb*rho*T)/(mu*m_H)*unit
        return prs 
    
    if not rho:
        unit = (u.kg)/(u.m**3)
        rho = (prs*mu*m_H)/(T*kb)*unit
        return rho 

def central_sound_speed(rho_0,T):
   prs_0 = EOS(rho =rho_0,T = T).value
   nonrel_gamma = 5/3
   unit = u.m / u.s
   return (np.sqrt((nonrel_gamma * prs_0) / (rho_0)))*unit

def get_inlet_speed(rho_0,T,wind_vxx):
   inlet_vxx = []
   for vx in wind_vxx:
      inlet_vxx.append((vx*central_sound_speed(rho_0=rho_0,T=T)).to(u.kpc / u.Myr))
   return inlet_vxx

def locate_injection_region(rho_0,T,wind_vxx,sim_time):
    sim_time = sim_time * u.Myr
    inlet_vxx = get_inlet_speed(rho_0=rho_0,T=T,wind_vxx=wind_vxx)
    inj_xyz = []
    for vx in inlet_vxx:
        inj_xyz.append(-vx*sim_time)
    return inj_xyz



#---Jet angle---#
def binned_mean_tracer_mask(bin_size, x, y, tr_array, tr_cut=None, side=None, **kwargs):
    """
    Calculates a mean value of x/y across some number of bins for a given tracer cutoff.
    """
    x_mean = [0]
    y_mean = [0]
    bin_counter = 0

    for i in range(0, len(x), bin_size):
        x_slice = x[i:i+bin_size]
        y_slice = y[i:i+bin_size]
        tr_slice = tr_array[i:i+bin_size]

        # simple mask
        if side == "top":
            mask = (
                (y_slice > 0) &
                (tr_slice >= tr_cut)
            )
        elif side == "bot":
            mask = (
                (y_slice < 0) &
                (tr_slice >= tr_cut)
            )
        else:
            raise ValueError("side must be 'top' or 'bot'")

        x_cut = x_slice[mask]
        y_cut = y_slice[mask]

        if len(y_cut) == 0:
            continue

        x_next_mean = np.mean(x_cut)
        y_next_mean = np.mean(y_cut)
  
        last_y = y_mean[-1]
        last_x = x_mean[-1]


        if (np.abs(y_next_mean) / np.abs(last_y) > 0.95):
        # if x_next_mean/last_x > 0.33:
            x_mean.append(x_next_mean)
            y_mean.append(y_next_mean)


        bin_counter += 1

    return x_mean, y_mean

def dx_mean_tracer_mask(nslices,x, y, tr_array, tr_cut=None, side=None, **kwargs):
    """
    Calculates a mean value of x/y across some number of bins for a given tracer cutoff.
    """
    x_mean = [0]
    y_mean = [0]

    grid_slices = np.linspace(0,x[-1],nslices)
    for grid_start in grid_slices:
        dx = grid_slices[1] - grid_slices[0]

        # simple mask
        if side == "top":
            mask = (
                (y >= 0) &
                (tr_array >= tr_cut) &
                (x >= grid_start) &
                (x <= grid_start + dx)
            )
        elif side == "bot":
            mask = (
                (y <= 0) &
                (tr_array >= tr_cut) &
                (x >= grid_start) &
                (x <= grid_start + dx)
            )
        else:
            raise ValueError("side must be 'top' or 'bot'")

        x_cut = x[mask]
        y_cut = y[mask]

        if len(y_cut) == 0:
            continue

        # y_next_mean = np.max(y_cut) if side == "top" else np.min(y_cut)
        if side == "top":
            x_next_mean = np.mean(x_cut)
            # y_next_mean = np.percentile(y_cut,80)
            y_next_mean = np.max(y_cut)
            
        if side == "bot":
            x_next_mean = np.mean(x_cut)
            y_next_mean = np.min(y_cut)

            # y_next_mean = np.percentile(y_cut,20)

        if len(x_mean) > 0:
            last_y = y_mean[-1]
            last_x = x_mean[-1]

            x_mean.append(x_next_mean)
            y_mean.append(y_next_mean)

        else:
            # First valid data point
            x_mean.append(x_next_mean)
            y_mean.append(y_next_mean)


    return x_mean, y_mean

def jet_angle_linegress(sdata,load_outputs,bin_size,tr_cut = None,**kwargs):
    """
    Calculates the jet angle from the binned mean tracer cutoff using a linear regression
    """
    jet_angle = []
    angles_top, angles_bot = [],[]
    sdata.load_particles(load_outputs)
    part_output = sdata.particle_data
    part_files = sdata.particle_files

    for part_file in part_files:
        particle_data = part_output[part_file]
        x_bmt, y_bmt = binned_mean_tracer_mask(bin_size,particle_data["x1"],particle_data["x3"],particle_data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = binned_mean_tracer_mask(bin_size,particle_data["x1"],particle_data["x3"],particle_data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)
        slope_top = stats.linregress(x_bmt,y_bmt).slope
        slope_bot = stats.linregress(x_bmb,y_bmb).slope

        angle_top = np.abs(np.rad2deg(np.atan(slope_top)))
        angle_bot = np.abs(np.rad2deg(np.atan(slope_bot)))

        # print("Angle of top lobe WRT horizontal:",f"{angle_top:.2f} deg")
        # print("Angle of bottom lobe WRT horizontal:",f"{angle_bot:.2f} deg")

        angles_top.append(angle_top)
        angles_bot.append(angle_bot)
        jet_angle.append(angle_top+angle_bot)
        # print("Angle of jet:",f"{angle_top+angle_bot:.2f} deg")
        print("Top angle:",f"{angle_top:.2f}","Bottom angle:",f"{angle_bot:.2f}")

        mask_vars = ["x1", "x2", "x3","tracer"]
        for mask_var in mask_vars:
            tracer_mask = particle_data["tracer"] > tr_cut # change to tr_cut to actually represent

            particle_data[mask_var] = particle_data[mask_var][tracer_mask]


    returns = {
        "part_output":part_output,"part_files":part_files,"angles_top":angles_top,"angles_bot":angles_bot,"jet_angle":jet_angle,"tr_cut":tr_cut
    }
    return returns

def jet_angle_vector_old(sdata,load_outputs,nslices,tr_cut = None,**kwargs):
    """
    Calculates the jet angle from the binned mean tracer cutoff using a linear regression
    """
    jet_angle = []
    angles_top, angles_bot = [],[]
    sdata.load_particles(load_outputs)
    part_output = sdata.particle_data
    part_files = sdata.particle_files

    for part_file in part_files:
        particle_data = part_output[part_file]
        x_bmt, y_bmt = dx_mean_tracer_mask(nslices=nslices,x=particle_data["x1"],y=particle_data["x3"],tr_array=particle_data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = dx_mean_tracer_mask(nslices=nslices,x=particle_data["x1"],y=particle_data["x3"],tr_array=particle_data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)

        A = np.array([x_bmt, y_bmt]) 
        B = np.array([x_bmb, y_bmb]) 

        # min_len = min(A.shape[1], B.shape[1])
        # A_trim = A[:, :min_len]  # shape (2, min_len)
        # B_trim = B[:, :min_len]

        y_max_idx_A = np.argmax(A[1,:])
        y_min_idx_B = np.argmin(B[1,:])

        start_A = A[:,0]
        start_B = B[:,0]
        end_A = A[:,y_max_idx_A]
        end_B = B[:,y_min_idx_B]

        vec_A = end_A - start_A
        vec_B = end_B - start_B

        # Dot product & angle
        dot_product = np.dot(vec_A, vec_B)
        magnitude_A = np.linalg.norm(vec_A)
        magnitude_B = np.linalg.norm(vec_B)
        angle_radians = np.arccos(dot_product / (magnitude_A * magnitude_B))
        angle_degrees = np.degrees(angle_radians)
        jet_angle.append(angle_degrees)
        # print(f"Angle between top and bottom curves: {angle_degrees:.2f} degrees")


        mask_vars = ["x1", "x2", "x3","tracer"]
        for mask_var in mask_vars:
            tracer_mask = particle_data["tracer"] > tr_cut # change to tr_cut to actually represent

            particle_data[mask_var] = particle_data[mask_var][tracer_mask]


    returns = {
        "part_output":part_output,"part_files":part_files,"jet_angle":jet_angle,"tr_cut":tr_cut
    }
    return returns

def jet_angle_vector(sdata,load_outputs,nslices,tr_cut = None,**kwargs):
    """
    Calculates the jet angle from the binned mean tracer cutoff using a linear regression
    """
    jet_angle = []
    angles_top, angles_bot = [],[]
    sdata.load_particles(load_outputs)
    part_output = sdata.particle_data
    part_files = sdata.particle_files

    for part_file in part_files:
        particle_data = part_output[part_file]
        x_bmt, y_bmt = dx_mean_tracer_mask(nslices=nslices,x=particle_data["x1"],y=particle_data["x3"],tr_array=particle_data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = dx_mean_tracer_mask(nslices=nslices,x=particle_data["x1"],y=particle_data["x3"],tr_array=particle_data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)

        A = np.array([x_bmt, y_bmt]) 
        B = np.array([x_bmb, y_bmb]) 

        # min_len = min(A.shape[1], B.shape[1])
        # A_trim = A[:, :min_len]  # shape (2, min_len)
        # B_trim = B[:, :min_len]

        y_max_idx_A = np.argmax(A[1,:])
        y_min_idx_B = np.argmin(B[1,:])

        # y_max_idx_A = np.argmin(np.abs(A[1,:] - np.median(A[1,:])))
        # y_min_idx_B = np.argmin(np.abs(B[1,:] - np.median(B[1,:])))
        start_A = A[:,0]
        start_B = B[:,0]
        end_A = A[:,y_max_idx_A]
        end_B = B[:,y_min_idx_B]

        vec_A = end_A - start_A
        vec_B = end_B - start_B

        # Dot product & angle
        dot_product = np.dot(vec_A, vec_B)
        magnitude_A = np.linalg.norm(vec_A)
        magnitude_B = np.linalg.norm(vec_B)
        angle_radians = np.arccos(dot_product / (magnitude_A * magnitude_B))
        angle_degrees = np.degrees(angle_radians)
        jet_angle.append(angle_degrees)
        # print(f"Angle between top and bottom curves: {angle_degrees:.2f} degrees")


        mask_vars = ["x1", "x2", "x3","tracer"]
        for mask_var in mask_vars:
            tracer_mask = particle_data["tracer"] > tr_cut # change to tr_cut to actually represent

            particle_data[mask_var] = particle_data[mask_var][tracer_mask]


    returns = {
        "part_output":part_output,"part_files":part_files,"jet_angle":jet_angle,"tr_cut":tr_cut
    }
    return returns

def plot_jet_angle_particles(sdata,outputs,tr_cut,bin_size,show_means=True,**kwargs):
    """
    Plots the jet angle across simulation outputs using the two helper functions:
    `binned_mean_tracer_mask`, `jet_angle_particles`
    """
    jet_angle_data = jet_angle_linegress(sdata,outputs,bin_size=bin_size,tr_cut=tr_cut,**kwargs) #smaller cuttoff  is better for later data 
    part_files = jet_angle_data["part_files"]
    part_output = jet_angle_data["part_output"]
    jet_angles = jet_angle_data["jet_angle"]

    pdata = pp.PlotData(**kwargs)
    pdata.axes,pdata.fig = pp.subplot_base(sdata,fig_resize = 5)

    for ax, part_file,angle in zip(pdata.axes, part_files,jet_angles):
        data = part_output[part_file]

        # # scatter plot using pre masked data
        im = ax.scatter(
            data["x1"], data["x3"],
            c=data["tracer"],
            s=2.5
        )

        ax.text(
                0.05, 0.98, f"Angle ={angle:.2f}°",
                transform=ax.transAxes,   # relative to axis (0–1 coords)
                fontsize=8,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
            )
        
        # median points and regression of median points
        x_bmt, y_bmt = binned_mean_tracer_mask(bin_size,data["x1"],data["x3"],data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = binned_mean_tracer_mask(bin_size,data["x1"],data["x3"],data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)
        
        z1 = np.polyfit(x_bmt, y_bmt, 1)
        p1 = np.poly1d(z1)
        ax.plot(x_bmt, p1(x_bmt), "r--")

        z2 = np.polyfit(x_bmb, y_bmb, 1)
        p2 = np.poly1d(z2)
        ax.plot(x_bmb, p2(x_bmb), "r--")

        #additional scatter of means
        if show_means:
            ax.scatter(x_bmt, y_bmt, color="r", s=2.5) 
            ax.scatter(x_bmb, y_bmb, color="r", s=2.5)

        #title etc
        # ax.set_aspect("equal")
        ax.set_xlabel("X / kpc")
        ax.set_ylabel("Z / kpc")
        ax.set_title(f"{part_file}")

    # plt.tight_layout()
    pdata.fig.colorbar(im, ax=pdata.axes, label="Tracer")#,fraction = 0.05)
    plt.show()

def plot_jet_angle_particles_vector(sdata,outputs,tr_cut,nslices,show_means=True,**kwargs):
    jet_angle_data = jet_angle_vector(sdata,outputs,nslices=nslices,tr_cut=tr_cut,**kwargs) #smaller cuttoff  is better for later data 
    part_files = jet_angle_data["part_files"]
    part_output = jet_angle_data["part_output"]
    jet_angles = jet_angle_data["jet_angle"]

    pdata = pp.PlotData(**kwargs)
    pdata.axes,pdata.fig = pp.subplot_base(sdata,fig_resize = 5)

    for ax, part_file,angle in zip(pdata.axes, part_files,jet_angles):
        data = part_output[part_file]

        # # scatter plot using pre masked data
        im = ax.scatter(
            data["x1"], data["x3"],
            c=data["tracer"],
            s=2.5
        )

        ax.text(
                0.05, 0.98, f"Angle ={angle:.2f}°",
                transform=ax.transAxes,   # relative to axis (0–1 coords)
                fontsize=8,
                verticalalignment="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
            )
        
        # median points and regression of median points
        x_bmt, y_bmt = dx_mean_tracer_mask(nslices=nslices,x=data["x1"],y=data["x3"],tr_array=data["tracer"],side = "top",tr_cut=tr_cut,**kwargs)
        x_bmb, y_bmb = dx_mean_tracer_mask(nslices=nslices,x=data["x1"],y=data["x3"],tr_array=data["tracer"],side = "bot",tr_cut=tr_cut,**kwargs)
        
        A = np.array([x_bmt, y_bmt]) 
        B = np.array([x_bmb, y_bmb]) 

        y_max_idx_A = np.argmax(A[1,:])
        y_min_idx_B = np.argmin(B[1,:])

        # y_max_idx_A = np.argmin(np.abs(A[1,:] - np.median(A[1,:])))
        # y_min_idx_B = np.argmin(np.abs(B[1,:] - np.median(B[1,:])))

        start_A = A[:,0]
        start_B = B[:,0]
        end_A = A[:,y_max_idx_A]
        end_B = B[:,y_min_idx_B]
        vec_A = end_A - start_A
        vec_B = end_B - start_B

        origin = A[0]
        ax.quiver(*start_A, *vec_A, angles='xy', scale_units='xy', scale=1, color="black",width = 1.5e-2)
        ax.quiver(*start_B, *vec_B, angles='xy', scale_units='xy', scale=1, color="black",width = 1.5e-2)

        #additional scatter of means
        if show_means:
            ax.scatter(A[0,:], A[1,:], color="r", s=2.0,alpha = 0.4)
            ax.scatter(B[0,:], B[1,:], color="r", s=2.0,alpha = 0.4)

        #title etc
        ax.set_aspect("equal")
        ax.set_xlabel("X / kpc")
        ax.set_ylabel("Z / kpc")
        ax.set_title(f"{part_file}")

    # plt.tight_layout()
    pdata.fig.colorbar(im, ax=pdata.axes[-1], label="Tracer",pad = 0.0001)#,fraction = 0.05)

    plt.show()
    pp.plot_save(sdata,pdata,**kwargs) 

def jet_angle_tprog_linegress(sdata,outputs,tr_cut,bin_size):
    """
    Plots the calculated jet angle across simulation outputs
    """
    jet_angle_data = jet_angle_linegress(sdata,outputs,tr_cut=tr_cut,bin_size=bin_size) #smaller cuttoff  is better for later data 
    part_files = jet_angle_data["part_files"]
    part_output = jet_angle_data["part_output"]
    jet_angles = jet_angle_data["jet_angle"]

    times = []
    for output in outputs:
        times.append(output / 10)

    slope, intercept, r_value, p_value, std_err = stats.linregress(times, jet_angles)
    eqn = f'$\\text{{slope}} = {{{slope:.2f}}} \\pm {std_err:.2f}\\;\\;[\\mathrm{{deg/Myr}}]$'
    plt.title("Jet angle vs Time ")
    plt.ylabel("Angle [deg]")
    plt.xlabel("Time [Myr]")
    plt.text(0.05, 0.05, eqn, transform=plt.gca().transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 

    plt.plot(times,jet_angles)   

def jet_angle_tprog_vector(sdata,outputs,tr_cut,nslices):
    """
    Plots the calculated jet angle across simulation outputs
    """
    jet_angle_data = jet_angle_vector(sdata,outputs,tr_cut=tr_cut,nslices=nslices) #smaller cuttoff  is better for later data 
    part_files = jet_angle_data["part_files"]
    part_output = jet_angle_data["part_output"]
    jet_angles = jet_angle_data["jet_angle"]

    times = []
    for output in outputs:
        times.append(output / 10)

    slope, intercept, r_value, p_value, std_err = stats.linregress(times, jet_angles)
    eqn = f'$\\text{{slope}} = {{{slope:.2f}}} \\pm {std_err:.2f}\\;\\;[\\mathrm{{deg/Myr}}]$'
    plt.title("Jet angle vs Time ")
    plt.ylabel("Angle [deg]")
    plt.xlabel("Time [Myr]")
    plt.text(0.05, 0.05, eqn, transform=plt.gca().transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 

    plt.plot(times,jet_angles) 

def calc_var_prof_old(sdata,sel_coord,ndim = 3,**kwargs):
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file', sdata.ini_file)
    ndim = ndim #TODO replace this with a try except loop
    value = 0 if not kwargs.get('value') else kwargs.get('value')
    x,y,z = pu.get_coord_names()

    sel_coord = pu.unmap_coord_name(sel_coord,arr_type=arr_type)
    # --- Determine whether to use find_nearest or grid midpoints ---
    
    use_find_nearest = (
        (value is not None and value != 0) #use find_nearest if value is not 0 
        or ('idx' in kwargs and kwargs['idx'] is not None)
    )

    if value == 0 and not use_find_nearest: #using ini grid values if value is 0, -> midpoint
        x_mid = sdata.grid_setup["x1-grid"]["origin_idx"]
        y_mid = sdata.grid_setup["x2-grid"]["origin_idx"] if ndim > 1 else None
        z_mid = sdata.grid_setup["x3-grid"]["origin_idx"] if ndim > 2 else None


    elif value !=0 and use_find_nearest: #using value/idx kwargs only if value is not 0
        print("calc_var_prof: using find_nearest")
        target = value  if 'value' in kwargs and value  is not None else 0

        if 'idx' in kwargs:
            idx = kwargs['idx']
        else:
            # 1D slices for coordinate lookup
            value_slice_map = {
                x : (slice(None), 0, 0),
                y : (0, slice(None), 0),
                z : (0, 0, slice(None)),
            }
            coords_1D = {
                coord: (
                    sdata.fluid_data(coord,load_slice=None)[coord][value_slice_map[coord]]
                    if arr_type in ('nc', 'cc')
                    else sdata.sdata.fluid_data(coord,load_slice=None)[coord]
                )
                for coord in list(pu.get_coord_names()) #returns ncx,ncy,ncz etc depending on arr_type
            }

            idx = find_nearest(coords_1D[sel_coord], target)['idx']

        # nearest indices for each axis
        x_mid = find_nearest(coords_1D[x], target)['idx']
        y_mid = find_nearest(coords_1D[y], target)['idx'] if ndim > 1 else None
        z_mid = find_nearest(coords_1D[z], target)['idx'] if ndim > 2 else None

    # --- Define slicing maps ---
    if ndim > 2:
        slice_map_1D = {
            x : (slice(None), y_mid, z_mid),
            y: (x_mid, slice(None), z_mid),
            z: (x_mid, y_mid, slice(None)),
        }

        slice_map_2D = {
            x: (x_mid, slice(None), slice(None)),
            y: (slice(None), y_mid, slice(None)),
            z: (slice(None), slice(None), z_mid),
        }

    else:
        slice_map_1D = {
            x: (slice(None), y_mid),
            y: (x_mid, slice(None)),
        }
        slice_map_2D = None

    slice_1D = slice_map_1D[sel_coord]
    slice_2D = slice_map_2D[sel_coord] if ndim > 2 else None

    coord_sliced = None
    if use_find_nearest:
        if arr_type in ('nc', 'cc'):
            coord_sliced = sdata.fluid_data(sel_coord)[sel_coord][slice_1D][idx]
        else:
            coord_sliced = sdata.fluid_data(sel_coord)[sel_coord][idx]

    return {
        "slice_1D": slice_1D,
        "slice_2D": slice_2D,
        "coord_sliced": coord_sliced,
    }
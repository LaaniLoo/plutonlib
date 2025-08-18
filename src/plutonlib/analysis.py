import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl
import plutonlib.plot as pp
from plutonlib.colours import pcolours 

import numpy as np
import scipy as sp
from scipy import stats
from scipy import constants
import matplotlib.pyplot as plt
from collections import defaultdict 
from IPython.display import display, Latex
import inspect


def find_nearest(array, value):
    """Find closes value in array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return {"idx":idx, "value": array[idx]}

def calc_var_prof(sdata,sel_coord,**kwargs):
    """
    automatically calculates the required array slice for an array of >=2 dimensions
    """
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = pl.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)
    
    vars_last = sdata.get_vars(sdata.d_last)
    ndim = vars_last["rho"].ndim #NOTE using rho to find ndim as it is often multi-dimensional 

    #TODO add 3d case if statement
    # used to slice at custom index or specified value
    if 'value' in kwargs:
        idx = find_nearest(sdata.get_coords()[sel_coord],kwargs['value'])['idx']

    elif 'idx' in kwargs:
        idx = kwargs['idx']

    else:
        # print('Neither idx or value kwargs were given: slicing at idx = 0')
        idx = 0

    if ndim >2:
        try:
            if arr_type == 'nc': #best method for 3D arrays
                x_mid = vars_last["x1"].shape[0] // 2
                y_mid = vars_last["x2"].shape[1] // 2
                z_mid = vars_last["x3"].shape[2] // 2
    
            else: #NOTE not sure what this method is?
                x_mid = len(vars_last["x1"])//2 
                y_mid = len(vars_last["x2"])//2 
                z_mid = len(vars_last["x3"])//2 
        except KeyError:
            raise ValueError("all coord data was not loaded, make sure profile_choice = 'all'")
        
        slice_map_1D = { #slices in shape of coord
        "x1": (slice(None), y_mid, z_mid),
        "x2": (x_mid, slice(None), z_mid),
        "x3": (x_mid, y_mid, slice(None))
        }

        slice_map_2D = { #slices in shape of coord
        "x1": (x_mid, slice(None), slice(None)),
        "x2": (slice(None), y_mid, slice(None)),
        "x3": (slice(None), slice(None), z_mid)
        } 

    else:
        slice_map_1D = { #slices in shape of coord
        "x1": (slice(None), idx),
        "x2": (idx, slice(None)),
        }

    slice_1D = slice_map_1D[sel_coord]
    slice_2D = slice_map_2D[sel_coord] if ndim >2 else None

    coord_sliced = sdata.get_coords()[sel_coord][idx] if ndim <=2 else sdata.get_coords()[sel_coord][slice_1D][idx]

    returns = {
        "slice_1D": slice_1D,"slice_2D": slice_2D,"coord_sliced": coord_sliced
    }
    return returns

#---Peak Finding---#
def peak_findr(sel_coord,sel_var,sdata,**kwargs):
    """
    Calculates the max values of an array and their location, e.g. use to find max values of x2 for vx2 to find jet radius
    """

    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = pl.SimulationData(
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

        coord_array = var[sel_coord]   

        peak_info.append(f"{d_file} Radius: {coord_array[max_loc][0]:.2e} m, {sel_var}: {var_sliced[max_loc][0]:.2e}")
        peak_var.append(var_sliced[max_loc][0])
        radius.append(coord_array[max_loc][0])

    return {"peak_info": peak_info,"radius": radius, "peak_var": peak_var,"locs": locs } 

def graph_peaks(sel_coord,sel_var,sdata,**kwargs): #TODO Put in peak findr 
    """Follows a similar process to peak_findr() except it uses scipy signal peak finding, good for visual representation"""
    
    loaded_outputs = kwargs.get('load_outputs', sdata.load_outputs)
    arr_type = kwargs.get('arr_type', sdata.arr_type)
    ini_file = kwargs.get('ini_file',sdata.ini_file)

    sdata = pl.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    coord_units = (sdata.get_var_info(sel_coord)["si"]).to_string('latex')
    var_units = (sdata.get_var_info(sel_var)["si"]).to_string('latex')

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

    sdata = pl.SimulationData(
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

    var_profile = calc_var_prof(sdata,sel_coord,**kwargs)["slice_1D"]
    var_sliced = var[var_profile]

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

    sdata = pl.SimulationData(
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
    peak_coords = peak_data["peak_coords"]
    peak_vars = peak_data["peak_vars"]

    is_log = sel_var in ('rho','prs')
    base_plot_data = np.log10(var_sliced) if is_log else var_sliced
    peak_plot_data = np.log10(peak_vars) if is_log else peak_vars

    xlab = f"{sdata.get_var_info(sel_coord)['coord_name']} [{sdata.get_var_info(sel_coord)['si']}]"
    ylab = f"log10({sdata.get_var_info(sel_var)['var_name']}) [{sdata.get_var_info(sel_var)['si']}]" if is_log else f"{sdata.get_var_info(sel_var)['var_name']} [{sdata.get_var_info(sel_var)['si']}]"
    label = f"Peak {ylab}"
    title = f"{sdata.sim_type} Peak {ylab} Across {xlab}"


    f,a = plt.subplots(figsize = (7,7))
    a.plot(vars_last[sel_coord],base_plot_data) # base plot
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

    sdata = pl.SimulationData(
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

    xlab = f"{sdata.get_var_info(sel_coord)['coord_name']} [{sdata.get_var_info(sel_coord)['si']}]"
    ylab = f"log10({sdata.get_var_info(sel_var)['var_name']}) [{sdata.get_var_info(sel_var)['si']}]" if is_log else f"{sdata.get_var_info(sel_var)['var_name']} [{sdata.get_var_info(sel_var)['si']}]"
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

    sdata = pl.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)


    var_info = sdata.get_var_info(sel_coord)

    xlab = f"sim_time [{sdata.get_var_info('sim_time')['cgs']}]"
    ylab = f"{var_info['coord_name']}-Radius [{var_info['si']}]"
    title = f"{sdata.sim_type} {ylab} across {xlab}"

    t_yr = sdata.get_vars(sdata.d_last)["sim_time"]

    #Legend assignment based on sim_time
    if sdata.sim_type == "Jet":
        longest_array = get_jet_length_dim(sdata) # used to avoid confusion btwn jet length and width
        measurement = "length" if sel_coord == longest_array else "width"
        legend_base = f"{sdata.sim_type} {var_info['coord_name']}-Radius ({measurement})" #names legend based on width or length

    elif sdata.sim_type == "Stellar_Wind":
        legend_base = f"{sdata.sim_type} {var_info['coord_name']}-Radius"
    
    if type == "def": #default type plot
        f,a = plt.subplots()
        a.plot(t_yr, r, color = "darkorchid") # base plot
        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
        a.set_title(title)

        a.legend([legend_base])
        a.text(0.05, 0.8,f"R = {r[-1]:.2e} m", transform=a.transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 

        a.plot(t_yr,r,"x", label = sdata.d_files)
        for i, d_file in enumerate(sdata.d_files):
            a.annotate(d_file.strip("data_"), (t_yr[i], r[i]), textcoords="offset points", xytext=(1,1), ha='right')

        pdata = pp.PlotData()
        pdata.fig = plt.gcf()  
        pp.plot_save(sdata,pdata,custom=1,**kwargs)
    
        return None
    
    elif type == "log": #log-log type plot
        d_files = sdata.d_files[1:]
        t_yr = np.log10(t_yr[1:])
        r = np.log10(r[1:])

        slope, intercept, r_value, p_value, std_err = stats.linregress(t_yr, r)
        eqn = f'$R_{var_info["coord_name"]} \\propto t^{{{slope:.2f}}} \\pm {std_err:.2f} \\; [m]$'
        display(Latex(eqn))

        f,a = plt.subplots()
        a.plot(t_yr, r, color = "orange") # base plot

        # r_ideal = (t_yr ** 0.6)
        # a.plot(t_yr, r_ideal, color="hotpink")

        a.set_xlabel(xlab)
        a.set_ylabel(ylab)
        a.set_title(title)

        a.legend([legend_base,r'Ideal: $t^{0.6}$'])
        a.text(0.05, 0.8, eqn, transform=a.transAxes, fontsize=11,bbox=dict(facecolor='white', alpha=0.8)) 
        # a.text(slope)
        a.plot(t_yr,r,"x", label = d_files)
        for i, d_file in enumerate(d_files):
            a.annotate(d_file.strip("data_"), (t_yr[i], r[i]), textcoords="offset points", xytext=(1,1), ha='right')

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

    sdata = pl.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)


    r = []


    if sdata.sim_type == "Jet":
        sel_var = "rho" #graph peaks doesn't care which var is used, a peak is a peak?
        peak_data = graph_peaks(sel_coord,sel_var,sdata) 
        var_peak_idx = peak_data["var_peak_idx"]


        for d_file in sdata.d_files:
            coord = sdata.get_vars(d_file)[sel_coord]

            if np.any(var_peak_idx[d_file]):
                r.append(coord[var_peak_idx[d_file]][-1])
            else:
                r.append(0)

    elif sdata.sim_type == "Stellar_Wind":
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

    sdata = pl.SimulationData(
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

    sdata = pl.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)

    rt_sim, rt_calc = [], []
    rho_0 = 1 * pc.value_norm_conv("rho",sdata.d_files,self = 1)["si"]

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


    r_0 = 1 * pc.value_norm_conv("x1",sdata.d_files,self = 1)["si"]
    v_wind = 1 * pc.value_norm_conv("vx1",sdata.d_files,self = 1)["si"]
    rho_norm = 1 * pc.value_norm_conv("rho",sdata.d_files,self = 1)["si"]

    v_r = []

    all_vars = sdata.get_all_vars()

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

    sdata = pl.SimulationData(
        sim_type=sdata.sim_type,
        run_name=sdata.run_name,
        profile_choice="all",
        subdir_name = sdata.subdir_name,
        load_outputs=loaded_outputs,
        arr_type = arr_type,
        ini_file = ini_file)


    r_0 = 1 * pc.value_norm_conv("x1",sdata.d_files,self = 1)["si"]
    v_wind = 1 * pc.value_norm_conv("vx1",sdata.d_files,self = 1)["si"]
    rho_norm = 1 * pc.value_norm_conv("rho",sdata.d_files,self = 1)["si"]

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

def EOS(rho=None,prs=None,T=None,mu = 0.60364,prnt = 1):
    """
    Simple Equation of state calculator to get Temp for a given density and pressure etc...
    """
    m_H = constants.m_p
    kb = constants.k
    
    if not T:
        T = (prs*mu*m_H)/(rho*kb)
        T_prnt = f"Temperature = {T:.2e} K"
        return T if not prnt else T_prnt
    
    if not prs:
        prs = (kb*rho*T)/(mu*m_H)
        prs_prnt = f"Pressure = {prs:.2e} Pa"
        return prs if not prnt else prs_prnt
    
    if not rho:
        rho = (prs*mu*m_H)/(T*kb)
        rho_prnt = f"Density = {rho:.2e} kgm^-3"
        return rho if not prnt else rho_prnt


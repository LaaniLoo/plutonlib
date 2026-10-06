import plutonlib.utils as pu
# import plutonlib.config as pc
# import plutonlib.read_write as prw
# import plutonlib.simulations as ps

import numpy as np
from scipy import constants

from astropy import units as u
import astropy.constants as astro_const

from scipy.spatial.transform import Rotation

def find_nearest(array, value):
    """Find closes value in array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return {"idx":idx, "value": array[idx]}

#---Array profile slices---#
def get_grid_idx(value,sdata,coord):
    """
    Method for finding array index of a specific xyz value without loading full PLUTO grid arrays,
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
    value_2D (float): 
        Value to slice at for 2D slice, e.g x1 = 20kpc, defaults to value_2D = 0 (midpoint)
    value_1D (dict): 
        Used to make a slice at value for different coord to sel_coord for 1D slice, 
        e.g. slice at x1 = 20kpc and x2 = 0kpc -> value_1D = {"x1":20,"x2":0} 

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

#---Equations---#
def jet_kinetic_power(radius,rho,vel):
    eqn = 0.5*4*np.pi*(radius**2)*rho*(vel**3)
    return eqn.si

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

def calc_sound_speed(rho_0,T):
   prs_0 = EOS(rho =rho_0,T = T).value
   nonrel_gamma = 5/3
   unit = u.m / u.s
   return (np.sqrt((nonrel_gamma * prs_0) / (rho_0)))*unit

def calc_inlet_speed(rho_0,T,wind_vxx):
    """
    Gets speed in kpc/Myr for a jet with moving injection region

    Args:
        rho_0 (float): environment density in kg/m^3
        T (float): environment temperature in K
        wind_vxx (list): wind speed as multiple of environment sound speed e.g. WIND_VX1,WIND_VX2,WIND_VX3 = [2,0,0] -> 2*c_s in x

    Returns:
        inlet_vxx (list): List of inlet speeds in kpc/Myr per wind_vx component
    """

    inlet_vxx = []
    for vx in wind_vxx:
        inlet_vxx.append((vx*calc_sound_speed(rho_0=rho_0,T=T)).to(u.kpc / u.Myr))
    return inlet_vxx

def locate_injection_region(rho_0,T,wind_vxx,sim_time):
    """
    Gives location of jet injection region (in kpc) for a given timestep

    Args:
        rho_0 (float): environment density in kg/m^3
        T (float): environment temperature in K
        wind_vxx (list): wind speed as multiple of environment sound speed e.g. WIND_VX1,WIND_VX2,WIND_VX3 = [2,0,0] -> 2*c_s in x
        sim_time (float): simulation time in Myr e.g 35

    Returns:
        inj_xyz (list): list with 4 elements, x,y,z location of injection region, then timestep
    """
    sim_time = sim_time * u.Myr
    inlet_vxx = calc_inlet_speed(rho_0=rho_0,T=T,wind_vxx=wind_vxx)
    inj_xyz = []
    for vx in inlet_vxx:
        inj_xyz.append(-vx*sim_time)
    inj_xyz.append(sim_time)
    return inj_xyz

def calc_length_scales(Q,rho,v_jet,theta,T,v_wind = 0):
    """
    Calculates the length scales from Krause (2012). 
    L1: Length at which the jet density becomes compariable to the external density
    L1a: Jet recollimation, sideways ram pressure = ambient pressure
    L1b: cocoon formation
    L1c: terminal shock
    L2: buoyancy scale

    Args:
        Q (float): Jet kinetic power [W]
        rho (float): Environment density [kgm^-3]
        v_jet (float): Jet injection velocity [c]
        theta (float): Half opening angle of jet in degreees
        T (float): Environment temperature [K]
        v_wind (float): Velocity of environment cross wind [ms^-1], defaults to 0 (no wind)

    Returns:
        lscale_dict (dict): dictionary with L1x as keys (length scales in kpc)
    """
    Q, rho, v_jet, theta, T,v_wind = [v.value if isinstance(v, u.Quantity) else v for v in [Q, rho, v_jet, np.deg2rad(theta), T,v_wind]]
    
    to_kpc = astro_const.kpc.value / u.kpc
    # to_kpc = 1

    gamma = 5/3
    Omega = 2*np.pi*(1-np.cos(theta))
    c_s = calc_sound_speed(rho,T).value
    prs_env = EOS(rho=rho,T=T).value

    M_jet = v_jet/c_s
    M_wind = v_wind/c_s

    L1 = (2 * np.sqrt(2) * np.sqrt(Q / (rho * v_jet ** 3)))
    L1a = (np.sqrt(((gamma)/(4*Omega)) * M_jet**2 * np.sin(theta)**2 * L1**2)) 
    L1b = (np.sqrt((1/(4*Omega)) * L1**2)) 
    L1c = (np.sqrt((gamma/(4*Omega)) * M_jet**2 * L1**2))
    L2 = (np.sqrt(Q /(rho * c_s**3)))

    r_jet = L1a * np.tan(theta)
    # L_bend = (gamma * np.pi * (1-np.cos(theta)) * r_jet * prs_env )**-1 * M_wind**-2 * (Q/v_jet)
    L_bend = (L1b**2/r_jet) * (M_jet**2/M_wind**2)

    eta = (L1b/L1a)**2

    lscale_dict = {
            "L1": L1 / to_kpc,
            "L1a": L1a / to_kpc,
            "L1b": L1b / to_kpc,
            "L1c": L1c / to_kpc,
            "L2": L2 / to_kpc,
            "L_bend": L_bend / to_kpc,
            "r_jet": r_jet / to_kpc,
            "eta": eta,

        }

    return lscale_dict

def l2s(lorentz):
    return np.sqrt(astro_const.c**2 * (1 - (1 / lorentz) ** 2))

def s2l(speed):
    return (1 / np.sqrt(1 - (speed / astro_const.c) ** 2)).value

def calc_jet_area(theta, radius):
    theta = np.deg2rad(theta)
    return (2 * np.pi * (1 - np.cos(theta))) * (radius**2)

def calc_jet_density(Q_jet, v_jet, theta, r_jet):
    adiab_ind = 5.0 / 3.0
    area = calc_jet_area(theta, r_jet)

    return 2 * Q_jet / (v_jet**3 * area)

def calc_jet_density_rel(power, speed, half_opening_angle, radius, adiab_ind=5.0 / 3.0, prs=None, chi=None):
    area = calc_jet_area(half_opening_angle, radius)
    lorentz = s2l(speed)

    if chi is None:
        return (1 / (lorentz * (lorentz - 1) * astro_const.c**2)) * (
            (power / (speed * area)) - lorentz**2 * (adiab_ind) / (adiab_ind - 1) * prs
        )

    elif prs is None:
        return (power) / (
            (speed * area * astro_const.c**2)
            * (lorentz * (lorentz - 1) + (lorentz**2) / (chi))
        )

def calc_chi(density, pressure, adiabatic_ind):
    return ((adiabatic_ind - 1) / (adiabatic_ind)) * (density * astro_const.c**2) / (pressure)

def rjet_from_theta(theta, Q_jet, v_jet, prs_env, gamma=5/3):
    """Calculates jet radius from opening angle

    Args:
        theta (float): Jet opening angle
        Q_jet (float): Jet power
        v_jet (float): Jet velocity
        prs_env (float): Environment pressure
        gamma (float, optional): Adaibatic index. Defaults to 5/3.

    Returns:
        float: opening angle of jet 
    """
    theta = np.deg2rad(theta)
    return (gamma**-0.5) * (Q_jet/v_jet)**0.5 * prs_env**-0.5 * np.tan(theta)

def bending_params(L_bend, prs_env, theta, v_jet=None, M_wind=None, Q_jet=None,r_jet=None):
    """Determines jet parameters e.g. jet velocity, wind speed or jet power required to produce L_bend

    Args:
        L_bend (float): Bending length scale in kpc or m
        prs_env (float): Environment pressure
        theta (float): Jet opening angle
        v_jet (float, optional): Velcotiy of jet in m/s. Defaults to None.
        M_wind (float, optional): Environment wind speed as mach number. Defaults to None.
        Q_jet (float, optional): Jet power in W. Defaults to None.
        r_jet (float, optional): Jet radius in kpc or m. Defaults to None.

    Raises:
        ValueError: Error if specified all parameters without leaving 1 to be found
        ValueError: Some parameters require jet radius to be calculated -> Q_jet and v_jet

    Returns:
        Returns one of the empty "None" parameters
    """
    
    gamma = 5/3
    theta_rad = np.deg2rad(theta)

    optional_params = {'v_jet': v_jet, 'M_wind': M_wind, 'Q_jet': Q_jet}
    none_params = [k for k, v in optional_params.items() if v is None]

    if len(none_params) != 1:
        raise ValueError(f"Exactly 1 optional parameter must be None to solve for, got {len(none_params)}: {none_params}")

    if L_bend < 1e18: #convert from kpc to m
        L_bend = L_bend * astro_const.kpc.value
        r_jet = r_jet * astro_const.kpc.value if r_jet else None

    # r_jet always derived from theta once we have Q_jet and v_jet
    if Q_jet is not None and v_jet is not None:
        r_jet = rjet_from_theta(theta, Q_jet, v_jet, prs_env)

    if v_jet is None:
        print("Calculating jet velocity")
        if r_jet is None:
            raise ValueError("Need to provide r_jet to calculate jet power")

        v_jet = (gamma * np.pi * (1-np.cos(theta_rad)) * r_jet * prs_env)**-1 * M_wind**-2 * (Q_jet/L_bend)
        return v_jet / constants.c 

    if M_wind is None:
        print("Calculating wind mach")
        M_wind = np.sqrt(Q_jet / (gamma * np.pi * (1-np.cos(theta_rad)) * r_jet * prs_env * v_jet * L_bend))
        return M_wind

    if Q_jet is None:
        print("Calculating jet power")
        if r_jet is None:
            raise ValueError("Need to provide r_jet to calculate jet power")

        Q_jet = gamma * np.pi * (1-np.cos(theta_rad)) * r_jet * prs_env * M_wind**2 * v_jet * L_bend
        return Q_jet

def calc_jet_bending_angle(L1c,L_bend):
    """Uses Jet termination length and jet bending length to find aproximate bending angle between the jets

    Args:
        L1c (float): Jet termination length
        L_bend (float): Jet bending length
    
    Returns:
        angle_btwn (float): angle between both jets in degrees
    """

    angle_vert = np.rad2deg(L1c/L_bend) #using arc length
    angle_btwn = 180 - 2*angle_vert

    return angle_btwn

#---Praise Setup---#
def rotation_matrix(axis_str="xyz",angles=[0,0,0]):
    if len(axis_str) != len(angles):
        raise ValueError(f"Axis and angle dimensions are unequal (axes: {len(axis_str)}, angles: {len(angles)})")
    rot_mat = Rotation.from_euler(axis_str, angles, degrees=True).as_matrix()
    return rot_mat

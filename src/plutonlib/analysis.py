import plutonlib.utils as pu
import plutonlib.config as pc
import plutonlib.load as pl
import plutonlib.simulations as ps
import plutonlib.plot as pp
from plutonlib.colours import pcolours 

import plutokore.radio as pk_radio

import numpy as np
import scipy as sp
from scipy import stats
from scipy import constants

from astropy import units as u
from astropy import cosmology as cosmo  # Astropy cosmology
import astropy.constants as astro_const

from scipy.spatial.transform import Rotation
from scipy.interpolate import splprep,splev
from astropy.convolution import convolve, Gaussian2DKernel  # Astropy convolutions

import matplotlib.pyplot as plt

from collections import defaultdict 
from IPython.display import display, Latex
import inspect

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
def setup_obs_properties_praise(sdata,redshift,angle,plane="xz"):
    #--------------------------------------------------------#
    #            Set up the observing properties             #  
    #--------------------------------------------------------#

    pixel_size = 1 * u.arcsec
    beam_fwhm = 3 * u.arcsec  # generally this should be ~3x bigger than pixel size

    arcsec2kpc = cosmo.Planck15.kpc_proper_per_arcmin(redshift).to(u.kpc/u.arcsec)
    pixel_size_kpc = (pixel_size * arcsec2kpc).to(u.kpc)
    beam_kpc = (beam_fwhm * arcsec2kpc).to(u.kpc)

    # beam equations
    fwhm_to_sigma = 1 / (8 * np.log(2)) ** 0.5
    beam_sigma = beam_fwhm * fwhm_to_sigma
    omega_beam = 2 * np.pi * beam_sigma ** 2  # Area for a circular 2D gaussian

    # part_ind = part_outputs   

    ## integration grid cell size
    grid_spacing = pixel_size_kpc.value * 1.0 #0th redshift?

    # set a min and max of the grid
    grid_setup = sdata.grid_setup
    plane_grid_map = {
    "xy" : [grid_setup["x1-grid"]["grid_extent"],grid_setup["x2-grid"]["grid_extent"]],
    "xz" : [grid_setup["x1-grid"]["grid_extent"],grid_setup["x3-grid"]["grid_extent"]],
    "yz" : [grid_setup["x2-grid"]["grid_extent"],grid_setup["x3-grid"]["grid_extent"]],
    }
    grid_lim_x = plane_grid_map[plane][0] #e.g if xz-plane 0th element is x limits e.g. (-80,80)
    grid_lim_y = plane_grid_map[plane][1]

    # ray properties
    delta_r = 0.3         
    ray_depth_min = -200
    ray_depth_max = 200

    # specify our grid
    grid_x = np.arange(grid_lim_x[0], grid_lim_x[1], grid_spacing)
    grid_y = np.arange(grid_lim_y[0], grid_lim_y[1], grid_spacing)
    grid_mx = np.diff(grid_x) * 0.5 + grid_x[:-1]
    grid_my = np.diff(grid_y) * 0.5 + grid_y[:-1]

    # grid_mx = test.fluid_data("ccx",load_slice=(0,0,slice(None)))["ccx"]
    # grid_my = test.fluid_data("ccz",load_slice=(0,0,slice(None)))["ccz"]

    rot_mat = Rotation.from_euler("X", angle, degrees=True).as_matrix()
    gaussian_sigma = (beam_kpc.value * fwhm_to_sigma) / grid_spacing
    gaussian_kernel = Gaussian2DKernel(gaussian_sigma)  # create our gaussian convlution kernel

    returns = {
        "grid_x":grid_x,
        "grid_y":grid_y,
        "grid_mx":grid_mx,
        "grid_my":grid_my,
        "delta_r":delta_r,
        "omega_beam":omega_beam,
        "ray_depth_min":ray_depth_min,
        "ray_depth_max":ray_depth_max,
        "rot_mat":rot_mat,
        "gaussian_kernel": gaussian_kernel
    }

    return returns

def calc_surface_brightness_praise(sdata,freqs=[1.4],redshift=0.05,particle_outputs="last",angle=0,plane="xz"):
    """
    Calculates the particle emssion using PRAiSE (plutokore: pk_radio) under adiabatic, sychrotron and inverse compton losses.
    Surface brightness is then calculated by integrating emissivity with raytracing.    
    :param sdata: Description
    :param freqs: Description
    :param redshift: Description
    :param particle_outputs: Description
    :param angle: Description
    """
    pk_sim = sdata.to_plutokore() #convert SimulationData object to plutokore PlutoSimulation
    particle_outputs = [pl.get_particle_outputs(sdata.wdir)] if particle_outputs == "last" else particle_outputs
    particle_spacing = sdata.part_to_simtime(particle_outputs[0]) / particle_outputs[0]
    s=2.2   # for injection spectral index alpha=-0.55. NOTE: the PRAiSE default is also 2.2

    particle_data = sdata.load_particle_data(output=particle_outputs[-1])
    particle_times = particle_data["particle_times"]
    particle_emis = pk_radio.praise2.praise(
        sim=pk_sim,
        max_output=particle_outputs[-1],
        emit_outputs=particle_outputs, #calc emission for these outputs
        output_system="particles", #idx in grid or particles
        freqs=(freqs*u.GHz).si.value, #list of GHz freqs to calc for 
        part_data=particle_data, 
        part_times=particle_times,
        particle_spacing=particle_spacing * u.Myr, #particle outputs per Myr 
        redshift=redshift, # 0.05 redshift
        lst_index=2, #last shock time index, lowest to highest str -> 2 = only strong shocks
        losses=4, #idx to include losses: adiab, synch, inverse compton losses -> 4 = include all losses
    )

    # Remove the NaNs for each output 
    # create an empty list of particle coordinates, velocities and the nan masks which will all be different for each simulation output. 
    all_part_coords = []  
    all_vel_vec = []
    all_nan_masks = []
    sb_arr = []
    integ_emis = []
    for i in  range(0, len(particle_outputs), 1):    
        part_ind = particle_outputs[i]
        nan_mask = ~np.isnan(particle_data["id"][:, part_ind]) 
        all_nan_masks.append(nan_mask)
        
        part_coords = np.c_[
            (
                particle_data["x1"][:, part_ind][nan_mask],
                particle_data["x2"][:, part_ind][nan_mask],
                particle_data["x3"][:, part_ind][nan_mask],
            )
        ]
        all_part_coords.append(part_coords)
        
        # set up particle velocities
        vel_vec = np.c_[
            particle_data["vx1"][:, part_ind][nan_mask],
            particle_data["vx2"][:, part_ind][nan_mask],
            particle_data["vx3"][:, part_ind][nan_mask],
        ]
        all_vel_vec.append(vel_vec)

        obs_properties = setup_obs_properties_praise(sdata=sdata,redshift=redshift,angle=angle,plane=plane)
        integrated_emissivity = pk_radio.raytracing.raytrace_particles_multiple_freq(
            grid=(obs_properties["grid_mx"], obs_properties["grid_my"]),
            ray_depth_lim=(obs_properties["ray_depth_min"], obs_properties["ray_depth_max"]),
            rot_mat=obs_properties["rot_mat"],
            s= s,
            delta_r=obs_properties["delta_r"],
            coords=all_part_coords[i],
            dist_upper_bound = 2,      # <-- NOTE the praise default is 20... this will mess up your results. 2 is better
            vel_vec=all_vel_vec[i],
            obs_normal=[0, 1, 0],
            # particle_emissivities=part_emis['full'][i]['emis'][all_nan_masks[i],:,:]  #only the non-nan particles, all frequencies, and the ith snapshot
            particle_emissivities=particle_emis['full'][i]['emis'][all_nan_masks[i],:,0] #only the non-nan particles, all frequencies, and the ith snapshot 
        )
        integ_emis.append(integrated_emissivity)

        # we multiple our integrated emissivity by kpc (to account for integration), and divide by 4pi to account for solid angle
        surface_brightness = (integrated_emissivity * u.kpc / (4 * np.pi)).to(u.mJy / u.beam, equivalencies=u.beam_angular_area(obs_properties["omega_beam"]))
        sb_arr.append(surface_brightness)
    
    for freq_ind in range(len(freqs)):
        sb_arr[i][:,:,freq_ind] = np.nan_to_num(sb_arr[i][:,:,freq_ind], copy=True, nan=0.0, posinf=0.0, neginf=0.0) # get rid of NaNs (replace with zero)

    return sb_arr

#---Jet splines---#
def get_jet_splines(sdata,output,tr_cut):
    """Fits ridgepoints along the jet length by looking at a tracer slice in the jet radius 
    and weighting the x,z coordinates by the maximum tracer, stops when a window of 5 points has 
    reached a an average of some cutoff value e.g. 0.2. Will also stop if the dot product is (-) 
    (chaing directions). Splines are then fitted to these ridgepoints to resample at a higher resolution 
    (based on the grid resolution) using scipy splprep.

    Args:
        sdata (SimulationData): SimulationData object
        output (int): PLUTO file output
        tr_cut (float): tracer cuttoff value to truncate the data #NOTE not sure if required

    Returns:
        dict: 
            spline_points: x,z array e.g. x = spline_points[:,0] of fitted splines 
            spline_slice_map: x,z indecies that match spline points to grid cells
            ridgepoints: x,z array of the ridgepoints found along the jet
            
    """
    # sim_time = round(sdata.get_metadata(output).sim_time)
    sim_time = output
    part_time = round(sdata.simtime_to_part(sim_time))

    particle_data = sdata.load_particle_data(output=(part_time,),tr_cut=tr_cut)
    
    #was int but that causes rounding error
    def calc_ridgepoints(jet_side,particle_data,sim_time):
        window = [] #windowed averages
        window_size = 5
        step_size = 1 #kpc
        tr_peak = 0
        tr_min = 1 

        inj_array = sdata.get_injection_region(output=sim_time) #gets the location of the injection region -> converts particle time to simtime
        xz_start = [inj_array[0].value, inj_array[2].value] # start at injection region coords
        z_current = xz_start[1]
        x_current = xz_start[0]
        r_jet = sdata.jet.radius.value #jet analytic radius
        ridgepoints = [xz_start.copy()] #initial ridgepoint at inj region

        while True:
            if len(ridgepoints) >=2:
                dx = ridgepoints[-1][0] - ridgepoints[-2][0] #difference btwn last two x ridgepoints
                dz = ridgepoints[-1][1] - ridgepoints[-2][1] #difference btwn last two z ridgepoints
                norm = np.sqrt(dx**2 + dz**2)
                x_current = ridgepoints[-1][0] + (dx/norm)*step_size #update xz position vector with some step size
                z_current = ridgepoints[-1][1] + (dz/norm)*step_size

            else:
                if jet_side == "top":
                    z_current += step_size #initial move, move down 1 kpc
                if jet_side == "bot":
                    z_current -= step_size #initial move, move down 1 kpc

            tr_plane = (np.abs(particle_data["x3"] - z_current) < r_jet) & \
                    (np.abs(particle_data["x1"] - x_current) < r_jet) #tracers at the ridgepoint with plane width of jet radius
            
            if tr_plane.sum() == 0: #if no tracers
                continue

            tr_val = particle_data['tracer'][tr_plane][np.argmax(particle_data['tracer'][tr_plane])] #current max tracer in plane
            tr_peak = max(tr_peak, tr_val) #maximum recorded tracer
            tr_min = min(tr_min,tr_val)

            window.append(tr_min)
            if len(window) > window_size: #shift window values over to fit new value
                window.pop(0)

            if len(window) == window_size and np.mean(window) < 0.2: #NOTE once window is full and if min tracer drops below some value, stop
                print(f"Jet head detected @ z= {z_current:.3f} kpc")
                break
            
            x_new = np.average(particle_data["x1"][tr_plane], weights=particle_data['tracer'][tr_plane]) #average all xz points (midpoint) in plane weighted by value of tracer
            z_new = np.average(particle_data["x3"][tr_plane], weights=particle_data['tracer'][tr_plane])

            if len(ridgepoints) > 5:
                # Vector of the current step
                prev = np.array(ridgepoints[-1])
                current_step = np.array([x_new, z_new]) - prev

                # Mean direction of recent N steps
                recent = np.array(ridgepoints[-5:])
                recent_dir = recent[-1] - recent[0]  # overall direction over last 5 points

                # Dot product — negative means we've reversed direction
                if np.dot(current_step, recent_dir) < 0:
                    print(f"Direction reversal detected @ z = {z_current:.3f} kpc")
                    break
            ridgepoints.append([x_new, z_new])
        ridgepoints = np.array(ridgepoints)
        return ridgepoints

    ridgepoints_top = calc_ridgepoints(jet_side = "top",particle_data=particle_data,sim_time=sim_time)
    ridgepoints_bot = calc_ridgepoints(jet_side = "bot",particle_data=particle_data,sim_time=sim_time)
    top_length = np.sum(np.sqrt(np.diff(ridgepoints_top[:,0])**2 + np.diff(ridgepoints_top[:,1])**2))
    bot_length = np.sum(np.sqrt(np.diff(ridgepoints_bot[:,0])**2 + np.diff(ridgepoints_bot[:,1])**2))


    ridgepoints_all = np.vstack((ridgepoints_bot[::-1],ridgepoints_top[1:]))

    arc_length = np.sum(np.sqrt(np.diff(ridgepoints_all[:,0])**2 + np.diff(ridgepoints_all[:,1])**2)) #length of jet spline
    max_resolution = min([sdata.grid_setup['x1-grid']['dx'],sdata.grid_setup['x2-grid']['dx'],sdata.grid_setup['x3-grid']['dx']])
    n_points = int(arc_length / max_resolution) #determines the resampling resolution

    tck, u = splprep([ridgepoints_all[:,0],ridgepoints_all[:,1]],s=0)
    u2 = np.linspace(u[0],u[-1],n_points) #use the spline array to generate a higher resolution array
    spline_points = splev(u2,tck)
    spline_points = np.column_stack(spline_points) #resampled ridgepoints with higher resolution

    x1_data = sdata.load_fluid_data(["ccx"], output=sim_time, load_slice=sdata.quick_slice_1D("yz"))["ccx"]
    x3_data = sdata.load_fluid_data(["ccz"], output=sim_time, load_slice=sdata.quick_slice_1D("xy"))["ccz"]
    # spline_slice_map = (np.searchsorted(x1_data, np.sort(spline_points[:, 0])), np.searchsorted(x3_data, np.sort(spline_points[:, 1])))   # direct numpy index tuple

    x_indices = np.searchsorted(x1_data, spline_points[:, 0])
    z_indices = np.searchsorted(x3_data, spline_points[:, 1])

    # Clip to valid array bounds for safety
    x_indices = np.clip(x_indices, 0, len(x1_data) - 1)
    z_indices = np.clip(z_indices, 0, len(x3_data) - 1)

    spline_slice_map = (x_indices, z_indices)


    returns = {
        "spline_points": spline_points,
        "spline_slice_map": spline_slice_map,
        "ridgepoints": ridgepoints_all,
        "jet_length": [top_length,bot_length],
    }
    return returns

#---Analysis Plotting---#
def plot_surface_brightness(sdata,freqs=[1.4],redshift=0.05,particle_outputs="last",angle=0,plane="xz",**kwargs):
    # if pdata is None:
    # loop over all outputs and all frequencies
    particle_outputs = [pl.get_particle_outputs(sdata.wdir)] if particle_outputs == "last" else particle_outputs
    pdata = pp.PlotData(var_choice=freqs,plane=plane,load_outputs = particle_outputs,**kwargs)
    pdata.plot_idx = 0
    pdata.axes, pdata.fig = pp.subplot_base(sdata=sdata,pdata=pdata,load_outputs=particle_outputs)
    extras = pp.plot_extras(sdata,pdata)

    obs_properties = setup_obs_properties_praise(sdata=sdata,redshift=redshift,angle=angle,plane=plane)
    sb_arr = calc_surface_brightness_praise(
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

def plot_ram_pressure(sdata,load_outputs= None,**kwargs):

    pdata = pp.PlotData(var_choice=["vx3"],**kwargs)
    pdata.arr_type = "cc"
    pdata.var_name = "vx3"

    load_outputs = sdata.load_outputs if load_outputs is None else load_outputs
    axes, fig = pp.subplot_base(sdata,pdata,load_outputs=pdata.load_outputs,**kwargs)
    plot_idx = 0  # Keep track of which subplot index we are using

    for output in load_outputs:
        pdata.output = output
        extras_data = pp.plot_extras(sdata,pdata)
        xy_labels = extras_data["xy_labels"]
        var_label = sdata.get_var_info("prs").var_name
        var_units = sdata.get_var_info("prs").usr_uv
        coord_units = sdata.get_var_info("z").usr_uv

        inj_loc = sdata.get_injection_region(output=output)[0].value #location of the injection region for moving injection regions
        value_dict = {"x1":inj_loc} #dict of coord then location of inj region
        slices = calc_var_prof(sdata,"x3",value_1D=value_dict)
        fluid_data_1d = sdata.load_fluid_data(var_choice=["rho","vx3","ccz"],output=output,load_slice=slices['slice_1D'])

        ram_prs = fluid_data_1d['rho']*np.pow(fluid_data_1d["vx3"],2)
        inv_z = 1/fluid_data_1d["ccz"]**2
        scale_factor = ram_prs[np.argmax(ram_prs)] / inv_z[np.argmax(ram_prs)]

        ax = axes[plot_idx]
        ax.set_yscale("log")
        ax.plot(fluid_data_1d["ccz"],ram_prs)
        ax.plot(fluid_data_1d["ccz"],inv_z * scale_factor,color = "g",linestyle = ":")
        ax.plot(fluid_data_1d["ccz"],inv_z * -1*scale_factor,color = "g",linestyle = ":")
        
        ax.set_xlabel(f"{xy_labels['ccz']}")
        ax.set_ylabel(f"{var_label} [{var_units}]")

        legend_coord, = value_dict.keys() #get the first key from value_dict, NOTE: works assuming that only 1 slice arg is used 
        value = f"{(value_dict.get(legend_coord)):.2f}" #scaling factor makes it easier to read
        legend_str = f"$P_{{\\rm{{ram}}}}$ @ {sdata.get_var_info(legend_coord).coord_name} = {value} {coord_units}"
        fit_str = '$1/z^{{2}}$'

        ax.legend([legend_str,fit_str])
        ax.set_title(f"Plot of $P_{{\\rm{{ram}}}}$ along jet axis ($z$) with $1/z^{{2}}$ fit")
        pp.plot_axlim(ax,kwargs)
        plot_idx += 1
    pp.plot_save(sdata,pdata,**kwargs)

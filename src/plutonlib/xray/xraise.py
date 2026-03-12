from numba import njit, prange, jit
import numpy as np
import astropy.units as u
from scipy.special import gamma, zeta
from scipy.signal import savgol_filter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import time

__all__ = [
    "ffill",
    "bfill",
    "get_adiabatic_ind",
    "get_ap",
    "get_a3",
    "get_a4",
    "get_ub",
    "get_a2",
    "get_gamma_larmor",
    "get_gamma",
    "get_kappa_nonumba",
    "get_j_corr",
    "get_K",
    "get_emissivity",
    "get_emissivity_lossless",
]

# --------------- #
# Helper functions
# --------------- #

@njit(parallel=True, fastmath=False)
def ffill(arr):
    for row_idx in prange(arr.shape[0]):
        for col_idx in range(1, arr.shape[1]):
            if np.isnan(arr[row_idx, col_idx]):
                arr[row_idx, col_idx] = arr[row_idx, col_idx - 1]


@njit(parallel=True, fastmath=False)
def bfill(arr):
    for row_idx in prange(arr.shape[0]):
        for col_idx in range(arr.shape[1] - 2, -1, -1):
            if np.isnan(arr[row_idx, col_idx]):  # or arr[row_idx, col_idx] == 0.0:
                arr[row_idx, col_idx] = arr[row_idx, col_idx + 1]


@njit("f8[:, :, :](f8[:, :, :], f8[:, :, :])")
def get_adiabatic_ind_arr(p, rho):
    th = p / (rho * (2.99792458e8 ** 2))
    h = (5.0 / 2.0) * th + np.sqrt((9.0 / 4.0) * np.power(th, 2) + 1)
    return (h - 1) / (h - 1 - th)


@njit("f8(f8, f8)")
def get_adiabatic_ind(p, rho):
    th = p / (rho * (2.99792458e8 ** 2))
    h = (5.0 / 2.0) * th + np.sqrt((9.0 / 4.0) * np.power(th, 2) + 1)
    return (h - 1) / (h - 1 - th)


@njit("f8(f8, f8, f8, f8)")
def get_ap(pn, pn1, tn, tn1):
    return (np.log10(pn) - np.log10(pn1)) / (np.log10(tn) - np.log10(tn1))


@njit("f8(f8, f8)")
def get_a3(ap, gc):
    return 1 + ap * (1 + 1 / (3 * gc))

@njit("f8(f8, f8)")
def get_a4(ap, gc):
    return 1 + ap / (3 * gc)

@njit("f8(f8, f8, f8)")
def get_ub(p, gc, eta):
    return (eta * p) / ((gc - 1) * (eta + 1))

@njit("f8[:, :, :](f8[:, :, :], f8[:, :, :], f8)")
def get_ub_arr(p, gc, eta):
    return (eta * p) / ((gc - 1) * (eta + 1))


@njit("f8(f8, f8, f8, f8, f8, f8, f8, i4)")
def get_a2(p, tn, tn1, ap, gc, eta, z, losses):
    # c1 = 3.2479641514432146e-07 # in SI
    c1 = 7.719133e-21 # in SI (Myr)
    energy_density_in_myr = 9.9588212e26 # conversion factor to time in terms of Myr
    # losses: 0 -> lossless
    #         1 -> adiabatic only
    #         2 -> adiabatic + synch
    #         3 -> adiabatic + IC
    #         4 -> full

    uc = 0.0
    ub = 0.0

    # synch losses enabled:
    if losses == 2 or losses == 4:
        # ub = get_ub(p, gc, eta) # in SI
        ub = get_ub(p, gc, eta) * energy_density_in_myr # in SI (Myr)
    # IC losses enabled:
    if losses >= 3:
        # uc = 4.005e-14 * ((1 + z) ** 4) # in SI
        uc = 4.005e-14 * ((1 + z) ** 4) * energy_density_in_myr # in SI (Myr)

    a3 = get_a3(ap, gc)
    a4 = get_a4(ap, gc)

    return c1 * (
        (ub / a3) * np.power(tn, -ap) * (np.power(tn1, a3) - np.power(tn, a3))
        + (uc / a4) * (np.power(tn1, a4) - np.power(tn, a4))
    )
    

@njit("f8(f8, f8, f8, f8)")
def get_gamma_larmor(p, gc, eta, f):
    mu0 = 1.25663706212e-06
    me = 9.1093837015e-31
    e = 1.602176634e-19
    ub = get_ub(p, gc, eta)
    B = np.sqrt(2 * ub * mu0)
    #     return np.sqrt((2*np.pi*f*me)/(3*e*B))
    return np.sqrt((2 * np.pi * f * me) / (3 * e * B))


@njit("f8(f8, f8, f8, f8, f8, f8, i4)")
def get_gamma(gn1, tn, tn1, ap, a2, gc, losses):
    # if we only have adiabatic losses (losses==1), we modify the recurrent lorentz equation to 
    # make it slightly more numerically stable
    if losses == 1:
        return (gn1 * np.power(tn/tn1, ap / (3 * gc)))
    else:
        return (gn1 * np.power(tn, ap / (3 * gc))) / (
            np.power(tn1, ap / (3 * gc)) - (a2 * gn1)
        )

# don't njit this because of gamma func
def get_kappa_nonumba(s):
    return (
        gamma(0.25 * s + (19.0 / 12.0))
        * gamma(0.25 * s - (1.0 / 12))
        * gamma(0.25 * s + (5.0 / 4.0))
        / gamma(0.25 * s + (7.0 / 4.0))
    )

# don't njit this because of gamma and zeta func
def get_j_corr(s): 
    return ( np.power(np.pi, 4) / 
            (15*gamma( (s+5) / 2 ) * zeta( (s + 5) / 2) )
           )


# @njit("f8[:, :, :, :](f8, f8, f8[:, :, :, :], f8, f8)")
# def get_K(kp, s, gc, gmin, gmax):
#     c = 299792458.0
#     mu0 = 1.25663706212e-06
#     me = 9.1093837015e-31
#     e = 1.602176634e-19

#     return (
#         (kp)
#         / (np.power(me, (s + 3) / 2) * c * (s + 1))
#         * (np.power(((e ** 2) * mu0) / (2 * (gc - 1)), (s + 5) / 4))
#         * (np.power(3 / np.pi, s / 2))
#         / (
#             (np.power(gmin, 2 - s) - np.power(gmax, 2 - s)) / (s - 2)
#             - (np.power(gmin, 1 - s) - np.power(gmax, 1 - s)) / (s - 1)
#         )
#     )


def to_counts_per_ks(F_nu, A_eff): 
    """
    Compute photon count rate (counts per second)
    from monochromatic flux density F_nu (W/m^2/Hz)
    and telescope collecting area in m^2.
    """
    h = 6.62607015e-34  # Planck's constant (J*s)

    return(
        ( (F_nu /h) * A_eff) * 1000
    )


def flux_to_counts_per_ks(F_nu, A_eff, freq):
    photon_energy = get_photon_energies(freq)
    
    photons_per_second = (F_nu * freq * A_eff / photon_energy)  # units of 1/s

    return (
        photons_per_second * 1000         # units of 1/ks
    )
    
@njit("f8[:, :, :](f8, f8, f8[:, :, :], f8, f8)")
def get_K(kp, s, gc, gmin, gmax):
    c = 299792458.0
    mu0 = 1.25663706212e-06
    me = 9.1093837015e-31
    e = 1.602176634e-19

    constants = (
        (kp) / 
        (np.power(me, (s + 3) / 2) * c * (s + 1)) 
        * (np.power((((e ** 2) * mu0) / 2), (s + 5) / 4) )
        * (np.power(3 / np.pi, s / 2)) 
        / ( ((np.power(gmin, 2 - s) - np.power(gmax, 2 - s)) / (s - 2)) - ((np.power(gmin, 1 - s) - np.power(gmax, 1 - s)) / (s - 1)) )
    )
    arr_bit = np.power(np.subtract(gc, 1), (s + 5) / 4)

    return (np.divide(constants, arr_bit))
    

def get_emissivity(p, pinj, g, ginj, gc, eta, f, s, trc, K):
    pquot = pinj / p
    gquot = ginj / g

    return (
        (K / (4 * np.pi))
        * (np.power(f, (1 - s) / 2))
        * (np.power(eta, (s + 1) / 4) / np.power(eta + 1, (s + 5) / 4))
        * (np.power(p, (s + 5) / 4) * trc)
        * (np.power(pquot, 1 - 4 / (3 * gc)))
        * (np.power(gquot, 2 - s))
    )
    

def get_doppler(grid_vel_vec, rot_matrix): 
    """
    The grid velocity vector (n, n, n, 3) 
    """

    gamma = np.divide(1,  
                      np.sqrt(1 - np.einsum("...i,...i->...", grid_vel_vec, grid_vel_vec))
                     )

    obs_normal = [0, 1, 0]

    rot_obs_normal = rot_matrix.dot(obs_normal)

    return (1 / (
        np.multiply(gamma, (1 - np.dot(grid_vel_vec, rot_obs_normal)))
    ))
    


def get_emissivity_lossless(p, eta, f, s, trc, K):
    return (
        (K / (4 * np.pi))
        * (np.power(f, (1 - s) / 2))
        * (np.power(eta, (s + 1) / 4) / np.power(eta + 1, (s + 5) / 4))
        * (np.power(p, (s + 5) / 4) * trc)
    )


### Inverse-Compton-specific emission
def get_ic_emissivity(p, pinj, g, ginj, gc, eta, f, s, trc, K, z, j_corr): 
    e = 1.602176634e-19
    uc0 = 0.25e6*e
    
    pquot = pinj / p
    gquot = ginj / g

    f_sync = get_f_sync(f, p, z, gc, eta)

    return (
        (K / (4 * np.pi))
        * (np.power(f_sync, (1 - s) / 2) / j_corr)
        * (f_sync / f)
        * (np.power(eta, (s - 3) / 4) / np.power(eta +1, (s + 1) / 4))
        * (np.power(p, (s + 1) / 4) * trc) ## <-- Do we need to weight by tracer???
        * (gc - 1)
        * (uc0*np.power(z+1, 4))
        * (np.power(pquot, 1-4 / (3 * gc)))
        * (np.power(gquot, 2-s))
    )


def get_f_sync(f, p, z, gc, eta):
    mu0 = 1.25663706212e-06
    me = 9.1093837015e-31
    e = 1.602176634e-19

    ub = get_ub_arr(p, gc, eta)

    f_cmb = 5.879e10 * (2.73)*(1+z) # from page 3 in RAiSE X (Turner + Shabala 2020)

    return ( (3*e*f*np.sqrt(2*mu0*ub)) / 
            (2 * np.pi * me * f_cmb)
           )

def get_ic_emissivity_lossless(p_arrays, gc, eta, frequencies, s, trc_arrays, K, redshifts, j_corr): 
    e = 1.602176634e-19
    uc0 = 0.25e6*e
    
    # ensure redshift is an array
    redshifts = np.atleast_1d(redshifts)
    #     frequencies = frequencies * (1 + z)
    # correct observing frequencies for redshift (emitted is higher)
    corrected_frequencies = np.empty((frequencies.shape[0], redshifts.shape[0]))
    for i in range(corrected_frequencies.shape[0]):
        for j in range(corrected_frequencies.shape[1]):
            corrected_frequencies[i, j] = frequencies[i] * (1 + redshifts[j])

    # this has shape (frequency, redshift)

    # set up the ic emission array which has shape (grid_ouput, frequencies, redshifts, data_z, data_y, data_x)
    ic_emission = np.zeros((p_arrays.shape[0], p_arrays.shape[1], p_arrays.shape[2], corrected_frequencies.shape[0], corrected_frequencies.shape[1]))
    #print(ic_emission.shape)

    # calulate the ic emission 
    for i in range(corrected_frequencies.shape[0]): # loop through the frequencies
        for j in range(corrected_frequencies.shape[1]):  # loop through the redshifts
            f_sync = get_f_sync(corrected_frequencies[i, j],  
                                p_arrays, 
                                redshifts[j], 
                                gc, 
                                eta) # get the synchrotron frequency
            ic_emis = (
                (np.divide(K, (4 * np.pi)))
                * (np.power(f_sync, (1 - s) / 2) / j_corr)
                * (np.divide(f_sync, corrected_frequencies[i, j]))
                * (np.power(eta, (s - 3) / 4) / np.power(eta +1, (s + 1) / 4))
                * (np.power(p_arrays, (s + 1) / 4) * trc_arrays) ## <-- Do we need to weight by tracer???
                * (np.subtract(gc, 1))
                * (uc0*np.power(redshifts[j]+1, 4))
            ) # this should have shape (dataz, datay, datax)
            
            ic_emission[:, :, :, i, j] = ic_emis
                
    return ic_emission


def get_sync_emissivity_lossless(p_arrays, eta, frequencies, s, trc_arrays, K, redshifts): 
    
    # ensure redshift is an array
    redshifts = np.atleast_1d(redshifts)
    #     frequencies = frequencies * (1 + z)
    # correct observing frequencies for redshift (emitted is higher)
    corrected_frequencies = np.empty((frequencies.shape[0], redshifts.shape[0]))
    for i in range(corrected_frequencies.shape[0]):
        for j in range(corrected_frequencies.shape[1]):
            corrected_frequencies[i, j] = frequencies[i] * (1 + redshifts[j])


    print(corrected_frequencies)
    # set up the ic emission array which has shape (grid_ouput, frequencies, redshifts, data_z, data_y, data_x)
    sync_emission = np.zeros((p_arrays.shape[0], p_arrays.shape[1], p_arrays.shape[2], corrected_frequencies.shape[0], corrected_frequencies.shape[1]))
    #print(sync_emission.shape)

    # calulate the ic emission 
    for i in range(corrected_frequencies.shape[0]): # loop through the frequencies
        for j in range(corrected_frequencies.shape[1]):  # loop through the redshifts
            sync_emis = (
                (np.divide(K, (4 * np.pi)))
                * (np.power(corrected_frequencies[i, j], (1 - s) / 2))
                * (np.power(eta, (s + 1) / 4) / np.power(eta + 1, (s + 5) / 4))
                * (np.power(p_arrays, (s + 5) / 4) * trc_arrays)
            ) # this should have shape (dataz, datay, datax)

            sync_emission[:, :, :, i, j] = sync_emis
            
    return sync_emission
            

            
def get_emissivity_lossless(p, eta, f, s, trc, K):
    return (
        (K / (4 * np.pi))
        * (np.power(f, (1 - s) / 2))
        * (np.power(eta, (s + 1) / 4) / np.power(eta + 1, (s + 5) / 4))
        * (np.power(p, (s + 5) / 4) * trc)
    )




### Thermal Bremsstrahlung emission from the grid ###
def get_T_array(p_array, rho_array):
    """
    calculates the temperature array from the raw simulation hydro values
    Make sure p and rho arrays are in SI units
    input p_array in Pa
    input rho_array in kg/m^3
    """
    mp = 1.6726e-27
    kb=  1.380649e-23
    # grab some constants first
    constants = (0.60364 * mp) / kb  # SI units
    
    return ( 
        (p_array / rho_array)  
        * constants
    )

## Inputting the equations to calculate the bremsstrahlung from the grid.
def get_grid_volumes(data): 
    """
    Calculate the grid volume for the simulation grid
    """

    kpc3_to_m3 = 2.938e58
    
    dz = data.dx1[:, np.newaxis, np.newaxis]
    dy = data.dx2[np.newaxis, :, np.newaxis]
    dx = data.dx3[np.newaxis, np.newaxis, :]

    cell_volumes = dx * dy * dz # this is in cubic kpc
    cell_volumes = np.multiply(cell_volumes, 2.938e58)  # convert to SI units (m^3) 
    
    return cell_volumes

def get_g_nu_T(T_array, gauss_num, f):
    """
    Whatever that random bit is on the end of the emissivity function is
    """
    kb = 1.380649e-23
    h = 6.62607015e-34

    return ( 
        (np.sqrt(3) / np.pi) * 
        np.log( (4 * kb * T_array) 
               / (gauss_num * h * f) )
    )


# ------------------------------------------------------------- #
# The main functions for calculating the emission from the grid
# ------------------------------------------------------------- #
def grid_bremsstrahlung_shit(rho_arrays, p_arrays, frequencies, redshifts): 
    """vi
    Main function: 
    Z:         average atomic number: 
    rho_array: density_array:          from simulations in units of kg/m**3
    p_array:   pressure array :        from simulations in units of Pa 
    freqs:     observing frequencies:  Chandra sees anywhere from 0.12–12 nm (0.1–10 keV) (2.5x 10^16 -- 2.5x10^18).    Input is in Hz.
    gauss_num: Gauss's number :        default is 1.78
    """
    e=1.602176634e-19
    h = 6.62607015e-34
    me = 9.1093837015e-31
    kb = 1.380649e-23
    c = 299792458.0
    eps0=8.8541878128e-12
    gauss_num = 1.78       # changeable?
    Z=1.04                 # changeable?

    if gauss_num == None: 
        gauss_num = 1.78
        
    scaling_factor = 10**10

    # pile all the constants together 
    loads_of_consts =  ( 
         (np.power(e,6) * np.sqrt(np.pi*me)) / (3 * np.power(np.pi*me,2) * (np.power(eps0*c,3)) * np.sqrt(6)* np.power(kb, 5/2))
    )
    
    # ensure redshift is an array
    redshifts = np.atleast_1d(redshifts)
    #     frequencies = frequencies * (1 + z)
    # correct observing frequencies for redshift (emitted is higher)
    corrected_frequencies = np.empty((frequencies.shape[0], redshifts.shape[0]))
    for i in range(corrected_frequencies.shape[0]):
        for j in range(corrected_frequencies.shape[1]):
            corrected_frequencies[i, j] = frequencies[i] * (1 + redshifts[j])

    # set up the bremsstrahlung emission array which has shape (frequencies, redshifts, data_z, data_y, data_x)
    brem_emission = np.zeros((rho_arrays.shape[0], rho_arrays.shape[1], rho_arrays.shape[2], corrected_frequencies.shape[0], corrected_frequencies.shape[1]))
    #print(brem_emission.shape)

   #brem_emission = np.zeros((rho_arrays[:, 0].shape[0], grid_range.shape[0], corrected_frequencies.shape[0], corrected_frequencies.shape[1]))


    for i in range(corrected_frequencies.shape[0]): # loop through the frequencies
        for j in range(corrected_frequencies.shape[1]):  # loop through the redshifts
            T_array = get_T_array(p_arrays, rho_arrays)
            #T_array = get_T_array(p_arrays[:, g], rho_arrays[:, g])
            gaunt = get_g_nu_T(T_array, gauss_num, corrected_frequencies[i, j])
            power_bit = (h * corrected_frequencies[i,j]) / (kb * T_array)
            most_of_the_stuff = loads_of_consts * gaunt * np.power(Z, 2) * np.exp(-power_bit)
            frac_top = np.power(p_arrays,2) * scaling_factor #<-- we use this momentarily to stop underflow..or something. Will divide later on.
            #frac_top = np.power(p_arrays[:, g],2) * scaling_factor
            frac_denom = np.power(T_array, 5/2)
            thermal_emis_per_unit_volume = np.divide(frac_top, frac_denom)
            thermal_emis_per_unit_volume = np.divide(thermal_emis_per_unit_volume, np.multiply(scaling_factor, most_of_the_stuff))  #<-- remove the scaling factor
            brem_emission[:, :, :, i, j] = thermal_emis_per_unit_volume / (4 * np.pi)
            #brem_emission[:, i, j] = thermal_emis_per_unit_volume

    return brem_emission

def grid_bremsstrahlung(rho_arrays, p_arrays, frequencies, redshifts): 
    """
    Main function: 
    Z:         average atomic number: 
    rho_array: density_array:          from simulations in units of kg/m**3
    p_array:   pressure array :        from simulations in units of Pa 
    freqs:     observing frequencies:  Chandra sees anywhere from 0.12–12 nm (0.1–10 keV) (2.5x 10^16 -- 2.5x10^18).    Input is in Hz.
    gauss_num: Gauss's number :        default is 1.78
    """
    e=1.602176634e-19
    h = 6.62607015e-34
    me = 9.1093837015e-31
    kb = 1.380649e-23
    c = 299792458.0
    eps0=8.8541878128e-12
    gauss_num = 1.78       # changeable?
    Z=1.04                 # changeable?

    if gauss_num == None: 
        gauss_num = 1.78

    # ensure redshift is an array
    redshifts = np.atleast_1d(redshifts)
    #     frequencies = frequencies * (1 + z)
    # correct observing frequencies for redshift (emitted is higher)
    corrected_frequencies = frequencies[:, None] * (1 + redshifts[None, :])

    # set up the bremsstrahlung emission array which has shape (frequencies, redshifts, data_z, data_y, data_x)
    brem_emission = np.zeros((rho_arrays.shape[0], rho_arrays.shape[1], rho_arrays.shape[2], corrected_frequencies.shape[0], corrected_frequencies.shape[1]))
    T_array = get_T_array(p_arrays, rho_arrays)
    p_squared = p_arrays * p_arrays
        
    # pile all the stuff that doesn't depend on frequency together 
    consts = (
        (Z**2 * e**6 / (3 * np.pi**2 * eps0**3 * me**2 * c**3)) * (np.pi * me / 6)**(1/2) * (kb*T_array)**(-5/2) * p_squared
    )

    # ok now we loop through frequencies and redshifts
    for i in range(corrected_frequencies.shape[0]): # loop through the frequencies
        for j in range(corrected_frequencies.shape[1]):  # loop through the redshifts
            exp = np.exp(-(h * corrected_frequencies[i,j]) / (kb * T_array))
            #gaunt = get_g_nu_T(T_array, gauss_num, corrected_frequencies[i, j]) ## We get this from a look-up table now
            
            brem_emission[:, :, :, i, j] = ((consts * exp) / (4 * np.pi))
    
    return brem_emission, T_array

    
# -----------------------------------------
#           KERNEL FUNCTION
# -----------------------------------------

@njit
def njit_compute_particle_radii(particle_x1, particle_x2, particle_x3, n): 
    num_particles = len(particle_x1)
    mean_radius = np.zeros(num_particles)
    
    for i in range(num_particles):
        separations = np.empty(num_particles)
        
        for j in range(num_particles):
            if i == j:
                separations[j] = 1e99  # A large value to ignore self-distance
            else:
                dx = particle_x1[i] - particle_x1[j]
                dy = particle_x2[i] - particle_x2[j]
                dz = particle_x3[i] - particle_x3[j]
                separations[j] = np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # Find the n smallest separations and compute their mean
        smallest_separations = np.full(n, 1e99)
        
        for k in range(n):
            min_index = -1
            min_separation = 1e99
            for j in range(num_particles):
                if separations[j] < min_separation:
                    min_separation = separations[j]
                    min_index = j
            
            smallest_separations[k] = min_separation
            separations[min_index] = 1e99  # Exclude this index in the next iteration
        
        mean_radius[i] = np.sum(smallest_separations) / (2 * n)
    
    return mean_radius
    

@njit
def compute_kernel(grid_x1_chunk, grid_x2_chunk, grid_x3_chunk, 
                   particle_x1, particle_x2, particle_x3, 
                   particle_radii, n, p):
    
    kernel_weightings = np.zeros((len(grid_x1_chunk), n))
    particle_ids = np.zeros((len(grid_x1_chunk), n), dtype=np.int32)
    
    for i in range(len(grid_x1_chunk)):
        distances = np.empty(len(particle_x1))
        
        # Calculate squared distances for all particles at once
        for j in range(len(particle_x1)):
            distances[j] = (grid_x1_chunk[i] - particle_x1[j])**2 + \
                           (grid_x2_chunk[i] - particle_x2[j])**2 + \
                           (grid_x3_chunk[i] - particle_x3[j])**2
        
        # Manually find the n smallest distances and their indices
        nearest_indices = np.full(n, -1, dtype=np.int32)
        nearest_distances = np.full(n, 1e99)
        
        for k in range(n):
            min_index = -1
            min_distance = 1e99
            for j in range(len(distances)):
                if distances[j] < min_distance:
                    min_distance = distances[j]
                    min_index = j
            
            nearest_distances[k] = min_distance
            nearest_indices[k] = min_index
            distances[min_index] = 1e99  # Exclude this index in the next iteration
        
        # Compute the kernel radius
        weighted_radii_sum = 0.0
        distance_sum = 0.0
        for k in range(n):
            radius_p = particle_radii[nearest_indices[k]]**p
            weighted_radii_sum += nearest_distances[k] / 2 * radius_p
            distance_sum += radius_p
        kernel_radii = weighted_radii_sum / distance_sum
        
        # Compute the kernel weightings
        sum_weightings = 0.0
        for k in range(n):
            exp_val = np.exp(-nearest_distances[k]**2 / (2 * kernel_radii**2))
            weighting = exp_val * particle_radii[nearest_indices[k]]**p
            kernel_weightings[i, k] = weighting
            sum_weightings += weighting
        
        # Normalize the kernel weightings
        for k in range(n):
            kernel_weightings[i, k] /= sum_weightings
        
        # Store the particle IDs
        particle_ids[i, :] = nearest_indices
    
    return kernel_weightings, particle_ids



def loss_factor_kernel(grid_x1, grid_x2, grid_x3, particle_x1, particle_x2, particle_x3, n=6, p=3):
    # Compute particle radii using all available cores
    # with Pool(cpu_count()) as pool:
    #     chunk_size = len(particle_x1) // cpu_count()
    #     particle_chunks = [(particle_x1[i:i+chunk_size], particle_x2[i:i+chunk_size], particle_x3[i:i+chunk_size], n)
    #                        for i in range(0, len(particle_x1), chunk_size)]
    #     particle_radii_list = pool.starmap(compute_particle_radii, particle_chunks)
    
    # particle_radii = np.concatenate(particle_radii_list)
    particle_radii = njit_compute_particle_radii(particle_x1, particle_x2, particle_x3, n)

    
    # Compute kernel radii and weightings using all available cores
    with Pool(cpu_count()) as pool:
        chunk_size = len(grid_x1) // cpu_count()
        print(cpu_count, chunk_size)
        grid_chunks = [(grid_x1[i:i+chunk_size], grid_x2[i:i+chunk_size], grid_x3[i:i+chunk_size], 
                        particle_x1, particle_x2, particle_x3, particle_radii, n, p)
                       for i in range(0, len(grid_x1), chunk_size)]
        results = pool.starmap(compute_kernel, grid_chunks)
    
    kernel_weightings = np.concatenate([r[0] for r in results], axis=0)
    particle_ids = np.concatenate([r[1] for r in results], axis=0)
    
    return kernel_weightings, particle_ids


# -------------------------------------- #
#  Map the losses to the emission array  #
# -------------------------------------- #

@njit(parallel=True)
def weighted_emission(loss_factors, jet_lobe_emission, particle_ids, kernel_weightings, grid_indices): 
    # initialise the jet_lobe_emission_weight0ed array
    jet_lobe_emission_weighted = np.empty_like(jet_lobe_emission)
    for i in range(loss_factors.shape[1]): 
        for j in range(loss_factors.shape[2]):
            particle_losses = loss_factors[:, i, j]
            for grid_cell in prange(kernel_weightings.shape[0]): 
                pids = particle_ids[grid_cell, :]                # get the particle ids that are associated with that cell
                weights = kernel_weightings[grid_cell, :]        # get the weightings that are associated with the particles  
                indxs = grid_indices[grid_cell]                  # get the indices for that grid cell that will allow us to shove it back into emission array
                # if the particle doesn't exist in the loss array, it's not emitting!! 
                weighted_losses = particle_losses[pids] * weights   # multiply the particle losses with their corresponding weights
                mean_loss = np.nansum(weighted_losses)          # Compute the mean loss for this grid cell
                jet_lobe_emission_weighted[indxs[2], indxs[1], indxs[0], i, j] = jet_lobe_emission[indxs[2], indxs[1], indxs[0], i, j] * mean_loss#.value  # multiply the existing emission value with the loss term.
    return jet_lobe_emission_weighted


# ------------------------------------- #
#  The stuff that does particle losses
# ------------------------------------- #

@njit("f8(f8, f8, f8, f8)")
def get_loss_term(p_quot, g, ginj, s):
    """
    Just calculates the lossy bit from the particles.
    """
    gquot = ginj / g

    #pressure_term = np.power(pquot, 1 - (4 / (3 * gc)))
    #gamma_term = np.power(gquot, 2 - s)

    return ( p_quot * (np.power(gquot, 2 - s)) ) # (P_acc / P_current)^(1-4/3Gamma_c) * (gam_acc / gam_cur)^(2-s)


@njit("f8(f8)")
def get_loss_term_lossless(p_quot):
    return p_quot





@njit(parallel=True, fastmath=False)
def praise_aio(
    *,
    time,
    density,
    pressure,
    lst,
    trc,
    frequencies,
    redshifts,
    eta,
    s,
    gmin,
    gmax,
    losses,
    emitting_time_index,
    debug=False,
):
    freq_cmb = 5.879e10 # frequency of cosmic microwave background at z = 0
    temp_cmb = 2.725 # temperature of cosmic microwave background at z = 0
    
    # ensure redshift is an array
    redshifts = np.atleast_1d(redshifts)

    #     frequencies = frequencies * (1 + z)

    # correct observing frequencies for redshift (emitted is higher)
    corrected_frequencies = np.empty((frequencies.shape[0], redshifts.shape[0]))
    for i in range(corrected_frequencies.shape[0]):
        for j in range(corrected_frequencies.shape[1]):
            if frequencies[i] > 10**12: 
                cor_f = frequencies[i] * (1 + redshifts[j])
                lorentz = np.sqrt(cor_f/(freq_cmb*temp_cmb*(1 + redshifts[j])))
                corrected_frequencies[i, j] = cor_f / (lorentz**2) # down convert the frequency
            else:
                corrected_frequencies[i, j] = frequencies[i] * (1 + redshifts[j])

    # set up Lorentz factor arrays
    # shape is (n_particles,n_frequencies,n_redshifts)
    cur_gn = np.zeros((pressure.shape[0], frequencies.shape[0], redshifts.shape[0]))
    gn = np.zeros_like(cur_gn)  # (n_particles,n_frequencies,n_redshifts)

    # set up injected pressure array
    pinj = np.zeros((pressure.shape[0]))  # (n_particles)

    # set up emissivity arrays
    is_emitting = np.full((pressure.shape[0]), True, dtype=np.bool_)  # (n_particles)
    emissivity = np.zeros_like(cur_gn)  # (n_particles,n_frequencies,n_redshifts)
    # brem_emis = np.zeros(( grid size,frequencies.shape[0], redshifts.shape[0])) #n_cells, n_frequencies, n_redshifts

    # set up debug arrays
    if debug:
        lorentz_factors = np.zeros(
            (
                pressure.shape[0],
                pressure.shape[1],
                frequencies.shape[0],
                redshifts.shape[0],
            )  # (n_particles,n_outputs,n_frequencies,n_redshifts)
        )
        ap_arr = np.zeros(
            (pressure.shape[0], pressure.shape[1])
        )  # (n_particles,n_outputs)
        a2_arr = np.zeros(
            (pressure.shape[0], pressure.shape[1], redshifts.shape[0])
        )  # (n_particles,n_outputs,n_redshifts)

    # calculate current lorentz factor
    for i in prange(cur_gn.shape[0]):  # loop over particles
        gc = get_adiabatic_ind(
            pressure[i, emitting_time_index], density[i, emitting_time_index]
        )
        for j in range(cur_gn.shape[1]):  # loop over frequencies
            for k in range(cur_gn.shape[2]):  # loop over redshifts
                gam = get_gamma_larmor(
                    pressure[i, emitting_time_index],
                    gc,
                    eta,
                    corrected_frequencies[j, k],
                )
                cur_gn[i, j, k] = gam
                gn[i, j, k] = gam
                if debug:
                    lorentz_factors[i, emitting_time_index, j, k] = cur_gn[i, j, k]

    for i in prange(cur_gn.shape[0]):  # loop over particles
        # boolean to track whether this particle is still emitting at ANY frequencies we're looking at
        still_emitting = True

        # check that this particle exists. if it doesn't, continue (emissivity remains 0)
        if pressure[i, emitting_time_index] == np.nan:
            continue

        # also check that this particle has actually been shocked (lst > 0).
        # if not, continue (emissivity remains 0)
        if lst[i] <= 0:
            continue

        if debug:
            print("Looping times for particle " + str(i))



        # -------------------------- #
        #  PARTICLE PRESSURE HISTORY
        # -------------------------- #
        ## We trace backwards through time 

        # initialise the pressure quotient term 
        pressure_quotient = 1
        
        for j in range(emitting_time_index - 1, -1, -1):  # loop over times starting from the current time, and working backwards until we get to -1. 
            ## We are only concerned with looping back in time until we reach where the particle was last shocked! 
            # if we are lossless -- we still want to carry on
            #if losses < 1:
            #    break

            # break if we have gone past the last shock time for this particle
            if lst[i] > time[j]:
                # should check out of bounds access here?
                if debug:
                    print("Reached lst time for particle " + str(i))
                    print(j, gn[i, 0, 0])
                pinj[i] = pressure[i, j + 1] # We have found the pressure of last shock 
                break

            # set break flag
            still_emitting = False

            # variable to store result of get_gamma
            gam = 0

            # set time variables
            tn = time[j] - lst[i]
            tn1 = time[j + 1] - lst[i]

            # set pressure variables
            pn = pressure[i, j]  # current pressure 
            pn1 = pressure[i, j + 1]  # future pressure

            # set adiabatic index
            if losses >= 1: 
                gc = get_adiabatic_ind(pn, density[i, j]) # current adiabatic index


            # ---------------------------- #
            # Calculate the pressure ratio
            ## In the lossless calculation we have a term (P_acc / P)^(1-(4/3Gamma_c)). This ratio assumes that Gamma_c is the same for P_acc and P. This is a safe assumption for where P_acc and P 
            ## are not temporally different. Where they are temporally different, it is likely to be inaccurate for a variable value of Gamma_c. Here, we will calculate the ratio  (P_acc / P)^(1-(4/3Gamma_c)) 
            ## by clycling backwards through time, multiplying the new pressure ratio to the old one each time. So what we get is  (P_acc / P_acc+1)^(1-(4/3Gamma_c)) x.....x  (P_acc+k / P_acc+k+1)^(1-(4/3Gamma_c, k))

            if losses <= 1:
                # lossless pressure quotient
                pressure_quotient *= np.power((pn / pn1), 1) # we update the pressure quotent with each timestep.  
            else: 
                pressure_quotient *= np.power((pn / pn1), 1 - (4 / (3 * gc)))
            

            # calculate ap, a2
            # losses: 0 -> lossless
            #         1 -> adiabatic only
            #         2 -> adiabatic + synch
            #         3 -> adiabatic + IC
            #         4 -> full

            ap = 0.0
            #             a2 = 0.0
            a2_redshift = np.zeros((redshifts.shape[0]))

            # adiabtic losses enabled
            if losses >= 1:
                ap = get_ap(pn=pn, pn1=pn1, tn=tn, tn1=tn1)
            # radiative losses enabled
            if losses >= 2:
                for k in range(redshifts.shape[0]):  # loop over redshifts
                    a2_redshift[k] = get_a2(
                        pn, tn, tn1, ap, gc, eta, redshifts[k], losses
                    )

            if debug:
                ap_arr[i, j] = ap
                for k in range(redshifts.shape[0]):  # loop over redshifts
                    a2_arr[i, j, k] = a2_redshift[k]

            if debug and j % 1000 == 0:
                print(j, tn, tn1, pn, pn1, gc, ap, a2_redshift[0], gn[i, 0, 0])
            for k in range(cur_gn.shape[1]):  # loop over frequencies
                for l in range(cur_gn.shape[2]):  # loop over redshifts
                    gam = get_gamma(gn[i, k, l], tn, tn1, ap, a2_redshift[l], gc, losses)
                    gn[i, k, l] = gam
                    if gn[i, k, l] > 0:
                        still_emitting = True
                    else:
                        gn[i, k, l] = 0
                    if debug:
                        lorentz_factors[i, j, k, l] = gn[i, k, l]

            # break if no longer emitting
            if still_emitting == False:
                if debug:
                    print(
                        "Particle "
                        + str(i)
                        + " is no longer emitting, timestep "
                        + str(j)
                    )
                    print(j, tn, tn1, pn, pn1, gc, ap, a2_redshift[0], gn[i, 0, 0])
                is_emitting[i] = False
                break

        # calculate emissivity if necessary
        if is_emitting[i]:
            # get adiabatic index
            gc = get_adiabatic_ind(
                pressure[i, emitting_time_index], density[i, emitting_time_index]
            )

            if debug:
                print("Particle " + str(i) + " is emitting")
                print(
                    pinj[i],
                    pressure[i, emitting_time_index],
                    density[i, emitting_time_index],
                    gn[i, 0, 0],
                    cur_gn[i, 0, 0],
                    gc,
                )

            for k in range(cur_gn.shape[1]):  # loop over frequencies

                for l in range(cur_gn.shape[2]):  # loop over redshifts

                    # calculate the loss factor - lossy case
                    if gn[i, k, l] > 0:
                        if losses > 1:
                            emissivity[i, k, l] = get_loss_term(p_quot = pressure_quotient,
                                                                g= cur_gn[i, k, l], 
                                                                ginj=gn[i, k, l], 
                                                                s=s, 
                                                               )

        
                        else:
                            # calculate the loss factor, lossless case - just returns the pressure quotient
                            emissivity[i, k, l] = get_loss_term_lossless(p_quot = pressure_quotient)
                            
                            
    return emissivity


def praise(
    sim,
    part_data,
    part_times,
    max_output,
    emit_outputs,
    redshift=None,
    freqs=None,
    eta=0.03,
    s=2.2,
    gmin=500,
    gmax=1e5,
    lst_index=2,
    losses=4,
    window_width=5,
    smooth_order=3,
    particle_spacing=0.01 * u.Myr,
    debug=False,
    time_execution=False,
    particle_slice=np.s_[:],
    output_system="grid",
):
    loss_name_mapping = {
        0: "lossless",
        1: "adiabatic",
        2: "adiabatic+synch",
        3: "adiabatic+ic",
        4: "full",
        5: "synch",
        6: "ic",
        7: "radiative",
    }

    required_loss_mapping = {
        5: [0, 1, 2],
        6: [0, 1, 3],
        7: [0, 1, 4],
    }

    if time_execution:
        start = time.perf_counter()


    # particle time array with units
    pt = part_times * sim.unit_values.time

    # evenly spaced particle times
    # particle_spacing should be set to the particle output rate, 0.01Myr by default
    desired_particle_times = (
        np.arange(0, pt[-1].value, step=particle_spacing.value) * u.Myr
    )

    # indicies of elements in the particle time array from above that match with the evenly spaced particle times
    particle_uniform_indices = np.argmin(
        np.abs(np.subtract.outer(pt, desired_particle_times)), axis=0
    )

    # check we are monotonically increasing
    if np.any(np.diff(particle_uniform_indices) < 0):
        raise Exception(
            "Uniform particle time indicies are not monotonically increasing"
        )
    particle_uniform_indices = np.unique(particle_uniform_indices)

    # evenly spaced particle times, using the indices calculated above. This should be **nearly** identical to desired_particle_times
    particle_uniform_time = pt[particle_uniform_indices]
    # same as above but unitless
    particle_uniform_time_unitless = part_times[particle_uniform_indices]

    # handle output system for max output
    if output_system == "grid":
        pt_even_output = np.argmin(
            np.abs(particle_uniform_time - sim.times[max_output])
        )
        max_output = pt_even_output
    elif output_system == "particles":
        pt_even_output = np.argmin(np.abs(particle_uniform_time - pt[max_output]))
        max_output = pt_even_output

    if time_execution:
        print(f"Setup evenly spaced times: {time.perf_counter() - start:.2f}s")

    # set up frequency
    if freqs is None:
        freqs = np.array(([0.15, 0.3, 0.9, 1.4, 5.5, 100] * u.GHz).si.value)
    # set up redshift
    if redshift is None:
        redshift = 0.05

    # set up time array
    if time_execution:
        start = time.perf_counter()
    all_time = (
        particle_uniform_time_unitless[: max_output + 1].astype(np.float64)
        * sim.unit_values.time.to(u.Myr).value
    )
    if time_execution:
        print(f"Create time array: {time.perf_counter() - start:.2f}s")

    # set up density array
    if time_execution:
        start = time.perf_counter()
    all_density = (
        part_data["density"][particle_slice, :][
            :, particle_uniform_indices[: max_output + 1]
        ].astype(np.float64)
        * sim.unit_values.density.si.value
    )
    if time_execution:
        print(f"Create density array: {time.perf_counter() - start:.2f}s")

    # backfill density
    if time_execution:
        start = time.perf_counter()
    bfill(all_density)
    if time_execution:
        print(f"Backfill density array: {time.perf_counter() - start:.2f}s")

    # set up pressure array
    if time_execution:
        start = time.perf_counter()
    all_pressure = (
        part_data["pressure"][particle_slice, :][
            :, particle_uniform_indices[: max_output + 1]
        ].astype(np.float64)
        * sim.unit_values.pressure.si.value
    )
    if time_execution:
        print(f"Create pressure array: {time.perf_counter() - start:.2f}s")

    # backfill pressure
    if time_execution:
        start = time.perf_counter()
    bfill(all_pressure)
    if time_execution:
        print(f"Backfill pressure array: {time.perf_counter() - start:.2f}s")

    # set up last shock array
    if time_execution:
        start = time.perf_counter()
    all_lst = (
        part_data["last_shock_time"][particle_slice, :, lst_index][
            :, particle_uniform_indices[: max_output + 1]
        ].astype(np.float64)
        * sim.unit_values.time.to(u.Myr).value
    )
    if time_execution:
        print(f"Create LST array: {time.perf_counter() - start:.2f}s")

    # set up tracer array
    if time_execution:
        start = time.perf_counter()
    all_trc = part_data["tracer"][particle_slice, :][
        :, particle_uniform_indices[: max_output + 1]
    ]
    if time_execution:
        print(f"Create trc array: {time.perf_counter() - start:.2f}s")

    # smooth density and pressure data, if our window width is greater than 0
    if window_width > 0:
        if time_execution:
            start = time.perf_counter()
        all_density_smoothed = savgol_filter(
            all_density, window_width, smooth_order, mode="nearest"
        )
        if time_execution:
            print(f"Smooth density array: {time.perf_counter() - start:.2f}s")

        if time_execution:
            start = time.perf_counter()
        all_pressure_smoothed = savgol_filter(
            all_pressure, window_width, smooth_order, mode="nearest"
        )
        if time_execution:
            print(f"Smooth pressure array: {time.perf_counter() - start:.2f}s")
    else:
        all_density_smoothed = all_density
        all_pressure_smoothed = all_pressure

    # ensure losses, emit_outputs, and redshift are iterable
    try:
        _ = iter(losses)
    except TypeError:
        losses = [losses]

    # for losses >=5, we also need to have lossless (0), adiabatic (1), and corresponding adibatic+... (2,3, or 4)
    fixed_losses = []
    for i in losses:
        fixed_losses.append(i)
        if i in required_loss_mapping:
            fixed_losses.extend(required_loss_mapping[i])
    fixed_losses = sorted(list(set(fixed_losses)))

    # we only need to pass losses <= 4 to the actual praise function -- the others are handled in post-processing
    losses_to_calculate = [loss for loss in fixed_losses if loss <= 4]

    try:
        _ = iter(emit_outputs)
    except TypeError:
        emit_outputs = [emit_outputs]

    try:
        _ = iter(redshift)
    except TypeError:
        redshift = [redshift]
    redshift = np.array(redshift)

    # handle output system for emit_outputs
    if output_system == "grid":
        praise_outputs = [
            np.argmin(np.abs(particle_uniform_time - sim.times[eo]))
            for eo in emit_outputs
        ]
    elif output_system == "particles":
        praise_outputs = [
            np.argmin(np.abs(particle_uniform_time - pt[eo])) for eo in emit_outputs
        ]

    emissivity_dict = {}
    for loss_switch in losses_to_calculate:
        emissivity_list = []
        for i, start_output in enumerate(praise_outputs):
            if time_execution:
                start = time.perf_counter()
            result_dict = {
                "emis": praise_aio(
                    time=all_time,
                    density=all_density_smoothed,
                    pressure=all_pressure_smoothed,
                    lst=all_lst[:, start_output],
                    trc=all_trc[:, start_output],
                    frequencies=freqs,
                    redshifts=redshift,
                    eta=eta,
                    s=s,
                    gmin=gmin,
                    gmax=gmax,
                    losses=loss_switch,
                    emitting_time_index=start_output,
                    debug=debug,
                )
                * (u.W / (u.Hz * u.m ** 3 * u.sr)),  # in these units!!
                "part_output": particle_uniform_indices[start_output],
            }
            if output_system == "grid":
                result_dict["grid_output"] = emit_outputs[i]
            emissivity_list.append(result_dict)
            if time_execution:
                print(
                    f"Calculated emissivities ({loss_switch=}, t={particle_uniform_time[start_output]:.2f}): {time.perf_counter() - start:.2f}s"
                )
        emissivity_dict[loss_name_mapping[loss_switch]] = emissivity_list

    # handle losses >= 5 in post processing
    ret_emissivity_dict = {}
    for loss_switch in losses:
        # begin by adding any losses asked for that don't need extra calculation
        if loss_switch <= 4:
            ret_emissivity_dict[loss_name_mapping[loss_switch]] = emissivity_dict[
                loss_name_mapping[loss_switch]
            ]
        else:
            # now we handle more complicated situations
            new_emissivity_list = []
            loss_indicies = required_loss_mapping[loss_switch]
            print(
                f"Calculating losses for {loss_name_mapping[loss_switch]} using {[loss_name_mapping[loss] for loss in loss_indicies]}"
            )
            for i, start_output in enumerate(praise_outputs):
                lossless = emissivity_dict[loss_name_mapping[loss_indicies[0]]][i]
                adiabatic = emissivity_dict[loss_name_mapping[loss_indicies[1]]][i]
                loss_other = emissivity_dict[loss_name_mapping[loss_indicies[2]]][i]
                result_dict = {
                    "emis": np.nan_to_num(
                        lossless["emis"] * (loss_other["emis"] / adiabatic["emis"])
                    ),
                    "part_output": lossless["part_output"],
                }
                if output_system == "grid":
                    result_dict["grid_output"] = lossless["grid_output"]

                new_emissivity_list.append(result_dict)
            ret_emissivity_dict[loss_name_mapping[loss_switch]] = new_emissivity_list

    return ret_emissivity_dict

# ---------------------------- #
#   INTERPOLATION FUNCTIONS
# ---------------------------- #

@njit
def get_cell_overlap(x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max):
    """
    x1, y1 are the coordinates of the uniform grid. 
    x2, y2 are the coordinates of the stretched grid
    """
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) 
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    return x_overlap * y_overlap

@njit(parallel=True)
def get_uniform_emis_grid(x_stretched, y_stretched, x_uniform, y_uniform, values_stretched):
    interpolated_values = np.zeros((len(y_uniform) - 1, len(x_uniform) - 1))
   
    # Loop over each cell in the uniform grid
    for i in prange(len(y_uniform) - 1):
        for j in prange(len(x_uniform) - 1):
            x_min_uniform = x_uniform[j]
            x_max_uniform = x_uniform[j + 1]
            y_min_uniform = y_uniform[i]
            y_max_uniform = y_uniform[i + 1]
            
            # Loop over each cell in the stretched grid
            for m in range(len(y_stretched) - 1):
                for n in range(len(x_stretched) - 1):
                    x_min_stretched = x_stretched[n]
                    x_max_stretched = x_stretched[n + 1]
                    y_min_stretched = y_stretched[m]
                    y_max_stretched = y_stretched[m + 1]
                    
                    # Calculate the overlap between stretched and uniform grid cells
                    overlap_area = get_cell_overlap(x_min_stretched, x_max_stretched, y_min_stretched, y_max_stretched,
                                                     x_min_uniform, x_max_uniform, y_min_uniform, y_max_uniform)
                    
                    # Find the fraction of the stretched cell that contributes to the uniform cell
                    stretched_cell_area = (x_max_stretched - x_min_stretched) * (y_max_stretched - y_min_stretched)

                    # Calculate the area of the uniform grid cell
                    uniform_cell_area = (x_max_uniform - x_min_uniform) * (y_max_uniform - y_min_uniform)
                    
                    
                    if stretched_cell_area > 0:
                        fraction = overlap_area / stretched_cell_area
                    else:
                        fraction = 0
                    
                    # Accumulate the contribution to the uniform grid cell
                    if uniform_cell_area > 0:
                        interpolated_values[i, j] += fraction * values_stretched[m, n] * (stretched_cell_area / uniform_cell_area)
    
                
    
    return interpolated_values


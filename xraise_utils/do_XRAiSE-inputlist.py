#!/usr/bin/env python

## This script produces the full X-ray images!
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# Imports 

# system
import os
import sys
import argparse
import gc as garbage
import yaml


# plutokore 
import plutokore as pk
import plutokore.radio as radio
import plutonlib.xray.xraise as xraise
#from plutokore import PlutoSim

# utilities
from numba import njit
from pathlib import Path
import h5py
from IPython.display import display, clear_output
import time as time


# science imports
import numpy as np
import numpy.ma as ma
import math as m
import scipy 
import pandas as pd

# matplotlib imports
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# astropy imports
from astropy.table import Table
from astropy import units as u  # Astropy units
from astropy import cosmology as cosmo  # Astropy cosmology
from astropy import constants as const  # Astropy constants
from astropy.convolution import convolve, Gaussian2DKernel  # Astropy convolutions

from IPython.display import clear_output, display
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.interpolate import RegularGridInterpolator

src_path = os.path.join(os.path.expanduser('~'),'plutonlib/src/plutonlib')
main_path = os.path.join(os.path.expanduser('~'),'plutonlib/')
xraise_utils = os.path.join(main_path,"xraise_utils/")
# ---------------
#   Useful Units
# ---------------

import plutokore.radio as radio
from plutokore.jet import UnitValues

unit_length = 1 * u.kpc 
unit_density = (0.60364 * u.u / (u.cm ** 3)).to(u.g / u.cm ** 3)
unit_speed = const.c
unit_time = (unit_length / unit_speed).to(u.Myr)
unit_pressure = (unit_density * (unit_speed ** 2)).to(u.Pa)
unit_mass = (unit_density * (unit_length ** 3)).to(u.kg)
unit_energy = (unit_mass * (unit_length ** 2) / (unit_time ** 2)).to(u.J)

uv = UnitValues(
    density=unit_density,
    length=unit_length,
    time=unit_time,
    mass=unit_mass,
    pressure=unit_pressure,
    energy=unit_energy,
    speed=unit_speed,
)

def make_surface_brightness_xray(sim, 
                                 parent_sim,
                                 praise_params,
                                 grid_range,
                                 particle_data, 
                                 particle_times, 
                                 out_directory): 
    # STEP 1 calculate the emissivities for all grid outputs - ready to go for after lunch
    if praise_params["emis_mode"] == 'radio':
        print("This script is only for x-ray stuff you silly goose")
        return 
    

    print(" ")
    print("CALCULATING GRID EMISSION FOR", grid_range)
    print(" ")
    start = time.time()

    # set a few variables if they have been missed 
    if praise_params["redshift"] is None: 
        redshifts = [0.05]
    else: 
        redshifts = praise_params["redshift"] 
        
    if praise_params["freqs"] is None:
        freqs = np.array(([2.5e16, 2.5e17] * u.Hz).si.value)  ### <----Corresponds to 0.1, and 1 keV.. values that Chandra can observe.
        
    else: 
        freqs = praise_params["freqs"].si.value
    # calculate kappa and the correction term for the xray equation. 

    ## ------------------------------------- ##
    #### LOAD IN THE APEC CORRECTION TABLE ####
    ## ------------------------------------- ##
    
    apec_corr_file = f"{xraise_utils}/apec_corrections.txt"
    apec_df = pd.read_csv(apec_corr_file, index_col=0)
    apec_temps = apec_df.index.values.astype(float)     
    apec_reds  = apec_df.columns.values.astype(float)    
    apec_corr_grid = apec_df.values                      

    interp_corr_apec = RegularGridInterpolator(
        (apec_temps, apec_reds),
        apec_corr_grid,
        bounds_error=False,
        fill_value=None   # extrapolate if slightly outside range
    )

    ## ------------------------------------- ##
    #### LOAD IN THE GAUNT CORRECTION TABLE ####
    ## ------------------------------------- ##
    gaunt_corr_file = np.load(f"{xraise_utils}/gaunt_lut_2d.npz")
    gaunt_T_vals = gaunt_corr_file['T']
    gaunt_f_vals = gaunt_corr_file['f']
    g_table = gaunt_corr_file['g']
    
    g_interp2d = RegularGridInterpolator((gaunt_T_vals, gaunt_f_vals), 
                                     g_table, 
                                     bounds_error=False, 
                                     fill_value=None)

    # --------------------------------------------------- #
    #                0. PARTICLE LOSSES
    #  We first calculate the particle loss factor array 
    #     for all particles at all times following 
    #               standard praise code
    # --------------------------------------------------- #
    
    # As per equation 7 in Turner + Shabala (2020) and equation ... in RAiSE II, we need to multiply the lossless emission calcualted above by [p_inj / pi]^(1-4/3Gamma_c) [gamma_inj / gamma_i]^(2-s). This is becaue we obtain loss  histories from the particles whereas the emission can easily be determined from the grid values and we don't run into issues with lobe 'undersampling' or 'oversampling' based on particle number. For each particle we calculate the loss factor at all considered frequencies and redshifts. 

    
    ## much of this is left over from the old praise scripts. It's been cleaned up a little to only include the things that are relevant for the losses

    print('CALCULATING PARTICLE LOSSES FOR ALL TIMESTEPS')
    start = time.time()
    
    particle_loss_dict = xraise.praise(
        sim=sim,
        part_data=particle_data,
        part_times=particle_times,
        max_output=grid_range[-1],
        emit_outputs=grid_range,
        redshift=praise_params["redshift"],
        freqs=praise_params["freqs"].si.value,
        eta=praise_params["eta"],
        s=praise_params["s"],
        gmin=praise_params["gmin"],
        gmax=praise_params["gmax"],
        lst_index=praise_params["lst_index"],
        losses=praise_params["losses"],
        particle_spacing=praise_params["particle_sampling"]*u.Myr,
        time_execution=False,
        output_system="grid",
    )

    loss_keys = list(particle_loss_dict.keys())
    emissivity_count = len(particle_loss_dict[loss_keys[0]])
    print(' -- emissivity count')
        
    #print(particle_loss_dict.values())
    print(' -- time to calculate losses for all particles', time.time() - start, 's.')

    # This creates an emissivity object particle_loss_dict['full'][i]['emis'] where part_emis['full'] [i] ['emis'] = [full losses], [the simulation/particle output], [the particle data].
    # particle_loss_dict['full'][i]['emis'] object is a (i, j, k) array where i=particles, j=frequencies, k=redshifts.
    # This structure is leftover from the original praise calcualtion. The praise calcualtion has been modified to just determine the particle loss factor rather than the emission. So, 
    # there will always be 'full' losses... once I neaten this up, it should be particle_loss_dict[i]['loss_factor']...or something like that. It should still store an (i, j, k) object because 
    # the loss term calculation is frequency and redshift dependent. 



    # ---------------------------------------------------------------- #
    #                   1.  OBSERVING PROPERTIES
    #    setup the observing grid and apply rotation if necessary
    #   The user should specify the angular resolution of the beam!
    #  We work in physical units (i.e. kpc) throughout until the last 
    # moment. Then we output in terms of arcseconds 
    # ---------------------------------------------------------------- #

    print('SETTING UP OBSERVING PROPERTIES')

    # set up observation properties
    arcsec2kpc = cosmo.Planck15.kpc_proper_per_arcmin(praise_params["redshift"]).to(u.kpc / u.arcsec)
    # The pixel size of the observing grid should be 1/3 of the size of the beam
    #pixel_size_arcsec = praise_params["beam_fwhm_angular"] * 1/3

    # we tailor our data outputs to the input requirements. 
    ## if user has selected kpc then the angular size of the beam is scaled down appropriately with redshift. 
    ## if user has selected arcsec then the angular size of the beam will capture a different amount of kpc as redshift increases.

    
    grid_resolution_kpc = praise_params["pixel_size"] * u.kpc
    grid_resolution_angular = grid_resolution_kpc / arcsec2kpc
    grid_resolution_angular = grid_resolution_angular[0]
    
    beam_fwhm_angular = praise_params["beam_fwhm"] * u.arcsec
    beam_fwhm_kpc = beam_fwhm_angular * arcsec2kpc

    
    fwhm_to_sigma = 1 / (8 * np.log(2)) ** 0.5   
    gaussian_sigma = (beam_fwhm_kpc * fwhm_to_sigma) / grid_resolution_kpc
    omega_beam_angular = (2 * np.pi * (beam_fwhm_angular.value * fwhm_to_sigma) ** 2 * u.arcsec**2)


    ## set up the coordinates for the projected emission grid based on the emission for the last particle output
    # determine where the maximum extend of the emitting grid needs to be
    # use the particles as before as this also makes a useful comparison between praise

    last_emissivity_output = list(particle_loss_dict.values())[0][-1]
    particle_output = last_emissivity_output["part_output"]
    nan_mask = ~np.isnan(particle_data["id"][:, particle_output])

    part_coords = np.c_[
            (
                particle_data["x1"][:, particle_output][nan_mask],
                particle_data["x2"][:, particle_output][nan_mask],
                particle_data["x3"][:, particle_output][nan_mask],
            )
        ]

    if praise_params["emis_mode"] == 'radio':
        coord_lim = np.max(np.abs(part_coords)) * 1.2 # the limit of our observing grid

    elif praise_params["emis_mode"] == 'xray':
        coord_lim = np.max(np.abs(part_coords)) * 1.8 # the limit of our observing grid
        
    # create a three dimensional cube that goes a bit beyond the limit.
    grid_lim = (
            -coord_lim - 2 * grid_resolution_kpc.value,  # use the physical grid resolution 
            coord_lim + 2 * grid_resolution_kpc.value,
    )

    grid_uniform = np.arange(grid_lim[0], grid_lim[1], grid_resolution_kpc.value)
    grid_uniform_mid = np.diff(grid_uniform) * 0.5 + grid_uniform[:-1]
    print('shape of grid', grid_uniform.shape)

    
    print(' .. arcsecond to kpc conversion', arcsec2kpc)
    print(' .. grid resolution (kpc)', grid_resolution_kpc)
    print(' .. grid resolution (arcsec)', grid_resolution_angular)
    print(' .. beam fwhm (arcsec)', beam_fwhm_angular)
    print(' .. gaussian_sigma', gaussian_sigma)
    print(' .. omega beam angular', omega_beam_angular)
    print(' .. limit of observing grid (x and y)', grid_lim)

    
    
    
    ## set up the rotation matrix if we need to turn the thing around 
    rot_mat = Rotation.from_euler(
        praise_params["rotation_axes"], praise_params["angle"], degrees=True
    ).as_matrix()


    
    # ------------------------------------------------ #
    #           2. LOOP THROUGH GRID OUTPUTS
    #  set up all the arrays we will need to calcualte 
    #            the grid cell emission 
    # ------------------------------------------------ #
    print('LOOPING THROUGH GRID OUTPUTS')


    print(emissivity_count)
    for GO, emis_output in zip(grid_range, range(emissivity_count)):

        PROJECTIONS = np.zeros((grid_uniform_mid.shape[0], grid_uniform_mid.shape[0], len(freqs), len(redshifts))) # has shape observing gridy, observing gridx, freqs, redshifts, num outputs!
        
        print(f'COMPUTING FOR GRID OUTPUT {GO}')
        print(' -- setting up pressure and density range') 

        data = sim.load_output(GO)
  
        # load in some useful quantities
        mx = data.mx
        my = data.mz  

        ex = data.ex
        ez = data.ez

        # create the dr for integration
        dr = (data.dx2*u.kpc).to(u.m)
        dr = dr.value
        dr_3d = dr.reshape(1, dr.shape[0], 1)

        grid_vels = np.stack((data.vx3[:, :, :], data.vx2[:, :, :], data.vx1[:, :, :]), axis=3)
                                                                                                     

        # grab density and pressure in SI units and tracer value
        rho = (
            np.multiply(data.rho[:,:,:], sim.unit_values.density.si.value)
        ).astype(np.float64) # in kg / m**3 now 
        
        prs = (
            np.multiply(data.prs[:,:,:], sim.unit_values.pressure.si.value)
        ).astype(np.float64) # in Pa
        
        trc = data.tr1[:, :, :] # unitless between 0 and 1
        

        print('   -- prs array shape', prs.shape)
        print('   -- rho array shape', rho.shape)

        # -------------------------- #
        # 2.1 calculate grid emission
        # -------------------------- #

        print(' -- calculating grid emission')
        start = time.time()
        # These things go into both the xray and radio calculations so do them here
        kappa_s = xraise.get_kappa_nonumba(praise_params["s"])
        # calculate adiabatic index for each cell
        gc_cells = xraise.get_adiabatic_ind_arr(prs, 
                                                rho)

        print("   -- gamma shape", gc_cells.shape)

        K = xraise.get_K(kappa_s, 
                     praise_params["s"], 
                     gc_cells, 
                     praise_params["gmin"], 
                     praise_params["gmax"])

        
        print("   -- K shape", K.shape)

        # Now calculate bremstrahlung for the lobe

        grid_emis, T_array = xraise.grid_bremsstrahlung(rho, 
                                                prs, 
                                                frequencies = freqs, 
                                                redshifts = redshifts)

        # IC emission now
        # calculate correction factor
        j_corr = xraise.get_j_corr(praise_params["s"])

        
        jet_lobe_emission = xraise.get_ic_emissivity_lossless(prs, 
                                                              gc_cells, 
                                                              praise_params["eta"], 
                                                              freqs, 
                                                              praise_params["s"], 
                                                              trc, 
                                                              K, 
                                                              redshifts, 
                                                              j_corr
                                                             )
        print('   -- bremsstrahlung emissivity array shape', grid_emis.shape) 
        print('   -- jet lobe emission shape', jet_lobe_emission.shape)

        # # diagnostics
        # mid = grid_emis.shape[1]//2
        # fig = plt.figure(figsize = (5, 5))
        # im = plt.pcolormesh(ex, 
        #                ez,
        #                np.log10(grid_emis[:, mid, :,  0, 0]),  # at the midplane slice
        #                cmap = "viridis", 
        #                    rasterized=True)
        
        # cbar = fig.colorbar(im)
        # cbar.set_label(r"Surface Brightness (midplane)")
        # plt.savefig(f'/u/gscs/processed_sim_data/x-ray_diagnostics/grid_emis_midplane_{GO}.pdf', dpi=50, bbox_inches="tight")
            
        jet_lobe_emission = np.nan_to_num(jet_lobe_emission, nan=0.0, posinf=0.0, neginf=0.0) 
        grid_emis = np.nan_to_num(grid_emis, nan=0.0, posinf=0.0, neginf=0.0) #UNITS (u.W / (u.Hz * u.m ** 3 * u.sr)) 

        ######################################################
            # APEC AND GAUNT CORRECTION TO BREMSTRAHLUNG
        ######################################################
        # We need to figure out the contribution from the line emission (whether abosoption or extra emission.
        # We returned the temperature array with the bremstrahlung emission so now we convert this to keV and use that to index the correction grid. 
        # We also want to get the gaunt factor for the bremstraglung emission. Because some temperatures on the grids can be cooler than the 
        # analytical expectation in Turner + Shabala 2020, we have created a look-up table for the gaunt value
        
        T_kev = T_array *  8.617333262e-8 # the conversion from kelvin to keV

        for b in range(grid_emis.shape[4]): #loop through the redshifts
            z_grid = np.full_like(T_kev, redshifts[b])  # fill an array with the same redshift
            pts = np.stack([T_kev, z_grid], axis=-1)
            corr_grid = interp_corr_apec(pts) # use the interpolator to find the best spot. 
            print(corr_grid.shape)

            # we have the corrections that should match to the grid now. We want to apply that correction
            for d in range(grid_emis.shape[3]):  # we're going to apply this correction across all frequencies
                grid_emis[:, :, :, d, b] *= corr_grid[:, :, :]  # apply the apec correction (not frequency dependent)
                ## And here do the Gaunt factor 
                corr_f = freqs[d] * (1 + redshifts[b])
                f_grid = np.full_like(T_array, freqs[d])
                pts = np.stack([T_array, f_grid], axis=-1)
                corr_grid = g_interp2d(pts) # re-write the corr_grid
                grid_emis[:, :, :, d, b] *= corr_grid[:, :, :] # add the Gaunt factor in.
            
        print('Grid emission shape after corection', grid_emis.shape) 
        

        del prs, rho, K, gc_cells, pts, corr_grid, T_kev, z_grid
        garbage.collect()
        print('   -- time to calculate emission from grid', time.time() - start, 's.')

        

        # --------------------------- #
        #  2.2  particle loss kernel
        # --------------------------- #
        #  We need to determine which particles are closest to a grid cell
        #  and what the weights of the loss factor should be 
        #  before we map back to the grid emission

        print(' -- calculating the loss kernel')
        
        # create grid coordinates that the jet occupies
        indices = np.where(trc >= praise_params["tracer_cutoff"])
        z_indices, y_indices, x_indices = indices
        x_coords = data.mx[x_indices]
        y_coords = data.my[y_indices]
        z_coords = data.mz[z_indices]
        grid_stacked = np.stack((x_coords, y_coords, z_coords), axis=1)
        grid_indices = np.stack((x_indices, y_indices, z_indices), axis=1)
        
        print('   -- grid coordinates shape after removing ambient crap', grid_stacked.shape)
        base_emis_dict = list(particle_loss_dict.values())[0][emis_output]  # only include the radiating particles
        particle_output = base_emis_dict["part_output"]

        print('   -- particle output', particle_output)
         
        nan_mask = ~np.isnan(particle_data["id"][:, particle_output]) 
        particle_x1 = particle_data["x1"][:, particle_output][nan_mask]
        particle_x2 = particle_data["x2"][:, particle_output][nan_mask]
        particle_x3 = particle_data["x3"][:, particle_output][nan_mask]

        print('   -- number of particles', len(particle_x3))

        ## Diagnostics - loss factors
        # fig  = plt.figure(figsize=(10, 7))

        particle_loss_arr = particle_loss_dict['full'][0]['emis'][:, 0, 0]
        last_inj = len(particle_x1)
        inj_p_loss = particle_loss_arr[:last_inj]
        particle_x2_indices = np.where((particle_x2 > -2) & (particle_x2 < 2))
        particle_x1_slice = particle_x1[particle_x2_indices]
        particle_x3_slice = particle_x3[particle_x2_indices]
        loss_slice = inj_p_loss[particle_x2_indices].value
                
        # im0 = plt.scatter(particle_x1_slice, 
        #          particle_x3_slice,
        #          cmap = 'magma',
        #          c=loss_slice, 
        #                   rasterized=True
        #          # vmin = 0, 
        #          # vmax = 1, 
        #          )
        
        
        # plt.xlabel('kpc', font='serif', fontsize=18)
        # plt.ylabel('kpc', font='serif', fontsize=18)
        
        # plt.gca().set_aspect("equal")
        # cbar = fig.colorbar(im0, orientation='vertical')
        # cbar.set_label(r"Loss Factors")
        # plt.savefig(f'/u/gscs/processed_sim_data/x-ray_diagnostics/loss_factors_{GO}.pdf', dpi=50, bbox_inches="tight")

        # ----- HISTOGRAM ---------

        # fig  = plt.subplots(figsize=(10, 7), sharey=True, sharex=True)
        
        # particle_loss_arr = particle_loss_dict['full'][0]['emis'][:, 0, 0]
        
        # last_inj = len(particle_x1)
        # inj_p_loss = particle_loss_arr[:last_inj]
        
        
        # particle_x2_indices = np.where((particle_x2 > -5) & (particle_x2 < 5))
        # particle_x1_slice = particle_x1[particle_x2_indices]
        # particle_x3_slice = particle_x3[particle_x2_indices]
        # loss_slice = inj_p_loss[particle_x2_indices].value
                
        # plt.hist(loss_slice)
        
        # plt.ylabel('count', font='serif', fontsize=18)
        # plt.xlabel('loss value', font='serif', fontsize=18)
        
        # print(np.amin(loss_slice))
        # plt.savefig(f'/u/gscs/processed_sim_data/x-ray_diagnostics/loss_factors_hist_{GO}.pdf', dpi=50, bbox_inches="tight")
            
        # -------------------------------------------------------------------------- #
        #  2.2.1 calcualte kernel weights and particle ids that belong to each cell  #
        # -------------------------------------------------------------------------- #
        start = time.time()
        kernel_weightings, particle_ids = xraise.loss_factor_kernel(grid_x1=grid_stacked[:, 0], 
                                                             grid_x2=grid_stacked[:, 1], 
                                                             grid_x3=grid_stacked[:, 2], 
                                                             particle_x1=particle_x1, 
                                                             particle_x2=particle_x2, 
                                                             particle_x3=particle_x3, 
                                                             n=6, 
                                                             p=3)
        
        print('   -- kernel weightings shape', kernel_weightings.shape)
        print('   -- particle id shape', particle_ids.shape)

        print('   -- time to find kernel weightings and particle ids', time.time() - start, 's.')
        #print(kernel_weightings)
        #print(particle_ids)

        # ------------------------------- #
        #  2.2.2 mapping losses to grid   #
        # ------------------------------- #
        #  We now take the particle losses and map these to the grid... we can use the stacked index array to map the cells back to the emission.
        #  For each grid cell, we have the particle ids that belong to the cell and the weightings that we attribute to them. We also have a grid index array
        print(' -- mapping losses')
        start = time.time()
        #print(particle_loss_dict['full'][emis_output]['emis'][:, :, :].shape)
                             
        jet_lobe_emission = xraise.weighted_emission(loss_factors = particle_loss_dict['full'][emis_output]['emis'][:, :, :], 
                                                    jet_lobe_emission = jet_lobe_emission, 
                                                    particle_ids = particle_ids, 
                                                    kernel_weightings = kernel_weightings, 
                                                    grid_indices = grid_indices)
        
        jet_lobe_emission = np.where(jet_lobe_emission < 0, 0.0, jet_lobe_emission) 
         
        print('time to map grid losses', time.time() - start, 's.') 

        # diagnostics
        fig = plt.figure(figsize = (5, 5))
        mid = jet_lobe_emission.shape[2]//2
        im = plt.pcolormesh(ex, 
                       ez,
                       np.log10(jet_lobe_emission[:, mid, :,  0, 0]),  # at the midplane slice
                       cmap = "viridis",
                           rasterized=True)
        
        cbar = fig.colorbar(im)
        cbar.set_label(r"Surface Brightness (midplane)")
        os.makedirs('./x-ray_diagnostics', exist_ok=True)
        plt.savefig(f'./x-ray_diagnostics/lossy_jet_lobe_emis_{GO}.png', dpi=50, bbox_inches="tight")

        # --------------------------------- #
        # 2.3 Total emission and projection
        # --------------------------------- #

        # jet and lobe emission is in the rest-frame of the emitting particles. Need to convert out of that. 
        print(' -- calculating total emission and projecting')
        start = time.time()


        # ------------------------------ #
        #  2.3.1  calculate doppler term #
        # ------------------------------ #
        # This is done with j = D^(2+alpha)*j' where j' is the emissivity in the rest-frame of the emitting particles and D is the relativistic Doppler factor. 
        # take the grid velocity array and rotation matrix we defined earlier.
        doppler = xraise.get_doppler(grid_vels, rot_mat)
        print('   -- doppler shape', doppler.shape, 'with nonzero values', np.count_nonzero(doppler), np.nanmax(doppler), np.nanmin(doppler))
        doppler_term = np.power(doppler, 2 + ((praise_params["s"] - 1) / 2))
        print('   -- doppler bit shape', doppler_term.shape, 'number of nonzero values', np.count_nonzero(doppler_term), 'number of nan values', np.count_nonzero(np.isnan(doppler_term)))

        for i in range(jet_lobe_emission.shape[3]): # loop through frequencies
            for j in range(jet_lobe_emission.shape[4]): # loop through redshifts  
                print('   -- total emission in prf', jet_lobe_emission.shape, 'with nonzero values', np.count_nonzero(jet_lobe_emission), np.nanmin(jet_lobe_emission), np.nanmax(jet_lobe_emission))

                # if we're considering xray emission, we need to add the two sources of emission together.
                total_emis = np.add(grid_emis[:, :, :, i, j], jet_lobe_emission[:, :, :, i, j] ) # 2 sources of emission

                # # Diagnostics
                # fig = plt.figure(figsize = (5, 5))
                # mid = total_emis.shape[2]//2
                # im = plt.pcolormesh(ex, 
                #                ez,
                #                np.log10(total_emis[:, 280, :]),  # at the midplane slice
                #                cmap = "viridis", 
                #                 vmin = -40,
                #                 vmax = -33, 
                #                    rasterized=True)
                
                # cbar = fig.colorbar(im)
                # cbar.set_label(r"Surface Brightness (midplane)")
                # plt.savefig(f'/u/gscs/processed_sim_data/x-ray_diagnostics/total_emis_{GO}.pdf', dpi=50, bbox_inches="tight")

                # multiply this with the total emission 
            
                total_emis = np.multiply(total_emis, doppler_term) # this is now just a i, j, thing

                # # diagnostics
                # fig = plt.figure(figsize = (5, 5))
                # mid = total_emis.shape[2]//2
                # im = plt.pcolormesh(ex, 
                #                ez,
                #                np.log10(total_emis[:, 280, :]),  # at the midplane slice
                #                cmap = "viridis", 
                #                 vmin = -40,
                #                 vmax = -33, 
                #                    rasterized=True)
                
                # cbar = fig.colorbar(im)
                # cbar.set_label(r"Surface Brightness (midplane)")
                # plt.savefig(f'/u/gscs/processed_sim_data/x-ray_diagnostics/total_emis_doppler_{GO}.pdf', dpi=50, bbox_inches="tight")
            
                # print(total_emis.shape)

                # with h5py.File('/u/gscs/scripts/X-RAiSE/nondoppler_test.hdf5', "w") as data_file:
                #     data_file.create_dataset('nodoppler', data=total_emis, dtype='float32')

                
                # sum down an axis 
                dr = np.ones((total_emis.shape[0], total_emis.shape[1], total_emis.shape[2])) * dr_3d
                print('dr shape', dr.shape)
                #total_emis = np.where(total_emis < 0, 0.0, total_emis) 
                total_emis_dr = np.multiply(total_emis, dr)
                
                integral = np.nansum(total_emis_dr, axis=1)

                # fig = plt.figure(figsize = (5, 5))
                # mid = total_emis.shape[2]//2
                # im = plt.pcolormesh(ex, 
                #                ez,
                #                np.log10(integral[:, :]),  # at the midplane slice
                #                cmap = "viridis", 
                #                vmin = -16.8, 
                #                vmax = -15.8, 
                #                    rasterized=True)
                
                # cbar = fig.colorbar(im)
                # cbar.set_label(r"Surface Brightness (midplane)")
                # plt.savefig(f'/u/gscs/processed_sim_data/x-ray_diagnostics/integrated_{GO}.pdf', dpi=50, bbox_inches="tight")

                 # --------------------------------------------- #
                 #  2.3.2  Interpolate and convolve if required  #
                 # --------------------------------------------- #
                
                interpolated_values = xraise.get_uniform_emis_grid(x_stretched = ex, 
                                                y_stretched = ez, 
                                                x_uniform = grid_uniform, # same in each direction 
                                                y_uniform = grid_uniform, 
                                                values_stretched = integral)
                
                # convert to mJy/beam
                projection =  (
                    (interpolated_values*u.W/(u.Hz * u.m**2 * u.sr)).to(
                        u.mJy / u.beam, 
                        equivalencies=u.beam_angular_area(
                        omega_beam_angular),
                    )
                )

                # convolve
                kern = Gaussian2DKernel(gaussian_sigma[j])
                
                projection_conv = (
                    convolve(
                        projection,
                        kern,
                        boundary='extend',
                    )
                )
                
                # ------------------------ #
                # 2.3.3 Save to the array
                # ------------------------ #

                PROJECTIONS[:, :, i, j] = projection_conv
                
                
        # ------------------------------------#
        #          3. SAVE PROJECTIONS
        # ----------------------------------- #
        write_SB_data(
            convolved_SB = PROJECTIONS,
            hdf5_file_path = out_directory,  
            praise_params = praise_params,
            sim = sim, 
            grid_output = GO,
            grid_x_kpc = grid_uniform, 
            grid_y_kpc = grid_uniform, 
            grid_mx_kpc = grid_uniform_mid, 
            grid_my_kpc = grid_uniform_mid, 
        )
        
    return 


def write_SB_data(convolved_SB,
                 hdf5_file_path,  
                 praise_params,
                 sim, 
                 grid_output,
                 grid_x_kpc, 
                 grid_y_kpc, 
                 grid_mx_kpc, 
                 grid_my_kpc, 
                 ): 
    """
    Writes the surface brightness data to hdf5 files. One for every simulation. 
    """
    with h5py.File(hdf5_file_path, "a") as data_file:
    
        for n, r in enumerate(praise_params['redshift']): # cycle redshifts
            for m, f in enumerate(praise_params['freqs']):  # cycle frequencies 
                dset_convolved = f'/{praise_params["config"]}/{r}/{f.value}/{praise_params["losses"][0]}/{praise_params["angle"]}'
                convolved_group = data_file.require_group(dset_convolved)
                grp = data_file[f'{praise_params["config"]}'][f'{r}'][f'{f.value}'][f'{praise_params["losses"][0]}'][f'{praise_params["angle"]}']
                
                sim_time = f'{sim.times[grid_output]:.2f}'
                print(f"Saving surface brightness data for {sim_time}")
                
                # # Add metadata for beam
                # beam_attrs = ['eta', 'gmin', 'gmax', 'lst_index', 's', 'beam_fwhm', 'obs_unit_input', 'grid_unit_input']
                # for attr in beam_attrs:
                #     grp.attrs[attr] = praise_params[attr]
                
                # print(f'Metadata created for {grp}')
                
                # Remove existing datasets if they exist
                group_path = f'/{praise_params["config"]}/{r}/{f.value}/{praise_params["losses"][0]}/{praise_params["angle"]}'
                if sim_time in data_file[group_path]:
                    print(f"Dataset {sim_time} exists in {group_path}. Deleting...")
                    del data_file[group_path][sim_time]
            
                # Create new datasets for convolved and nonconvolved
                grp.create_dataset(sim_time, data=convolved_SB[:, :, m, n])
                
                print(f'Dataset and metadata created for {sim_time}')
                
                # Save grid data 

                try:
                    print(f'----> Saving the grid midpoints and edges for {sim_time}')
                    for grid_name, grid_data in zip(['mx_kpc', 'my_kpc', 'x_kpc', 'y_kpc'], 
                                                    [grid_mx_kpc, grid_my_kpc, grid_x_kpc, grid_y_kpc]):
                        grid_group = data_file.require_group(f'/grid/{praise_params["config"]}/{grid_name}')
                        grid_group = data_file['grid'][f'{praise_params["config"]}'][f'{grid_name}']
                        # if exists, delete and remake
                        if sim_time in grid_group:
                            print(f"Dataset {sim_time} exists in {grid_group}. Deleting...")
                            del grid_group[sim_time]
                        #recreate
                        grid_group.create_dataset(sim_time, data=grid_data)
                        
                except ValueError:
                    pass
                print('Save complete')
                print(' ')

        
        
def main(): 
    parser = argparse.ArgumentParser(
        description = 'Calculates the convolved surface brightness data for a simulation'
    )
    
    # setup the arguments
    parser.add_argument("-s", '--sim_name', required = True, help='Simulation run code', type=str)
    parser.add_argument("-gr", '--grid_range', required = True, help='the range of grid outputs to save for', nargs='*', type=int)
    parser.add_argument("-m", '--emission_config', required = True, help='the emission configuration we want. Load this into the emission_params.yml file', type=str)
    parser.add_argument("-o", '--out_dir', required = False, help='output directory if a special one is required', type=str)
    
    args=parser.parse_args()
    
    print(logo)

    ## load in the directories for the sim we're processing. 
    ## set these directories in the yaml file 
    sim_stuff = f"{xraise_utils}/sim_directories.yml"
    emission_params = f"{xraise_utils}/emission_params.yml"
    
    with open(sim_stuff, "r") as f: 
        c = yaml.safe_load(f)
        name = c[args.sim_name]['name']
        sim_dir = c[args.sim_name]['sim_dir']
        parent_dir = c[args.sim_name]['parent_dir']
        pp = c[args.sim_name]['pp']
        pg = c[args.sim_name]['pg']
        particle_path =  c[args.sim_name]['particle_path']
        # Set standard directory if out_dir is not provided
        if args.out_dir is None:
            args.out_dir = f'/u/gscs/processed_sim_data/PRAiSE2_SB_data/{name}.hdf5'

        print(f'Processing for {name}')
        print(' -- simulation directory', sim_dir)
        print(' -- parent directory', parent_dir)
        print(' -- restarted particle file', pp)
        print(' -- restarted parent grid file', pg)
        print(' -- particle directory', particle_path)

    if parent_dir != 'None':
        print(" ")
        print("Parent directory found...loading sims and particle file")
        parent_sim = pk.PlutoSimulation(
            simulation_name='thing',                 
            simulation_directory=parent_dir,          
            simulation_description="thing",        
            datatype="float",                      
            dimensions=3,                           
        )

        sim = pk.PlutoSimulation(
            simulation_name='thing',                 
            simulation_directory=sim_dir,          
            simulation_description="thing",        
            datatype="float",                      
            dimensions=3,
            parent_simulation = parent_sim, 
            parent_grid_output = pg, 
            parent_particle_output = pp,
        )

    else: 
        print(" ")
        print("No parent directory found...loading single sim and particle file")
        sim = pk.PlutoSimulation(
            simulation_name='thing',                 
            simulation_directory=sim_dir,          
            simulation_description="thing",        
            datatype="float",                      
            dimensions=3,
        )

        parent_sim = 'None'
    
    
    particle_data, particle_times = pk.particles.load_particle_data_hdf5(sim, 
                                                                         particle_path)
    
    
    grid_range = [int(x) for x in args.grid_range]

    # construct the parameter dictionary
    
    
    print(' ')
    print('Particle Data Contains the Following')
    print(' ')
    print(particle_data)  # shows you what variables are contained within
    print(' -- The last time processed', particle_times[-1]*unit_time)
    print(' -- Number of grid outputs', len(sim.times))
    print(' -- Last grid output', len(sim.times) -1)


    praise_config = {}
    with open(emission_params, "r") as f: 
        # load in the emission mode/parameters we want 
        c = yaml.safe_load(f)
        pc = c["configs"][f'{args.emission_config}']
        print(pc)
        
        praise_config["config"] = f'{args.emission_config}'
        praise_config["angle"] =  pc.get("angle", c["angle"])
        praise_config["eta"] = pc.get("eta", c["eta"])
        praise_config["freqs"] = (np.array(
            pc.get("frequencies")) * u.GHz
                                 )
        praise_config["gmin"] = pc.get("gmin", c["gmin"])
        praise_config["gmax"] = pc.get("gmax", c["gmax"])
        praise_config["losses"] = pc.get("losses", c["losses"])
        praise_config["lst_index"] = pc.get("lst_index", c["lst_index"])
        praise_config["particle_sampling"] = pc.get("particle_sampling", c["particle_sampling"])
        praise_config["pixel_size"] = pc.get("pixel_size")
        praise_config["beam_fwhm"] = pc.get("beam_fwhm")
        praise_config["obs_unit_input"] = pc.get("obs_unit_input")
        praise_config["grid_unit_input"] = pc.get("grid_unit_input")
        praise_config["redshift"] = pc.get("redshifts")
        praise_config["s"] = pc.get("s", c["s"])
        praise_config["rotation_axes"] = pc.get("rotation_axes", c["rotation_axes"])
        praise_config["emis_mode"] = pc.get("emis_mode")
        praise_config["tracer_cutoff"] = pc.get("tracer_cutoff", c["tracer_cutoff"])
        if praise_config["emis_mode"]  == "xray":
            praise_config["tscope_collecting_area_m2"] = pc.get("tscope_collecting_area_m2", c["tscope_collecting_area_m2"])
            # we need this one for calculating the effective telescope area stuff. 
        
    print(praise_config)
    
    make_surface_brightness_xray(sim = sim, 
                                 parent_sim = parent_sim,
                            praise_params = praise_config,
                            grid_range = grid_range,
                            particle_data = particle_data, 
                            particle_times = particle_times, 
                            out_directory = args.out_dir)
    
    
logo = """
  _____ _____ _____ _____ _____ _____     __   __   _____
 |  _  | __  |  _  |     |   __|   __|   |  | |  | |__   |
 |   __|    -|     |-   -|__   |   __| -- |  |  |  |   __|
 |__|  |__|__|__|__|_____|_____|_____|     |_|_|   |_____|
"""
if __name__ == "__main__": 
    main()






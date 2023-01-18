# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:33:38 2021

@author: BVH
"""

import numpy as np
import datetime
import os
import scipy.stats
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import shutil

from scipy import interpolate

from useful_funcs import data_extractor, telescope_effects, optics_transmission, wavelength_array_maker, \
        sky_spectrum_load,SNR_calc,nearest,mag_calc, \
            data_extractor_TLUSTY, data_extractor_TLUSTY_joint_spec,spec_seeing, \
                atmospheric_effects,rebinner_with_bins, \
                    R_value,redshifter,order_merge_reg_grid,grid_plotter,SNR_calc_grid,SNR_calc_pred_grid,grid_plotter_opp, \
                        model_interpolator,model_interpolator_sky,fwhm_fitter_lorentzian,continuum_removal, \
                            rebinner_2d, fwhm_fitter_lorentzian_double, rebin_and_calc_SNR, \
                                pixel_sums_to_order_wavelength_converter,reduced_chi_test
from parameters import *
from flux_to_photons_conversion import photons_conversion,flux_conversion,flux_conversion_3
from apply_QE import QE
from grating import grating_orders_2_arms,grating_binning_high_enough_R
from MKID_gaussians import MKID_response_V2,recreator

#plt.rcParams.update({'font.size': 32}) #sets the fontsize of any plots
time_start = datetime.datetime.now() #beginning timer




#####################################################################################################################################################
#SPECTRUM DATA FILE IMPORT AND CONVERTING TO PHOTONS
######################################################################################################################################################
def KSIM_loop(return_only_snr=False):
    print('\nImporting spectrum from data file.')
    if TLUSTY == True:
        model_spec = data_extractor_TLUSTY_joint_spec(object_file_1,object_file_2,row,plotting=extra_plots)  
    elif supported_file_extraction == True:
        #If data that will be used is from the DRs of XShooter then set XShooter to True, since their FITS files are setup differently to the ESO-XShooter archive
        #Spectrum is in units of Flux / $ergcm^{-2}s^{-1}\AA^{-1}$, the AA is angstrom, wavelength array is in nm
        model_spec = data_extractor(object_file,Seyfert=False,plotting=extra_plots)
    else:
        model_spec_initial = np.loadtxt('%s/%s'%(folder,object_file)) #previous text file containing two columns, wavelength in nm and flux in ergcm^{-2}s^{-1}\AA^{-1} 
        
        if np.shape(model_spec_initial) != (len(model_spec_initial[:,0]),2):
            raise Exception('Format for input file should be 2 columns, one for wavelength in nm and one for flux in ergcm^{-2}s^{-1}\AA^{-1}')
                                        
        
        model_spec = np.zeros((2,len(model_spec_initial)))
        model_spec[0] += model_spec_initial[:,0]
        model_spec[1] += model_spec_initial[:,1]
    
    
    original_spec = np.copy(model_spec)
    
    if redshift > 0:
        print('\n Redshifting.')
        model_spec = redshifter(model_spec,redshift_orig,redshift)
    print('\nSimulating observation of %s.'%object_name)
    
    #sectioning the spectrum to chosen KIDSpec bandpass
    model_spec = (model_spec[0],model_spec[1] / mag_reduce) 
    
    low_c = nearest(model_spec[0],lambda_low_val,'coord')               
    high_c = nearest(model_spec[0],lambda_high_val,'coord')
    model_spec = (model_spec[0][low_c+1:high_c],model_spec[1][low_c+1:high_c])
    
    #converting data spectrum to photons and loading incoming sky spectrum from ESO sky model
    photon_spec_no_eff_original = photons_conversion(model_spec,model_spec,plotting=extra_plots)
    
    #increasing number of points in model spectrum
    photon_spec_no_eff = model_interpolator(photon_spec_no_eff_original,200000)
    
    #generating sky
    photon_spec_of_sky_orig = sky_spectrum_load(plotting=extra_plots)
    photon_spec_of_sky = model_interpolator_sky(photon_spec_of_sky_orig,200000)
    
    #calculating magnitudes of model spectrum
    SIM_obj_mags = mag_calc(model_spec,plotting=False,wls_check=True)
    
    ##############################################################################################################################################################################
    #ATMOSPHERE TRANSMISSION, TELESCOPE TRANSMISSION, SLIT EFFECTS, QE
    ##############################################################################################################################################################################
    
    print('\nApplying atmospheric, telescope, slit, and QE effects.')
    
    photon_spec_post_atmos = atmospheric_effects(photon_spec_no_eff,plotting=extra_plots,return_trans=False)
    photon_sky_post_atmos = atmospheric_effects(photon_spec_of_sky,plotting=False,return_trans=False)
    
    photon_spec_pre_optics = telescope_effects(photon_spec_post_atmos,plotting=extra_plots) 
    photon_sky_pre_optics = telescope_effects(photon_sky_post_atmos,plotting=False)
    
    photon_spec_to_instr = optics_transmission(photon_spec_pre_optics,opt_surfaces) #number input is how many optical surfaces in path
    photon_sky_to_instr = optics_transmission(photon_sky_pre_optics,opt_surfaces)
    
    if gen_model_seeing_eff == True:
        photon_spec_post_slit,seeing_transmiss_model = spec_seeing(photon_spec_to_instr,plotting=extra_plots)
        np.save('Misc/%s.npy'%model_seeing_eff_file_save_or_load,seeing_transmiss_model)
    else:
        photon_spec_post_slit = np.copy(photon_spec_to_instr)
        seeing_transmiss_model = np.load('Misc/%s.npy'%model_seeing_eff_file_save_or_load)
        photon_spec_post_slit[1] *= seeing_transmiss_model[1]
        
    print('\nModel spectrum complete')
    
    if gen_sky_seeing_eff == True:
        photon_sky_post_slit,seeing_transmiss_sky = spec_seeing(photon_sky_to_instr)
        np.save('Misc/%s.npy'%sky_seeing_eff_file_save_or_load,seeing_transmiss_sky)
    else:
        photon_sky_post_slit = np.copy(photon_sky_to_instr)
        seeing_transmiss_sky = np.load('Misc/%s.npy'%sky_seeing_eff_file_save_or_load)
        photon_sky_post_slit[1] *= seeing_transmiss_sky[1]
    print('\nSky spectrum complete.')
    
    if extra_plots == True:
            plt.figure()
            plt.plot(photon_spec_to_instr[0],photon_spec_to_instr[1],'r-',label='Pre slit')
            plt.plot(photon_spec_post_slit[0],photon_spec_post_slit[1],'b-',alpha=0.7,label='Post slit')
            plt.xlabel('Wavelength / nm')
            plt.ylabel('Photon count')
            plt.legend(loc='best')
            
            plt.figure()
            plt.plot(seeing_transmiss_model[0],seeing_transmiss_model[1],label='Slit transmission')
            plt.xlabel('Wavelength / nm')
            plt.ylabel('Transmission') 
            plt.legend(loc='best')
    
    spec_QE = QE(photon_spec_post_slit,constant=False,plotting=False)
    sky_QE = QE(photon_sky_post_slit,constant=False,plotting=False)
    
    
    
    ############################################################################################################################################################################################
    #GRATING
    ##########################################################################################################################################################################################
    
    
    print('\nSimulating the grating orders and calculating efficiencies.')
    orders,order_wavelengths,grating_efficiency,orders_opt,order_wavelengths_opt, \
        efficiencies_opt,orders_ir,order_wavelengths_ir,efficiencies_ir, \
             = grating_orders_2_arms('Casini',cutoff,plotting=extra_plots)
            
    
    #############################################################################################################################################################################################
    #BINNING PHOTONS ONTO MKIDS AND THEIR ORDER WAVELENGTHS FOR >>OPTICAL<< ARM
    #################################################################################################################################################################################################
    
    if reset_dead_pixels == True:
        try:
            print('\nRemoving dead pixel lists.')
            os.remove('DEAD_PIXEL_LISTS/%s/%i_PERC_DEAD/DEAD_PIXELS_OPT_ARM.npy'%(folder_name_dead_pixel_array,dead_pixel_perc))
            os.remove('DEAD_PIXEL_LISTS/%s/%i_PERC_DEAD/DEAD_PIXELS_IR_ARM.npy'%(folder_name_dead_pixel_array,dead_pixel_perc))
            os.remove('DEAD_PIXEL_LISTS/%s/%i_PERC_DEAD'%(folder_name_dead_pixel_array,dead_pixel_perc))
            os.remove('DEAD_PIXEL_LISTS/%s/%i_PERC_DEAD'%(folder_name_dead_pixel_array,dead_pixel_perc))
        except:
            raise Exception('Unable to locate dead pixel arrays. Double check "DEAD_PIXEL_LISTS" directory')
            
    
    if orders_opt[0] != 1:
        print('\nBinning photons for OPT/Single arm (incoming object photons).')
        
        pixel_sums_opt,order_wavelength_bins_opt,pixel_sums_opt_pre_dead_pix = grating_binning_high_enough_R(spec_QE,order_wavelengths_opt,order_wavelengths,
                                                                                                                          orders_opt,efficiencies_opt,cutoff,IR=False,OPT=True,plotting=extra_plots)
        print('\nOPT/Single arm sky photons.')
        pixel_sums_opt_sky,_,pixel_sums_opt_pre_dead_pix_sky = grating_binning_high_enough_R(sky_QE,order_wavelengths_opt,order_wavelengths,
                                                      orders_opt,efficiencies_opt,cutoff,IR=False,OPT=True,plotting=extra_plots)
        
        #adding the object and sky grids together
        pixel_sums_opt_no_sky = np.zeros_like(pixel_sums_opt)
        pixel_sums_opt_no_sky += np.copy(pixel_sums_opt)
        pixel_sums_opt += np.copy(pixel_sums_opt_sky)
        
        print('\nChecking for saturated pixels in OPT/Single arm')
        sat_pix_opt = []
        for i in range(n_pixels):
            sum_ph = np.sum(pixel_sums_opt[i]) #checking the MKIDs arent seeing too many photons
            if sum_ph > 100000*exposure_t: 
                sat_pix_opt.append([i+1,int(sum_ph)])
                print('WARNING: Pixel %i sees too many photons, %i/%i'%(i+1,int(sum_ph),int(100000*exposure_t)) )
            
            
    
    
    
    
    #############################################################################################################################################################################################
    #BINNING PHOTONS ONTO MKIDS AND THEIR ORDER WAVELENGTHS FOR >>NIR<< ARM
    #################################################################################################################################################################################################
    
    if len(orders_ir) == 0:
        orders_ir = np.append(orders_ir,200)
    if orders_ir[0] != 1:
        
        #bins the photons onto relevant MKIDs and orders
        print('\nBinning photons for NIR arm (incoming object photons).')
        
        
        pixel_sums_ir,order_wavelength_bins_ir,pixel_sums_ir_pre_dead_pix = grating_binning_high_enough_R(spec_QE,order_wavelengths_ir,order_wavelengths,
                                                                                                                    orders_ir,efficiencies_ir,cutoff,IR=True,OPT=False,
                                                                                                                        plotting=extra_plots)
        print('\nNIR arm sky photons.')
        pixel_sums_ir_sky,_,pixel_sums_ir_pre_dead_pix_sky = grating_binning_high_enough_R(sky_QE,order_wavelengths_ir,
                                                                    order_wavelengths,orders_ir,efficiencies_ir,
                                                                    cutoff,IR=True,OPT=False,plotting=extra_plots)
        
        #adding the object and sky grids together
        pixel_sums_ir_no_sky = np.zeros_like(pixel_sums_ir)
        pixel_sums_ir_no_sky += pixel_sums_ir
        pixel_sums_ir += pixel_sums_ir_sky
        
        print('\nChecking for saturated pixels in NIR arm')
        sat_pix_ir = []
        for i in range(n_pixels):
            sum_ph = np.sum(pixel_sums_ir[i]) #checking the MKIDs arent seeing too many photons
            if sum_ph > 100000*exposure_t: 
                sat_pix_opt.append([i+1,int(sum_ph)])
                print('WARNING: Pixel %i sees too many photons, %i/%i'%(i+1,int(sum_ph),int(100000*exposure_t)) )
    
    
    #############################################################################################################################################################################################
    #SIMULATING MKID RESPONSES USING GAUSSIAN METHOD
    #################################################################################################################################################################################################
    
    if reset_R_Es == True:
        print('\nResetting MKID energy resolution spread files.')
        
        try:
            os.remove('R_E_PIXELS/%s/%i_SCALE/R_E_PIXELS_IR.npy'%(folder_name_R_E_spread_array,r_e_spread))
            os.remove('R_E_PIXELS/%s/%i_SCALE/R_E_PIXELS_OPT.npy'%(folder_name_R_E_spread_array,r_e_spread))
        except:
            try:
                os.mkdir('R_E_PIXELS/%s/%i_SCALE/'%(folder_name_R_E_spread_array,r_e_spread))
            except:
                os.mkdir('R_E_PIXELS/%s/'%(folder_name_R_E_spread_array))
                os.mkdir('R_E_PIXELS/%s/%i_SCALE/'%(folder_name_R_E_spread_array,r_e_spread))
        default_R_Es = np.ones(n_pixels)*ER_band_low
        np.save('R_E_PIXELS/%s/%i_SCALE/R_E_PIXELS_IR.npy'%(folder_name_R_E_spread_array,r_e_spread),default_R_Es)
        np.save('R_E_PIXELS/%s/%i_SCALE/R_E_PIXELS_OPT.npy'%(folder_name_R_E_spread_array,r_e_spread),default_R_Es)
    
    print('\nBeginning MKID response simulation for each arm and simultaneous sky exposure.')
    
    #OPT object mkid response
    kidspec_resp_opt,kidspec_mis_opt = MKID_response_V2(spec_QE,orders_opt,order_wavelengths,order_wavelengths_opt,n_pixels,pixel_sums_opt,
                      IR=False,sky=False,dual_arm_=IR_arm)
    print('\nOPT object observation complete. 1/4')
    
    #OPT sky mkid response
    kidspec_sky_resp_opt,kidspec_sky_mis_opt = MKID_response_V2(sky_QE,orders_opt,order_wavelengths,order_wavelengths_opt,n_pixels,pixel_sums_opt_sky,
                      IR=False,sky=True,dual_arm_=IR_arm)
    print('\nOPT sky observation complete. 2/4')
    
    
    if IR_arm == True:
        #NIR object mkid response
        kidspec_resp_ir,kidspec_mis_ir = MKID_response_V2(spec_QE,orders_ir,order_wavelengths,order_wavelengths_ir,n_pixels,pixel_sums_ir,
                          IR=True,sky=False,dual_arm_=IR_arm)
        print('\nNIR object observation complete. 3/4')
        
        #NIR sky mkid response
        kidspec_sky_resp_ir,kidspec_sky_mis_ir = MKID_response_V2(sky_QE,orders_ir,order_wavelengths,order_wavelengths_ir,n_pixels,pixel_sums_ir_sky,
                          IR=True,sky=True,dual_arm_=IR_arm)
        print('\nNIR sky observation complete. 4/4')
    else:
        print('\nNIR arm not selected, observations complete. 4/4')
    
    
    print('\nFinalising MKID response grids.')
    if IR_arm == True:
        kidspec_raw_output = np.zeros_like(order_wavelengths)
        incoming_photon_per_pixels = np.zeros_like(order_wavelengths)
        
        kidspec_raw_output[:len(order_list_ir)] += kidspec_resp_ir
        kidspec_raw_output[len(order_list_ir):] += kidspec_resp_opt
        incoming_photon_per_pixels[:len(order_list_ir)] += np.rot90(pixel_sums_ir_pre_dead_pix,k=3)+np.rot90(pixel_sums_ir_pre_dead_pix_sky,k=3)
        incoming_photon_per_pixels[len(order_list_ir):] += np.rot90(pixel_sums_opt_pre_dead_pix,k=3)+np.rot90(pixel_sums_opt_pre_dead_pix_sky,k=3)
        
        misidentified_spectrum = np.zeros_like(order_wavelengths)
        misidentified_spectrum[:len(order_list_ir)] += kidspec_mis_ir
        misidentified_spectrum[len(order_list_ir):] += kidspec_mis_opt
        
        kidspec_raw_output_sky = np.zeros_like(order_wavelengths)
        kidspec_raw_output_sky[:len(order_list_ir)] += kidspec_sky_resp_ir
        kidspec_raw_output_sky[len(order_list_ir):] += kidspec_sky_resp_opt
        
        misidentified_sky_spectrum = np.zeros_like(order_wavelengths)
        misidentified_sky_spectrum[:len(order_list_ir)] += kidspec_sky_mis_ir
        misidentified_sky_spectrum[len(order_list_ir):] += kidspec_sky_mis_opt
        
        
        
    else:
        kidspec_raw_output = np.zeros_like(order_wavelengths)
        misidentified_spectrum = np.zeros_like(order_wavelengths)
        kidspec_raw_output_sky = np.zeros_like(order_wavelengths)
        misidentified_sky_spectrum = np.zeros_like(order_wavelengths)
        incoming_photon_per_pixels = np.zeros_like(order_wavelengths)
        
        kidspec_raw_output += kidspec_resp_opt
        misidentified_spectrum += kidspec_mis_opt
        kidspec_raw_output_sky += kidspec_sky_resp_opt
        misidentified_sky_spectrum += kidspec_sky_mis_opt
        incoming_photon_per_pixels += pixel_sums_to_order_wavelength_converter(pixel_sums_opt_pre_dead_pix)+pixel_sums_to_order_wavelength_converter(pixel_sums_opt_pre_dead_pix_sky)
            
    #percentage_misidentified_tot = (abs(np.sum(kidspec_raw_output+kidspec_raw_output_sky) \
    #                                    - np.sum(misidentified_spectrum+misidentified_sky_spectrum)) \
    #                                        / np.sum(kidspec_raw_output+kidspec_raw_output_sky) )*100
        
    #percentage_misidentified_pp = np.mean(abs((kidspec_raw_output+kidspec_raw_output_sky) \
    #                                    - (misidentified_spectrum+misidentified_sky_spectrum)) \
    #                                        / (kidspec_raw_output+kidspec_raw_output_sky) )*100
    
    percentage_misidentified_tot = (np.sum(abs(kidspec_raw_output-incoming_photon_per_pixels)) / np.sum(abs(incoming_photon_per_pixels)) )*100
    per_pixel = np.sum(abs(kidspec_raw_output-incoming_photon_per_pixels),axis=0) / np.sum(abs(incoming_photon_per_pixels),axis=0)
    percentage_misidentified_pp = np.median(per_pixel*100)
    
    
    
    #############################################################################################################################################################################################
    #SKY SUBTRACTION, ORDER MERGING AND SNR CALCULATION
    #################################################################################################################################################################################################
    
    print('\nSubtracting sky.')
    #subtracting sky
    raw_sky_subbed_spec_pre_ord_merge = np.zeros_like(kidspec_raw_output)
    raw_sky_subbed_spec_pre_ord_merge += (kidspec_raw_output - kidspec_raw_output_sky)
    
    if extra_plots == True:
        grid_plotter(order_wavelengths,raw_sky_subbed_spec_pre_ord_merge)
        grid_plotter(order_wavelengths,kidspec_raw_output_sky)
    
    #SNR calculation
    SNR_total,SNRs = SNR_calc_grid(raw_sky_subbed_spec_pre_ord_merge,kidspec_raw_output_sky,plotting=False)
    predicted_SNRs_x = SNR_calc_pred_grid(raw_sky_subbed_spec_pre_ord_merge,kidspec_raw_output_sky,SOX=False,plotting=False) #X-Shooter
    predicted_SNRs_s = SNR_calc_pred_grid(raw_sky_subbed_spec_pre_ord_merge,kidspec_raw_output_sky,SOX=True,plotting=False) #SOXS
    av_SNR_x = np.median(predicted_SNRs_x[(order_wavelengths > model_spec[0][0])*(order_wavelengths < model_spec[0][-1])])
    av_SNR_s = np.median(predicted_SNRs_s[(order_wavelengths > model_spec[0][0])*(order_wavelengths < model_spec[0][-1])])
    SNR_av = np.median(SNRs[(order_wavelengths > model_spec[0][0])*(order_wavelengths < model_spec[0][-1])])
    
    if return_only_snr == True:
        return SNR_av,SNR_total
    
    #calculating spectral resolution of KIDSpec setup
    Rs = []
    for i in range(len(order_wavelengths)):
        R1 = order_wavelengths[i][0] / (order_wavelengths[i][1]-order_wavelengths[i][0])
        Rs.append(R1)
        R2 = order_wavelengths[i][-1] / (order_wavelengths[i][-1]-order_wavelengths[i][-2])
        Rs.append(R2)
    Rs = np.asarray(Rs)/slit_R_factor
    R_high = np.max(Rs)
    R_low = np.min(Rs)
    spec_R_pre_cutoff = np.arange(lambda_low_val,cutoff,lambda_low_val/R_high)
    spec_R_post_cutoff = np.arange(cutoff,lambda_high_val,lambda_high_val/R_low)
    spec_R = np.append(spec_R_pre_cutoff,spec_R_post_cutoff)
    
    print('\nMerging orders.')
    #merging orders here onto large regular grid
    raw_sky_subbed_spec = order_merge_reg_grid(order_wavelengths,raw_sky_subbed_spec_pre_ord_merge) 
    
    #rebinning to KIDSpec's current spectral resolution
    #raw_sky_subbed_spec = rebinner_with_bins(raw_sky_subbed_spec_pre_bin,spec_R)
    
    #############################################################################################################################################################################################
    #STANDARD STAR FACTORS GENERATION OR APPLICATION
    #################################################################################################################################################################################################
    
    SIM_total_flux_spectrum = flux_conversion_3(raw_sky_subbed_spec)
    
    if stand_star_factors_run == False: #loading standard star factors
        print('\nLoading standard star weights and applying them.')
        stand_star_spec = np.load('STANDARD_STAR_FACTORS_STORE/DATA_SPECTRUM_GD71_%i_%spix%s.npy'%(reg_grid_factor,n_pixels,stand_star_filename_detail))
        factors = np.zeros_like(stand_star_spec)
        factors[0] += stand_star_spec[0]
        factors[1] += np.load('STANDARD_STAR_FACTORS_STORE/FACTORS_STAND_STAR_GD71_%i_%spix%s.npy'%(reg_grid_factor,n_pixels,stand_star_filename_detail))[1]
        
        SIM_flux_pre_weights = np.copy(SIM_total_flux_spectrum)
        
        #applying standard star weights and flux conversion
        SIM_total_flux_spectrum[1] /= factors[1]
        
        corrected_KS_spec = np.copy(raw_sky_subbed_spec)
        corrected_KS_spec[1] /= factors[1]
        
    elif stand_star_factors_run == True: #generating standard star factors
        print('\nGenerating standard star weights.')
        model_func = np.zeros_like(SIM_total_flux_spectrum)
        model_func[0] += SIM_total_flux_spectrum[0]
        model_func[1] += np.interp(SIM_total_flux_spectrum[0],model_spec[0],model_spec[1])
        
        factors = np.zeros_like(SIM_total_flux_spectrum)
        factors[0] += SIM_total_flux_spectrum[0]
        factors[1] += SIM_total_flux_spectrum[1] / model_func[1]
        
        np.save('STANDARD_STAR_FACTORS_STORE/FACTORS_STAND_STAR_GD71_%i_%spix%s.npy'%(reg_grid_factor,n_pixels,stand_star_filename_detail),factors)
        np.save('STANDARD_STAR_FACTORS_STORE/DATA_SPECTRUM_GD71_%i_%spix%s.npy'%(reg_grid_factor,n_pixels,stand_star_filename_detail),model_func)
        
        SIM_flux_pre_weights = np.copy(SIM_total_flux_spectrum)
    
        SIM_total_flux_spectrum[1] /= factors[1]
        
        corrected_KS_spec = np.copy(raw_sky_subbed_spec)
        corrected_KS_spec[1] /= factors[1]
    
    else:
        raise Exception('STANDARD STAR FACTOR RUN OPTION NOT SELECTED: TRUE OR FALSE')
    
    '''
    #plotting the effects of the standard star factors
    plt.figure()
    plt.plot(SIM_flux_pre_weights[0],SIM_flux_pre_weights[1],'b-',label='Raw KSIM output')
    plt.plot(model_spec[0],model_spec[1],'r-',label='Model spectrum')
    plt.plot(SIM_total_flux_spectrum[0],SIM_total_flux_spectrum[1],'g-',alpha=0.6,label='Standard star weights applied')
    plt.legend(loc='best')
    plt.xlabel('Wavelength / nm')
    plt.ylabel(object_y)
    '''
    
    #############################################################################################################################################################################################
    #MISCELLEANOUS ANALYSIS
    #################################################################################################################################################################################################
    
    coord_low = nearest(corrected_KS_spec[0],model_spec[0][0],'coord')
    coord_high = nearest(corrected_KS_spec[0],model_spec[0][-1],'coord')
    
    corrected_KS_spec = corrected_KS_spec[:,coord_low+1:coord_high]
    SIM_total_flux_spectrum_before_binning = np.copy(SIM_total_flux_spectrum)
    SIM_total_flux_spectrum = SIM_total_flux_spectrum[:,coord_low+1:coord_high]
    
    #rebinning back to model spectrum 
    try:
        SIM_rebin_to_data = rebinner_with_bins(corrected_KS_spec,model_spec[0])
    
    except:
        SIM_rebin_to_data = rebinner_with_bins(corrected_KS_spec[:,1:-1],model_spec[0])
    
    #flux conversion
    SIM_total_flux_spectrum_model_bins_pre_filt = flux_conversion_3(SIM_rebin_to_data)
    SIM_total_flux_spectrum_model_bins = np.zeros_like(SIM_total_flux_spectrum_model_bins_pre_filt)
    SIM_total_flux_spectrum_model_bins[0] += SIM_total_flux_spectrum_model_bins_pre_filt[0]
    SIM_total_flux_spectrum_model_bins[1] += SIM_total_flux_spectrum_model_bins_pre_filt[1]
    
    #magnitude calculation from simulation result
    SIM_out_mags = mag_calc(SIM_total_flux_spectrum,plotting=False,wls_check=False)
    
    #R value statistic between model and simulation output
    R_value_stat = R_value(SIM_total_flux_spectrum_model_bins,model_spec,plotting=extra_plots)
    
    #FWHM calculator
    if fwhm_fitter == True:
        SIM_spec_continuum_removed = continuum_removal(SIM_total_flux_spectrum_model_bins,poly=cont_rem_poly)
        model_spec_continuum_removed = continuum_removal(model_spec,poly=cont_rem_poly)
        
        
        nan_removed_spec = np.copy(SIM_spec_continuum_removed)
        nan_removed_model = np.copy(model_spec_continuum_removed)
        nan_coords = np.isnan(nan_removed_spec[1])
        nan_coords +=  np.isinf(nan_removed_spec[1])
        nan_coords = [not t for t in nan_coords]
        nan_removed_spec_y = nan_removed_spec[1][nan_coords]
        nan_removed_spec_x = nan_removed_spec[0][nan_coords]
        nan_removed_model_y = nan_removed_model[1][nan_coords]
        nan_removed_model_x = nan_removed_model[0][nan_coords]
        
        SIM_spec_continuum_removed = np.array([nan_removed_spec_x,nan_removed_spec_y])
        model_spec_continuum_removed = np.array([nan_removed_model_x,nan_removed_model_y])
        
        coord_feature = nearest(model_spec_continuum_removed[0],cen_wl,'coord')
        coord_range = coord_feature - nearest(model_spec_continuum_removed[0],cen_wl-12,'coord')
        
        if double_fit == True:
            print('\nFitting result for KSIM:')
            coord_feature_2 =  nearest(SIM_total_flux_spectrum_model_bins[0],cen_wl_2,'coord')
            fwhm,fwhm_err,r_val_fit,chi_val_fit = fwhm_fitter_lorentzian_double(SIM_spec_continuum_removed[0,coord_feature-coord_range:coord_feature+coord_range],
                                               SIM_spec_continuum_removed[1,coord_feature-coord_range:coord_feature+coord_range],
                                               cen_wl,SIM_spec_continuum_removed[1][coord_feature],
                                               SIM_spec_continuum_removed[1][coord_feature],0.5,
                                               cen_wl_2,SIM_spec_continuum_removed[1][coord_feature_2],
                                               SIM_spec_continuum_removed[1][coord_feature_2],0.5)
            
            print('\nFitting result for model:')
            fwhm_model,fwhm_err_model,_,_ = fwhm_fitter_lorentzian_double(model_spec_continuum_removed[0][coord_feature-coord_range:coord_feature+coord_range],
                                                   model_spec_continuum_removed[1][coord_feature-coord_range:coord_feature+coord_range],
                                                   cen_wl,model_spec_continuum_removed[1][coord_feature],
                                                   model_spec_continuum_removed[1][coord_feature],0.5,
                                                   cen_wl_2,model_spec_continuum_removed[1][coord_feature_2],
                                                   model_spec_continuum_removed[1][coord_feature_2],0.5)
        
        else:
            print('\nFitting result for KSIM:')
            fwhm,fwhm_err,r_val_fit,chi_val_fit = fwhm_fitter_lorentzian(SIM_spec_continuum_removed[0,coord_feature-coord_range:coord_feature+coord_range],
                                                   SIM_spec_continuum_removed[1,coord_feature-coord_range:coord_feature+coord_range],
                                                   cen_wl,SIM_spec_continuum_removed[1][coord_feature],
                                                   SIM_spec_continuum_removed[1][coord_feature],0.5)
            print('\nFitting result for model:')
            fwhm_model,fwhm_err_model,_,_ = fwhm_fitter_lorentzian(model_spec_continuum_removed[0][coord_feature-coord_range:coord_feature+coord_range],
                                                   model_spec_continuum_removed[1][coord_feature-coord_range:coord_feature+coord_range],
                                                   cen_wl,model_spec_continuum_removed[1][coord_feature],
                                                   model_spec_continuum_removed[1][coord_feature],0.5)
    
    
    #############################################################################################################################################################################################
    #RESIDUALS
    #################################################################################################################################################################################################
    residuals = (np.nan_to_num(SIM_total_flux_spectrum_model_bins[1]) - model_spec[1]) / model_spec[1]
    res1 = abs(residuals)
    nans= np.isnan(res1)
    res1[nans] = 0
    infs = np.isinf(res1)
    res1[infs] = 0
    
    residuals_av = np.median(res1[res1!=1.0])
    residuals_spread = scipy.stats.median_absolute_deviation(res1)
    
    res2 = np.copy(residuals)
    nans= np.isnan(res2)
    res2[nans] = 0
    infs = np.isinf(res2)
    res2[infs] = 0
    res2 = res2[res2!=0]
    #plt.figure()
    #plt.hist(res2,bins=100)
    #plt.xlabel('Fractional Residuals')
    #plt.ylabel('Count')
    
    residuals_fwhm = (2.355*np.sqrt(np.var(res2[res2!=1.0])))
    
    
    
    time_took = datetime.datetime.now() - time_start
    
    print('\n Simulation took', time_took,'(hours:minutes:seconds)')
    


    nan_coords = np.isnan(SIM_total_flux_spectrum[1])
    nan_coords = [not t for t in nan_coords]
    from useful_funcs import reduced_chi_test_2
    upscaled_model = np.zeros((2,len(SIM_total_flux_spectrum[0])))
    upscaled_model[0] += SIM_total_flux_spectrum[0]
    upscaled_model[1] += np.interp(SIM_total_flux_spectrum[0],model_spec[0],model_spec[1])
    sky_ph = model_interpolator_sky(photon_spec_of_sky_orig,len(SIM_total_flux_spectrum[0]))
    factors_2 = factors[:,coord_low+1:coord_high]
    data_ph = photons_conversion(SIM_total_flux_spectrum,SIM_total_flux_spectrum)
    model_ph = photons_conversion(upscaled_model,upscaled_model)
    error_ph = np.sqrt((data_ph[1]/factors_2[1]) + (sky_ph[1]/factors_2[1]))*reg_grid_factor
    R_unbinned = R_value(data_ph,model_ph)

    return R_unbinned,R_value_stat,SNR_av,SNR_total
    






snrs_only = False
loops = 5





r_unbins = np.zeros(loops)
r_binneds = np.zeros(loops)
SNRs = np.zeros(loops)
SNR_tots = np.zeros(loops)
for i in range(loops):
    if snrs_only == True:
        snr,snr_tot = KSIM_loop(return_only_snr=True)
    else:
        r_unbin,r_binned,snr,snr_tot = KSIM_loop()
        r_unbins[i] += r_unbin
        r_binneds[i] += r_binned
        print(r_unbins[i])
    SNRs[i] += snr
    SNR_tots[i] += snr_tot

    print('\n\n################################################\n')
    print('\n %i / %i loops complete.'%(i+1,loops))
    print('\n\n################################################\n')
    
print('\nR unbinned\n')
print(r_unbins)

print('\nR binned\n')
print(r_binneds)


print('Dead pix perc:',dead_pixel_perc)
print('Re spread:',r_e_spread)

np.save(r'C:\Users\BVH\Documents\KineticInductanceDetectors\Own Papers\Paper_17032022\r_unbins_saved_%ivar_%i_dead_1800s.npy'%(r_e_spread,dead_pixel_perc),r_unbins)
np.save(r'C:\Users\BVH\Documents\KineticInductanceDetectors\Own Papers\Paper_17032022\r_bins_saved_%ivar_%i_dead_1800s.npy'%(r_e_spread,dead_pixel_perc),r_binneds)
np.save(r'C:\Users\BVH\Documents\KineticInductanceDetectors\Own Papers\Paper_17032022\snrs_saved_%ivar_%i_dead_1800s.npy'%(r_e_spread,dead_pixel_perc),SNRs)
np.save(r'C:\Users\BVH\Documents\KineticInductanceDetectors\Own Papers\Paper_17032022\snrs_tots_saved_%ivar_%i_dead_1800s.npy'%(r_e_spread,dead_pixel_perc),SNR_tots)




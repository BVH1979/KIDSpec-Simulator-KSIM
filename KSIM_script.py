# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:49:08 2023

@author: BVH
"""





import datetime
import pickle
import numpy as np
#import matplotlib.pyplot as plt
from grating import *
from parameters import *
from useful_funcs import rebin_and_calc_SNR,nearest

################################################################################################################


binary = False
steps = 100
filename_low_res = 'ksim_spectra_100steps_T498g775_T100g825_14_4_Vmag_4500specR'
filename_high_res = 'ksim_spectra_100steps_T498g775_T100g825_14_4_Vmag'

lim_mag_run = True


################################################################################################################


time_start = datetime.datetime.now()

if binary == True:
    
    max_steps = 400
    row = 0
    
    step_skip = int(np.round(max_steps/steps,decimals=0))
    
    ksim_sim_spectra = []
    ksim_sim_spectra_low_res = []
    
    for i in range(steps):
        from KSIM_module import KSIM_main
        ksim = KSIM_main(suppress_plots=True)
        ksim.run_sim(row_change=row)
        
        ksim_sim_spectra_low_res.append(ksim.SIM_rebin_lowerR)
        ksim_sim_spectra.append(ksim.SIM_total_flux_spectrum)
        
        file_param = open('SETUP_KSIM/KSIM_INPUT_PARAMETERS_B.txt','r')
        params = file_param.readlines()
        file_param.close()
        params[38] = params[38].replace(params[38].split(' ')[2],str(row+step_skip))
        file_param = open('SETUP_KSIM/KSIM_INPUT_PARAMETERS_B.txt','w')
        file_param.writelines(params)
        file_param.close()
        
        row += step_skip
        
        print('\n##############################################################\n\n')
        print(' %i / %i spectra complete...'%(i+1,steps))
        print(' Time taken so far:',datetime.datetime.now() - time_start)
        print('\n\n##############################################################\n')
        
        if row > max_steps-1:
            break
    
    with open(r'C:\Users\BVH\Documents\KineticInductanceDetectors\KIDSpec_Simulator\binary_system_analysis\binary_analysis_data_and_info\%s.pkl'%filename_low_res, 'wb') as file:  
        pickle.dump(ksim_sim_spectra_low_res, file)
        file.close()
    with open(r'C:\Users\BVH\Documents\KineticInductanceDetectors\KIDSpec_Simulator\binary_system_analysis\binary_analysis_data_and_info\%s.pkl'%filename_high_res, 'wb') as file:  
        pickle.dump(ksim_sim_spectra, file)
        file.close()



elif lim_mag_run == True:
    from KSIM_module import KSIM_main
    print('\nBeginning limiting magnitude simulation...\n')
    print('\nExposure time:',exposure_t)
    print('\nTelescope diameter:',mirr_diam)
    print('\nMKID pixels:',n_pixels)
    print('\nRe:',ER_band_low,'@',ER_band_wl,'nm')
    
    print('\n Simulating the grating orders and calculating efficiencies.')
    orders,order_wavelengths,grating_efficiency,orders_opt,order_wavelengths_opt, \
        efficiencies_opt,orders_ir,order_wavelengths_ir,efficiencies_ir, \
             = grating_orders_2_arms('Casini',cutoff,plotting=extra_plots)
    
    mag_central_wls = np.array([  358.5       ,   458.        ,   598.5       ,   820.        ,
         943.        ,   999.5       ,  1020.        ,  1250.12503051,
         1650.13000489,  2150.2199707 ,  3800.70495605,  4700.94506836,
         10500.        , 20000.        ])
    
    
    blaze_coord = int(np.round(n_pixels/2,decimals=0))
    
    mag_central_wls = np.array([  358.5       ,   458.        ,   598.5       ,   820.        ,
             943.        ,   999.5       ,  1020.        ,  1250.12503051,
             1650.13000489,  2150.2199707 ,  3800.70495605,  4700.94506836,
             10500.        , 20000.        ])
    
    
    
    blaze_wls_mag = np.zeros((len(order_wavelengths[:,0]),3))
    blaze_wls_mag[:,0] += order_wavelengths[:,blaze_coord]
    
    mag_coord_match_for_wls = np.zeros(len(order_wavelengths[:,0]))
    for i in range(len(order_wavelengths[:,0])):
        mag_coord_match_for_wls[i] += int(nearest(mag_central_wls,order_wavelengths[i,blaze_coord],'coord'))
    
    
    stop_condition = 0
    blaze_wl_mag_found_check = np.zeros(len(order_wavelengths[:,0]))
    
    mag_reduce_fac = mag_reduce #starting magnitude reduction
    
    while stop_condition != len(blaze_wl_mag_found_check):
    
        print('\nCurrent magnitude reduction factor: %.3f'%mag_reduce_fac)
        print('\nCurrent wavelengths found: %i / %i'%(stop_condition,len(blaze_wl_mag_found_check)))
        print(blaze_wls_mag)
        print('\nCurrent SNRs found:')
        try:
            print(SNRs_blaze)
        except:
            print('\nWill begin printing SNRs found after initial run.')
        

        ksim = KSIM_main(suppress_plots=True,input_mag_reduce=mag_reduce_fac)
        ksim.run_sim()
        SNRs = ksim.SNRs
        current_mag = ksim.SIM_obj_mags
        
        SNRs_blaze = np.zeros((len(order_wavelengths[:,0]),2))
        SNRs_blaze[:,0] += order_wavelengths[:,blaze_coord]
        SNRs_blaze[:,1] += SNRs[:,blaze_coord]
        
        if rebin_lower_specR == True:
            SNRs_rebin,wls_rebin = rebin_and_calc_SNR(ksim.raw_sky_subbed_spec_pre_ord_merge,
                                                      ksim.kidspec_raw_output_sky,
                                                      order_wavelengths,rebin_specR)
            new_blaze_coord = int(len(wls_rebin[0])/2)
            SNRs_blaze = np.zeros((len(order_wavelengths[:,0]),2))
            SNRs_blaze[:,0] += wls_rebin[:,new_blaze_coord]
            SNRs_blaze[:,1] += SNRs_rebin[:,new_blaze_coord]
            SNRs = np.copy(SNRs_rebin)
        
        for i in range(len(SNRs_blaze[:,0])):
            if SNRs_blaze[i,1] < 5 and blaze_wl_mag_found_check[i] == 0:
                blaze_wls_mag[i,1] += current_mag[0][int(mag_coord_match_for_wls[i])]
                blaze_wls_mag[i,2] += SNRs_blaze[i,1]
                blaze_wl_mag_found_check[i] += 1
                stop_condition += 1
        
        mag_reduce_fac_prev = np.copy(mag_reduce_fac)
        if mag_reduce_fac >= 100:
            mag_reduce_fac += 20
        if mag_reduce_fac >= 500:
            mag_reduce_fac += 30
        if mag_reduce_fac >= 1000:
            mag_reduce_fac += 200
        if mag_reduce_fac >= 5000:
            mag_reduce_fac += 250
        if mag_reduce_fac >= 15000:
            mag_reduce_fac += 50
        if mag_reduce_fac >= 20000:
            mag_reduce_fac += 2000
        if mag_reduce_fac_prev == mag_reduce_fac:
            mag_reduce_fac += 10.0
                
        print('\nCurrent time taken:',datetime.datetime.now() - time_start)
    
    time_took = datetime.datetime.now() - time_start

    print('\n Simulation took', time_took,'(hours:minutes:seconds)')

else:
    from KSIM_module import KSIM_main
    ksim = KSIM_main(suppress_plots=False)
    ksim.run_sim()







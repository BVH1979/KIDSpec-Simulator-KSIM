# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:32:18 2023

@author: BVH
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import pickle
from scipy.interpolate import interp1d



class order_optimiser:
    
    def __init__(self):
        
        ###################################################################################################################
        #Loading in data files for bands and sky/atmos data
        #####################################################################################################################
        self.sky_radiance = np.loadtxt('data_files/radiance_sky.txt')
        self.sky_transmission = np.loadtxt('data_files/transmission_sky.txt')
        self.sky_rad_func = interp1d(self.sky_radiance[:,0],1-self.sky_radiance[:,1]/max(self.sky_radiance[:,1]))
        self.sky_trans_func = interp1d(self.sky_transmission[:,0],self.sky_transmission[:,1])

        self.h_band = np.loadtxt(r'data_files\Passbands\H_bandpass.txt')
        self.h_band[:,0] *= 1000
        self.j_band = np.loadtxt(r'data_files\Passbands\J_bandpass.txt')
        self.j_band[:,0] *= 1000
        self.u_band = np.loadtxt(r'data_files\Passbands\bandpass_u.txt')
        self.u_band[:,1] /= 100
        self.b_band = np.loadtxt(r'data_files\Passbands\bandpass_b.txt')
        self.b_band[:,1] /= 100
        self.g_band = np.loadtxt(r'data_files\Passbands\bandpass_g.txt')
        self.g_band[:,1] /= 100
        self.v_band = np.loadtxt(r'data_files\Passbands\bandpass_v.txt')[375:,:]
        self.v_band[:,1] /= 100
        self.r_band = np.loadtxt(r'data_files\Passbands\bandpass_r.txt')
        self.i_band = np.loadtxt(r'data_files\Passbands\bandpass_i.txt')
        self.k_band = np.loadtxt(r'data_files\Passbands\K_bandpass.txt')
        self.k_band[:,0] *= 1000
        
        return
    
    

    def overlap_counter(orders,wls): #tracks how many wavelengths overlap across grating orders
        counter = 0
        for i in range(len(orders[:,0])):
            if orders[i,1]+orders[i,3] > wls[1] and orders[i,1]-orders[i,3] < wls[0]:
                counter += 1
        if counter == 0:
            counter += 1 
        return counter
    
    def nearest(x,value,val_or_coord): #finds nearest value in array to set value or coordinate
        coord = np.abs(x-value).argmin()
        val = x[coord]
        if val_or_coord == 'coord':
            return coord
        if val_or_coord == 'val':
            return val
    
    def order_information(energy_res,n_pix,order_max_wl,spectral_res,sig): #calculates various grating orders information
        max_order = int(energy_res/sig)
        order_1_wl = max_order * order_max_wl
        orders = np.zeros((max_order,11))
        orders[:,0] += np.arange(1,max_order+1,1)
        
        for i in range(max_order):
            orders[i,1] += order_1_wl / orders[i,0] #central wavelength
            orders[i,2] += orders[i,1] / (2*spectral_res) #2 is for nyquist sampling, this is the dispersion
            orders[i,4] += orders[i,2] * n_pix #wavelength range from pixels
            orders[i,3] += (n_pix*orders[i,2]) / 2 #wavelength range in order from pixels from one half 
        
        for i in range(max_order):
            orders[i,5] +=  (0.5 * ( orders[i,1]/(orders[i,2]*orders[i,0]) ))*orders[i,2] #FSR
            orders[i,6] +=  (1 * ( orders[i,1]/(orders[i,2]*orders[i,0]) )) #number of pixels in FSR
            orders[i,7] +=  (orders[i,1]-orders[i,3]) / orders[i,2]#
            orders[i,8] +=  (orders[i,1]+orders[i,3]) / orders[i,2]#spectral R at each end of pixel array in order
            orders[i,9] +=  (orders[i,1]-orders[i,5]) / orders[i,2]#
            orders[i,10] +=  (orders[i,1]+orders[i,5]) / orders[i,2]#spectral R at each end of FSR in order
        
        return orders
    
    def scorer(orders,spec_r,n_pix,re,ord_opt_obj,weight_transmission=None,weight_radiance=None,h_band=False,k_band=False): #calculates the score for a given setup of number of number of pixels, spectral resolution, and energy resolution
        prevent_nan = False
        max_sky_rad = max(ord_opt_obj.sky_radiance[:,1])
        
        if weight_transmission is None:
            weight_transmission = 3
        if weight_radiance is None:
            weight_radiance = 1
        
        if h_band == True:
            max_wl = 1900 
        elif k_band == True:
            max_wl = 2450 
        else:
            max_wl = 1500 
        min_wl = 330
        
        score = 0 
        sky_coverage = np.zeros(len(ord_opt_obj.sky_transmission[:,0]))
        for i in range(len(orders[:,4])):
            #scoring whether there is a consistent wavelength coverage, bright sky lines, atmospheric transmission, the amount of pixels within the FSR
            
            if orders[i,1]+orders[i,3] < max_wl and orders[i,1]-orders[i,3] > min_wl: #checking whether the current order is completely within the wavelength range
                coord_low = order_optimiser.nearest(ord_opt_obj.sky_transmission[:,0],orders[i,1]-orders[i,3],'coord')
                coord_high = order_optimiser.nearest(ord_opt_obj.sky_transmission[:,0],orders[i,1]+orders[i,3],'coord')
                sky_coverage[coord_low:coord_high] += 1
                fsr_factor = orders[i,6] / n_pix 
                if fsr_factor > 1:
                    fsr_factor = 1
                order_appearance_count = order_optimiser.overlap_counter(orders,[orders[i,1]-orders[i,3],orders[i,1]+orders[i,3]])
                score += spec_r/(
                    (1 + (np.mean((ord_opt_obj.sky_radiance[coord_low:coord_high,1]/max_sky_rad))*weight_radiance)
                    + np.mean(1-ord_opt_obj.sky_transmission[coord_low:coord_high,1])*weight_transmission ) * order_appearance_count / (fsr_factor)
                    )
            
            elif min_wl < orders[i,1]+orders[i,3] < max_wl: #if the current order wavelength range spans outside the set minimum wavelength
                coord_low = order_optimiser.nearest(ord_opt_obj.sky_transmission[:,0],min_wl,'coord')
                coord_high = order_optimiser.nearest(ord_opt_obj.sky_transmission[:,0],orders[i,1]+orders[i,3],'coord')
                fraction_order_in = ((orders[i,1]+orders[i,3])-min_wl) / ((orders[i,1]+orders[i,3])-(orders[i,1]-orders[i,3]))
                sky_coverage[coord_low:coord_high] += 1
                fsr_factor = orders[i,6] / n_pix 
                if fsr_factor > 1:
                    fsr_factor = 1
                order_appearance_count = order_optimiser.overlap_counter(orders,[orders[i,1]-orders[i,3],orders[i,1]+orders[i,3]])
                score += spec_r/(
                    (1 + (np.mean((ord_opt_obj.sky_radiance[coord_low:coord_high,1]/max_sky_rad))*weight_radiance)
                    + np.mean(1-ord_opt_obj.sky_transmission[coord_low:coord_high,1])*weight_transmission ) * order_appearance_count / (fraction_order_in*fsr_factor)
                    )
                
            elif min_wl < orders[i,1]-orders[i,3] < max_wl: #if the current order wavelength range spans outside the set maximum wavelength
                coord_low = order_optimiser.nearest(ord_opt_obj.sky_transmission[:,0],orders[i,1]-orders[i,3],'coord')
                coord_high = order_optimiser.nearest(ord_opt_obj.sky_transmission[:,0],max_wl,'coord')
                fraction_order_in = (max_wl-(orders[i,1]-orders[i,3])) / ((orders[i,1]+orders[i,3])-(orders[i,1]-orders[i,3]))
                sky_coverage[coord_low:coord_high] += 1
                fsr_factor = orders[i,6] / n_pix 
                if fsr_factor > 1:
                    fsr_factor = 1
                order_appearance_count = order_optimiser.overlap_counter(orders,[orders[i,1]-orders[i,3],orders[i,1]+orders[i,3]])
                score += spec_r/(
                    (1 + (np.mean((ord_opt_obj.sky_radiance[coord_low:coord_high,1]/max_sky_rad))*weight_radiance)
                    + np.mean(1-ord_opt_obj.sky_transmission[coord_low:coord_high,1])*weight_transmission ) * order_appearance_count / (fraction_order_in*fsr_factor)
                    )
                
    
        coord_max = order_optimiser.nearest(ord_opt_obj.sky_transmission[:,0],max_wl,'coord')
        sky_coverage = sky_coverage[:coord_max]
        sky_rad_short = np.copy(ord_opt_obj.sky_radiance)[:coord_max,:]
        sky_trans_short = np.copy(ord_opt_obj.sky_transmission)[:coord_max,:]
        
        len_sky_short_0 = len(sky_rad_short[sky_coverage==0,1])
        if len_sky_short_0 == 0:
            len_sky_short_0 = 1
            prevent_nan = True
        
        #rewarding the setup if 'bad' bits of sky are not covered
        scorer_not_covered = np.zeros(len_sky_short_0)
        scorer_not_covered += 1-len_sky_short_0
        scorer_not_covered[scorer_not_covered>0.5] = 1
        scorer_not_covered[scorer_not_covered<0.5] = 0
        score *= 1+(np.sum(scorer_not_covered)/len_sky_short_0)
        scorer_not_covered = np.zeros(len_sky_short_0)
        
        try: 
            scorer_not_covered += sky_trans_short[sky_coverage==0,1]
        except:
            scorer_not_covered += 0
            
        scorer_not_covered[scorer_not_covered>0.5] = 1
        scorer_not_covered[scorer_not_covered<0.5] = 0
        score *= 1+(np.sum(len(scorer_not_covered[scorer_not_covered==0])) / len_sky_short_0)
        
        #adding score for covering as close to the min wavelength as possible
        coverage_factor = 1
        for i in range(len(sky_coverage)):
            if sky_coverage[i] == 0:
                coverage_factor += 0
            else:
                break
        if coverage_factor > 1:
            score /= ((coverage_factor))
        #reducing score for poor coverage
        if prevent_nan == False:
            score /= ( 1 * ( 
                np.mean((1-(sky_rad_short[sky_coverage==0,1]/max_sky_rad))
                        + ((sky_trans_short[sky_coverage==0,1]))))
                        )
        
        return score   
    
    def order_optimiser_varying_number_of_mkids(order_max_wl,var,energy_res,no_points,sig,ord_opt_obj,
                                                weight_transmission=None,weight_radiance=None,h=False,k=False):
        
        spectral_res = var #varying parameter is entered here
        scores_spec_res = np.zeros(no_points)
        n_pix = np.linspace(500,10000,no_points)
        
        for i in range(no_points):
            orders_info = order_optimiser.order_information(energy_res,n_pix[i],order_max_wl,spectral_res,sig)
            scores_spec_res[i] += order_optimiser.scorer(orders_info,spectral_res,n_pix[i],energy_res,ord_opt_obj,
                                                         weight_transmission=None,weight_radiance=None,h_band=h,k_band=k)
            print('\rCurrent wavelength progress: %i%%'%(i*100/no_points),end='',flush=True)
            
        score_results = np.zeros((2,no_points))
        score_results[0] += n_pix
        score_results[1] += scores_spec_res
        
        return score_results
    
    
    
    def order_optimiser_varying_spectral_resolution(order_max_wl,var,energy_res,no_points,sig,ord_opt_obj,
                                                    weight_transmission=None,weight_radiance=None,h=False,k=False):
        
        spectral_res = np.linspace(100,15000,no_points)
        scores_spec_res = np.zeros(no_points)
        n_pix = var
        
        for i in range(no_points):
            orders_info = order_optimiser.order_information(energy_res,n_pix,order_max_wl,spectral_res[i],sig)
            scores_spec_res[i] += order_optimiser.scorer(orders_info,spectral_res[i],n_pix,energy_res,ord_opt_obj,
                                                         weight_transmission=None,weight_radiance=None,h_band=h,k_band=k)
            print('\rCurrent wavelength progress: %i%%'%(i*100/no_points),end='',flush=True)
            
        score_results = np.zeros((2,no_points))
        score_results[0] += spectral_res
        score_results[1] += scores_spec_res
        
        return score_results
        
        
        
    
    
    
    def run_order_optimiser(ord_opt_obj,energy_res,x_axis_vars,specr_or_npixels,no_points_x,no_points_y,sig_sep,
                            weight_transmission=None,weight_radiance=None,h_band_active=True,k_band_active=False):
        
        count = 0
        
        if specr_or_npixels == 1:
            what_varying = 'Number of MKIDs'
            what_not_varying = 'Spectral Resolution'
        elif specr_or_npixels == 0:
            what_varying = 'Spectral Resolution'
            what_not_varying = 'Number of MKIDs'
        else:
            raise Exception('`specr_or_npixels` must be 0 or 1, for varying number of MKIDs and spectral resolution respectively.')
        
        for var in x_axis_vars:
                
            y_var = np.linspace(300,500,no_points_y) 
            scores_wls = []
                
            if specr_or_npixels == 1:
                for i in range(len(y_var)):
                    scores_wls.append(order_optimiser.order_optimiser_varying_number_of_mkids(y_var[i],var,energy_res,no_points_x,sig_sep,ord_opt_obj,
                                                                                              weight_transmission=None,weight_radiance=None,h=h_band_active,k=k_band_active))    
                    print('\nY variables complete: %i / %i'%(i+1,len(y_var)))
                    
            elif specr_or_npixels == 0:
                for i in range(len(y_var)):
                    scores_wls.append(order_optimiser.order_optimiser_varying_spectral_resolution(y_var[i],var,energy_res,no_points_x,sig_sep,ord_opt_obj,
                                                                                                  weight_transmission=None,weight_radiance=None,h=h_band_active,k=k_band_active))    
                    print('\nY variables complete: %i / %i'%(i+1,len(y_var)))
            
            
            scores_2d_array = np.zeros((len(scores_wls),len(scores_wls[0][1])))
            
            for i in range(len(scores_wls)):
                scores_2d_array[i] += np.nan_to_num(scores_wls[i][1]) 
                scores_2d_array[i][scores_2d_array[i] > 1e308] = 0
                
            np.save('scorer_grating_orders_%i_%s_%i_re_%i_sig.npy'%(var,what_varying,energy_res,sig_sep),scores_2d_array)
            count += 1
            print('\n##################################################################')
            print('\nSets complete %i / %i'%(count,len(x_axis_vars)))
            print('\n##################################################################')


        max_score_vals = []
        if specr_or_npixels == 1:
            max_score_vals.append(['Spectral resolution','Y variable','Number of MKID pixels'])
        elif specr_or_npixels == 0:
            max_score_vals.append(['Number of MKID pixels','Y variable','Spectral resolution'])
        
        max_coords = []

        y_var *= int(energy_res/sig_sep)

        for i in range(len(x_axis_vars)):
            
            if specr_or_npixels == 1:
                x_axis = np.linspace(500,10000,no_points_x)
            elif specr_or_npixels == 0:
                x_axis = np.linspace(100,15000,no_points_x)
            
            max_coords.append(np.unravel_index(np.argmax(scores_2d_array, axis=None), scores_2d_array.shape))
            max_score_vals.append([x_axis_vars[i],y_var[max_coords[i][0]],x_axis[max_coords[i][1]]])
            
            
            plt.figure()
            plt.imshow(scores_2d_array, extent=(x_axis.min(), x_axis.max(), y_var.max(), y_var.min()),interpolation='none', cmap='plasma',aspect='auto',norm=colors.LogNorm())
            plt.colorbar()
            plt.ylabel('1st order central wavelength / nm')
            plt.xlabel(what_varying)
            plt.title('%i %s with energy resolution %i'%(x_axis_vars[i],what_not_varying,energy_res))
            

        return scores_2d_array,max_score_vals









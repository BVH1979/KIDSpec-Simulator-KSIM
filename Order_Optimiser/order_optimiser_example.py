# -*- coding: utf-8 -*-
"""
Created on Sun May 28 21:51:17 2023

@author: BVH
"""

import numpy as np
from order_optimiser_core import order_optimiser


energy_res = 30 #energy resolution of MKIDs at 350nm 
constant_parameter = [3000] #values of parameter which is constant, i.e. spectral resolution or number of mkids
specr_or_npixels = 0 #set to 1 for varying number of MKIDs, 0 for varying spectral resolution
no_points_x = 10 #amount of points to sample for either spectral resolution or number of MKIDs
no_points_y = 10 #amount of points to sameple for grating order wavelength placements 
sig_sep = 2 #sigma separation of orders on MKIDs, functionally decides how many orders the MKIDs will separate
weight_transmission=3 #weighting of atmospheric transmission score, the value here multiplies the score recieved for atmospheric transmission
weight_radiance=1     #same as weight_transmission, instead for the sky radiance score

ord_opt = order_optimiser()

scores_2d_array,max_scores_vals = order_optimiser.run_order_optimiser(ord_opt,energy_res,constant_parameter,
                                                      specr_or_npixels,no_points_x,
                                                      no_points_y,sig_sep,
                                                      weight_transmission=None,weight_radiance=None,
                                                      h_band_active=True,k_band_active=False)















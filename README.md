# KIDSpec-Simulator-KSIM
Observation simulator for the upcoming KIDSpec instrument, described in the paper 'KSIM: simulating KIDSpec, a Microwave Kinetic Inductance Detector
spectrograph for the optical/NIR'.


# KSIM how to use

## 05/06/2023

Please send feedback to `benedict.hofmann@durham.ac.uk`.<br>


## Initialising desired simulation
<br>

When simulating an observation with KSIM all changes can be done within the `SETUP_KSIM/KSIM_INPUT_PARAMETERS.txt` file. Included below are all parameters which can be altered for a astronomical target object observation simulation using KSIM. Where appropriate, a requirement or range for the value of the parameter is also included. <br><br>

|Parameter|Description|
|------|-------|
| object_name | Name of astronomical target object being simulated within KSIM. |
|object_file | Name of file containing spectrum of astronomical target object, structured in the form of two columns with wavelength (nm) and flux ($ergcm^{-2}s^{-1}\text{\r{A}}^{-1}$). |
|binstep | The size in nm of the bins in the Object File spectrum.|
|mirr_diam | Diameter in cm of the primary mirror of the telescope.|
|central_obscuration} | Percentage obscuration to the primary mirror of the telescope. |
|seeing | Value of the atmospheric seeing, in arcseconds.  |
|exposure_t | Exposure time of simulated observation in seconds. |
|tele_file | Text file containing two columns, wavelength in nm and percentage reflectance of telescope mirror material. |
|lambda_low_val | Minimum wavelength for simulated KIDSpec bandpass. Minimum value of 350nm and maximum value of 2999nm.|
|lambda_high_val | Maximum wavelength for simulated KIDspec bandpass. Minimum value of 351nm and maximum value of 3000nm.|
|n_pixels | Number of MKID pixels in linear array for KIDSpec. Minimum value greater than zero.|
|alpha_val | Incidence angle of incoming light to grating in degrees. |
|phi_val | Reflected central angle of incoming light to grating in degrees.|
|refl_deg | Reflected angle range of incoming light passed to MKIDs in degrees.     |                                                        
|grooves | Number of grooves on grating per mm.         |                          
|norders | Number of grating orders to test for incoming wavelengths. | 
|number_optical_surfaces | Number of optical surfaces in KIDSpec instrument between primary mirror and MKIDs. The GEMINI silver mirrors reflectance is used here.|
|folder_dir | Folder path where all other files can be found and where results are saved to.|
|fudicial_energy_res | Energy resolution used to calculate energy resolution at all other wavelengths. |
|fudicial_wavelength | Wavelength used to calculate energy resolution at all other wavelengths. |              
|coincidence_rejection_time | The coincidence rejection time, in $\mu$s, used for MKID saturation calculations for both the PTS and Order Gaussian methods. |
|raw_sky_file | FITS file containing the sky background, can be generated using ESO SKYCALC.|       
|slit_width | Width of slit in arcseconds. |                    
|pixel_fov | FoV of MKID pixels in arcseconds. |          
|off_centre | Sets the distance target object is from the centre of the slit in arcseconds. Can be set to zero or greater.|  
|airmass | Airmass of atmosphere. |                    
|dead_pixel_perc | Percentage of MKIDs which are considered dead. Value can be set in the range 0-100.|        
|R_E_spread | Standard deviation value of normal distribution used to generate spread of $R_{E}$. Can be set to zero or greater.|               
|redshift | Desired redshift of target object.| 
|redshift_orig | Original redshift of target object.|
|mag_reduce | Factor which reduces incoming flux from simulated target. Can be set to <1 for an increase in flux.|       
|generate_sky_seeing_eff | Generates transmission file, containing transmission of sky spectrum though slit. |      
|sky_seeing_eff_file_save_or_load | Name of sky seeing transmission file to either save or load.|      
|generate_model_seeing_eff | Generates transmission file, containing transmission of target object spectrum though slit|                               
|model_seeing_eff_file_save_or_load | Name of target object seeing transmission file to either save or load.|    
|generate_additional_plots | Plots additional steps throughout KSIM, including photon spectra at various stages such as atmosphere, telescope, and grating orders.|                
|generate_standard_star_factors | Generates standard star spectral weights.|          
|stand_star_run_filename_details | Name of standard star spectral weights to either save or load.|
|fwhm_fitter | Option to use a Lorentzian shape fitter for spectral features, up to two features at once.|
|fwhm_fitter_central_wavelength | Central wavelength of a Lorentzian shaped line.|
|fwhm_fitter_central_wavelength_2 | Central wavelength of a second Lorentzian shaped line.|  
|double_fitter | If two lines are to be fitted then this is set to True.|
|continuum_removal_use_polynomial | If True a polynomial will be fitted to the spectrum to remove the spectrum continuum. If False a linear fit is used. |    
|reset_R_E_spread_array | When True generates new energy resolution spreads. |         
|reset_dead_pixel_array | When True generates dead pixel spreads. |           


## Using KSIM
<br>

KSIM can be run using the `KSIM_script.py` script which will produce plots and a results text file. The structure of KSIM is modular so can be altered easily. For example included is `KSIM_Lim_Mag_Determine.py` can be used to determine what limiting magnitudes a particular KIDSpec design could achieve.<br><br>

If the observation parameters are changed beyond the object being 'observed' (such as number of MKIDs) or this is the first run using KSIM, new standard star factors will need to be generated. This can be done by setting `generate_standard_star_factors` to True and simulating the standard star GD71 using the GD71.txt file in `STANDARD_STAR/`. Note the other changes to the parameters file which will be needed such as the correct `folder_dir` input. <br><br>

Once these factors have been generated any other appropriate file can be simulated. If two TLUSTY spectra for a binary simulation in a `.npy` are chosen then this can be entered in the parameters file. Otherwise combine them externally and enter it as a `.txt` file normally in `object_file`.<br><br>








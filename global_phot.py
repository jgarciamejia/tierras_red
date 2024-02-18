import numpy as np 
from glob import glob 
import os 
import pandas as pd 
import copy 
from scipy.stats import sigmaclip
import matplotlib.pyplot as plt 
plt.ion()

'''
	Scripts for doing global light curves of Tierras targets
'''

def return_dataframe_onedate(mainpath,targetname,obsdate,ffname): #JGM MODIFIED NOV 9/2023
	datepath = os.path.join(mainpath,obsdate,targetname,ffname)
	optimal_lc_fname = os.path.join(datepath,'optimal_lc.txt') #returns only optimal aperture lc
	try:
		optimal_lc_csv = open(optimal_lc_fname).read().rstrip('\n')
		df = pd.read_csv(optimal_lc_csv)
	except FileNotFoundError:
		print ("No photometric extraction for {} on {}".format(targetname,obsdate))
		return None
	return df, optimal_lc_csv

def return_dataframe_onedate_forapradius(mainpath,targetname,obsdate,ffname,ap_radius='optimal'): #JGM MODIFIED JAN3/2024: return dataframe for a user-defined aperture.
	if ap_radius == 'optimal': # simply use optimal radius according to ap_phot. Needs work. 
			df,lc_fname = return_dataframe_onedate(mainpath,targetname,obsdate,ffname)
			print('Optimal ap radius: ',lc_fname.split('_')[-1].split('.csv')[0])
			return df,lc_fname
	else:
			datepath = os.path.join(mainpath,obsdate,targetname,ffname)
			lc_fname = os.path.join(datepath,'circular_fixed_ap_phot_{}.csv'.format(str(ap_radius)))
			print (ap_radius)
			try:
				df = pd.read_csv(lc_fname)
			except:
				print ("No photometric extraction for {} with aperture radius of {} pixels on {}".format(targetname,ap_radius,obsdate))
				return None
			return df, lc_fname

def make_global_lists(mainpath,targetname,ffname,exclude_dates=[], ap_radius='optimal'):
	# arrays to hold the full dataset
	full_bjd = []
	# full_flux = []
	# full_err = []
	# full_flux_div_expt = [] # sometimes data from one star has different exposure times in a given night or between nights
	# full_err_div_expt = []
	
	full_flux = None
	full_err = None 
	full_flux_div_expt = None 
	full_err_div_expt = None 
	# full_reg = None
	# full_reg_err = None

	full_relflux = []
	full_exptime = []
	full_sky = None
	full_x = None
	full_y = None
	full_airmass = []
	full_fwhm_x = None
	full_fwhm_y = None
	full_humidity = []
	full_dome_humidity = []
	full_ccd_temp = []
	full_dome_temp = []
	full_focus = []
	full_sec_temp = []
	full_ret_temp = []
	full_pri_temp = []
	full_rod_temp = []
	full_cab_temp = []
	full_inst_temp = []
	full_temp = []
	full_dewpoint = []
	full_sky_temp = []
	full_pressure = []
	full_ret_pressure = []
	full_supp_pressure = []
	full_ha = []
	full_dome_az = []
	full_wind_spd = []
	full_wind_gust = []
	full_wind_dir = []

	#full_corr_relflux = [] 

	# array to hold individual nights
	bjd_save = []
	lcfolderlist = np.sort(glob(mainpath+"/**/"+target))
	lcdatelist = [lcfolderlist[ind].split("/")[4] for ind in range(len(lcfolderlist))] 
	breakpoint()
	for ii,lcfolder in enumerate(lcfolderlist):
		print("Processing", lcdatelist[ii])

		# if date excluded, skip
		if np.any(exclude_dates == lcdatelist[ii]):
			print ("{} :  Excluded".format(lcdatelist[ii]))
			continue

		# read the .csv file
		df, optimal_lc = return_dataframe_onedate_forapradius(mainpath,targetname, lcdatelist[ii], ffname, ap_radius)

		n_sources = int(df.keys()[-1].split(' ')[1])

		bjds = df['BJD TDB'].to_numpy()
		flux = np.zeros((n_sources,len(bjds)))
		err = np.zeros_like(flux)
		x = np.zeros_like(flux)
		y = np.zeros_like(flux)
		sky = np.zeros_like(flux)
		fwhm_x = np.zeros_like(flux)
		fwhm_y = np.zeros_like(flux)
		
		expt = df['Exposure Time']
		for i in range(n_sources):
			if i == 0:
				source = 'Target'
			else:
				source = f'Ref {i}'
			flux[i] = df[f'{source} Source-Sky ADU'] / expt
			err[i] = df[f'{source} Source-Sky Error ADU'] / expt
			x[i] = df[f'{source} X']
			y[i] = df[f'{source} Y']
			sky[i] = df[f'{source} Sky ADU'] / expt
			fwhm_x[i] = df[f'{source} X FWHM Arcsec']
			fwhm_y[i] = df[f'{source} Y FWHM Arcsec']

		airmass = df['Airmass']
		humidity = df['Humidity']
		dome_humidity = df['Dome Humidity']
		ccd_temp = df['CCD Temperature']
		dome_temp = df['Dome Temperature']
		focus = df['Focus']
		sec_temp = df['Sec Temperature']
		pri_temp = df['Pri Temperature']
		ret_temp = df['Ret Temperature']
		rod_temp = df['Rod Temperature']
		cab_temp = df['Cab Temperature']
		inst_temp  = df['Instrument Temperature']
		temp = df['Temperature']
		dewpoint = df['Dewpoint']
		sky_temp = df['Sky Temperature']
		pressure = df['Pressure']
		ret_pressure = df['Return Pressure']
		supp_pressure = df['Supply Pressure']
		ha = df['Hour Angle']
		dome_az = df['Dome Azimuth']
		wind_spd = df['Wind Speed']
		wind_gust = df['Wind Gust']
		wind_dir = df['Wind Direction']

		# # get the comparison fluxes.
		# comps = {}
		# comps_err = {}
		# fwhm_x = {}
		# fwhm_y = {}
		# fwhm_x[0] = df[f'Source {source_ind+1} X FWHM Arcsec']
		# fwhm_y[0] = df[f'Source {source_ind+1} Y FWHM Arcsec']
		# for comp_num in complist:
		# 	try:
		# 		fwhm_x[comp_num] = df['Source '+str(comp_num+1)+' X FWHM Arcsec']
		# 		fwhm_y[comp_num] = df['Source '+str(comp_num+1)+' Y FWHM Arcsec']
		# 		comps[comp_num] = df['Source '+str(comp_num+1)+' Source-Sky ADU'] / expt  # divide by exposure time since it can vary between nights
		# 		comps_err[comp_num] = df['Source '+str(comp_num+1)+' Source-Sky Error ADU'] / expt
		# 	except:
		# 		print("Error with comp", str(comp_num+1))
		# 		continue

		# # make a list of all the comps
		# regressors = []
		# regressors_err = []
		# fwhm_x_ = []
		# fwhm_y_ = []
		# fwhm_x_.append(fwhm_x[0])
		# fwhm_y_.append(fwhm_y[0])
		# for key in comps.keys():
		# 	regressors.append(comps[key])
		# 	regressors_err.append(comps_err[key])
		# 	fwhm_x_.append(fwhm_x[key])
		# 	fwhm_y_.append(fwhm_y[key])


		# regressors = np.array(regressors)
		# regressors_err = np.array(regressors_err)

		# fwhm_x = np.array(fwhm_x_)
		# fwhm_y = np.array(fwhm_y_)

		# add this night of data to the full data set
		full_bjd.extend(bjds)
		# full_flux.extend(flux)
		# full_err.extend(err)
		# full_flux_div_expt.extend(flux/expt)
		# full_err_div_expt.extend(err/expt)		
		bjd_save.append(bjds)

		# full_relflux.extend(relflux)
		full_exptime.extend(expt)
		# full_sky.extend(sky/expt)
		# full_x.extend(x)
		# full_y.extend(y)
		
		full_airmass.extend(airmass)
		full_humidity.extend(humidity)
		full_dome_humidity.extend(dome_humidity)
		full_ccd_temp.extend(ccd_temp)
		full_dome_temp.extend(dome_temp)
		full_focus.extend(focus)
		full_sec_temp.extend(sec_temp)
		full_pri_temp.extend(pri_temp)
		full_ret_temp.extend(ret_temp)
		full_rod_temp.extend(rod_temp)
		full_cab_temp.extend(cab_temp)
		full_inst_temp.extend(inst_temp)
		full_temp.extend(temp)
		full_dewpoint.extend(dewpoint)
		full_sky_temp.extend(sky_temp)
		full_pressure.extend(pressure)
		full_ret_pressure.extend(ret_pressure)
		full_supp_pressure.extend(supp_pressure)
		full_ha.extend(ha)
		full_dome_az.extend(dome_az)
		full_wind_spd.extend(wind_spd)
		full_wind_gust.extend(wind_gust)
		full_wind_dir.extend(wind_dir)

		if full_flux is None:
			full_flux = flux
			full_flux_err = err
			full_x_pos = x
			full_y_pos = y 
			full_fwhm_x = fwhm_x
			full_fwhm_y = fwhm_y
		else:
			full_flux = np.concatenate((full_flux, flux), axis=1) 
			full_flux_err = np.concatenate((full_flux_err, err), axis=1)
			full_x_pos = np.concatenate((full_x_pos, x), axis=1)
			full_y_pos = np.concatenate((full_y_pos, y), axis=1)
			full_fwhm_x = np.concatenate((full_fwhm_x, fwhm_x),axis=1)
			full_fwhm_y = np.concatenate((full_fwhm_y, fwhm_y),axis=1)

	# convert from lists to arrays
	full_bjd = np.array(full_bjd)
	full_flux = np.array(full_flux)
	full_flux_err = np.array(full_flux_err)
	#full_reg_err = np.array(full_reg_err)
	# full_flux_div_expt = np.array(full_flux_div_expt)
	# full_err_div_expt =np.array(full_err_div_expt)
	# full_relflux = np.array(full_relflux)
	full_exptime = np.array(full_exptime)
	full_sky = np.array(full_sky)
	full_x = np.array(full_x_pos)
	full_y = np.array(full_y_pos)
	full_airmass = np.array(full_airmass)
	full_fwhm_x = np.array(full_fwhm_x)
	full_fwhm_y = np.array(full_fwhm_y)
	full_humidity = np.array(full_humidity)
	full_dome_humidity = np.array(full_dome_humidity)
	full_ccd_temp = np.array(full_ccd_temp)
	full_dome_temp = np.array(full_dome_temp)
	full_focus = np.array(full_focus)
	full_sec_temp = np.array(full_sec_temp)
	full_pri_temp = np.array(full_pri_temp)
	full_ret_temp = np.array(full_ret_temp)
	full_rod_temp = np.array(full_rod_temp)
	full_cab_temp = np.array(full_cab_temp)
	full_inst_temp = np.array(full_inst_temp)
	full_temp = np.array(full_temp)
	full_dewpoint = np.array(full_dewpoint)
	full_sky_temp = np.array(full_sky_temp)
	full_pressure = np.array(full_pressure)
	full_ret_pressure  = np.array(full_ret_pressure)
	full_supp_pressure  = np.array(full_supp_pressure)
	full_ha  = np.array(full_ha)
	full_dome_az  = np.array(full_dome_az)
	full_wind_spd  = np.array(full_wind_spd)
	full_wind_gust  = np.array(full_wind_gust)
	full_wind_dir = np.array(full_wind_dir)

	output_dict = {'BJD':full_bjd, 'BJD List':bjd_save, 'Flux':full_flux, 'Flux Error':full_flux_err, 
				'Exptime':full_exptime, 'Sky':full_sky, 'X':full_x, 'Y':full_y, 'Airmass':full_airmass,
				'FWHM X':full_fwhm_x, 'FWHM Y':full_fwhm_y, 'Humidity':full_humidity, 'Dome Humidity':full_dome_humidity, 'CCD Temp':full_ccd_temp, 'Dome Temp':full_dome_temp,
				'Focus':full_focus, 'Secondary Temp':full_sec_temp, 'Primary Temp':full_pri_temp,
				'Return Temp':full_ret_temp, 'Rod Temp':full_rod_temp, 'Cabinet Temp':full_cab_temp,
				'Instrument Temp':full_inst_temp, 'Temp': full_temp, 'Dewpoint':full_dewpoint, 
				'Sky Temp':full_sky_temp, 'Pressure':full_pressure, 'Return Pressure':full_ret_pressure,
				'Supply Pressure':full_supp_pressure, 'Hour Angle':full_ha, 'Dome Azimuth':full_dome_az,
				'Wind Speed':full_wind_spd, 'Wind Gust':full_wind_gust, 'Wind Direction':full_wind_dir}

	return  output_dict

def mearth_style_pat_weighted(data_dict, cluster_ids):
	""" Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
	""" it's called "mearth_style" because it's inspired by the mearth pipeline """
	cluster_ids = np.array(cluster_ids)

	bjds = data_dict['BJD']
	flux = data_dict['Flux']
	flux_err = data_dict['Flux Error']

	# mask any cadences where the flux is negative for any of the sources 
	mask = np.ones_like(bjds, dtype='bool')  # initialize a bad data mask
	for i in range(len(flux)):
		mask[np.where(flux[i] <= 0)[0]] = 0  

	flux_corr_save = np.zeros_like(flux)
	flux_err_corr_save = np.zeros_like(flux)
	mask_save = np.zeros_like(flux)
	weights_save = np.zeros((len(flux),len(flux)-1))

	# loop over each star, calculate its zero-point correction using the other stars
	for i in range(len(flux)):
		target_source_id = cluster_ids[i] # this represents the ID of the "target" *in the photometry files
		regressor_inds = [j for j in np.arange(len(flux)) if i != j] # get the indices of the stars to use as the zero point calibrators; these represent the indices of the calibrators *in the data_dict arrays*
		regressor_source_ids = cluster_ids[regressor_inds] # these represent the IDs of the calibrators *in the photometry files*  

		# grab target and source fluxes and apply initial mask 
		target_flux = data_dict['Flux'][i]
		target_flux_err = data_dict['Flux Error'][i]
		regressors = data_dict['Flux'][regressor_inds]
		regressors_err = data_dict['Flux'][regressor_inds]

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 

		tot_regressor = np.sum(regressors, axis=0)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure
		tot_regressor[~mask] = np.nan

		c0s = -2.5*np.log10(np.nanpercentile(tot_regressor, 90)/tot_regressor)  # initial guess of magnitude zero points
		
		mask = np.ones_like(c0s, dtype='bool')  # initialize another bad data mask
		mask[np.where(c0s < -0.24)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 

		# repeat the cs estimate now that we've masked out the bad cadences
		phot_regressor = np.nanpercentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
		cs = -2.5*np.log10(phot_regressor[:,None]/regressors)  # estimate c for each star
		c_noise = np.nanstd(cs, axis=0)  # estimate the error in c
		c_unc = (np.nanpercentile(cs, 84, axis=0) - np.nanpercentile(cs, 16, axis=0)) / 2.  # error estimate that ignores outliers

		''' c_unc will overestimate error introduced by zero-point offset because it is correlated. Attempt to correct
		for this by only considering the additional error compared to the cadence where c_unc is minimized '''
		c_unc_best = np.nanmin(c_unc)
		c_unc = np.sqrt(c_unc**2 - c_unc_best**2)

		# Initialize weights using average fluxes of the regressors
		# weights_init = np.nanmean(regressors, axis=1)
		# weights_init /= np.nansum(weights_init) # Normalize weights to sum to 1

		# give all stars equal weights at first
		weights_init = np.ones(len(regressors))/len(regressors)

		cs = np.matmul(weights_init, cs) # Take the *weighted mean* across all regressors

		# one more bad data mask: don't trust cadences where the regressors have big discrepancies
		mask = np.ones_like(target_flux, dtype='bool')
		mask[np.where(c_noise > 3*np.median(c_noise))[0]] = 0

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 
		c_unc[~mask] = np.nan

		cs_original = cs
		delta_weights = np.zeros(len(regressors))+999 # initialize
		threshold = 1e-4 # delta_weights must converge to this value for the loop to stop
		weights_old = weights_init
		full_ref_inds = np.arange(len(regressors))
		while len(np.where(delta_weights>threshold)[0]) > 0:
			stddevs = np.zeros(len(regressors))
			cs = -2.5*np.log10(phot_regressor[:,None]/regressors)

			for jj in range(len(regressors)):
				use_inds = np.delete(full_ref_inds, jj)
				weights_wo_jj = weights_old[use_inds]
				weights_wo_jj /= np.nansum(weights_wo_jj)
				cs_wo_jj = np.matmul(weights_wo_jj, cs[use_inds])
				corr_jj = regressors[jj] * 10**(-cs_wo_jj/2.5)
				# reg_flux, intercept, coeffs, ancillary_dict_return = regression(corr_jj, ancillary_dict, pval_threshold=1e-3, verbose=False)
				corr_jj /= np.nanmedian(corr_jj)
				stddevs[jj] = np.nanstd(corr_jj)

			weights_new = 1/stddevs**2
			weights_new /= np.nansum(weights_new)
			delta_weights = abs(weights_new-weights_old)
			weights_old = weights_new

		weights = weights_new

		# determine if any references should be totally thrown out based on the ratio of their measured/expected noise
		regressors_err_norm = (regressors_err.T / np.nanmedian(regressors,axis=1)).T
		noise_ratios = stddevs / np.nanmedian(regressors_err_norm)      

		# the noise ratio threshold will depend on how many bad/variable reference stars were used in the ALC
		# sigmaclip the noise ratios and set the upper limit to the n-sigma upper bound 
		v, l, h = sigmaclip(noise_ratios, 2, 2)
		weights[np.where(noise_ratios>h)[0]] = 0
		weights /= sum(weights)
		
		if len(np.where(weights == 0)[0]) > 0:
			# now repeat the weighting loop with the bad refs removed 
			delta_weights = np.zeros(len(regressors))+999 # initialize
			threshold = 1e-6 # delta_weights must converge to this value for the loop to stop
			weights_old = weights
			full_ref_inds = np.arange(len(regressors))
			count = 0
			while len(np.where(delta_weights>threshold)[0]) > 0:
				stddevs = np.zeros(len(regressors))
				cs = -2.5*np.log10(phot_regressor[:,None]/regressors)

				for jj in range(len(regressors)):
					if weights_old[jj] == 0:
						continue
					use_inds = np.delete(full_ref_inds, jj)
					weights_wo_jj = weights_old[use_inds]
					weights_wo_jj /= np.nansum(weights_wo_jj)
					cs_wo_jj = np.matmul(weights_wo_jj, cs[use_inds])
					corr_jj = regressors[jj] * 10**(-cs_wo_jj/2.5)
					corr_jj /= np.nanmean(corr_jj)
					stddevs[jj] = np.nanstd(corr_jj)
				weights_new = 1/(stddevs**2)
				weights_new /= np.sum(weights_new[~np.isinf(weights_new)])
				weights_new[np.isinf(weights_new)] = 0
				delta_weights = abs(weights_new-weights_old)
				weights_old = weights_new
				count += 1

		weights = weights_new
		
		# calculate the zero-point correction
		cs = -2.5*np.log10(phot_regressor[:,None]/regressors)
		cs = np.matmul(weights, cs)
		
		corrected_regressors = regressors * 10**(-cs/2.5)
		
		# flux_original = copy.deepcopy(flux)
		err_corr = 10**(cs/(-2.5)) * np.sqrt(target_flux_err**2 + (c_unc*target_flux*np.log(10)/(-2.5))**2)  # propagate error
		flux_corr = target_flux*10**(cs/(-2.5))  #cs, adjust the flux based on the calculated zero points

		mask_save[i] = ~np.isnan(flux_corr)
		flux_corr_save[i] = flux_corr
		flux_err_corr_save[i] = err_corr
		weights_save[i] = weights

	output_dict = copy.deepcopy(data_dict)
	output_dict['ZP Mask'] = mask_save
	output_dict['Corrected Flux'] = flux_corr_save
	output_dict['Corrected Flux Error'] = flux_err_corr_save
	output_dict['Weights'] = weights_save

	return output_dict

if __name__ == '__main__':
	target = 'TIC384984325'
	data_path = '/data/tierras/lightcurves/'
	ffname = 'flat0000'
	ap_radius = 8
	
	data_dict = make_global_lists(data_path, target, ffname, ap_radius=ap_radius)
#!/usr/bin/env python

import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import ImageNormalize, ZScaleInterval, simple_norm
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, interpolate_replace_nans
from astropy.coordinates import SkyCoord, get_moon
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from astropy.table import Table, QTable
from astropy.nddata import NDData
import astropy.units as u 
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
from photutils import make_source_mask
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import BasicPSFPhotometry, IntegratedGaussianPRF, DAOGroup, extract_stars, EPSFBuilder
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, EllipticalAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic
from scipy.stats import sigmaclip, pearsonr, linregress
from scipy.spatial.distance import cdist
from scipy.signal import correlate2d, fftconvolve, savgol_filter
from copy import deepcopy
import argparse
import os 
import stat
import sys
import lfa
import time
import astroalign as aa
#import reproject as rp
import sep 
from fitsutil import *
from pathlib import Path
from sklearn import linear_model
import copy
import batman
from glob import glob
import pickle
import shutil

def psf_photometry(file_list, targ_and_refs, an_in=30, an_out=50, centroid=False, live_plot=False):
	ffname = file_list[0].parent.name	
	target = file_list[0].parent.parent.name
	date = file_list[0].parent.parent.parent.name 

	#file_list = file_list[-2:] #TESTING!!!
	
	DARK_CURRENT = 0.19 #e- pix^-1 s^-1
	NONLINEAR_THRESHOLD = 40000. #ADU
	SATURATION_THRESHOLD = 55000. #ADU
	PLATE_SCALE = 0.43 #arcsec pix^-1, from Juliana's dissertation Table 1.1
	
	#Set up arrays for doing photometry 

	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH FILE
	filenames = []
	mjd_utc = np.zeros(len(file_list),dtype='float')
	jd_utc = np.zeros(len(file_list),dtype='float')
	bjd_tdb = np.zeros(len(file_list),dtype='float')
	airmasses = np.zeros(len(file_list),dtype='float16')
	ccd_temps = np.zeros(len(file_list),dtype='float16')
	exp_times = np.zeros(len(file_list),dtype='float16')
	dome_temps = np.zeros(len(file_list),dtype='float16')
	focuses = np.zeros(len(file_list),dtype='float16')
	dome_humidities = np.zeros(len(file_list),dtype='float16')
	sec_temps = np.zeros(len(file_list),dtype='float16')
	ret_temps = np.zeros(len(file_list),dtype='float16')
	pri_temps = np.zeros(len(file_list),dtype='float16')
	rod_temps = np.zeros(len(file_list),dtype='float16')
	cab_temps = np.zeros(len(file_list),dtype='float16')
	inst_temps = np.zeros(len(file_list),dtype='float16')
	temps = np.zeros(len(file_list),dtype='float16')
	humidities = np.zeros(len(file_list),dtype='float16')
	dewpoints = np.zeros(len(file_list),dtype='float16')
	sky_temps = np.zeros(len(file_list),dtype='float16')
	#lunar_distance = np.zeros(len(file_list),dtype='float16')
	
	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH SOURCE IN EACH FILE
	source_x = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_y = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_sky_ADU = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_sky_e = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_x_fwhm_arcsec = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_y_fwhm_arcsec = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_theta_radians = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')

	#ARRAYS THAT CONTAIN DATA PERTAININING TO EACH APERTURE RADIUS FOR EACH SOURCE FOR EACH FILE
	source_minus_sky_ADU = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_minus_sky_e = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_minus_sky_err_ADU = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_minus_sky_err_e = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	non_linear_flags = np.zeros((len(targ_and_refs),len(file_list)),dtype='bool')
	saturated_flags = np.zeros((len(targ_and_refs),len(file_list)),dtype='bool')
	ensemble_alc_ADU = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	ensemble_alc_e = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	ensemble_alc_err_ADU = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	ensemble_alc_err_e = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	relative_flux = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	relative_flux_err = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	
	#Load in the stacked image of the field that was used for source identification. 
	#All images will be cross-correlated with this to determine aperture positions. 
	bpm = load_bad_pixel_mask()
	#Extra masking in case you need to fall back on astroalign, which does source detection. 
	bpm[0:1032, 1447:1464] = True
	bpm[1023:, 1788:1801]  = True
	#25-pixel mask on all edges
	bpm[:, 0:25+1] = True
	bpm[:,4096-1-25:] = True
	bpm[0:25+1,:] = True
	bpm[2048-1-25:,:] = True

	reference_image_hdu = fits.open('/data/tierras/targets/'+target+'/'+target+'_stacked_image.fits')[0] #TODO: should match image from target/reference csv file, and that should be loaded automatically.

	#reference_image_hdu = fits.open(file_list[1])[0]

	reference_image_header = reference_image_hdu.header
	reference_wcs = WCS(reference_image_header)
	reference_world_coordinates = [reference_wcs.pixel_to_world(targ_and_refs['x'][i],targ_and_refs['y'][i]) for i in range(len(targ_and_refs))] #Get world coordinates of target and reference stars in the reference image. 

	#reference_image_data = np.ma.array(reference_image_hdu.data, mask=bpm)
	reference_image_data = reference_image_hdu.data
	reference_image_data[np.where(bpm==1)] = np.nan

	#Background-subtract the data for figuring out shifts	
	try:
		bkg = sep.Background(reference_image_data)
	except:
		bkg = sep.Background(reference_image_data.byteswap().newbyteorder())

	reference_image_data -= bkg.back()
	
	n_files = len(file_list)
	if live_plot:
		#fig, ax = plt.subplots(2,2,figsize=(16,9))
		fig = plt.figure(figsize=(16,9))
		gs = gridspec.GridSpec(2,4,figure=fig)
		ax1 = fig.add_subplot(gs[0,0:2])
		ax2 = fig.add_subplot(gs[1,0])
		ax3 = fig.add_subplot(gs[1,1])
		ax4 = fig.add_subplot(gs[0,2:])
		ax5 = fig.add_subplot(gs[1,2:])

	print(f'Doing PSF photometry on {n_files} images.\n')
	time.sleep(2)
	for i in range(n_files):
		print(f'{i+1} of {n_files}')
		source_hdu = fits.open(file_list[i])[0]
		source_header = source_hdu.header
		source_data = source_hdu.data #TODO: Should we ignore BPM pixels?

		GAIN = source_header['GAIN'] #e- ADU^-1
		READ_NOISE = source_header['READNOIS'] #e-
		EXPTIME = source_header['EXPTIME']
		RA = source_header['RA']
		DEC = source_header['DEC']

		#SAVE ANCILLARY DATA
		filenames.append(file_list[i].name)
		mjd_utc[i] = source_header['MJD-OBS'] + (EXPTIME/2)/(24*60*60) #MJD-OBS is the modified julian date at the start of the exposure. Add on half the exposure time in days to get the time at mid-exposure. 
		jd_utc[i] = mjd_utc[i]+2400000.5 #Convert MJD_UTC to JD_UTC
		bjd_tdb[i] = jd_utc_to_bjd_tdb(jd_utc[i], RA, DEC)
		airmasses[i] = source_header['AIRMASS']
		ccd_temps[i] = source_header['CCDTEMP']
		exp_times[i] = source_header['EXPTIME']
		focuses[i] = source_header['FOCUS']
		#These keywords are sometimes missing
		try:
			dome_humidities[i] = source_header['DOMEHUMI']
			dome_temps[i] = source_header['DOMETEMP']
			sec_temps[i] = source_header['SECTEMP']
			rod_temps[i] = source_header['RODTEMP']
			cab_temps[i] = source_header['CABTEMP']
			inst_temps[i] = source_header['INSTTEMP']
			ret_temps[i] = source_header['RETTEMP']
			pri_temps[i] = source_header['PRITEMP']
		except:
			dome_humidities[i] = np.nan
			dome_temps[i] = np.nan
			sec_temps[i] = np.nan
			rod_temps[i] = np.nan
			cab_temps[i] = np.nan
			inst_temps[i] = np.nan
			ret_temps[i] = np.nan
			pri_temps[i] = np.nan

		temps[i] = source_header['TEMPERAT']
		humidities[i] = source_header['HUMIDITY']
		dewpoints[i] = source_header['DEWPOINT']
		sky_temps[i] = source_header['SKYTEMP']

		#lunar_distance[i] = get_lunar_distance(RA, DEC, bjd_tdb[i])

		#UPDATE SOURCE POSITIONS
		#METHOD 1: WCS
		source_wcs = WCS(source_header)
		transformed_pixel_coordinates = [source_wcs.world_to_pixel(reference_world_coordinates[i]) for i in range(len(reference_world_coordinates))]
			
		
		#Save transformed pixel coordinates of sources
		for j in range(len(targ_and_refs)):
			source_x[j,i] = transformed_pixel_coordinates[j][0]
			source_y[j,i] = transformed_pixel_coordinates[j][1]
		
		# fig2, ax2 = plot_image(source_data)
		# for j in range(len(source_x[:,i])):
		# 	ax2.plot(source_x[j,i],source_y[j,i],'rx')
		# breakpoint()

		#DO PHOTOMETRY AT UPDATED SOURCE POSITIONS FOR ALL SOURCES AND ALL APERTURES
		for j in range(len(targ_and_refs)):
			x_pos_image = source_x[j,i]
			y_pos_image = source_y[j,i]

			#Check that the source position falls on the chip. If not, set its measured fluxes to NaNs.
			#TODO: NaN all the quantities you want to ignore. 
			if (x_pos_image < 0) or (x_pos_image > 4095) or (y_pos_image < 0) or (y_pos_image > 2047):
				source_minus_sky_ADU[j,i] = np.nan
				continue
			
			#Set up the source cutout
			cutout_y_start = int(y_pos_image-an_out)
			if cutout_y_start < 0:
				cutout_y_start = 0
			cutout_y_end = int(y_pos_image+an_out)
			if cutout_y_end > 2047:
				cutout_y_end = 2047
			cutout_x_start = int(x_pos_image-an_out)
			if cutout_x_start < 0:
				cutout_x_start = 0
			cutout_x_end = int(x_pos_image+an_out)
			if cutout_x_end > 4095:
				cutout_x_end = 4095

			cutout = source_data[cutout_y_start:cutout_y_end+1,cutout_x_start:cutout_x_end+1]
			#cutout = cutout[40:60,40:60] #TESTING
			cutout = cutout.copy(order='C')
			xx,yy = np.meshgrid(np.arange(cutout.shape[1]),np.arange(cutout.shape[0]))

			#x_pos_cutout = x_pos_image-int(x_pos_image)+an_out
			#y_pos_cutout = y_pos_image-int(y_pos_image)+an_out
			x_pos_cutout = x_pos_image-int(x_pos_image)+int(cutout.shape[1]/2)
			y_pos_cutout = y_pos_image-int(y_pos_image)+int(cutout.shape[0]/2)
			#breakpoint()
			if j == 0 and live_plot:
				norm = simple_norm(cutout,'linear',min_percent=0,max_percent=98.)
				ax2.imshow(cutout,origin='lower',interpolation='none',norm=norm,cmap='Greys_r')
				#ax[1,0].imshow(cutout,origin='lower',interpolation='none',norm=norm)
				ax2.plot(x_pos_cutout,y_pos_cutout, color='m', marker='x',mew=1.5,ms=8)
				#ap_circle = plt.Circle((x_pos_cutout,y_pos_cutout),ap_radii[k],fill=False,color='m',lw=2)
				an_in_circle = plt.Circle((x_pos_cutout,y_pos_cutout),an_in,fill=False,color='m',lw=2)
				an_out_circle = plt.Circle((x_pos_cutout,y_pos_cutout),an_out,fill=False,color='m',lw=2)
				#ax2.add_patch(ap_circle)
				ax2.add_patch(an_in_circle)
				ax2.add_patch(an_out_circle)
				ax2.set_xlim(0,cutout.shape[1])
				ax2.set_ylim(0,cutout.shape[0])
				ax2.grid(False)
				ax2.set_title('Target')

				ax1.imshow(source_data,origin='lower',interpolation='none',norm=simple_norm(source_data,'linear',min_percent=1,max_percent=99.9), cmap='Greys_r')
				ax1.grid(False)
				ax1.set_title(file_list[i].name)
				for l in range(len(source_x)):
					if l == 0:
						color = 'm'
						name = 'T'
					else:
						color = 'tab:red'
						name = f'R{l}'
					ap_circle = plt.Circle((source_x[l,i],source_y[l,i]),30,fill=False,color=color,lw=1)
					ax1.add_patch(ap_circle)
					ax1.text(source_x[l,i]+15,source_y[l,i]+15,name,color=color,fontsize=14)

			#Do PSF photometry on x/y cutout position

			# #METHOD 1: IntegratedGaussianPRF
			# #TODO: the IntegratedGaussianPRF model does not include a rotation term...
			# init_params = QTable()
			# init_params['x_0'] = [x_pos_cutout]
			# init_params['y_0'] = [y_pos_cutout]
			# psf_model = IntegratedGaussianPRF()
			# psf_model.sigma.fixed = False
			# psf_model.x_0.fixed = True
			# psf_model.y_0.fixed = True
			# psfphot = BasicPSFPhotometry(DAOGroup(8), MedianBackground(), psf_model, fitshape=(5,5))
			# result = psfphot(cutout, init_guesses=init_params)
			# if len(result) > 1:
			# 	#TODO: handling if more than one source is modeled....
			# 	breakpoint()
			# psf_flux = result['flux_fit'].value[0]
			# psf_flux_err = result['flux_unc'].value[0]

			#METHOD 2: Effective Point Spead Function
			if j == 0: #Only build once per image
				stars_tbl = Table()
				stars_tbl['x'] = targ_and_refs['x'][1:]
				stars_tbl['y'] = targ_and_refs['y'][1:]
				mean_val, median_val, std_val = sigma_clipped_stats(source_data, sigma=2.0)  
				nddata = NDData(data=source_data-median_val)
				stars = extract_stars(nddata, stars_tbl, size=20) 
				epsf_builder = EPSFBuilder(oversampling=2, maxiters=3,progress_bar=False, recentering_func=centroid_com)  
				psf_model, fitted_stars = epsf_builder(stars)  

			init_params = QTable()
			init_params['x_0'] = [x_pos_cutout]
			init_params['y_0'] = [y_pos_cutout]
			psf_model.x_0.fixed = False
			psf_model.x_0.bounds = (x_pos_cutout-1,x_pos_cutout+1)
			psf_model.y_0.fixed = False
			psf_model.y_0.bounds = (y_pos_cutout-1,y_pos_cutout+1)

			psfphot = BasicPSFPhotometry(DAOGroup(4), MedianBackground(), psf_model, fitshape=(11,11), aperture_radius=3)
			result = psfphot(cutout, init_guesses=init_params)
			breakpoint()
			if len(result) > 1:
				#TODO: handling if more than one source is modeled....
				breakpoint()
			psf_flux = result['flux_fit'].value[0]
			psf_flux_err = result['flux_unc'].value[0]


			source_minus_sky_ADU[j,i] = psf_flux
			source_minus_sky_e[j,i] = source_minus_sky_ADU[j,i]*GAIN
			
			if j == 0 and live_plot:
				#psf_model_fit = IntegratedGaussianPRF(flux=result['flux_fit'].value[0], x_0=result['x_fit'].value[0], y_0=result['y_fit'].value[0], sigma=result['sigma_fit'].value[0])
				psf_model.flux = result['flux_fit'].value[0]
				psf_model.x_0 = result['x_fit'].value[0]
				psf_model.y_0 = result['y_fit'].value[0]
				ax3.imshow(cutout-psf_model(xx,yy),origin='lower',norm=norm, cmap='Greys_r')
				ax3.set_title('Target Residual Image')
				breakpoint()

			# #Check for non-linear/saturated pixels in the aperture
			#TODO: Don't know how to do this with PSF photometry...
			# max_pix = np.max(ap.to_mask().multiply(cutout))
			# if max_pix >= SATURATION_TRESHOLD:
			# 	saturated_flags[k,j,i] = 1
			# if max_pix >= NONLINEAR_THRESHOLD:
			# 	non_linear_flags[k,j,i] = 1

			#Estimate background 
			#Determine which pixels are in the annulus using their distances from (x_pos_cutout,y_pos_cutout)
			dists = np.sqrt((x_pos_cutout-xx)**2+(y_pos_cutout-yy)**2)
			an_inds = np.where((dists<an_out)&(dists>an_in))  
			an_data = cutout[an_inds]
			v, l, h = sigmaclip(an_data[an_data!=0],2,2)
			bkg = np.mean(v)
			source_sky_ADU[j,i] = bkg
			source_sky_e[j,i] = bkg*GAIN
			
			#Calculate uncertainty
			#Right now, use sqrt of flux from the PSF fit as the source photon noise uncertainty estimate. 
			#Could alternatively use the 'flux_unc' entry from the result table, not sure which is better
			
			scintillation_rel = 0.09*(130)**(-2/3)*airmasses[i]**(7/4)*(2*EXPTIME)**(-1/2)*np.exp(-2306/8000)
			scintillation_abs_e = scintillation_rel * source_minus_sky_e[j,i] #Don't know if this should be multiplied by the average source flux or the flux in this frame

			#TODO: not sure about these error calculations. How do you incoporate read noise/ sky noise/ dark current into PSF photometry?  
			source_minus_sky_err_e[j,i] = np.sqrt((psf_flux_err*GAIN)**2 + scintillation_abs_e**2)
			source_minus_sky_err_ADU[j,i] = source_minus_sky_err_e[j,i]/GAIN

			#Save PSF shape information		
			x_stddev_pix = result['x_fit'].value[0]
			y_stddev_pix = result['y_fit'].value[0]
			x_fwhm_pix = x_stddev_pix * 2*np.sqrt(2*np.log(2))
			y_fwhm_pix = y_stddev_pix * 2*np.sqrt(2*np.log(2))
			x_fwhm_arcsec = x_fwhm_pix * PLATE_SCALE
			y_fwhm_arcsec = y_fwhm_pix * PLATE_SCALE
			#theta_rad = g.theta.value
			source_x_fwhm_arcsec[j,i] = x_fwhm_arcsec
			source_y_fwhm_arcsec[j,i] = y_fwhm_arcsec
			#source_theta_radians[j,i] = theta_rad

			#Plot normalized target source-sky as you go along
			if live_plot and j == 0:
				target_renorm_factor = np.nanmean(source_minus_sky_ADU[j,0:i+1])
				targ_norm = source_minus_sky_ADU[j,0:i+1]/target_renorm_factor
				targ_norm_err = source_minus_sky_err_ADU[j,0:i+1]/target_renorm_factor
				
				ax4.errorbar(bjd_tdb[0:i+1]-int(bjd_tdb[0]),targ_norm,targ_norm_err,color='k',marker='.',ls='',ecolor='k',label='Normalized target flux')
				#plt.ylim(380000,440000)
				ax4.set_ylabel('Normalized Flux')

		#Create ensemble ALCs (summed reference fluxes with no weighting) for each source
		for l in range(len(targ_and_refs)):
			#For the target, use all reference stars
			ref_inds = np.arange(1,len(targ_and_refs))
			#For the reference stars, use all other references and NOT the target
			if l != 0:
				ref_inds = np.delete(ref_inds,l-1)
			ensemble_alc_ADU[l,i] = sum(source_minus_sky_ADU[ref_inds,i])
			ensemble_alc_err_ADU[l,i] = np.sqrt(np.sum(source_minus_sky_err_ADU[ref_inds,i]**2))
			ensemble_alc_e[l,i] = sum(source_minus_sky_e[ref_inds,i])
			ensemble_alc_err_e[l,i] = np.sqrt(np.sum(source_minus_sky_err_e[ref_inds,i]**2))

			relative_flux[l,i] = source_minus_sky_ADU[l,i]/ensemble_alc_ADU[l,i]
			relative_flux_err[l,i] = np.sqrt((source_minus_sky_err_ADU[l,i]/ensemble_alc_ADU[l,i])**2+(source_minus_sky_ADU[l,i]*ensemble_alc_err_ADU[l,i]/(ensemble_alc_ADU[l,i]**2))**2)

		if live_plot:
			alc_renorm_factor = np.nanmean(ensemble_alc_ADU[0,0:i+1]) #This means, grab the ALC associated with the ap_plot_ind'th aperture for the 0th source (the target) in all images up to and including this one.
			alc_norm = ensemble_alc_ADU[0,0:i+1]/alc_renorm_factor
			alc_norm_err = ensemble_alc_err_ADU[0,0:i+1]/alc_renorm_factor
			v,l,h=sigmaclip(alc_norm[~np.isnan(alc_norm)])
			ax4.errorbar(bjd_tdb[0:i+1]-int(bjd_tdb[0]),alc_norm, alc_norm_err,color='r',marker='.',ls='',ecolor='r', label='Normalized ALC flux')
			try:
				ax4.set_ylim(l,h)
			except:
				breakpoint()
			ax4.legend() 

			corrected_flux = targ_norm/alc_norm
			corrected_flux_err = np.sqrt((targ_norm_err/alc_norm)**2+(targ_norm*alc_norm_err/(alc_norm**2))**2)
			v,l,h=sigmaclip(corrected_flux)
			ax5.errorbar(bjd_tdb[0:i+1]-int(bjd_tdb[0]),corrected_flux, corrected_flux_err, color='k', marker='.', ls='', ecolor='k', label='Corrected target flux')
			ax5.set_ylim(l,h)
			ax5.legend()
			ax5.set_ylabel('Normalized Flux')
			ax5.set_xlabel(f'Time - {int(bjd_tdb[0]):d}'+' (BJD$_{TDB}$)')
			#plt.tight_layout()
			plt.suptitle(f'{i+1} of {n_files}')
			#plt.savefig(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/live_plots/{str(i+1).zfill(4)}.jpg',dpi=100)
			#breakpoint()
			plt.pause(0.01)
			ax1.cla()
			ax2.cla()
			ax3.cla()
			ax4.cla()
			ax5.cla()

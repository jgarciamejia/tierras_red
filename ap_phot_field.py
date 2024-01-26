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
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u 
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
from astroquery.vizier import Vizier
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
from ap_phot import load_bad_pixel_mask, jd_utc_to_bjd_tdb, plot_image, set_tierras_permissions
import warnings 

'''
	Scripts for doing photometry for all objects in a Tierras field. 
'''

def do_all_photometry(file_list, target, ap_radii, an_in, an_out, centroid=True, type='fixed'):
	''''
		Does photometry for *all* objects in a Tierras field. 
	'''
	# Query the field for sources in Gaia, cross-match with 2MASS
	sources = tierras_gaia_crossmatch(target, rp_mag_limit=17)


	ffname = file_list[0].parent.name	
	target = file_list[0].parent.parent.name
	date = file_list[0].parent.parent.parent.name 

	#file_list = file_list[383:] #TESTING!!!
	
	DARK_CURRENT = 0.19 #e- pix^-1 s^-1
	NONLINEAR_THRESHOLD = 40000. #ADU
	SATURATION_THRESHOLD = 55000. #ADU
	PLATE_SCALE = 0.43 #arcsec pix^-1, from Juliana's dissertation Table 1.1

	centroid_mask_half_l = 11 #If centroiding is performed, a box of size 2*centroid_mask_half_l x 2*centroid_mask_half_l is used to mask out around the source's expected position (reduces chance of measuring a bad centroid)
	
	#If doing variable aperture photometry, read in FWHM X/Y data and set aperture radii 
	if type == 'variable':
		fwhm_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/fwhm.csv'
		if not os.path.exists(fwhm_path):
			raise RuntimeError(f'{fwhm_path} does not exist! You have to run ap_range first.')
		else:
			fwhm_df = pd.read_csv(fwhm_path)
			fwhm_x = np.array(fwhm_df['FWHM X'])
			fwhm_y = np.array(fwhm_df['FWHM Y'])
			v1, l1, h1 = sigmaclip(fwhm_x)
			v2, l2, h2 = sigmaclip(fwhm_y)
			bad_inds = np.where((fwhm_x<l1)|(fwhm_x>h1)|(fwhm_y<l2)|(fwhm_y>h2))[0]
			fwhm_x[bad_inds] = np.nan
			fwhm_y[bad_inds] = np.nan
			fwhm_x = interpolate_replace_nans(fwhm_x, kernel=Gaussian1DKernel(5))
			fwhm_y = interpolate_replace_nans(fwhm_y, kernel=Gaussian1DKernel(5))
			mean_fwhm = np.mean([fwhm_x,fwhm_y],axis=0)
			smoothed_fwhm_arcsec = savgol_filter(mean_fwhm, 11, 3)
			smoothed_fwhm_pix = smoothed_fwhm_arcsec/PLATE_SCALE

			# plt.plot(mean_fwhm,label='Mean FWHM')
			# plt.plot(smoothed_fwhm_arcsec,label='Smoothed FWHM')
			# plt.ylabel('FWHM (")')
			# plt.legend()
			# breakpoint()

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
	loop_times = np.zeros(len(file_list),dtype='float16')
	#lunar_distance = np.zeros(len(file_list),dtype='float16')
	
	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH SOURCE IN EACH FILE
	source_x = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_y = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_sky_ADU = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_sky_e = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_x_fwhm_arcsec = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_y_fwhm_arcsec = np.zeros((len(sources),len(file_list)),dtype='float32')
	source_theta_radians = np.zeros((len(sources),len(file_list)),dtype='float32')

	#ARRAYS THAT CONTAIN DATA PERTAININING TO EACH APERTURE RADIUS FOR EACH SOURCE FOR EACH FILE
	source_minus_sky_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	source_minus_sky_e = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	source_minus_sky_err_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	source_minus_sky_err_e = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	non_linear_flags = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='bool')
	saturated_flags = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='bool')
	ensemble_alc_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	ensemble_alc_e = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	ensemble_alc_err_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	ensemble_alc_err_e = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	relative_flux = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	relative_flux_err = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	
	source_radii = np.zeros((len(ap_radii),len(file_list)),dtype='float16')
	an_in_radii = np.zeros((len(ap_radii),len(file_list)),dtype='float16')
	an_out_radii = np.zeros((len(ap_radii),len(file_list)),dtype='float16')
	
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
	# reference_world_coordinates = [reference_wcs.pixel_to_world(sources['X pix'][i],sources['Y pix'][i]) for i in range(len(targ_and_refs))] # Get world coordinates of target and reference stars in the reference image. 

	reference_world_coordinates = [SkyCoord(sources['ra_tierras'][i],sources['dec_tierras'][i], unit=(u.deg, u.deg)) for i in range(len(sources))] # Get world coordinates of target and reference stars in the reference image. 

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
	
	print(f'Doing fixed-radius circular aperture photometry on {n_files} images with aperture radii of {ap_radii} pixels, an inner annulus radius of {an_in} pixels, and an outer annulus radius of {an_out} pixels.\n')
	time.sleep(2)
	for i in range(n_files):
		if i > 0:
			loop_times[i-1]= time.time()-t1
			print(f'Avg loop time = {np.mean(loop_times[0:i]):.2f}s')
		t1 = time.time()
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
			dewpoints[i] = source_header['DEWPOINT']
			temps[i] = source_header['TEMPERAT']
			humidities[i] = source_header['HUMIDITY']
			sky_temps[i] = source_header['SKYTEMP']
		except:
			dome_humidities[i] = np.nan
			dome_temps[i] = np.nan
			sec_temps[i] = np.nan
			rod_temps[i] = np.nan
			cab_temps[i] = np.nan
			inst_temps[i] = np.nan
			ret_temps[i] = np.nan
			pri_temps[i] = np.nan
			dewpoints[i] = np.nan
			temps[i] = np.nan
			humidities[i] = np.nan
			sky_temps[i] = np.nan

		#lunar_distance[i] = get_lunar_distance(RA, DEC, bjd_tdb[i]) #Commented out because this is slow and the information can be generated at a later point if necessary
		
		#Calculate expected scintillation noise in this image
		scintillation_rel = 0.09*(130)**(-2/3)*airmasses[i]**(7/4)*(2*EXPTIME)**(-1/2)*np.exp(-2306/8000)

		#UPDATE SOURCE POSITIONS
		#METHOD 1: WCS
		source_wcs = WCS(source_header)
		transformed_pixel_coordinates = np.array([source_wcs.world_to_pixel(reference_world_coordinates[i]) for i in range(len(reference_world_coordinates))])
		
		#Save transformed pixel coordinates of sources
		source_x[:,i] = transformed_pixel_coordinates[:,0]
		source_y[:,i] = transformed_pixel_coordinates[:,1]

		# fig2, ax2 = plot_image(source_data)
		# for j in range(len(source_x[:,i])):
		# 	ax2.plot(source_x[j,i],source_y[j,i],'rx')
		# breakpoint()

		#DO PHOTOMETRY AT UPDATED SOURCE POSITIONS FOR ALL SOURCES AND ALL APERTURES
		for j in range(len(sources)):
			x_pos_image = source_x[j,i]
			y_pos_image = source_y[j,i]

			#Check that the source position falls on the chip. If not, set its measured fluxes to NaNs.
			#TODO: NaN all the quantities you want to ignore. 
			if (x_pos_image < 0) or (x_pos_image > 4095) or (y_pos_image < 0) or (y_pos_image > 2047):
				source_minus_sky_ADU[k,j,i] = np.nan
				continue
			
			#Set up the source cutout
			cutout_y_start = max([int(y_pos_image-an_out),0])
			cutout_y_end = min([int(y_pos_image+an_out),2047])
			cutout_x_start = max([int(x_pos_image-an_out),0])
			cutout_x_end = min([int(x_pos_image+an_out),4095])
			cutout = source_data[cutout_y_start:cutout_y_end+1,cutout_x_start:cutout_x_end+1]
			cutout = cutout.copy(order='C')
			cutout_x_len = cutout.shape[1]
			cutout_y_len = cutout.shape[0]
			

			#Set up grid of x/y pixel indices for the cutout
			xx,yy = np.meshgrid(np.arange(cutout_x_len),np.arange(cutout_y_len))

			#Transform the source position from image to cutout coordinates
			#TODO: This is not working when star is near an edge 
			if cutout_x_start == 0:
				x_pos_cutout = x_pos_image-int(x_pos_image)+an_out + (cutout.shape[1]-2*an_out-1)
			else:
				x_pos_cutout = x_pos_image-int(x_pos_image)+an_out
			if cutout_y_start == 0:
				y_pos_cutout = y_pos_image-int(y_pos_image)+an_out + (cutout.shape[0]-2*an_out-1)
			else:
				y_pos_cutout = y_pos_image-int(y_pos_image)+an_out


			#Optionally recompute the centroid
			if centroid:
				#TODO: What size should this mask be? Should we use the bad pixel mask instead? Should we just toss any stars that are remotely close to bad columns? 
				centroid_mask = np.ones(cutout.shape, dtype='bool')
				# x_mid = int(cutout_x_len/2)
				x_mid = int(x_pos_cutout)
				# y_mid = int(cutout_y_len/2)
				y_mid = int(y_pos_cutout)
				centroid_mask[y_mid-centroid_mask_half_l:y_mid+centroid_mask_half_l,x_mid-centroid_mask_half_l:x_mid+centroid_mask_half_l] = False
				
				#t2 = time.time()
				x_pos_cutout_centroid, y_pos_cutout_centroid = centroid_1dg(cutout-np.median(cutout), mask=centroid_mask)

				

				#Make sure the measured centroid is actually in the unmasked region
				#if (x_pos_cutout_centroid > 0) and (x_pos_cutout_centroid < cutout.shape[1]) and (y_pos_cutout_centroid > 0) and (y_pos_cutout_centroid < cutout.shape[0]):
				if (abs(x_pos_cutout_centroid-x_mid) < centroid_mask_half_l) and (abs(y_pos_cutout_centroid-y_mid) < centroid_mask_half_l):

					x_pos_cutout = x_pos_cutout_centroid
					y_pos_cutout = y_pos_cutout_centroid
				# else:
					# plt.figure()
					# plt.imshow(cutout,origin='lower',interpolation='none',norm=simple_norm(cutout,max_percent=97))
					# plt.plot(x_pos_cutout_centroid, y_pos_cutout_centroid, 'rx')
					# plt.plot(x_pos_cutout, y_pos_cutout, 'mx')
					# breakpoint()
					# plt.close()
				
				#Update position in full image using measured centroid
				x_pos_image = x_pos_cutout + int(x_pos_image) - an_out
				y_pos_image = y_pos_cutout + int(y_pos_image) - an_out
				source_x[j,i] = x_pos_image
				source_y[j,i] = y_pos_image
			 
			#Estimate background (NOTE: this can be performed at this loop level because we use the same background annulus regardless of the aperture size. If that assumption ever changes it will have to be moved inside of the aperture loop.)

			#Create array of pixel distances in the cutout from the source position
			dists = np.sqrt((x_pos_cutout-xx)**2+(y_pos_cutout-yy)**2)
			#Determine which pixels are in the annulus using their distances from (x_pos_cutout,y_pos_cutout)
			an_inds = np.where((dists<an_out)&(dists>an_in))  
			an_data = cutout[an_inds]
			v, l, h = sigmaclip(an_data[an_data!=0],2,2)
			bkg = np.mean(v)
			source_sky_ADU[j,i] = bkg
			source_sky_e[j,i] = bkg*GAIN

			for k in range(len(ap_radii)):
				if type == 'fixed':
					ap = CircularAperture((x_pos_cutout,y_pos_cutout),r=ap_radii[k])
				elif type == 'variable':
					ap = CircularAperture((x_pos_cutout,y_pos_cutout),r=ap_radii[k]*smoothed_fwhm_pix[i])

				#ap = EllipticalAperture((x_pos_cutout,y_pos_cutout),a=15,b=9, theta=90*np.pi/180)
				#an = CircularAnnulus((x_pos_cutout,y_pos_cutout),r_in=an_in,r_out=an_out)

				
				if type == 'fixed':
					source_radii[k,i] = ap_radii[k]
				elif type == 'variable':
					source_radii[k,i] = ap_radii[k]*smoothed_fwhm_pix[i]
				an_in_radii[k,i] = an_in
				an_out_radii[k,i] = an_out

				# ap.plot(color='r',lw=2.5)
				# an.plot(color='r',lw=2.5)

				#DO PHOTOMETRY
				#t1 = time.time()
				phot_table = aperture_photometry(cutout, ap)

				#Check for non-linear/saturated pixels in the aperture
				max_pix = np.max(ap.to_mask().multiply(cutout))
				if max_pix >= SATURATION_THRESHOLD:
					saturated_flags[k,j,i] = 1
				if max_pix >= NONLINEAR_THRESHOLD:
					non_linear_flags[k,j,i] = 1

				source_minus_sky_ADU[k,j,i] = phot_table['aperture_sum'][0]-bkg*ap.area 
				source_minus_sky_e[k,j,i] = source_minus_sky_ADU[k,j,i]*GAIN

				#Calculate uncertainty

				#scintillation_rel = 1.5*0.09*(130)**(-2/3)*airmasses[i]**(7/4)*(2*EXPTIME)**(-1/2)*np.exp(-2306/8000)*np.sqrt(1+1/(len(targ_and_refs)-1)) #Follows equation 2 from Tierras SPIE paper
				scintillation_abs_e = scintillation_rel * source_minus_sky_e[k,j,i] #Don't know if this should be multiplied by the average source flux or the flux in this frame
				source_minus_sky_err_e[k,j,i] = np.sqrt(phot_table['aperture_sum'][0]*GAIN + bkg*ap.area*GAIN + DARK_CURRENT*source_header['EXPTIME']*ap.area + ap.area*READ_NOISE**2 + scintillation_abs_e**2)
				source_minus_sky_err_ADU[k,j,i] = source_minus_sky_err_e[k,j,i]/GAIN

				#Measure shape by fitting a 2D Gaussian to the cutout.
				#Don't do for every aperture size, just do it once. 
				if j == 0 and k == 0:
					#g_2d_cutout = cutout[int(y_pos_cutout)-25:int(y_pos_cutout)+25,int(x_pos_cutout)-25:int(x_pos_cutout)+25]
					#t1 = time.time()
					g_2d_cutout = copy.deepcopy(cutout)
					xx2,yy2 = np.meshgrid(np.arange(g_2d_cutout.shape[1]),np.arange(g_2d_cutout.shape[0]))
					g_init = models.Gaussian2D(amplitude=g_2d_cutout[int(g_2d_cutout.shape[1]/2), int(g_2d_cutout.shape[0]/2)]-bkg,x_mean=g_2d_cutout.shape[1]/2,y_mean=g_2d_cutout.shape[0]/2, x_stddev=5, y_stddev=5)
					fit_g = fitting.LevMarLSQFitter()
					g = fit_g(g_init,xx2,yy2,g_2d_cutout-bkg)
					
					g.theta.value = g.theta.value % (2*np.pi) #Constrain from 0-2pi
					if g.y_stddev.value > g.x_stddev.value: 
						x_stddev_save = g.x_stddev.value
						y_stddev_save = g.y_stddev.value
						g.x_stddev = y_stddev_save
						g.y_stddev = x_stddev_save
						g.theta += np.pi/2

					x_stddev_pix = g.x_stddev.value
					y_stddev_pix = g.y_stddev.value 
					x_fwhm_pix = x_stddev_pix * 2*np.sqrt(2*np.log(2))
					y_fwhm_pix = y_stddev_pix * 2*np.sqrt(2*np.log(2))
					x_fwhm_arcsec = x_fwhm_pix * PLATE_SCALE
					y_fwhm_arcsec = y_fwhm_pix * PLATE_SCALE
					theta_rad = g.theta.value
					source_x_fwhm_arcsec[j,i] = x_fwhm_arcsec
					source_y_fwhm_arcsec[j,i] = y_fwhm_arcsec
					source_theta_radians[j,i] = theta_rad
				
		
		#Create ensemble ALCs (summed reference fluxes with no weighting) for each source
		for l in range(len(sources)):
			#For the target, use all reference stars
			ref_inds = np.arange(1,len(sources))
			#For the reference stars, use all other references and NOT the target
			if l != 0:
				ref_inds = np.delete(ref_inds,l-1)
			
			ensemble_alc_ADU[:,l] = np.sum(source_minus_sky_ADU[:,ref_inds],axis=1)
			ensemble_alc_err_ADU[:,l] = np.sqrt(np.sum(source_minus_sky_err_ADU[:,ref_inds]**2,axis=1))
			ensemble_alc_e[:,l] = ensemble_alc_ADU[:,l]*GAIN
			ensemble_alc_err_e[:,l] = ensemble_alc_err_ADU[:,l]*GAIN
			relative_flux[:,l] = source_minus_sky_ADU[:,l]/ensemble_alc_ADU[:,l]
			relative_flux_err[:,l] = np.sqrt((source_minus_sky_err_ADU[:,l]/ensemble_alc_ADU[:,l])**2+(source_minus_sky_ADU[:,l]*ensemble_alc_err_ADU[:,l]/(ensemble_alc_ADU[:,l]**2))**2)

	#Write out photometry. 
	for i in range(len(ap_radii)):
		if not os.path.exists('/data/tierras/lightcurves/'+date+'/'+target+'/'+ffname+'/full_field_photometry'):
			os.mkdir('/data/tierras/lightcurves/'+date+'/'+target+'/'+ffname+'/full_field_photometry')
		output_path = Path('/data/tierras/lightcurves/'+date+'/'+target+'/'+ffname+'/full_field_photometry'+f'/full_field_circular_{type}_ap_phot_{ap_radii[i]}.csv')
		
		output_list = []
		output_header = []
		
		output_list.append([f'{val}' for val in filenames])
		output_header.append('Filename')

		output_list.append([f'{val:.7f}' for val in mjd_utc])
		output_header.append('MJD UTC')
		output_list.append([f'{val:.7f}' for val in jd_utc])
		output_header.append('JD UTC')
		output_list.append([f'{val:.7f}' for val in bjd_tdb])
		output_header.append('BJD TDB')

		output_list.append([f'{val:.2f}' for val in source_radii[i]])
		output_header.append('Aperture Radius')
		output_list.append([f'{val:.2f}' for val in an_in_radii[i]])
		output_header.append('Inner Annulus Radius')
		output_list.append([f'{val:.2f}' for val in an_out_radii[i]])
		output_header.append('Outer Annulus Radius')

		output_list.append([f'{val:.2f}' for val in exp_times])
		output_header.append('Exposure Time')
		output_list.append([f'{val:.4f}' for val in airmasses])
		output_header.append('Airmass')
		output_list.append([f'{val:.1f}' for val in ccd_temps])
		output_header.append('CCD Temperature')
		output_list.append([f'{val:.2f}' for val in dome_temps])
		output_header.append('Dome Temperature')
		output_list.append([f'{val:.1f}' for val in focuses])
		output_header.append('Focus')
		output_list.append([f'{val:.2f}' for val in dome_humidities])
		output_header.append('Dome Humidity')
		output_list.append([f'{val:.1f}' for val in sec_temps])
		output_header.append('Sec Temperature')
		output_list.append([f'{val:.1f}' for val in ret_temps])
		output_header.append('Ret Temperature')
		output_list.append([f'{val:.1f}' for val in pri_temps])
		output_header.append('Pri Temperature')
		output_list.append([f'{val:.1f}' for val in rod_temps])
		output_header.append('Rod Temperature')
		output_list.append([f'{val:.2f}' for val in cab_temps])
		output_header.append('Cab Temperature')
		output_list.append([f'{val:.1f}' for val in inst_temps])
		output_header.append('Instrument Temperature')
		output_list.append([f'{val:.1f}' for val in temps])
		output_header.append('Temperature')
		output_list.append([f'{val:.1f}' for val in humidities])
		output_header.append('Humidity')
		output_list.append([f'{val:.2f}' for val in dewpoints])
		output_header.append('Dewpoint')
		output_list.append([f'{val:.1f}' for val in sky_temps])
		output_header.append('Sky Temperature')
		#output_list.append([f'{val:.5f}' for val in lunar_distance])
		#output_header.append('Lunar Distance')

		for j in range(len(sources)):
			source_name = f'Source {j+1}'
			output_list.append([f'{val:.4f}' for val in source_x[j]])
			output_header.append(source_name+' X')
			output_list.append([f'{val:.4f}' for val in source_y[j]])
			output_header.append(source_name+' Y')
			output_list.append([f'{val:.7f}' for val in source_minus_sky_ADU[i,j]])
			output_header.append(source_name+' Source-Sky ADU')
			output_list.append([f'{val:.7f}' for val in source_minus_sky_err_ADU[i,j]])
			output_header.append(source_name+' Source-Sky Error ADU')
			output_list.append([f'{val:.7f}' for val in source_minus_sky_e[i,j]])
			output_header.append(source_name+' Source-Sky e')
			output_list.append([f'{val:.7f}' for val in source_minus_sky_err_e[i,j]])
			output_header.append(source_name+' Source-Sky Error e')

			output_list.append([f'{val:.7f}' for val in ensemble_alc_ADU[i,j]])
			output_header.append(source_name+' Ensemble ALC ADU')
			output_list.append([f'{val:.7f}' for val in ensemble_alc_err_ADU[i,j]])
			output_header.append(source_name+' Ensemble ALC Error ADU')
			output_list.append([f'{val:.7f}' for val in ensemble_alc_e[i,j]])
			output_header.append(source_name+' Ensemble ALC e')
			output_list.append([f'{val:.7f}' for val in ensemble_alc_err_e[i,j]])
			output_header.append(source_name+' Ensemble ALC Error e')
			output_list.append([f'{val:.10f}' for val in relative_flux[i,j]])
			output_header.append(source_name+' Relative Flux')
			output_list.append([f'{val:.10f}' for val in relative_flux_err[i,j]])
			output_header.append(source_name+' Relative Flux Error')

			output_list.append([f'{val:.7f}' for val in source_sky_ADU[j]])
			output_header.append(source_name+' Sky ADU')
			output_list.append([f'{val:.7f}' for val in source_sky_e[j]])
			output_header.append(source_name+' Sky e')

			output_list.append([f'{val:.4f}' for val in source_x_fwhm_arcsec[j]])
			output_header.append(source_name+' X FWHM Arcsec')
			output_list.append([f'{val:.4f}' for val in source_y_fwhm_arcsec[j]])
			output_header.append(source_name+' Y FWHM Arcsec')
			output_list.append([f'{val:.4f}' for val in source_theta_radians[j]])
			output_header.append(source_name+' Theta Radians')

			output_list.append([f'{val:d}' for val in non_linear_flags[i,j]])
			output_header.append(source_name+' Non-Linear Flag')
			output_list.append([f'{val:d}' for val in saturated_flags[i,j]])
			output_header.append(source_name+' Saturated Flag')



		output_df = pd.DataFrame(np.transpose(output_list),columns=output_header)
		if not os.path.exists(output_path.parent.parent):
			os.mkdir(output_path.parent.parent)
			set_tierras_permissions(output_path.parent.parent)
		if not os.path.exists(output_path.parent):
			os.mkdir(output_path.parent)
			set_tierras_permissions(output_path.parent)
		output_df.to_csv(output_path,index=False)
		set_tierras_permissions(output_path)

	plt.close('all')
	return 



def tierras_gaia_crossmatch(target, rp_mag_limit=17):
	'''
		Search for objects in a Tierras field in Gaia DR3 out to rp_mag_limit. Code will cross match with 2MASS. 
	'''
	
	PLATE_SCALE = 0.432 
	stacked_image_path = f'/data/tierras/targets/{target}/{target}_stacked_image.fits'
	#stacked_image_path = f'/data/tierras/flattened/20230307/{target}/flat0000/20230307.1380.TIC362144730_red.fit'
	hdu = fits.open(stacked_image_path)
	data = hdu[0].data
	header = hdu[0].header
	wcs = WCS(header)
	tierras_epoch = Time(header['TELDATE'],format='decimalyear')

	#coord = SkyCoord(ra,dec,unit=(u.hourangle,u.degree),frame='icrs')
	coord = SkyCoord(wcs.pixel_to_world(int(data.shape[1]/2),int(data.shape[0]/2)))
	# width = u.Quantity(PLATE_SCALE*data.shape[0],u.arcsec)
	width = 1500*u.arcsec
	height = u.Quantity(PLATE_SCALE*data.shape[1],u.arcsec)

	job = Gaia.launch_job_async('''SELECT 
								source_id, ra, dec, ref_epoch, pmra, pmra_error, pmdec, pmdec_error, parallax, parallax_error, parallax_over_error, phot_bp_mean_mag, phot_g_mean_mag, phot_rp_mean_mag, bp_rp, phot_variable_flag, non_single_star, teff_gspphot, logg_gspphot, distance_gspphot
								FROM gaiadr3.gaia_source
								
								WHERE ra > {}
								AND ra < {}
								AND DEC > {}
								AND DEC < {}
								AND phot_rp_mean_mag <= {}
					
								ORDER BY phot_g_mean_mag ASC
						  '''.format(coord.ra.value-width.to(u.deg).value/2, coord.ra.value+width.to(u.deg).value/2, coord.dec.value-height.to(u.deg).value/2, coord.dec.value+height.to(u.deg).value/2, rp_mag_limit))
	res = job.get_results()
	#Cut to entries without masked pmra values; otherwise the crossmatch will break
	problem_inds = np.where(res['pmra'].mask)[0]

	#OPTION 1: Set the pmra, pmdec, and parallax of those indices to 0
	res['pmra'][problem_inds] = 0
	res['pmdec'][problem_inds] = 0
	res['parallax'][problem_inds] = 0

	# #OPTION 2: Drop them from the table
	# good_inds = np.where(~res['pmra'].mask)[0]
	# res = res[good_inds]

	gaia_coords = SkyCoord(ra=res['ra'], dec=res['dec'], pm_ra_cosdec=res['pmra'], pm_dec=res['pmdec'], obstime=Time('2016',format='decimalyear'))
	v = Vizier(catalog="II/246",columns=['*','Date'], row_limit=-1)
	twomass_res = v.query_region(coord, width=width, height=height)[0]
	twomass_coords = SkyCoord(twomass_res['RAJ2000'],twomass_res['DEJ2000'])
	twomass_epoch = Time('2000-01-01')
	gaia_coords_tm_epoch = gaia_coords.apply_space_motion(twomass_epoch)
	gaia_coords_tierras_epoch = gaia_coords.apply_space_motion(tierras_epoch)
	
	res.rename_column('ra', 'ra_gaia')
	res.rename_column('dec','dec_gaia')
	

	idx_gaia, sep2d_gaia, _ = gaia_coords_tm_epoch.match_to_catalog_sky(twomass_coords)
	#Now set problem indices back to NaNs
	res['pmra'][problem_inds] = np.nan
	res['pmdec'][problem_inds] = np.nan
	res['parallax'][problem_inds] = np.nan
	
	tierras_pixel_coords = wcs.world_to_pixel(gaia_coords_tierras_epoch)

	res.add_column(twomass_res['_2MASS'][idx_gaia],name='2MASS',index=1)
	res.add_column(tierras_pixel_coords[0],name='X pix', index=2)
	res.add_column(tierras_pixel_coords[1],name='Y pix', index=3)
	res.add_column(gaia_coords_tierras_epoch.ra, name='ra_tierras', index=4)
	res.add_column(gaia_coords_tierras_epoch.dec, name='dec_tierras', index=5)

	res['Jmag'] = twomass_res['Jmag'][idx_gaia]
	res['e_Jmag'] = twomass_res['e_Jmag'][idx_gaia]
	res['Hmag'] = twomass_res['Hmag'][idx_gaia]
	res['e_Hmag'] = twomass_res['e_Hmag'][idx_gaia]
	res['Kmag'] = twomass_res['Kmag'][idx_gaia]
	res['e_Kmag'] = twomass_res['e_Kmag'][idx_gaia]
	

	#Cut to sources that actually fall in the image
	use_inds = np.where((tierras_pixel_coords[0]>0)&(tierras_pixel_coords[0]<data.shape[1]-1)&(tierras_pixel_coords[1]>0)&(tierras_pixel_coords[1]<data.shape[0]-1))[0]
	res = res[use_inds]


	#Cut to sources that don't fall on the bottom/top couple of rows 
	use_inds = np.where((res['Y pix'] > 5) & (res['Y pix']<2041))[0]
	res = res[use_inds]

	# Cut to sources that are far away from the bad columns 
	bad_inds = np.where((res['Y pix'] < 1038) & (res['X pix'] > 1439) & (res['X pix'] < 1465))[0] # Bad column in lower half 
	res.remove_rows(bad_inds)

	bad_inds = np.where((res['Y pix'] > 1007) & (res['X pix'] > 1021) & (res['X pix'] < 1035))[0]
	res.remove_rows(bad_inds)

	# fig, ax = plot_image(data)
	# ax.plot(res['X pix'],res['Y pix'],'rx')
	# breakpoint()

	return res

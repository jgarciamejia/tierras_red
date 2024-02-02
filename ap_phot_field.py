#!/usr/bin/env python

import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import gridspec
from matplotlib import colors 
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
from ap_phot import load_bad_pixel_mask, jd_utc_to_bjd_tdb, plot_image, set_tierras_permissions, quotient_uncertainty, regression, tierras_binner, tierras_binner_inds, get_flattened_files, ap_range, t_or_f
import warnings 

'''
	Scripts for doing photometry for all objects in a Tierras field. 
'''

def annulus_mask(shape, center, inner_radius, outer_radius):
	y, x = np.ogrid[:shape[0], :shape[1]]
	distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
	mask = (distance_from_center >= inner_radius) & (distance_from_center <= outer_radius)
	return mask

def generate_square_cutout(image, position, size):
	height, width = image.shape
	
	# Calculate the bounds of the cutout
	half_size = size / 2
	top = max(0, int(position[0] - half_size))
	bottom = min(width, int(position[0] + half_size))
	left = max(0, int(position[1] - half_size))
	right = min(height, int(position[1] + half_size))
	
	# Calculate the actual position within the cutout
	adjusted_position = (int(half_size - (position[0] - top)), int(half_size - (position[1] - left)))
	
	# Generate the square cutout
	cutout = np.zeros((size, size), dtype=image.dtype)
	cutout[adjusted_position[1]:(adjusted_position[1] + right - left), adjusted_position[0]:(adjusted_position[0] + bottom - top),] = image[left:right, top:bottom]

	cutout[cutout == 0] = np.nan
	
	# Calculate the position of the input 'position' in the cutout
	# position_in_cutout = (half_size - adjusted_position[0], half_size - adjusted_position[1])
	position_in_cutout = (adjusted_position[0] + position[0] - top, adjusted_position[1] + position[1] - left)

	

	return cutout, position_in_cutout

def do_all_photometry(file_list, target, ap_radii, an_in, an_out, centroid=True, type='fixed'):
	''''
		Does photometry for *all* objects in a Tierras field. 
	'''
	# Query the field for sources in Gaia, cross-match with 2MASS
	sources = tierras_gaia_crossmatch(target, rp_mag_limit=17)

	ffname = file_list[0].parent.name	
	target = file_list[0].parent.parent.name
	date = file_list[0].parent.parent.parent.name 
	
	# Save the source detection table
	if not os.path.exists(Path('/data/tierras/lightcurves/'+date+'/'+target+'/'+ffname+'/full_field_photometry')):
		os.mkdir(Path('/data/tierras/lightcurves/'+date+'/'+target+'/'+ffname+'/full_field_photometry'))
	output_path = Path('/data/tierras/lightcurves/'+date+'/'+target+'/'+ffname+'/full_field_photometry'+f'/full_field_sources.csv')
	output_df = pd.DataFrame(dict(sources))
	output_df.to_csv(output_path, index=0)

	# file_list = file_list[322:] #TESTING!!!
	
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
	# source_sky_e = np.zeros((len(sources),len(file_list)),dtype='float32')
	# source_x_fwhm_arcsec = np.zeros((len(sources),len(file_list)),dtype='float32')
	# source_y_fwhm_arcsec = np.zeros((len(sources),len(file_list)),dtype='float32')
	# source_theta_radians = np.zeros((len(sources),len(file_list)),dtype='float32')

	#ARRAYS THAT CONTAIN DATA PERTAININING TO EACH APERTURE RADIUS FOR EACH SOURCE FOR EACH FILE
	source_minus_sky_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	# source_minus_sky_e = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	source_minus_sky_err_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	# source_minus_sky_err_e = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	non_linear_flags = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='bool')
	saturated_flags = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='bool')
	# ensemble_alc_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	# ensemble_alc_e = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	# ensemble_alc_err_ADU = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	# ensemble_alc_err_e = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	# relative_flux = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	# relative_flux_err = np.zeros((len(ap_radii),len(sources),len(file_list)),dtype='float32')
	
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

	# reference_world_coordinates = [SkyCoord(sources['ra_tierras'][i],sources['dec_tierras'][i], unit=(u.deg, u.deg)) for i in range(len(sources))] # Get world coordinates of target and reference stars in the reference image. 
	reference_world_coordinates = np.array([sources['ra_tierras'],sources['dec_tierras']]).T

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
	
	print(f'Doing fixed-radius circular aperture photometry for {len(sources)} sources in {n_files} images with aperture radii of {ap_radii} pixels, an inner annulus radius of {an_in} pixels, and an outer annulus radius of {an_out} pixels.\n')
	time.sleep(2)
	for i in range(n_files):
		if i > 0:
			loop_times[i-1]= time.time()-t_loop_start
			print(f'Avg loop time = {np.mean(loop_times[0:i]):.2f}s')
		t_loop_start = time.time()
		
		print(f'{i+1} of {n_files}')
		source_hdu = fits.open(file_list[i])[0]
		source_header = source_hdu.header
		source_data = source_hdu.data #TODO: Should we ignore BPM pixels?

		an_in_radii[:,i] = an_in
		an_out_radii[:,i] = an_out

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
		# transformed_pixel_coordinates = np.array([source_wcs.world_to_pixel(reference_world_coordinates[i]) for i in range(len(reference_world_coordinates))])
		transformed_pixel_coordinates = np.array(source_wcs.world_to_pixel_values(reference_world_coordinates))
		
		#Save transformed pixel coordinates of sources
		source_x[:,i] = transformed_pixel_coordinates[:,0]
		source_y[:,i] = transformed_pixel_coordinates[:,1]

		# create tuples of source positions in this frame
		source_positions = [(source_x[j,i], source_y[j,i]) for j in range(len(sources))]
		# fig, ax = plot_image(source_data)
		# ax.scatter(np.array([k[0] for k in source_positions]),np.array([k[1] for k in source_positions]), color='r', marker='x')
		# breakpoint()

		#DO PHOTOMETRY AT UPDATED SOURCE POSITIONS FOR ALL SOURCES AND ALL APERTURES

		# y, x = np.ogrid[:2*an_out, :2*an_out] # pixel indices for cutouts

		annuli = CircularAnnulus(source_positions, an_in, an_out)
		annulus_masks = annuli.to_mask(method='center')
		for j in range(len(annuli)):
			source_sky_ADU[j,i] = np.mean(sigmaclip(annulus_masks[j].get_values(source_data),2,2)[0])

		# Do photometry
		# Set up apertures
		if type == 'fixed':
			apertures = [CircularAperture(source_positions,r=ap_radii[k]) for k in range(len(ap_radii))]

		# check for non-linear/saturated pixels in the apertures 
		# just do in the smallest aperture for now  
		aperture_masks = apertures[0].to_mask(method='center')
		for j in range(len(apertures[0])):
			ap_cutout = aperture_masks[j].multiply(source_data)
			ap_pix_vals = ap_cutout[ap_cutout!=0]
			non_linear_flags[:,j,i] = int(np.sum(ap_pix_vals>NONLINEAR_THRESHOLD)>0)
			saturated_flags[:,j,i] = int(np.sum(ap_pix_vals>SATURATION_THRESHOLD)>0)
		

		#TODO: IMPLEMENT VARIABLE PHOT
		# elif type == 'variable':
		# 	apertures = [CircularAperture(source_positions,r=ap_radii[k]*smoothed_fwhm_pix[i]) for k in range(len(ap_radii))]
		
		phot_table = aperture_photometry(source_data, apertures)
		
		# Calculate sky-subtracted flux
		for k in range(len(ap_radii)):
			ap_area = apertures[k].area
			source_minus_sky_ADU[k,:,i] = phot_table[f'aperture_sum_{k}']-source_sky_ADU[:,i]*ap_area
			# source_minus_sky_e[k,:,i] = source_minus_sky_ADU[k,:,i]*GAIN

			#Calculation scintillation 
			scintillation_abs_e = scintillation_rel * source_minus_sky_ADU[k,:,i]*GAIN
			
			# Calculate uncertainty
			source_minus_sky_err_e = np.sqrt(source_minus_sky_ADU[k,:,i]*GAIN+ source_sky_ADU[:,i]*ap_area*GAIN + DARK_CURRENT*EXPTIME*ap_area + ap_area*READ_NOISE**2 + scintillation_abs_e**2)
			source_minus_sky_err_ADU[k,:,i] = source_minus_sky_err_e / GAIN
				
		
		# #Create ensemble ALCs (summed reference fluxes with no weighting) for each source
		# all_ref_inds = np.arange(len(sources))
		# for l in range(len(sources)):
		# 	# Use all other stars
		# 	ref_inds = np.delete(all_ref_inds,l)
			
		# 	ensemble_alc_ADU[:,l] = np.sum(source_minus_sky_ADU[:,ref_inds],axis=1)
		# 	ensemble_alc_err_ADU[:,l] = (np.sum(source_minus_sky_err_ADU[:,ref_inds]**2,axis=1))**0.5
		# 	ensemble_alc_e[:,l] = ensemble_alc_ADU[:,l]*GAIN
		# 	ensemble_alc_err_e[:,l] = ensemble_alc_err_ADU[:,l]*GAIN
		# 	relative_flux[:,l] = source_minus_sky_ADU[:,l]/ensemble_alc_ADU[:,l]
		# 	relative_flux_err[:,l] = ((source_minus_sky_err_ADU[:,l]/ensemble_alc_ADU[:,l])**2+(source_minus_sky_ADU[:,l]*ensemble_alc_err_ADU[:,l]/(ensemble_alc_ADU[:,l]**2))**2)**0.5

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
			# output_list.append([f'{val:.7f}' for val in source_minus_sky_e[i,j]])
			# output_header.append(source_name+' Source-Sky e')
			# output_list.append([f'{val:.7f}' for val in source_minus_sky_err_e[i,j]])
			# output_header.append(source_name+' Source-Sky Error e')

			# output_list.append([f'{val:.7f}' for val in ensemble_alc_ADU[i,j]])
			# output_header.append(source_name+' Ensemble ALC ADU')
			# output_list.append([f'{val:.7f}' for val in ensemble_alc_err_ADU[i,j]])
			# output_header.append(source_name+' Ensemble ALC Error ADU')
			# output_list.append([f'{val:.7f}' for val in ensemble_alc_e[i,j]])
			# output_header.append(source_name+' Ensemble ALC e')
			# output_list.append([f'{val:.7f}' for val in ensemble_alc_err_e[i,j]])
			# output_header.append(source_name+' Ensemble ALC Error e')
			# output_list.append([f'{val:.10f}' for val in relative_flux[i,j]])
			# output_header.append(source_name+' Relative Flux')
			# output_list.append([f'{val:.10f}' for val in relative_flux_err[i,j]])
			# output_header.append(source_name+' Relative Flux Error')

			output_list.append([f'{val:.7f}' for val in source_sky_ADU[j]])
			output_header.append(source_name+' Sky ADU')
			# output_list.append([f'{val:.7f}' for val in source_sky_e[j]])
			# output_header.append(source_name+' Sky e')

			# output_list.append([f'{val:.4f}' for val in source_x_fwhm_arcsec[j]])
			# output_header.append(source_name+' X FWHM Arcsec')
			# output_list.append([f'{val:.4f}' for val in source_y_fwhm_arcsec[j]])
			# output_header.append(source_name+' Y FWHM Arcsec')
			# output_list.append([f'{val:.4f}' for val in source_theta_radians[j]])
			# output_header.append(source_name+' Theta Radians')

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


	#Cut to sources that don't fall near the edges
	use_inds = np.where((res['Y pix'] > 30) & (res['Y pix']<2017) & (res['X pix'] > 30) & (res['X pix'] < 4065 ))[0]
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

def full_field_lc_post_processing(date, target, ffname, overwrite=False, min_refs=50, max_source_over_alc=0.05):

	GAIN = 5.9
	READ_NOISE = 18.5
	DARK_CURRENT = 0.19

	source_file = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/full_field_photometry/full_field_sources.csv'
	source_df = pd.read_csv(source_file)
	n_sources = len(source_df)
	lc_files = np.array(glob(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/full_field_photometry/*phot*.csv'))

	# If there are fewer than min_refs + 1 sources in the field, update the min_refs criterion to be the number of sources - 1
	if n_sources-1 < min_refs:
		min_refs = n_sources -1
	
	# # use snr to determine the "best" aperture file
	# snrs = np.zeros((len(lc_files),n_sources))
	# for i in range(len(lc_files)):
	# 	print(lc_files[i])
	# 	phot_df = pd.read_csv(lc_files[i])
	# 	for j in range(n_sources):
	# 		snrs[i,j] = np.median(phot_df[f'Source {j+1} Source-Sky ADU'] / phot_df[f'Source {j+1} Source-Sky Error ADU'])
	
	# breakpoint()

	#NOTE: fix this loop 
	for ii in range(3, len(lc_files)):
		phot_df = pd.read_csv(lc_files[ii])
		times = np.array(phot_df['BJD TDB'])
		fluxes = np.zeros((len(phot_df), len(source_df)))
		flux_errs = np.zeros_like(fluxes)
		times_alc = np.zeros_like(fluxes)
		corr_fluxes_alc = np.zeros_like(fluxes)
		corr_flux_errs_alc = np.zeros_like(fluxes)
		times_zp = np.zeros_like(fluxes)
		corr_fluxes_zp = np.zeros_like(fluxes)
		corr_flux_errs_zp = np.zeros_like(fluxes)
		skies = np.zeros_like(fluxes)
		nl_flags = np.zeros(len(source_df),dtype='int')
		for jj in range(len(source_df)):
			fluxes[:,jj] = np.array(phot_df[f'Source {jj+1} Source-Sky ADU'])
			flux_errs[:,jj] = np.array(phot_df[f'Source {jj+1} Source-Sky Error ADU'])
			nl_flags[jj] =  sum(phot_df[f'Source {jj+1} Non-Linear Flag']) > 0
			skies[:,jj] = np.array(phot_df[f'Source {jj+1} Sky ADU'])

		brightness_weights = 1/np.median(fluxes,axis=0)
		brightness_weights /= np.nansum(brightness_weights)

		avg_fluxes = np.zeros(n_sources)
		avg_skies = np.zeros(n_sources)
		measured_errs_alc = np.zeros(n_sources)
		measured_errs_zp = np.zeros(n_sources)
		expected_errs = np.zeros(n_sources)
		source_x_save = np.zeros(n_sources)
		source_y_save = np.zeros(n_sources)

		for jj in range(1078, n_sources):
			print(jj)
			source_bp_rp = source_df['bp_rp'][jj]
			source_x = source_df['X pix'][jj]
			source_y = source_df['Y pix'][jj]
			source_x_save[jj] = source_x
			source_y_save[jj] = source_y
			nl_inds = np.where(phot_df[f'Source {jj+1} Non-Linear Flag'] == 1)[0]
			fluxes[nl_inds, jj] = np.nan
			flux_errs[nl_inds, jj] = np.nan
			source_flux = fluxes[:, jj]
			source_flux_err = flux_errs[:, jj]

			if sum(np.isnan(source_flux)) >= len(source_flux)-1:
				continue

			# select reference stars for this source based on brightness, color, distance, raw flux variance, and 
			# in this scheme, lower weight means a given source is more likely to be chosen as a reference star

			# give lower weight to start with similar colors
			color_weights = np.array(abs(source_df['bp_rp']-source_bp_rp))
			color_weights /= np.nansum(color_weights)
			
			# give lower weight to nearby stars 
			distances = np.array(((source_df['X pix']-source_x)**2+(source_df['Y pix']-source_y)**2)**0.5)
			distance_weights = distances / np.nansum(distances)

			# give lower weight to stars with low variance in raw normalized flux
			var_weights = np.std(fluxes/np.median(fluxes,axis=0),axis=0)**2
			var_weights /= np.nansum(var_weights)

			total_weights = (brightness_weights**2 + color_weights**2 + distance_weights**2 + var_weights**2 + nl_flags)**0.5
			sort_inds = np.argsort(total_weights)
			sort_inds = np.delete(sort_inds, np.where(sort_inds == jj))

			# loop over reference stars sorted by the weighting scheme
			# keep going until min_refs have been used and the ensemble alc contains an avg 10x as many counts as the source flux
			kk = 0
			ensemble_alc = np.zeros_like(source_flux, dtype='float64')
			avg_source_over_alc = 999. # initialize
			ref_inds = []
			while (kk < min_refs) or (avg_source_over_alc > max_source_over_alc):
				ensemble_alc += fluxes[:, sort_inds[kk]]
				avg_source_over_alc = np.median(source_flux / ensemble_alc)
				ref_inds.extend([sort_inds[kk]])
				kk += 1
				if kk == n_sources - 1: # break out of the loop if you have used up all the refs
					break

			ref_inds = np.array(ref_inds)
			

			# Create an array of fluxes with the selected reference stars
			# By convention, insert the 'target' flux as the 0th entry in this array
			ref_inds = np.insert(ref_inds, 0, jj)
			flux_arr = fluxes[:,ref_inds]
			flux_err_arr = flux_errs[:,ref_inds]

			#CORRECTION METHOD 1: WEIGHTED ALC

			# generate weighted alc for this source with the selected reference stars
			weights, weighted_alc, weighted_alc_err = source_weights(flux_arr, flux_err_arr)

			# correct the source flux with the weighted alc
			alc_corr_flux = source_flux / weighted_alc
			alc_corr_flux_err = quotient_uncertainty(source_flux, source_flux_err, weighted_alc, weighted_alc_err)
			renorm = np.nanmedian(alc_corr_flux)
			alc_corr_flux /= renorm 
			alc_corr_flux_err /= renorm

			avg_fluxes[jj] = np.median(source_flux)*GAIN
			avg_skies[jj] = np.median(skies) * GAIN

			# sigma clip the alc-corrected flux
			v, l, h = sigmaclip(alc_corr_flux[~np.isnan(alc_corr_flux)]) 
			sc_inds = np.where((alc_corr_flux < l) | (alc_corr_flux > h))[0]
			alc_corr_flux[sc_inds] = np.nan
			alc_corr_flux_err[sc_inds] = np.nan

			# save the alc-corrected arrays 
			measured_errs_alc[jj] = np.nanstd(alc_corr_flux)*1e6
			expected_errs[jj] = np.nanmean(alc_corr_flux_err)*1e6
			times_alc[:,jj] = times
			corr_fluxes_alc[:,jj] = alc_corr_flux
			corr_flux_errs_alc[:,jj] = alc_corr_flux_err


			#CORRECTION METHOD 2: ZERO POINT CORRECTION
			bjds, flux, err, regressors, regressors_err, cs, c_unc, weights = mearth_style_pat_weighted(times, source_flux, source_flux_err,flux_arr[:,1:].T, flux_err_arr[:,1:].T)

			
			renorm = np.nanmedian(flux)
			flux /= renorm
			err /= renorm

			# sigma clip the zero-point-corrected flux
			v, l, h = sigmaclip(flux[~np.isnan(flux)])
			sc_inds = np.where((flux < l) | (flux > h))[0]

			breakpoint()
			flux[sc_inds] = np.nan
			err[sc_inds] = np.nan

			# save the zero-point-corrected arrays
			measured_errs_zp[jj] = np.nanstd(flux)*1e6
			times_zp[:,jj] = bjds
			corr_fluxes_zp[:,jj] = flux
			corr_flux_errs_zp[:,jj] = err

			breakpoint()


		# use_inds = np.where(avg_fluxes != 0)[0]
		# avg_fluxes = avg_fluxes[use_inds]
		# avg_skies = avg_skies[use_inds]
		# measured_errs_alc = measured_errs_alc[use_inds]
		# measured_errs_zp = measured_errs_zp[use_inds]
		# expected_errs = expected_errs[use_inds]

		# resort arrays by average source flux 
		sort = np.argsort(avg_fluxes)
		avg_fluxes = avg_fluxes[sort]
		avg_skies = avg_skies[sort]
		measured_errs_alc = measured_errs_alc[sort]
		measured_errs_zp = measured_errs_zp[sort]
		expected_errs = expected_errs[sort]

		times_alc = times_alc[:,sort]
		corr_fluxes_alc = corr_fluxes_alc[:,sort]
		corr_flux_errs_alc = corr_flux_errs_alc[:,sort]
		times_zp = times_zp[:,sort]
		corr_fluxes_zp = corr_fluxes_zp[:,sort]
		corr_flux_errs_zp = corr_flux_errs_zp[:,sort]

		breakpoint() 

		# calculate expected noise using average source fluxes, average sky, read noise, and scintillation
		# TODO: Have to sort by avg flux or use an array of generated fluxes
		ap_rad = int(lc_files[ii].split('/')[-1].split('.')[0].split('_')[-1])
		n_pix = np.pi*ap_rad**2

		expected_source_photon_noise = (avg_fluxes**-0.5)*1e6
		expected_sky_photon_noise = (avg_skies*n_pix)**0.5/(avg_fluxes) * 1e6
		expected_read_noise = (n_pix*READ_NOISE**2)**0.5/(avg_fluxes) * 1e6
		expected_scintillation_noise = 0.09*(130)**(-2/3)*np.mean(phot_df['Airmass'])**(7/4)*(2*np.median(phot_df['Exposure Time']))**(-1/2)*np.exp(-2306/8000) * 1e6

		total_expected_noise = ((avg_fluxes + n_pix*(avg_skies + READ_NOISE**2))**0.5 / avg_fluxes * 1e6) + expected_scintillation_noise

		
		# do some plots to compare the alc and zero-point correction approaches 
		fig, ax = plt.subplots(2,1,figsize=(9,8), sharex=True, sharey=True)

		ax[0].plot(avg_fluxes, expected_source_photon_noise, lw=2)
		ax[0].plot(avg_fluxes, expected_sky_photon_noise, lw=2)
		ax[0].plot(avg_fluxes, np.zeros_like(avg_fluxes)+expected_scintillation_noise, lw=2)
		ax[0].plot(avg_fluxes, total_expected_noise, lw=2)
		ax[0].plot(avg_fluxes, measured_errs_alc, marker='.', ls='',color='#b0b0b0', ms=5)
		# ax[0].plot(avg_fluxes, expected_errs, lw=2, color='k')
		
		h2d = ax[0].hist2d(avg_fluxes, measured_errs_alc, bins=[np.logspace(3, 7.2, 100), np.logspace(2,7,50)], cmin=2, norm=colors.PowerNorm(0.5), zorder=3, alpha=1, lw=0)
		ax[0].grid(True, alpha=0.5,lw=0.5)
		
		cb = fig.colorbar(h2d[3], ax=ax[0], pad=0.02, label='N$_{sources}$')

		ax[0].set_ylabel('$\sigma$ (ppm)', fontsize=14)

		ax[1].plot(avg_fluxes, expected_source_photon_noise, lw=2)
		ax[1].plot(avg_fluxes, expected_sky_photon_noise, lw=2)
		ax[1].plot(avg_fluxes, np.zeros_like(avg_fluxes)+expected_scintillation_noise, lw=2)
		ax[1].plot(avg_fluxes, total_expected_noise, lw=2)
		ax[1].plot(avg_fluxes, measured_errs_zp, marker='.', ls='',color='#b0b0b0', ms=5)
		h2d = ax[1].hist2d(avg_fluxes, measured_errs_zp, bins=[np.logspace(3, 7.2, 100), np.logspace(2,7,50)], cmin=2, norm=colors.PowerNorm(0.5), zorder=3, alpha=1, lw=0)
		cb = fig.colorbar(h2d[3], ax=ax[1], pad=0.02, label='N$_{sources}$')
		# ax[1].plot(avg_fluxes, expected_errs, lw=2, color='k')

		ax[1].grid(True, alpha=0.5, lw=0.5)
		ax[1].set_xlabel('Avg. source counts (photons)',fontsize=14)

		ax[0].invert_xaxis()
		ax[0].set_yscale('log')
		ax[0].set_xscale('log')
		
		plt.tight_layout()

		noise_sort = np.argsort(measured_errs_zp[~np.isnan(measured_errs_zp)])[::-1]
		breakpoint()
	return 

def source_weights(flux, flux_errs, crude_convergence=1e-4, fine_convergence=1e-6, max_iters=20, bad_ref_threshold=6):
	n_sources = flux.shape[1]-1
	n_ims = flux.shape[0]
	w_init = np.ones(n_sources) / n_sources

	#Do a 'crude' loop to first figure out which refs should be totally tossed out
	corr_fluxes = np.zeros((n_ims,n_sources))
	w_old = copy.deepcopy(w_init)
	delta_weights = np.ones(n_sources)
	count = 0 

	indices_array = np.array([[j for j in range(n_sources) if j != i] for i in range(n_sources)], dtype='uint16') # Each entry represents the indices of all the sources in the image excluding the entry in question

	#global_salc = np.sum(raw_fluxes*w_old,axis=1) # The ALC if every source is used
	while sum(delta_weights>crude_convergence)>0:
		#print(f'{count+1}')
		w_new = np.zeros(n_sources)


		salcs = np.array([np.nansum(flux[:,indices_array[i]]*(w_old[indices_array[i]]/np.nansum(w_old[indices_array[i]])),axis=1) for i in range(n_sources)])
		corr_fluxes = np.array([flux[:,i]/salcs[i] for i in range(n_sources)])
		corr_fluxes = (corr_fluxes.T/np.nanmedian(corr_fluxes,axis=1))
		w_new = 1/(np.nanstd(corr_fluxes,axis=0)**2)
		w_new /= np.nansum(w_new)
		
		delta_weights = abs(w_old - w_new)
		w_old = w_new
		count += 1
		if count == max_iters:
			break 

	w_crude = w_new 
	#Now determine which refs should be totally excluded based on the ratio of their measured/expected noise. 
	use_ref_inds = np.ones(n_sources,dtype='int')

	for i in range(1, n_sources):
		corr_flux = corr_fluxes[:,i]
		raw_flux = flux[:,i]
		raw_flux_err = flux_errs[:,i]
		corr_flux = corr_flux[np.where(corr_flux!=0)]
		v, l, h = sigmaclip(corr_flux[~np.isnan(corr_flux)])
		use_inds = np.where((corr_flux>l)&(corr_flux<h))[0]
		norm = np.nanmean(raw_flux[use_inds])
		#raw_flux_norm = raw_flux/norm 
		raw_flux_err_norm = raw_flux_err/norm
		expected = np.nanmean(raw_flux_err_norm)
		measured = np.nanstd(corr_flux[use_inds])
		if measured/expected > bad_ref_threshold:
			use_ref_inds[i] = 0
		#breakpoint()

	#Now do a more intensive loop with bad references given 0 weight. 
	w_old *= use_ref_inds
	w_old /= np.nansum(w_old)
	corr_fluxes = np.zeros((n_ims,n_sources))
	delta_weights = np.ones(n_sources)
	count = 0 
	while sum(delta_weights>fine_convergence)>0:
		#print(f'{count+1}')
		w_new = np.zeros(n_sources)
		salcs = np.array([np.nansum(flux[:,indices_array[i]]*(w_old[indices_array[i]]/np.nansum(w_old[indices_array[i]])),axis=1) for i in range(n_sources)])
		corr_fluxes = np.array([flux[:,i]/salcs[i] for i in range(n_sources)])
		corr_fluxes = (corr_fluxes.T/np.nanmedian(corr_fluxes,axis=1))
		w_new = 1/(np.nanstd(corr_fluxes,axis=0)**2)
		w_new /= sum(w_new)
	
		# w_new /= np.nansum(w_new)
		delta_weights = abs(w_old - w_new)
		w_old = w_new
		count += 1
		if count == max_iters:
			break
	
	alc = np.nansum(w_new*flux[:,1:],axis=1)
	alc_err = np.sqrt(np.nansum((w_new*flux_errs[:,1:])**2,axis=1))
	return w_new, alc, alc_err

def mearth_style_pat_weighted(bjds, flux, err, regressors, regressors_err):

	""" Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
	""" it's called "mearth_style" because it's inspired by the mearth pipeline """

	#TODO: Is the mask too agressive? 
	#TODO: The code is overwriting "source_flux" in the full_field_lc_post_processing function...
	bjds = copy.deepcopy(bjds)
	flux = copy.deepcopy(flux)
	err = copy.deepcopy(err)
	regressors = copy.deepcopy(regressors)
	regressors_err = copy.deepcopy(regressors_err)

	#pdb.set_trace()
	mask = np.ones_like(flux, dtype='bool')  # initialize a bad data mask

	mask[np.where(flux <= 0)[0]] = 0  # if target counts are less than or equal to 0, this cadence is bad

	# if one of the reference stars has negative flux, this cadence is also bad
	for ii in range(regressors.shape[0]): # for each comp star in 2D array of comps 
		mask[np.where(regressors[ii, :] <= 0)[0]] = 0 # mask out all the places where comp flux values below or equal to 0 


	# # OLD: DROP MASKED INDS
	# regressors = regressors[:, mask]
	# regressors_err = regressors_err[:, mask]
	# flux = flux[mask]
	# err = err[mask]
	# bjds = bjds[mask]
		
	# NEW: NaN masked inds
	regressors[:, ~mask] = np.nan
	regressors_err[:, ~mask] = np.nan
	flux[~mask] = np.nan
	err[~mask] = np.nan
	bjds[~mask] = np.nan

	tot_regressor = np.nansum(regressors, axis=0)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure
	c0s = -2.5*np.log10(np.nanpercentile(tot_regressor, 90)/tot_regressor)  # initial guess of magnitude zero points
	mask = np.ones_like(c0s, dtype='bool')  # initialize another bad data mask
	# mask[np.where(c0s < -0.24)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad
	mask[np.where(c0s < -1.5)[0]] = 0  # PT - I think 20% mask is maybe appropriate for a data set consisting of many nights, but I think it's too aggressive for a single night. Test using a threshold of -2.5 (should be like a 75% decrement)

	
	# # apply mask
	# regressors = regressors[:, mask]
	# regressors_err = regressors_err[:, mask]
	# flux = flux[mask]
	# err = err[mask]
	# bjds = bjds[mask]
	regressors[:, ~mask] = np.nan
	regressors_err[:, ~mask] = np.nan
	flux[~mask] = np.nan
	err[~mask] = np.nan
	bjds[~mask] = np.nan

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
	weights_init = np.nanmean(regressors, axis=1)
	weights_init /= np.nansum(weights_init) # Normalize weights to sum to 1

	
	cs = np.matmul(weights_init, cs) # Take the *weighted mean* across all regressors
	# cs = np.nanmedian(cs, axis=0)  # take the median across all regressors

	# one more bad data mask: don't trust cadences where the regressors have big discrepancies
	mask = np.ones_like(flux, dtype='bool')
	mask[np.where(c_noise > 3*np.median(c_noise))[0]] = 0

	# apply mask
	# flux = flux[mask]
	# err = err[mask]
	# bjds = bjds[mask]
	# cs = cs[mask]
	# c_unc = c_unc[mask]
	# regressors = regressors[:, mask]
	# regressors_err = regressors_err[:, mask]

	regressors[:, ~mask] = np.nan
	regressors_err[:, ~mask] = np.nan
	flux[~mask] = np.nan
	err[~mask] = np.nan
	bjds[~mask] = np.nan
	cs[~mask] = np.nan
	c_unc[~mask] = np.nan

	cs_original = cs
	delta_weights = np.zeros(len(regressors))+999 # initialize
	threshold = 1e-5 # delta_weights must converge to this value for the loop to stop
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
			corr_jj /= np.nanmean(corr_jj)
			stddevs[jj] = np.nanstd(corr_jj)

		# corrected_regressors = regressors * 10**(-cs/2.5)
		# corrected_regressors /= np.mean(corrected_regressors)
		# stddevs = np.std(corrected_regressors, axis=1)
		weights_new = 1/stddevs**2
		weights_new /= np.nansum(weights_new)
		delta_weights = abs(weights_new-weights_old)
		# cs = -2.5*np.log10(phot_regressor[:,None]/regressors)
		# cs = np.matmul(weights_new, cs)
		weights_old = weights_new

	cs = -2.5*np.log10(phot_regressor[:,None]/regressors)
	cs = np.matmul(weights_new, cs)
	
	weights = weights_new
	corrected_regressors = regressors * 10**(-cs/2.5)

	# flux_original = copy.deepcopy(flux)
	err = 10**(cs/(-2.5)) * np.sqrt(err**2 + (c_unc*flux*np.log(10)/(-2.5))**2)  # propagate error
	flux *= 10**(cs/(-2.5))  #cs, adjust the flux based on the calculated zero points

	return bjds, flux, err, regressors, regressors_err, cs, c_unc, weights



def main(raw_args=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
	ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")
	ap.add_argument("-ffname", required=True, help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
	ap.add_argument("-centroid",required=False,default=True,help="Whether or not to centroid during aperture photometry.",type=str)
	ap.add_argument("-ap_radius_low",required=False,default=5,help="Smallest aperture radius to try. All radii with sizes between ap_radius_low and ap_radius_high will be generated.")
	ap.add_argument("-ap_radius_high",required=False,default=15,help="Largest aperture radius to try. All radii with sizes between ap_radius_low and ap_radius_high will be generated.")
	ap.add_argument("-an_in",required=False,default=35,help="Inner background annulus radius.")
	ap.add_argument("-an_out",required=False,default=65,help="Outer background annulus radius.")
	args = ap.parse_args(raw_args)

	#Access observation info
	date = args.date
	target = args.target
	ffname = args.ffname
	centroid = t_or_f(args.centroid)
	ap_rad_low = args.ap_radius_low
	ap_rad_high = args.ap_radius_high
	an_in = args.an_in
	an_out = args.an_out
	ap_radii = np.arange(ap_rad_low, ap_rad_high+1)

	flattened_files = get_flattened_files(date, target, ffname)

	do_all_photometry(flattened_files, target, ap_radii=ap_radii, an_in=an_in, an_out=an_out, centroid=centroid, type='fixed')

	full_field_lc_post_processing(date, target, ffname, min_refs=50)

	return 

if __name__ == '__main__':
	main()
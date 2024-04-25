#!/usr/bin/env python

import logging
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
from astropy.coordinates import SkyCoord, get_body, AltAz
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
from astroquery.vizier import Vizier
from photutils import make_source_mask
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import BasicPSFPhotometry, IntegratedGaussianPRF, DAOGroup, extract_stars, EPSFBuilder
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, EllipticalAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_1dg, centroid_2dg, centroid_com, centroid_quadratic, centroid_sources
from scipy.stats import sigmaclip, pearsonr, linregress
from scipy.spatial.distance import cdist
from scipy.signal import correlate2d, fftconvolve, savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
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
import warnings
from etc_pat import etc 

# Suppress all Astropy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="astropy")

def get_flattened_files(date, target, ffname):
	#Get a list of data files sorted by exposure number
	'''
		PURPOSE: 
			Creates a list of flattened files associated with an input date, target, and ffname
		INPUTS:
			date (str): the dates of the observations in calendar YYYYMMDD format
			target (str): the name of the target
			ffname (str): the name of the flat field file
		OUTPUTS:
			sorted_files (numpy array): array of flattened files ordered by exposure number
	'''

	ffolder = '/data/tierras/flattened/'+date+'/'+target+'/'+ffname
	red_files = []
	for file in os.listdir(ffolder): 
		if '_red.fit' in file:
			red_files.append(ffolder+'/'+file)
	sorted_files = np.array(sorted(red_files, key=lambda x: int(x.split('.')[1])))
	sorted_files = np.array([Path(i) for i in sorted_files])

	# logger.debug(f'Found {len(sorted_files)} files for {target} on {date}')
	return sorted_files 

def plot_image(data,use_wcs=False,cmap_name='viridis'):
	'''
		PURPOSE: 
			Does a quick plot of a Tierras image (or any 2D array)
		INPUTS:
			data (2D array): array of data to be plotted 
			use_wcs (bool): whether or not to plot using WCS coordinates instead of pixel (TODO: this is currently broken)
			cmap_name (str): the name of whatever pyplot colormap you want to use
		OUTPUTS:
			fig, ax (matplotlib objects): the figure/axis objects associated with the plot

	'''

	#TODO: Do we want the image orientation to match the orientation on-sky? 
	#TODO: WCS and pixel coordinates simultaneously?
	
	#if use_wcs:
	#	wcs = WCS(header)
	
	#norm = ImageNormalize(data[4:2042,:], interval=interval) #Ignore a few rows near the top/bottom for the purpose of getting a good colormap

	norm = simple_norm(data, stretch='linear', min_percent=1,max_percent=99.5)
	cmap = get_cmap(cmap_name)
	im_scale = 2
	
	if use_wcs:
		fig, ax = plt.subplots(1,1,figsize=(im_scale*8,im_scale*4),subplot_kw={'projection':wcs})
		ra = ax.coords['ra']
		dec = ax.coords['dec']
		ra.set_major_formatter('hh:mm:ss.s')
		ra.set_separator(':')
		ra.set_ticks(spacing=2*u.arcmin)
		ra.set_ticklabel(exclude_overlapping=False)
		ra.display_minor_ticks(True)
		ra.set_minor_frequency(4)
		dec.set_major_formatter('dd:mm:ss.s')
		dec.set_separator(':')
		dec.set_ticks(spacing=2*u.arcmin)
		dec.set_ticklabel(exclude_overlapping=False)
		dec.display_minor_ticks(True)
		dec.set_minor_frequency(4)
	else:
		fig, ax = plt.subplots(1,1,figsize=(im_scale*8,im_scale*4))
		ax.set_xticks(np.arange(0,4500,250))
		ax.set_yticks(np.arange(0,2500,250))
		ax.set_xlim(0,data.shape[1])
		ax.set_ylim(0,data.shape[0])
	

	im = ax.imshow(data,origin='lower',norm=norm,cmap=cmap,interpolation='none')
	#im = ax.imshow(data,origin='lower',vmin=-15,vmax=30,cmap=cmap,interpolation='none')

	ax.grid(alpha=0.2,color='w',lw=1)
	ax.set_aspect('equal')
	
	#Add colorbar
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right',size='4%',pad=0.1)
	cb = fig.colorbar(im,cax=cax,orientation='vertical')
	cb.set_label('ADU',fontsize=14)
	
	plt.tight_layout()
	return fig, ax

def source_selection(file_list, logger, min_snr=10, edge_limit=20, plot=False, plate_scale=0.432, overwrite=False):	
	'''
		PURPOSE: identify sources in a Tierras field over a night
		INPUTS: 
			file_list (array): list of paths to Tierras images. If no existing stacked image exists in the /data/tierras/targets/{field} directory, one will be made using images from this list

			target_position (tuple, optional): the user-specified target pixel position. If (0,0), the code will use the RA/Dec coordinates of the target in the header of the stacked field image and the associated WCS to estimate its position. 

			plot (bool, optional): whether or not to produce/save plots of the selected reference stars in the field and a color-magnitude diagram

			overwrite (bool, optional): whether or not to restore previously saved output from the /data/tierras/targets/{field}/ directory

			dimness_limit (float, optional): the minimum mean flux ratio in Gaia Rp-band that a candidate reference can have to the target and still be retained as a reference

			nearness_limit (float, optional): the minimum distance (in pixels) that a candidate reference star must be from all other sources to still be retained as a reference

			edge_limit (float, optional): the minimum distance (in pixels) that a candidate reference star can be from the edge of the detector and still be retained as a reference

			targ_distance_limit (float, optional): the maximum distance (in pixels) that a candidate reference star can be from the target and still be retained as a reference
		
		OUTPUTS:
			output_df (pandas DataFrame): a data frame containing the target and reference stars, with the target as the first entry

	'''
	
	# write out the source csv file
	date = file_list[0].parent.parent.parent.name 
	target = file_list[0].parent.parent.name 
	ffname = file_list[0].parent.name 
	source_path = f'/data/tierras/photometry/{date}/{target}/{ffname}/{date}_{target}_sources.csv'

	if os.path.exists(source_path) and not overwrite:
		logger.info(f'Restoring existing sources from {source_path}.')
		source_df = pd.read_csv(source_path)
		return source_df

	# use the wcs to evaluate the coordinates of the central pixel in images over the night to determine average pointing
	central_ras = []
	central_decs = []
	for ii in range(len(file_list)):
		with fits.open(file_list[ii]) as hdul:
			wcs = WCS(hdul[0].header)
		im_shape = hdul[0].shape
		sc = wcs.pixel_to_world(im_shape[1]/2-1, im_shape[0]/2-1)
		central_ras.append(sc.ra.value)
		central_decs.append(sc.dec.value)

	# do a 3-sigma clipping and take the mean of the ra/dec lists to represent the average field center over the night 	
	v1, l1, h1 = sigmaclip(central_ras, 3, 3)
	avg_central_ra = np.mean(v1)
	v2, l2, h2 = sigmaclip(central_decs, 3, 3)
	avg_central_dec = np.mean(v2)

	logger.debug(f'Average central RA/Dec: {avg_central_ra:.6f}, {avg_central_dec:.6f}')	

	# identify the image closest to the average position 
	central_im_file = file_list[np.argmin(((avg_central_ra-central_ras)**2+(avg_central_dec-central_decs)**2)**0.5)]
	with fits.open(central_im_file) as hdul:
		central_im = hdul[0].data
		header = hdul[0].header
		wcs = WCS(header)
	
	if plot:
		fig, ax = plot_image(central_im)

	# get the epoch of these observations 
	tierras_epoch = Time(header['TELDATE'],format='decimalyear')

	logger.debug(f'Epoch of Tierras observations: {tierras_epoch.value:.6f}')
	# set up the region on sky that we'll query in Gaia
	# to be safe, set the width/height to be a bit larger than the estimates from plate scale alone, and cut to sources that actually fall on the chip after the query is complete
	#	after the query is complete
	coord = SkyCoord(avg_central_ra*u.deg, avg_central_dec*u.deg)
	width = u.Quantity(plate_scale*im_shape[0],u.arcsec)*1.5
	height = u.Quantity(plate_scale*im_shape[1],u.arcsec)*1.5

	# use the exposure time calculator to get an estimate for the rp magnitude limit that will give the desired minimum SNR 
	# this assumes that all the images are taken at the same exposure time
	rp_mags = np.arange(7,22,0.1)
	snrs = np.zeros(len(rp_mags))
	for ii in range(len(rp_mags)):
		rp_mag, snr, exptime = etc(rp_mag=rp_mags[ii], exptime=header['EXPTIME'])
		snrs[ii] = snr 
	mag_limit = rp_mags[np.argmin(abs(snrs-min_snr))]
	logger.debug(f'Using a Gaia Rp mag limit of {mag_limit:.1f} to get sources with a minimum SNR of {min_snr}')	

	# query Gaia DR3 for all the sources in the field brighter than the calculated magnitude limit
	job = Gaia.launch_job_async("""
									SELECT source_id, ra, ra_error, dec, dec_error, ref_epoch, pmra, pmra_error, pmdec, pmdec_error, parallax, parallax_error, parallax_over_error, ruwe, phot_bp_mean_mag, phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_g_mean_flux, phot_g_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error, bp_rp, bp_g, g_rp, phot_variable_flag,radial_velocity, radial_velocity_error, non_single_star, teff_gspphot, logg_gspphot, mh_gspphot
							 		FROM gaiadr3.gaia_source as gaia
									WHERE gaia.ra BETWEEN {} AND {} AND
											gaia.dec BETWEEN {} AND {} AND
							 				gaia.phot_rp_mean_mag <= {}
									ORDER BY phot_rp_mean_mag ASC
								""".format(coord.ra.value-width.to(u.deg).value/2, coord.ra.value+width.to(u.deg).value/2, coord.dec.value-height.to(u.deg).value/2, coord.dec.value+height.to(u.deg).value/2, mag_limit)
								)
	res = job.get_results()
	res['SOURCE_ID'].name = 'source_id' # why does this get returned in all caps? 

	# Do a separate search for objects in the Bailer-Jones 'photogeo' catalog
	job = Gaia.launch_job_async("""SELECT
								source_id, r_med_geo, r_lo_geo, r_hi_geo, r_med_photogeo, r_lo_photogeo, r_hi_photogeo,
								phot_g_mean_mag - 5 * LOG10(r_med_geo) + 5 AS qg_geo,
								phot_g_mean_mag - 5 * LOG10(r_med_photogeo) + 5 AS gq_photogeo
									FROM (
										SELECT * FROM gaiadr3.gaia_source as gaia

										WHERE gaia.ra BETWEEN {} AND {} AND
											  gaia.dec BETWEEN {} AND {} AND
							 				  gaia.phot_rp_mean_mag <= {}

										OFFSET 0
									) AS edr3
									JOIN external.gaiaedr3_distance using(source_id)
									ORDER BY phot_rp_mean_mag ASC
								""".format(coord.ra.value-width.to(u.deg).value/2, coord.ra.value+width.to(u.deg).value/2, coord.dec.value-height.to(u.deg).value/2, coord.dec.value+height.to(u.deg).value/2, mag_limit)
								)
	res2 = job.get_results()	
	
	# add the Bailer-Jones data into the main table 
	for key in res2.keys()[1:]:
		res[key] = np.zeros(len(res))

	for i in range(len(res)):
		if res['source_id'][i] in res2['source_id']:
			ind = np.where(res['source_id'][i] == res2['source_id'])[0][0]
			for key in res2.keys()[1:]:
				res[key][i] = res2[key][ind]
		else:
			for key in res2.keys()[1:]:
				res[key][i] = np.nan
	
	# cut to entries without masked pmra values; otherwise the crossmatch will break
	problem_inds = np.where(res['pmra'].mask)[0]

	# set the pmra, pmdec, and parallax of those indices to 0
	res['pmra'][problem_inds] = 0
	res['pmdec'][problem_inds] = 0
	res['parallax'][problem_inds] = 0

	# perform a crossmatch with 2MASS
	gaia_coords = SkyCoord(ra=res['ra'], dec=res['dec'], pm_ra_cosdec=res['pmra'], pm_dec=res['pmdec'], obstime=Time('2016',format='decimalyear'))
	v = Vizier(catalog="II/246",columns=['*','Date'], row_limit=-1)
	twomass_res = v.query_region(coord, width=width, height=height)[0]
	twomass_coords = SkyCoord(twomass_res['RAJ2000'],twomass_res['DEJ2000'])
	twomass_epoch = Time('2000-01-01')
	gaia_coords_tm_epoch = gaia_coords.apply_space_motion(twomass_epoch)
	gaia_coords_tierras_epoch = gaia_coords.apply_space_motion(tierras_epoch)

	idx_gaia, sep2d_gaia, _ = gaia_coords_tm_epoch.match_to_catalog_sky(twomass_coords)
	#Now set problem indices back to NaNs
	res['pmra'][problem_inds] = np.nan
	res['pmdec'][problem_inds] = np.nan
	res['parallax'][problem_inds] = np.nan
	
	# figure out source positions in the Tierras epoch 
	tierras_pixel_coords = wcs.world_to_pixel(gaia_coords_tierras_epoch)

	# add 2MASS data and pixel positions to the source table
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

	
	# determine which chip the sources fall on 
	# 0 = bottom, 1 = top 
	chip_inds = np.zeros(len(res),dtype='int')
	chip_inds[np.where(res['Y pix'] >= 1023)] = 1
	res.add_column(chip_inds, name='Chip')

	#Cut to sources that actually fall in the image
	use_inds = np.where((tierras_pixel_coords[0]>0)&(tierras_pixel_coords[0]<im_shape[1]-1)&(tierras_pixel_coords[1]>0)&(tierras_pixel_coords[1]<im_shape[0]-1))[0]
	res = res[use_inds]

	logger.debug(f'Found {len(res)} sources in Gaia query.')

	#Cut to sources that are away from the edges
	use_inds = np.where((res['Y pix'] > edge_limit) & (res['Y pix']<im_shape[0]-edge_limit-1) & (res['X pix'] > edge_limit) & (res['X pix'] < im_shape[1]-edge_limit-1))[0]
	logger.debug(f'Removed {len(res)-len(use_inds)} sources that are within {edge_limit} pixels of the detector edges.')
	res = res[use_inds]
	
	# remove ref stars that are too close to the bad columns or the divide between the detector halves
	bad_inds_col_1 = np.where((res['X pix'] >= 1431) & (res['X pix'] <= 1472) & (res['Y pix'] <= 1032))[0]
	res.remove_rows(bad_inds_col_1)
	logger.debug(f'Removed {len(bad_inds_col_1)} sources that were too near the bad pixel column in the lower detector half.')

	bad_inds_col_2 = np.where((res['X pix'] >= 1771) & (res['X pix'] <= 1813) & (res['Y pix'] >= 1023))[0]
	res.remove_rows(bad_inds_col_2)
	logger.debug(f'Removed {len(bad_inds_col_2)} sources that were too near the bad pixel column in the upper detector half.')

	bad_inds_half = np.where((res['Y pix'] >= 1019) & (res['Y pix'] <= 1032))[0]
	res.remove_rows(bad_inds_half)
	logger.debug(f'Removed {len(bad_inds_half)} sources that were too near the divide between the upper and lower detector halves.')

	logger.info(f'Found {len(res)} sources!')
	
	if plot:
		# ax.plot(all_field_sources['X pix'], all_field_sources['Y pix'], marker='x', ls='', color='tab:red')
		ax.plot(res['X pix'], res['Y pix'], marker='x', ls='', color='tab:red')

		fig1, ax1 = plt.subplots(1,1,figsize=(6,5))
		ax1.scatter(res['bp_rp'], res['gq_photogeo'], marker='x', color='tab:red')
		ax1.invert_yaxis()
		ax1.set_xlabel('B$_{p}-$R$_p$', fontsize=14)
		ax1.set_ylabel('M$_{G}$', fontsize=14)
		ax1.tick_params(labelsize=12)
		plt.tight_layout()
	
	# create the output dataframe consisting of the target as the 0th entry and the reference stars
	output_table = copy.deepcopy(res)

	output_df = output_table.to_pandas()
	output_df.to_csv(source_path, index=0)
	set_tierras_permissions(source_path)

	logger.debug(f'Saved source csv to {source_path}')
	return output_df

def load_bad_pixel_mask():
	#Load in the BPM. Code stolen from imred.py.
	bpm_path = '/home/jmejia/tierras/git/sicamd/config/badpix.mask'
	amplist = []
	sectlist = []
	vallist = []
	with open(bpm_path, "r") as mfp:
		for line in mfp:
			ls = line.strip()
			lc = ls.split("#", 1)
			ln = lc[0]
			if ln == "":
				continue
			amp, sect, value = ln.split()
			xl, xh, yl, yh = fits_section(sect)
			amplist.append(int(amp))
			sectlist.append([xl, xh, yl, yh])
			vallist.append(int(value))
	amplist = np.array(amplist, dtype='int')
	sectlist = np.array(sectlist, dtype='int')
	vallist = np.array(vallist, dtype='int')

	allamps = np.unique(amplist)

	namps = len(allamps)

	mask = [None] * namps

	for amp in allamps:
		ww = amplist == amp
		thissect = sectlist[ww,:]
		thisval = vallist[ww]

		nx = np.max(thissect[:,1])
		ny = np.max(thissect[:,3])

		img = np.ones([ny, nx], dtype=np.uint8)

		nsect = thissect.shape[0]

		for isect in range(nsect):
			xl, xh, yl, yh = thissect[isect,:]
			img[yl:yh,xl:xh] = thisval[isect]

		mask[amp-1] = img

	#Combine everything into one map.
	bad_pixel_mask = np.zeros((2048, 4096), dtype='uint8')
	bad_pixel_mask[0:1024,:] = mask[0]
	bad_pixel_mask[1024:,:] = mask[1]

	#Interchange 0s and 1s to match SEP/Astropy bad pixel mask convention. 
	bad_pixel_mask = np.where((bad_pixel_mask==0)|(bad_pixel_mask==1),bad_pixel_mask^1,bad_pixel_mask)

	return bad_pixel_mask

def align_and_stack_images(file_list, ref_image_buffer=10, n_ims_to_stack=20):
	#TODO: by default, will treat first image in the file list as the reference image, and stack the next 20 images to get a high snr image of the field.
	#	Not sure how to choose which should be the reference exposure programatically.
	#	Also not sure how many images we want to stack.
	target = file_list[0].name.split('.')[2].split('_')[0]

	bpm = load_bad_pixel_mask()

	#Determine which images to stack based on airmass
	airmasses = np.zeros(len(file_list))
	for i in range(len(file_list)):
		hdu = fits.open(file_list[i])[0]
		header = hdu.header
		airmasses[i] = header['AIRMASS']
	airmass_sort_inds = np.argsort(airmasses)
	reference_ind = airmass_sort_inds[0]
	
	#If the lowest airmass is within the first ref_image_buffer images, force it to ref_image_buffer
	if reference_ind < ref_image_buffer:
		reference_ind = ref_image_buffer

	#If it's within the last ref_image_buffer+n_ims_to_stack images, force it to be the closest image outside that range.
	if reference_ind > len(file_list)-ref_image_buffer-n_ims_to_stack:
		reference_ind = len(file_list)-ref_image_buffer-n_ims_to_stack

	if len(file_list) < n_ims_to_stack:
		reference_ind = int(len(file_list)/2)-1

	reference_hdu = fits.open(file_list[reference_ind])[0] 
	reference_image = reference_hdu.data
	reference_header = reference_hdu.header
	reference_header.append(('COMMENT',f'Reference image: {file_list[reference_ind].name}'), end=True)

	bkg = sep.Background(reference_image.byteswap().newbyteorder()) #TODO: why are the byteswap().newbyteorder() commands necessary?
	stacked_image_aa = np.zeros(reference_image.shape, dtype='float32')
	stacked_image_rp = np.zeros(reference_image.shape, dtype='float32')
	stacked_image_aa += reference_image - bkg.back()
	stacked_image_rp += reference_image - bkg.back()

	# #Do a loop over n_ims, aligning and stacking them. 
	# all_inds = np.arange(len(file_list))
	# inds_excluding_reference = np.delete(all_inds, ref_image_num) #Remove the reference image from the index list
	# #inds_to_stack = inds_excluding_reference[:n_ims_to_stack] #Count from 0...
	# inds_to_stack = inds_excluding_reference[ref_image_num:ref_image_num+n_ims_to_stack] #Count from ref_image_num...

	if len(airmass_sort_inds) < n_ims_to_stack:
		inds_to_stack = airmass_sort_inds
	else:
		inds_to_stack = airmass_sort_inds[reference_ind+1:reference_ind+1+n_ims_to_stack]
	print('Aligning and stacking images...')
	counter = 0 
	for i in inds_to_stack:
		print(f'{file_list[i]}, {counter+1} of {len(inds_to_stack)}.')
		source_hdu = fits.open(file_list[i])[0]
		source_image = source_hdu.data 
		
		#METHOD 1: using astroalign
		#Astroalign does image *REGISTRATION*, i.e., does not rely on header WCS.
		masked_source_image = np.ma.array(source_image,mask=bpm) #aa requires the use of numpy masked arrays to do bad pixel masking
		try:
			registered_image, footprint = aa.register(masked_source_image,reference_image)
		except:
			print(f'WARNING: no solution for {file_list[i]}, skipping.') #TODO: how to identify bad files BEFORE we try to stack?
			continue
		bkg_aa = sep.Background(registered_image)
		stacked_image_aa += registered_image-bkg_aa.back()
		
		# #METHOD 2: using reproject
		# #reproject uses WCS information in the fits headers to align images.
		# #It is much slower than aa and gives comparable results.
		# reprojected_image, footprint = rp.reproject_interp(source_hdu, reference_hdu.header)
		# print(f'rp time: {time.time()-t1:.1f} s')
		# bkg_rp = sep.Background(reprojected_image)
		# stacked_image_rp += reprojected_image - bkg_rp.back()

		counter += 1
		reference_header.append(('COMMENT',f'Stack image {counter}: {Path(file_list[i]).name}'), end=True)

	stacked_hdu = fits.PrimaryHDU(data=stacked_image_aa, header=reference_header)

	return stacked_hdu

def jd_utc_to_bjd_tdb(jd_utc, ra, dec, location='Whipple'):
	"""Converts Julian Date in UTC timescale to Barycentric Julian Date in TDB timescale. 

	:param jd_utc: julian date in utc
	:type jd_utc: float
	:param ra: the ra of the target as an hour angle (e.g., '23:06:30.0') or decimal degrees (e.g., 346.622013)
	:type ra: str or float
	:param dec: the dec of the target as a string (e.g, '-05:01:57') or decimal degrees (e.g., -5.041274)
	:type dec: str or float
	:param location: the site of observations, must match a location in the astropy .of_site json file, defaults to 'lowell'
	:type location: str, optional
	:return: time in bjd tdb
	:rtype: float
	"""

	if type(ra) == str:
		ra_unit = u.hourangle
	elif type(ra) == np.float64:
		ra_unit = u.deg

	if location == 'Whipple':
		#Specify the location of Whipple without needing to download the sites.json file.
		#The values used are the geocentric coordinates of Whipple in meters, and I got them from:
		#   coord.EarthLocation.of_site('Whipple')
		site = coord.EarthLocation.from_geocentric(-1936768.8080869*u.m, -5077878.69513142*u.m, 3331595.44464286*u.m)
	else:
		site = coord.EarthLocation.of_site(location)

	input_jd_utc = Time(jd_utc, format='jd', scale='utc', location=site)
	target = coord.SkyCoord(ra, dec, unit=(ra_unit, u.deg), frame='icrs')
	ltt_bary = input_jd_utc.light_travel_time(target)
	return (input_jd_utc.tdb + ltt_bary).value

def circular_footprint(radius, dtype=int):
	"""
	TAKEN DIRECTLY FROM PHOTUTILS!!!
	"""
	if ~np.isfinite(radius) or radius <= 0 or int(radius) != radius:
		raise ValueError('radius must be a positive, finite integer greater '
						 'than 0')

	x = np.arange(-radius, radius + 1)
	xx, yy = np.meshgrid(x, x)
	return np.array((xx**2 + yy**2) <= radius**2, dtype=dtype)

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

def circular_aperture_photometry(file_list, sources, ap_radii, logger, an_in=35, an_out=55, type='fixed', centroid=False, centroid_type='centroid_2dg', interpolate_cosmics=False):
	"""
	Does circular aperture photometry on sources in a list of reduced Tierras images for an array of aperture sizes. Writes out photometry csv files to /data/tierras/lightcurves/date/target/ffname/.

	Parameters:
	file_list (list): List of paths to reduced Tierras images. Generate with get_flattened_files.
	targ_and_refs (pandas DataFrame): DataFrame containing information about the target and reference stars. By convention, the target is the first target in this DataFrame. Generate with reference_star_chooser.
	ap_radii (list): List of circular aperture radii that you want to perform photometry for. See the 'type' parameter for how this input is interpreted. One output photometry file will be created for each radius in the list. 
	logger (Logger): python Logger object
	an_in (float): Inner annulus radius (in pixels) for measuring sky background around each source. This parameter only has an effect when bkg_type == '1d'.
	an_out (float): Outer annulus radius (in pixels) for measuring sky background around each source. This parameter only has an effect when bkg_type == '1d'. 
	type (str): The type of aperture photometry to perform, 'fixed' or 'variable'. If 'fixed', ap_radii is interpreted as a list of circular aperture radii (in pixels). If 'variable', ap_radii is interpreted as a list of multiplicative factors times the FWHM seeing in the images (i.e., the aperture radii will vary in time in accordance with seeing changes). 
	centroid (bool): Whether or not to perform centroiding on expected source positions. 
	centroid_type (str): The photutils centroiding function to use for centroiding if centroid == True. Can be 'centroid_1dg', 'centroid_2dg', 'centroid_com', or 'centroid_quadratic'. 
	
	Returns:
	None
	"""

	# set up centroid function
	if centroid:
		centroid_type = centroid_type.lower()
		if centroid_type == 'centroid_1dg':
			centroid_func = centroid_1dg
		elif centroid_type == 'centroid_2dg':
			centroid_func = centroid_2dg
		elif centroid_type == 'centroid_com':
			centroid_func = centroid_com
		elif centroid_type == 'centroid_quadratic':
			centroid_func = centroid_quadratic
		else:
			raise RuntimeError("centroid_type must be one of 'centroid_1dg', 'centroid_2dg', 'centroid_com', or 'centroid_quadratic'.")
	
	ffname = file_list[0].parent.name	
	target = file_list[0].parent.parent.name
	date = file_list[0].parent.parent.parent.name 
	
	# file_list = file_list[50:] #TESTING!!!	
	n_files = len(file_list)
	n_sources = len(sources)
	n_aps = len(ap_radii)

	# log input params
	logger.info(f'Target: {target}')
	logger.info(f'Date: {date}')
	logger.info(f'ffname: {ffname}')
	logger.info(f'Photometry type: {type}')
	logger.info(f'Ap radii: {ap_radii}')
	logger.info(f'An in: {an_in}')
	logger.info(f'An out: {an_out}')
	logger.info(f'Centroid: {centroid}')
	if centroid:
		logger.info(f'Centroid function: {centroid_type}')
	
	# file_list = file_list[98:] #TESTING!!!
	
	DARK_CURRENT = 0.00133 #e- pix^-1 s^-1, see Juliana's dissertation Table 4.1
	NONLINEAR_THRESHOLD = 40000. #ADU
	SATURATION_THRESHOLD = 55000. #ADU
	PLATE_SCALE = 0.432 #arcsec pix^-1, from Juliana's dissertation Table 1.1
	
	fit_g = fitting.LevMarLSQFitter() # fitter object for fitting 2D gaussians to measure FWHM

	#Set up arrays for doing photometry 

	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH FILE
	filenames = []
	wcs_flags = np.zeros(n_files, dtype='int')
	centroid_flags = np.zeros(n_files, dtype='int')
	mjd_utc = np.zeros(n_files, dtype='float')
	jd_utc = np.zeros(n_files,dtype='float')
	bjd_tdb = np.zeros(n_files,dtype='float')
	airmasses = np.zeros(n_files,dtype='float16')
	ccd_temps = np.zeros(n_files,dtype='float16')
	exp_times = np.zeros(n_files,dtype='float16')
	dome_temps = np.zeros(n_files,dtype='float16')
	focuses = np.zeros(n_files,dtype='float16')
	dome_humidities = np.zeros(n_files,dtype='float16')
	sec_temps = np.zeros(n_files,dtype='float16')
	ret_temps = np.zeros(n_files,dtype='float16')
	pri_temps = np.zeros(n_files,dtype='float16')
	rod_temps = np.zeros(n_files,dtype='float16')
	cab_temps = np.zeros(n_files,dtype='float16')
	inst_temps = np.zeros(n_files,dtype='float16')
	temps = np.zeros(n_files,dtype='float16')
	humidities = np.zeros(n_files,dtype='float16')
	dewpoints = np.zeros(n_files,dtype='float16')
	sky_temps = np.zeros(n_files,dtype='float16')
	pressures = np.zeros(n_files,dtype='float16')
	return_pressures = np.zeros(n_files,dtype='float16')
	supply_pressures = np.zeros(n_files,dtype='float16')
	hour_angles = np.zeros(n_files,dtype='float16')
	dome_azimuths = np.zeros(n_files,dtype='float16')
	wind_speeds = np.zeros(n_files,dtype='float16')
	wind_gusts = np.zeros(n_files,dtype='float16')
	wind_dirs = np.zeros(n_files,dtype='float16')

	loop_times = np.zeros(n_files,dtype='float16')
	lunar_distance = np.zeros(n_files,dtype='float16')
	
	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH SOURCE IN EACH FILE
	source_x = np.zeros((n_sources,n_files),dtype='float32')
	source_y = np.zeros((n_sources,n_files),dtype='float32')
	source_sky_ADU = np.zeros((n_sources,n_files),dtype='float32')
	source_x_fwhm_arcsec = np.zeros((n_sources,n_files),dtype='float32')
	source_y_fwhm_arcsec = np.zeros((n_sources,n_files),dtype='float32')
	source_theta_radians = np.zeros((n_sources,n_files),dtype='float32')

	#ARRAYS THAT CONTAIN DATA PERTAININING TO EACH APERTURE RADIUS FOR EACH SOURCE FOR EACH FILE
	source_minus_sky_ADU = np.zeros((n_aps,n_sources,n_files),dtype='float32')
	source_minus_sky_err_ADU = np.zeros((n_aps,n_sources,n_files),dtype='float32')
	non_linear_flags = np.zeros((n_aps,n_sources,n_files),dtype='bool')
	saturated_flags = np.zeros((n_aps,n_sources,n_files),dtype='bool')	
	interpolation_flags = np.zeros((n_aps,n_sources,n_files),dtype='bool')
	source_radii = np.zeros((n_aps,n_files),dtype='float16')
	an_in_radii = np.zeros((n_aps,n_files),dtype='float16')
	an_out_radii = np.zeros((n_aps,n_files),dtype='float16')
	
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

	if interpolate_cosmics:
		from astroscrappy import detect_cosmics 
		bp_inds = np.where(bpm == 1)
		

	# reference_image_hdu = fits.open('/data/tierras/targets/'+target+'/'+target+'_stacked_image.fits')[0] #TODO: should match image from target/reference csv file, and that should be loaded automatically.

	#reference_image_hdu = fits.open(file_list[1])[0]

	# declare a circular footprint in case centroiding is performed
	# only data within a radius of x pixels around the expected source positions from WCS will be considered for centroiding
	centroid_footprint = circular_footprint(5)

	logger.info(f'Doing fixed-radius circular aperture photometry on {n_files} images with aperture radii of {ap_radii} pixels, an inner annulus radius of {an_in} pixels, and an outer annulus radius of {an_out} pixels.')

	#TODO: this is only approximate since we measure background on sigma-clipped distribution
	nb = np.pi*an_out**2 - np.pi*an_in**2

	t1 = time.time()
	for i in range(n_files):
		if i > 0:
			loop_times[i-1]= time.time()-t1
			logger.debug(f'Avg loop time: {np.mean(loop_times[0:i]):.2f} s')
			t1 = time.time()
		
		logger.info(f'{file_list[i].name} ({i+1} of {n_files})')
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
		ha_str = source_header['HA']
		if ha_str[0] == '-':
			ha_decimal = int(ha_str.split(':')[0]) - int(ha_str.split(':')[1])/60 - float(ha_str.split(':')[2])/3600
		else:
			ha_decimal = int(ha_str.split(':')[0]) + int(ha_str.split(':')[1])/60 + float(ha_str.split(':')[2])/3600
		hour_angles[i] = ha_decimal

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
			pressures[i] = source_header['PRESSURE']
			return_pressures[i] = source_header['PSPRES1']
			supply_pressures[i] = source_header['PSPRES2']
			dome_azimuths[i] = source_header['DOMEAZ']
			wind_speeds[i] = source_header['WINDSPD']
			wind_gusts[i] = source_header['WINDGUST']
			wind_dirs[i] = source_header['WINDDIR']
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
			pressures[i] = np.nan
			return_pressures[i] = np.nan
			supply_pressures[i] = np.nan
			dome_azimuths[i] = np.nan
			wind_speeds[i] = np.nan
			wind_gusts[i] = np.nan
			wind_dirs[i] = np.nan

		lunar_distance[i] = get_lunar_distance(RA, DEC, bjd_tdb[i]) #Commented out because this is slow and the information can be generated at a later point if necessary	

		if interpolate_cosmics:
			data_copy = copy.deepcopy(source_data)
			cosmic_mask, cosmic_interpolated_data = detect_cosmics(data_copy, gain=GAIN, sigclip=7, sigfrac=0.1, objlim=1)


		#UPDATE SOURCE POSITIONS
		#METHOD 1: WCS
		tcent = time.time()
		source_wcs = WCS(source_header)

		logger.debug(f'RMS of astrometric solution: {source_header["STDCRMS"]:.4f} arcsec')

		if (source_header['STDCRMS'] > PLATE_SCALE/2):
			logger.info(f'WARNING: header indicates astrometric solution with RMS of {source_header["STDCRMS"]}, WCS positions may not be accurate.')
			wcs_flags[i] = 1

		transformed_pixel_coordinates = np.array([source_wcs.world_to_pixel((SkyCoord(sources['ra_tierras'][i]*u.deg, sources['dec_tierras'][i]*u.deg))) for i in range(n_sources)])
		
		#Save transformed pixel coordinates of sources
		source_x[:,i] = transformed_pixel_coordinates[:,0]
		source_y[:,i] = transformed_pixel_coordinates[:,1]

		# fig2, ax2 = plot_image(source_data)
		# for j in range(len(source_x[:,i])):
		# 	ax2.plot(source_x[j,i],source_y[j,i],'rx')
		# breakpoint()

		source_positions = [(source_x[j,i], source_y[j,i]) for j in range(n_sources)]

		if (sum(source_x[:,i] < 0) + sum(source_y[:,i] < 0) + sum(source_x[:,i] > source_data.shape[1]) + sum(source_y[:,i] > source_data.shape[0])) > 0:
			warnings.warn('Sources off chip! Skipping photometry.')
			continue 

		# # logger.debug(f'Source x (WCS): {[f"{item:.2f}" for item in source_x[:,i]]}')
		# logger.debug(f'Source y (WCS): {[f"{item:.2f}" for item in source_y[:,i]]}')
		if centroid:
			# mask any pixels in the image above the non-linear threshold
			mask = np.zeros(np.shape(source_data), dtype='bool')
			mask[np.where(source_data>NONLINEAR_THRESHOLD)] = 1

			# fig, ax = plot_image(source_data)
			# ax.scatter(source_x[:,i], source_y[:,i], marker='x', color='b')
			
			# if interpolate_cosmics is enabled, do centroiding on the interpolated data
			# otherwise a cosmic ray hit could throw off the centroiding 
			if interpolate_cosmics:
				centroid_x, centroid_y = centroid_sources(cosmic_interpolated_data,source_x[:,i], source_y[:,i], centroid_func=centroid_func, footprint=centroid_footprint, mask=mask)	
			# else, determine the centroid on un-interpolated data
			else:
				centroid_x, centroid_y = centroid_sources(source_data,source_x[:,i], source_y[:,i], centroid_func=centroid_func, footprint=centroid_footprint, mask=mask)	
			
			# ax.scatter(centroid_x, centroid_y, marker='x', color='r')
			# breakpoint()
			# plt.close()

			# update source positions
			source_x[:,i] = centroid_x 
			source_y[:,i] = centroid_y

			# logger.debug(f'Source x (centroid): {[f"{item:.2f}" for item in source_x[:,i]]}')
			# logger.debug(f'Source y (centroid): {[f"{item:.2f}" for item in source_y[:,i]]}')
			
			source_positions = [(source_x[j,i], source_y[j,i]) for j in range(n_sources)]
	
		logger.debug(f'Source position loop time: {time.time()-tcent:.2f} s')

		# Do photometry
		# Set up apertures
		if type == 'fixed':
			apertures = [CircularAperture(source_positions,r=ap_radii[k]) for k in range(len
			(ap_radii))]

		# TODO: need to implement variable apertures!
		# elif type == 'variable':
		# 	apertures = [CircularAperture(source_positions,r=ap_radii[k]*smoothed_fwhm_pix[i])]	

		tbkg = time.time()	
		# measure background
		annuli = CircularAnnulus(source_positions, an_in, an_out)
		annulus_masks = annuli.to_mask(method='center')
		for j in range(len(annuli)):
			source_sky_ADU[j,i] = np.mean(sigmaclip(annulus_masks[j].get_values(source_data),2,2)[0])
		logger.debug(f'Bkg loop time: {time.time()-tbkg:.2f} s')

		tphot = time.time()
		# do photometry on the *sources*
		phot_table = aperture_photometry(source_data, apertures)

		# Calculate sky-subtracted flux
		for k in range(n_aps):
			source_radii[k, i] = ap_radii[k]
			ap_area = apertures[k].area
			
			# for 1d background, subtract off the average bkg value measured in the annulus times the aperture area
			source_minus_sky_ADU[k,:,i] = phot_table[f'aperture_sum_{k}']-source_sky_ADU[:,i]*ap_area

			# calculate photometric uncertainty IN ADU following Stefansson et al. (2017) formalism
			v_star = source_minus_sky_ADU[k,:,i]*GAIN 
			v_sky = source_sky_ADU[:,i]*GAIN 
			v_dark = DARK_CURRENT*EXPTIME 
			v_read = READ_NOISE**2 
			sigma_ccd = np.sqrt(v_star + ap_area*(1+ap_area/nb)*(v_sky + v_dark + v_read))/GAIN 

			# Calculate uncertainty
			source_minus_sky_err_ADU[k,:,i] = sigma_ccd

		logger.debug(f'Aperture photometry time: {time.time()-tphot:.2f} s')
	
		# check for non-linear/saturated pixels in the apertures 
		# just do in the smallest aperture for now  
		aperture_masks = apertures[0].to_mask(method='center')
		for j in range(n_sources):
			# ap_cutout = aperture_masks[j].multiply(source_data)
			# ap_pix_vals = ap_cutout[ap_cutout!=0]
			ap_pix_vals = aperture_masks[j].get_values(source_data)
			non_linear_flags[:,j,i] = int(np.sum(ap_pix_vals>NONLINEAR_THRESHOLD)>0)
			saturated_flags[:,j,i] = int(np.sum(ap_pix_vals>SATURATION_THRESHOLD)>0)
		
		if interpolate_cosmics:
			tcosmic = time.time()
			for k in range(n_aps):
				aperture_masks = apertures[k].to_mask(method='center')
				for j in range(n_sources):
					cosmic_cutout = aperture_masks[j].multiply(cosmic_mask, fill_value=-1)
					# if cosmics fall inside the aperture, replace the phot_table with a measurement using new_data
					if len(np.where(cosmic_cutout == 1)[0]) > 0:
						# fig, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(8,4))
						# ax[0].imshow(aperture_masks[j].multiply(source_data), origin='lower')
						# ax[1].imshow(cosmic_cutout, origin='lower')
						# ax[2].imshow(aperture_masks[j].multiply(cosmic_interpolated_data), origin='lower')
						
						interpolated_measurement = aperture_photometry(cosmic_interpolated_data, apertures[k][j])['aperture_sum'] 
						interpolated_measurement -= source_sky_ADU[j,i]*np.pi*ap_radii[k]**2
						source_minus_sky_ADU[k,j,i] = interpolated_measurement
						interpolation_flags[k,j,i] = 1
			logger.debug(f'Cosmic interpolation time: {time.time()-tcosmic:.2f} s')
								
		tfwhm = time.time()
		#Measure FWHM 
		k = 0
		cutout_size = 30
		g_init = models.Gaussian2D(amplitude=1,x_mean=cutout_size/2,y_mean=cutout_size/2, x_stddev=3, y_stddev=3)
		g_init.theta.bounds = (0, np.pi)
		g_init.x_stddev.bounds = (1,10)
		g_init.y_stddev.bounds = (1,10)

		# pre-compute meshgrid of pixel indices for square cutouts (which will be the case for all sources except those near the edges)
		xx2, yy2 = np.meshgrid(np.arange(cutout_size),np.arange(cutout_size))

		# TODO: this loop appears to slow down over time, WHY!?
		# this is, in general, the slowest loop of the program. How can we make it faster??
		for j in range(len(source_positions)):
			g_2d_cutout, cutout_pos = generate_square_cutout(source_data, source_positions[j], cutout_size)
			cutout_shape = g_2d_cutout.shape

			bkg = np.median(g_2d_cutout)
			# bkg = np.mean(sigmaclip(g_2d_cutout,2,2)[0])

			g_2d_cutout -= bkg 
			
			# recompute the meshgrid only if you get a non-square cutout
			if g_2d_cutout.shape != (cutout_size, cutout_size):
				xx3, yy3 = np.meshgrid(np.arange(cutout_shape[1]),np.arange(cutout_shape[0]))
				xx = xx3
				yy = yy3
			else:
				xx = xx2
				yy = yy2 
			
			# intialize the model 
			g_init.amplitude = g_2d_cutout[int(cutout_pos[1]), int(cutout_pos[0])]

			# use the cutout position returned from generate_square_cutout to predict its location
			g_init.x_mean = cutout_pos[0]
			g_init.y_mean = cutout_pos[1]


			g = fit_g(g_init,xx,yy,g_2d_cutout)
			
			if g.y_stddev.value > g.x_stddev.value:
				x_stddev_save = g.x_stddev.value
				y_stddev_save = g.y_stddev.value
				g.x_stddev = y_stddev_save
				g.y_stddev = x_stddev_save
				g.theta += np.pi/2

			source_x_fwhm_arcsec[j,i] = g.x_stddev.value * 2.35482 * PLATE_SCALE
			source_y_fwhm_arcsec[j,i] = g.y_stddev.value * 2.35482 * PLATE_SCALE
			source_theta_radians[j,i] = g.theta.value
			#print(time.time()-t1)

			# fig, ax = plt.subplots(1,2,figsize=(12,8),sharex=True,sharey=True)
			# norm = ImageNormalize(g_2d_cutout-bkg,interval=ZScaleInterval())
			# ax[0].imshow(g_2d_cutout-bkg,origin='lower',interpolation='none',norm=norm)
			# ax[1].imshow(g(xx2,yy2),origin='lower',interpolation='none',norm=norm)
			# plt.tight_layout()
		logger.debug(f'FWHM loop time: {time.time()-tfwhm:.2f} s')
		# logger.debug(f'Major FWHM (arcsec): {[f"{item:.2f}" for item in source_x_fwhm_arcsec[:,i]]}')
		# logger.debug(f'Minor FWHM (arcsec): {[f"{item:.2f}" for item in source_y_fwhm_arcsec[:,i]]}')
		# logger.debug(f'Theta (rad): {[f"{item:.2f}" for item in source_theta_radians[:,i]]}')

			
	#Write out photometry. 
	for i in range(n_aps):
		output_path = Path('/data/tierras/photometry/'+date+'/'+target+'/'+ffname+f'/{date}_{target}_circular_{type}_ap_phot_{ap_radii[i]}.csv')
		
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
		output_list.append([f'{val:.5f}' for val in lunar_distance])
		output_header.append('Lunar Distance')
		output_list.append([f'{val:.1f}' for val in pressures])
		output_header.append('Pressure')
		output_list.append([f'{val:.1f}' for val in return_pressures])
		output_header.append('Return Pressure')
		output_list.append([f'{val:.1f}' for val in supply_pressures])
		output_header.append('Supply Pressure')
		output_list.append([f'{val:.2f}' for val in hour_angles])
		output_header.append('Hour Angle')
		output_list.append([f'{val:.2f}' for val in dome_azimuths])
		output_header.append('Dome Azimuth')
		output_list.append([f'{val:.2f}' for val in wind_speeds])
		output_header.append('Wind Speed')
		output_list.append([f'{val:.2f}' for val in wind_gusts])
		output_header.append('Wind Gust')
		output_list.append([f'{val:.1f}' for val in wind_dirs])
		output_header.append('Wind Direction')
		output_list.append([f'{val:d}' for val in wcs_flags])
		output_header.append('WCS Flag')

		for j in range(n_sources):
			source_name = f'S{j}'
			output_list.append([f'{val:.4f}' for val in source_x[j]])
			output_header.append(source_name+' X')
			output_list.append([f'{val:.4f}' for val in source_y[j]])
			output_header.append(source_name+' Y')
			output_list.append([f'{val:.7f}' for val in source_minus_sky_ADU[i,j]])
			output_header.append(source_name+' Source-Sky ADU')
			output_list.append([f'{val:.7f}' for val in source_minus_sky_err_ADU[i,j]])
			output_header.append(source_name+' Source-Sky Error ADU')
			output_list.append([f'{val:.7f}' for val in source_sky_ADU[j]])
			output_header.append(source_name+' Sky ADU')
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
			output_list.append([f'{val:d}' for val in interpolation_flags[i,j]])
			output_header.append(source_name+' Interpolation Flag')

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

def plot_target_summary(file_path,pval_threshold=0.01,bin_mins=15):
	date = file_path.parent.parent.parent.name
	target = file_path.parent.parent.name
	ffname = file_path.parent.name 

	df = pd.read_csv(file_path)
	times = np.array(df['BJD TDB'])
	x_offset =  int(np.floor(times[0]))
	times -= x_offset

	targ_flux = np.array(df['Target Source-Sky ADU'])
	targ_flux_err = np.array(df['Target Source-Sky Error ADU'])
	targ_pp_flux = np.array(df['Target Post-Processed Normalized Flux'])
	targ_pp_flux_err = np.array(df['Target Post-Processed Normalized Flux Error'])

	#NEW: generate ALC using reference star weights
	alc_flux, alc_flux_err = weighted_alc(file_path)

	# #OLD: use ensemble ALC from light curve file
	# alc_flux = np.array(df['Target Ensemble ALC ADU'])
	# alc_flux_err = np.array(df['Target Ensemble ALC Error ADU'])

	#Set up dictionary containing ancillary data to check for significant correlations
	ancillary_dict = {}
	ancillary_dict['Airmass'] = np.array(df['Airmass'])
	ancillary_dict['Target Sky ADU'] = np.array(df['Target Sky ADU'])
	ancillary_dict['Target X'] = np.array(df['Target X']) - np.median(df['Target X'])
	ancillary_dict['Target Y'] = np.array(df['Target Y']) - np.median(df['Target Y'])
	ancillary_dict['Target X FWHM Arcsec'] = np.array(df['Target X FWHM Arcsec'])
	ancillary_dict['Target Y FWHM Arcsec'] = np.array(df['Target Y FWHM Arcsec'])

	v1,l1,h1 = sigmaclip(targ_flux,3,3)
	v2,l2,h2 = sigmaclip(alc_flux,3,3)
	use_inds = np.where((targ_flux>l1)&(targ_flux<h1)&(alc_flux>l2)&(alc_flux<h2))[0]
	times = times[use_inds]
	targ_flux = targ_flux[use_inds]
	targ_flux_err = targ_flux_err[use_inds]
	targ_pp_flux = targ_pp_flux[use_inds]
	targ_pp_flux_err = targ_pp_flux_err[use_inds]
	alc_flux = alc_flux[use_inds]
	alc_flux_err = alc_flux_err[use_inds]
	#planet_model = planet_model[use_inds]

	for key in ancillary_dict.keys():
		ancillary_dict[key] = ancillary_dict[key][use_inds]

	targ_flux_norm_factor = np.median(targ_flux)
	targ_flux_norm = targ_flux / targ_flux_norm_factor
	targ_flux_err_norm = targ_flux_err / targ_flux_norm_factor

	alc_flux_norm_factor = np.median(alc_flux)
	alc_flux_norm = alc_flux/alc_flux_norm_factor
	alc_flux_err_norm = alc_flux_err/alc_flux_norm_factor

	corrected_targ_flux = targ_flux_norm/alc_flux_norm
	corrected_targ_flux_err = np.sqrt((targ_flux_err_norm/alc_flux_norm)**2 + (targ_flux_norm*alc_flux_err_norm/(alc_flux_norm**2))**2)

	v,l,h = sigmaclip(corrected_targ_flux)
	use_inds = np.where((corrected_targ_flux>l)&(corrected_targ_flux<h))[0]
	times = times[use_inds]
	targ_flux = targ_flux[use_inds]
	targ_flux_err = targ_flux_err[use_inds]
	targ_pp_flux = targ_pp_flux[use_inds]
	targ_pp_flux_err = targ_pp_flux_err[use_inds]
	alc_flux = alc_flux[use_inds]
	alc_flux_err = alc_flux_err[use_inds]
	targ_flux_norm = targ_flux_norm[use_inds]
	targ_flux_err_norm = targ_flux_err_norm[use_inds]
	alc_flux_norm = alc_flux_norm[use_inds]
	alc_flux_err_norm = alc_flux_err_norm[use_inds]
	corrected_targ_flux = corrected_targ_flux[use_inds]
	corrected_targ_flux_err = corrected_targ_flux_err[use_inds]
	for key in ancillary_dict.keys():
		ancillary_dict[key] = ancillary_dict[key][use_inds]

	norm = np.mean(corrected_targ_flux)
	corrected_targ_flux /= norm 
	corrected_targ_flux_err /= norm 

	#fig, ax = plt.subplots(7,1,figsize=(6,9), gridspec_kw={'height_ratios':[1,1,0.5,0.5,0.5,0.5,1],})

	fig = plt.figure(figsize=(8,9))	
	gs = gridspec.GridSpec(8,1,height_ratios=[0.75,1,1,0.75,0.75,0.75,0.75,1])
	ax1 = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1],sharex=ax1)
	ax3 = plt.subplot(gs[2],sharex=ax1,sharey=ax2)
	ax4 = plt.subplot(gs[3],sharex=ax1)
	ax5 = plt.subplot(gs[4],sharex=ax1)
	ax6 = plt.subplot(gs[5],sharex=ax1)
	ax7 = plt.subplot(gs[6],sharex=ax1)
	ax8 = plt.subplot(gs[7])


	label_size = 11
	ax1.errorbar(times, targ_flux_norm, targ_flux_err_norm, marker='.', color='k',ls='', ecolor='k', label='Norm. targ. flux')
	ax1.errorbar(times, alc_flux_norm, alc_flux_err_norm, marker='.', color='r',ls='', ecolor='r', label='Norm. ALC flux')
	ax1.tick_params(labelsize=label_size)
	ax1.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	
	#ax[0].grid(alpha=0.8)
	ax1.set_ylabel('Norm. Flux',fontsize=label_size)
	ax1.tick_params(labelbottom=False)

	ax2.plot(times, corrected_targ_flux, marker='.',color='#b0b0b0',ls='',label='Rel. targ. flux')
	#bin_mins=15
	bx, by, bye = tierras_binner(times, corrected_targ_flux, bin_mins=bin_mins)
	ax2.errorbar(bx,by,bye,marker='o',mfc='none',mec='k',mew=1.5,ecolor='k',ms=7,ls='',zorder=3,label=f'{bin_mins:d}-min bins')
	ax2.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	ax2.tick_params(labelsize=label_size)
	ax2.set_ylabel('Norm. Flux',fontsize=label_size)
	ax2.tick_params(labelbottom=False)

	ax3.plot(times, targ_pp_flux, marker='.',color='#b0b0b0',ls='',label='Post-processed flux')
	bx, by, bye = tierras_binner(times, targ_pp_flux, bin_mins=bin_mins)
	ax3.errorbar(bx,by,bye,marker='o',mfc='none',mec='k',mew=1.5,ecolor='k',ms=7,ls='',zorder=3,label=f'{bin_mins:d}-min bins')
	ax3.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	ax3.tick_params(labelsize=label_size)
	ax3.set_ylabel('Norm. Flux',fontsize=label_size)
	ax3.tick_params(labelbottom=False)

	ax4.plot(times,ancillary_dict['Airmass'], color='tab:blue',lw=2)
	ax4.tick_params(labelsize=label_size)
	ax4.set_ylabel('Airmass',fontsize=label_size)
	ax4.tick_params(labelbottom=False)

	ax5.plot(times,ancillary_dict['Target Sky ADU'],color='tab:orange',lw=2)
	ax5.tick_params(labelsize=label_size)
	ax5.set_ylabel('Sky\n(ADU)',fontsize=label_size)
	ax5.tick_params(labelbottom=False)

	ax6.plot(times,ancillary_dict['Target X'],color='tab:green',lw=2,label='X-med(X)')
	ax6.plot(times,ancillary_dict['Target Y'],color='tab:red',lw=2,label='Y-med(Y)')
	ax6.tick_params(labelsize=label_size)
	ax6.set_ylabel('Pos.',fontsize=label_size)
	ax6.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	v1,l1,h1 = sigmaclip(ancillary_dict['Target X'],5,5)
	v2,l2,h2 = sigmaclip(ancillary_dict['Target X'],5,5)
	ax6.set_ylim(np.min([l1,l2]),np.max([h1,h2]))
	ax6.tick_params(labelbottom=False)

	ax7.plot(times,ancillary_dict['Target X FWHM Arcsec'],color='tab:pink',lw=2,label='X')
	ax7.plot(times, ancillary_dict['Target Y FWHM Arcsec'], color='tab:purple', lw=2,label='Y')
	ax7.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	ax7.tick_params(labelsize=label_size)
	ax7.set_ylabel('FWHM\n(")',fontsize=label_size)
	ax7.set_xlabel(f'Time - {x_offset}'+' (BJD$_{TDB}$)',fontsize=label_size)

	#Do bin plot
	bins = np.arange(0.5,20.5,0.5)
	std, theo = juliana_binning(bins, times, corrected_targ_flux, corrected_targ_flux_err)

	ax8.plot(bins, std[1:]*1e6, lw=2,label='Measured')
	ax8.plot(bins, theo[1:]*1e6,lw=2,label='Theoretical')
	std, theo = juliana_binning(bins, times, targ_pp_flux, targ_pp_flux_err)
	ax8.plot(bins, std[1:]*1e6, lw=2,label='Measured post-processing')

	ax8.set_xlabel('Bin size (min)',fontsize=label_size)
	ax8.set_ylabel('$\sigma$ (ppm)',fontsize=label_size)
	ax8.set_yscale('log')
	ax8.legend(loc='center left', bbox_to_anchor=(1,0.5),fontsize=10)
	ax8.tick_params(labelsize=label_size)

	fig.align_labels()
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.7)

	summary_plot_output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/{date}_{target}_summary.png'
	plt.savefig(summary_plot_output_path,dpi=300)
	set_tierras_permissions(summary_plot_output_path)

	plt.close('all')
	return

def plot_target_lightcurve(lc_path,bin_mins=15):
	plt.ion()
	lc_path = Path(lc_path)
	ffname = lc_path.parent.name
	target = lc_path.parent.parent.name
	date = lc_path.parent.parent.parent.name

	df = pd.read_csv(lc_path)
	# targ_and_refs = pd.read_csv(f'/data/tierras/targets/{target}/{target}_target_and_ref_stars.csv')
	# n_refs = len(targ_and_refs)


	times = np.array(df['BJD TDB'])
	targ_flux = np.array(df['Target Source-Sky ADU'])
	targ_flux_err = np.array(df['Target Source-Sky Error ADU'])
	targ_pp_flux = np.array(df['Target Post-Processed Normalized Flux'])
	targ_pp_flux_err = np.array(df['Target Post-Processed Normalized Flux Error'])

	
	# NEW: Use weighted ALC
	alc_flux, alc_flux_err = weighted_alc(lc_path)
	rel_flux = targ_flux / alc_flux 
	rel_flux_err = np.sqrt((targ_flux_err/alc_flux)**2+(targ_flux*alc_flux_err/(alc_flux**2))**2)

	#Sigma clip
	v, l, h = sigmaclip(rel_flux)
	use_inds = np.where((rel_flux>l)&(rel_flux<h))[0]
	targ_flux = targ_flux[use_inds]
	targ_flux_err = targ_flux_err[use_inds]
	alc_flux = alc_flux[use_inds]
	alc_flux_err = alc_flux_err[use_inds]
	rel_flux = rel_flux[use_inds]
	rel_flux_err = rel_flux_err[use_inds]

	renorm = np.nanmean(rel_flux)
	rel_flux /= renorm
	rel_flux_err /= renorm

	renorm = np.nanmean(targ_flux)
	targ_flux /= renorm
	targ_flux_err /= renorm 

	renorm = np.nanmean(alc_flux)
	alc_flux /= renorm 
	alc_flux_err /= renorm

	#print(f"bp_rp: {targ_and_refs['bp_rp'][i+1]}")
	fig, ax = plt.subplots(3,1,figsize=(10,10),sharex=True)
	
	bx, by, bye = tierras_binner(times[use_inds],rel_flux,bin_mins=bin_mins)

	fig.suptitle(f'{target} on {date}',fontsize=16)
	ax[0].errorbar(times[use_inds], targ_flux, targ_flux_err, color='k',ecolor='k',marker='.',ls='',zorder=3,label='Target')
	ax[0].errorbar(times[use_inds], alc_flux, alc_flux_err, color='r', ecolor='r', marker='.',ls='',zorder=3,label='ALC')
	ax[0].set_ylabel('Normalized Flux', fontsize=16)
	ax[0].tick_params(labelsize=14)
	ax[0].grid(True, alpha=0.8)
	ax[0].legend()

	ax[1].plot(times[use_inds], rel_flux,  marker='.', ls='', color='#b0b0b0')
	ax[1].errorbar(bx,by,bye,marker='o',color='none',ecolor='k',mec='k',mew=2,ms=5,ls='',label=f'{bin_mins}-min binned photometry',zorder=3)
	ax[1].tick_params(labelsize=14)
	ax[1].set_ylabel('Normalized Flux',fontsize=16)
	ax[1].grid(True, alpha=0.8)
	ax[1].legend()
	
	ax[2].plot(times[use_inds],targ_pp_flux[use_inds],marker='.', ls='', color='#b0b0b0')
	bx, by, bye = tierras_binner(times[use_inds],targ_pp_flux[use_inds],bin_mins=bin_mins)
	ax[2].errorbar(bx,by,bye,marker='o',color='none',ecolor='k',mec='k',mew=2,ms=5,ls='',label=f'{bin_mins}-min binned photometry',zorder=3)
	ax[2].tick_params(labelsize=14)
	ax[2].set_ylabel('Normalized Flux',fontsize=16)
	ax[2].grid(True, alpha=0.8)
	ax[2].legend()
	ax[2].set_title('Post-Processed Flux',fontsize=16)
	ax[2].set_xlabel('Time (BJD$_{TDB}$)',fontsize=16)
	# model_times = np.arange(min(times[use_inds]),max(times[use_inds]),0.0005)
	# planet_flux = transit_model(model_times, 2459497.184957, 25.522952715243/2, np.sqrt(1179.7/1e6), 84.4, 90, 0, 90, 0.1, 0.3)
	# ax[1].plot(model_times, planet_flux)

	plt.tight_layout()
	#plt.subplots_adjust(hspace=0.04)
	output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/{date}_{target}_lc.png'
	plt.savefig(output_path,dpi=300)
	set_tierras_permissions(output_path)

	plt.close()
	return

def plot_ref_lightcurves(lc_path, bin_mins=15):
	plt.ioff()
	lc_path = Path(lc_path)
	parents = lc_path.parents
	ffname = parents[0].name
	target = parents[1].name
	date = parents[2].name
	output_path = lc_path.parent/'reference_lightcurves/'
	if not os.path.exists(output_path):
		os.mkdir(output_path)		
		set_tierras_permissions(output_path)

	#Clear out existing files
	existing_files = glob(str(output_path/'*.png'))
	for file in existing_files:
		os.remove(file)

	df = pd.read_csv(lc_path)
	# targ_and_refs = pd.read_csv(f'/data/tierras/targets/{target}/{target}_target_and_ref_stars.csv')
	# n_refs = len(targ_and_refs)-1

	n_refs = int(df.keys()[-1].split('Ref ')[1].split(' ')[0])

	if n_refs == 1:
		print('Only 1 reference, cannot create reference ALC.')
		return

	times = np.array(df['BJD TDB'])

	weight_df = pd.read_csv(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/night_weights.csv')
	weights = np.array(weight_df['Weight'])

	n_ims = len(df)
	ref_fluxes = np.zeros((n_ims, n_refs))
	ref_flux_errs = np.zeros((n_ims, n_refs))
	ref_pp_fluxes = np.zeros((n_ims,n_refs))
	ref_pp_flux_errs = np.zeros((n_ims,n_refs))
	for i in range(n_refs):
		ref_fluxes[:,i] = np.array(df[f'Ref {i+1} Source-Sky ADU'])
		ref_flux_errs[:,i] = np.array(df[f'Ref {i+1} Source-Sky Error ADU'])
		ref_pp_fluxes[:,i] = np.array(df[f'Ref {i+1} Post-Processed Normalized Flux'])
		ref_pp_flux_errs[:,i] = np.array(df[f'Ref {i+1} Post-Processed Normalized Flux Error'])

	for i in range(n_refs):
		print(f'Doing Ref {i+1} of {n_refs}')
		inds = np.array([j for j in range(n_refs) if j != i])

		#NEW: Calculate ALC using weights (excluding the weight of this particular ref star)
		weights_use = weights[inds] 
		weights_use /= sum(weights_use)

		alc = np.sum(weights_use*ref_fluxes[:,inds],axis=1)
		alc_err = np.sqrt(np.sum((weights_use*ref_flux_errs[:,inds])**2,axis=1))

		rel_flux = ref_fluxes[:,i]/alc
		rel_flux_err = np.sqrt((ref_flux_errs[:,i]/alc)**2+(ref_fluxes[:,i]*alc_err/(alc**2))**2)

		# #OLD: Use relative flux as calculated using ensemble ALC
		# rel_flux = np.array(df[f'Ref {i+1} Relative Flux'])
		# rel_flux_err = np.array(df[f'Ref {i+1} Relative Flux Error'])
		
		#Sigma clip
		v, l, h = sigmaclip(rel_flux)
		use_inds = np.where((rel_flux>l)&(rel_flux<h))[0]
		rel_flux = rel_flux[use_inds]
		rel_flux_err = rel_flux_err[use_inds]

		renorm = np.nanmean(rel_flux)
		rel_flux /= renorm
		rel_flux_err /= renorm

		#print(f"bp_rp: {targ_and_refs['bp_rp'][i+1]}")
		fig, ax = plt.subplots(2,1,figsize=(10,7),sharex=True)
		
		bx, by, bye = tierras_binner(times[use_inds],rel_flux,bin_mins=bin_mins)

		fig.suptitle(f'Reference {i+1}, Weight={weights[i]:.2g}',fontsize=16)

		ax[0].plot(times[use_inds], rel_flux,  marker='.', ls='', color='#b0b0b0')
		ax[0].errorbar(bx,by,bye,marker='o',color='none',ecolor='k',mec='k',mew=2,ms=5,ls='',label=f'{bin_mins}-min binned photometry',zorder=3)
		#ax.set_ylim(0.975,1.025)
		ax[0].tick_params(labelsize=14)
		ax[0].set_ylabel('Normalized Flux',fontsize=16)
		ax[0].grid(True, alpha=0.8)
		ax[0].legend()

		ax[1].plot(times[use_inds], ref_pp_fluxes[:,i][use_inds],  marker='.', ls='', color='#b0b0b0')
		bx,by,bye = tierras_binner(times[use_inds],ref_pp_fluxes[:,i][use_inds],bin_mins=bin_mins)
		ax[1].errorbar(bx,by,bye,marker='o',color='none',ecolor='k',mec='k',mew=2,ms=5,ls='',label=f'{bin_mins}-min binned photometry',zorder=3)
		#ax.set_ylim(0.975,1.025)
		ax[1].tick_params(labelsize=14)
		ax[1].set_xlabel('Time (BJD$_{TDB}$)',fontsize=16)
		ax[1].set_ylabel('Normalized Flux',fontsize=16)
		ax[1].grid(True, alpha=0.8)
		ax[1].legend()
		ax[1].set_title('Post-Processed Flux')
		
		plt.tight_layout()
		plt.savefig(output_path/f'ref_{i+1}.png',dpi=300)
		set_tierras_permissions(output_path/f'ref_{i+1}.png')

		plt.close()
	plt.ion()
	return

def plot_raw_fluxes(lc_path):
	lc_path = Path(lc_path)
	ffname = lc_path.parent.name
	target = lc_path.parent.parent.name
	date = lc_path.parent.parent.parent.name

	df = pd.read_csv(lc_path)
	times = np.array(df['BJD TDB'])
	x_offset = int(np.floor(times[0]))
	times -= x_offset 

	targ_flux = np.array(df['Target Source-Sky ADU'])
	#targ_flux /= np.median(targ_flux)

	n_refs = int(df.keys()[-1].split('Ref ')[1].split(' ')[0])

	plt.figure(figsize=(10,12))
	plt.plot(times, targ_flux, '.', color='k', label='Targ.')
	plt.text(times[0]-(times[-1]-times[0])/75,targ_flux[0],'Targ.',color='k',ha='right',va='center',fontsize=12)

	xvals = np.zeros(n_refs+1) + times[0] - (times[-1]-times[0])/200
	#xvals[0] = times[0]
	#markers = ['v','s','p','*','+','x','D','|','X']
	markers = ['.']
	#colors = plt.get_cmap('viridis_r')
	#offset = 0.125
	colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:grey','tab:olive','tab:cyan']
	for i in range(n_refs):
		ref_flux = np.array(df[f'Ref {i+1} Source-Sky ADU'])
		#ref_flux /= np.median(ref_flux)
		plt.plot(times, ref_flux, marker=markers[i%len(markers)],ls='',label=f'{i+1}',color=colors[i%len(colors)])
		if i % 2 == 0:
			plt.text(times[0]-(times[-1]-times[0])/25,ref_flux[0],f'{i+1}',color=colors[i%len(colors)],ha='right',va='center',fontsize=12)
		else:
			plt.text(times[0]-(times[-1]-times[0])/100,ref_flux[0],f'{i+1}',color=colors[i%len(colors)],ha='right',va='center',fontsize=12)
	#breakpoint()
	plt.yscale('log')
	#plt.legend(loc='center left', bbox_to_anchor=(1,0.5),ncol=2)
	plt.xlim(times[0]-(times[-1]-times[0])/15,times[-1]+(times[-1]-times[0])/100)

	plt.ylabel('Flux (ADU)',fontsize=14)
	plt.xlabel(f'Time - {x_offset}'+' (BJD$_{TDB}$)',fontsize=14)
	plt.tick_params(labelsize=14)
	plt.tight_layout()

	output_path = lc_path.parent/f'{date}_{target}_raw_flux.png'
	plt.savefig(output_path,dpi=300)
	set_tierras_permissions(output_path)
	plt.close()

	return 

def tierras_binner(t, y, bin_mins=15):
	x_offset = t[0]
	t = t - x_offset
	t = t*24*60
	n_bins = int(np.ceil(t[-1]/bin_mins))
	bx = np.zeros(n_bins)
	by = np.zeros(n_bins)
	bye = np.zeros(n_bins)
	for i in range(n_bins):
		if i == n_bins-1:
			time_start = i*bin_mins
			time_end = t[-1]+1
		else:
			time_start = i*bin_mins
			time_end = (i+1)*bin_mins
		bin_inds = np.where((t>=time_start)&(t<time_end))[0]
		bx[i] = np.mean(t[bin_inds])/(24*60) + x_offset
		by[i] = np.mean(y[bin_inds])
		bye[i] = np.std(y[bin_inds])/np.sqrt(len(bin_inds))

	return bx, by, bye 
	

def tierras_binner_inds(t, bin_mins=15):
	x_offset = t[0]
	t = t - x_offset
	t = t*24*60
	n_bins = int(np.ceil(t[-1]/bin_mins))
	bin_inds = []
	for i in range(n_bins):
		if i == n_bins-1:
			time_start = i*bin_mins
			time_end = t[-1]+1
		else:
			time_start = i*bin_mins
			time_end = (i+1)*bin_mins
		bin_inds.append(np.where((t>=time_start)&(t<time_end))[0])
	  
	
	return bin_inds

def juliana_binning(binsize, times, flux, flux_err):
	'''Bins up a Tierras light curve following example from testextract.py 

	times: array of times in days
	
	'''
	std = np.empty([len(binsize)+1])
	decstd = np.empty([len(binsize)+1])
	theo = np.empty([len(binsize)+1])

	for ibinsize, thisbinsize in enumerate(binsize):

		nbin = (times[-1] - times[0]) * 1440.0 / thisbinsize

		bins = times[0] + thisbinsize * np.arange(nbin+1) / 1440.0
		
		wt = 1.0 / np.square(flux_err)
		
		ybn = np.histogram(times, bins=bins, weights=flux*wt)[0]
		#ybnd = np.histogram(times, bins=bins, weights=decflux*wt)[0]
		ybd = np.histogram(times, bins=bins, weights=wt)[0]
		
		wb = ybd > 0
		
		binned_flux = ybn[wb] / ybd[wb]
		#binned_decflux = ybnd[wb] / ybd[wb]

		std[ibinsize+1] = np.std(binned_flux)
		#decstd[ibinsize+1] = np.std(binned_decflux)
		theo[ibinsize+1] = np.sqrt(np.mean(1.0 / ybd[wb]))
	return std, theo

def transit_model(times, T0,P,Rp,a,inc,ecc,w,u1,u2):
	params = batman.TransitParams()
	params.t0 = T0
	params.per = P #orbital period in days
	params.rp = Rp #planet radius in units of stellar radii 
	params.a = a #semi-major axis in units of stellar radii 
	params.inc = inc #inclination in degrees
	params.ecc = ecc
	params.w = w #longitude of periastron in degrees\
	params.u = [u1,u2] #limb darkening coeffs
	params.limb_dark = 'quadratic'
	m = batman.TransitModel(params, times)    #initializes model
	return m.light_curve(params)

def moving_average(x, w):
	return np.convolve(x, np.ones(w), 'same') / w

def optimal_lc_chooser(date, target, ffname, overwrite=True, start_time=0, stop_time=0, plot=False):

	optimum_lc_file = f'/data/tierras/photometry/{date}/{target}/{ffname}/optimal_lc.txt'
	weight_file = f'/data/tierras/photometry/{date}/{target}/{ffname}/night_weights.csv'

	if (os.path.exists(optimum_lc_file)) and (os.path.exists(weight_file)) and not overwrite:
		with open(optimum_lc_file) as f:
			best_lc_path = f.readline()
	else:
		lc_list = np.array(glob(f'/data/tierras/photometry/{date}/{target}/{ffname}/*phot*.csv'))
		sort_inds = np.argsort([float(i.split('/')[-1].split('_')[-1].split('.csv')[0]) for i in lc_list])
		lc_list = lc_list[sort_inds]
		if plot:
			fig, ax = plt.subplots(len(lc_list),1,figsize=(10,1.25*len(lc_list)),sharex=True,sharey=True)
		
		best_stddev = 9999.
		for i in range(len(lc_list)):
			type = lc_list[i].split('/')[-1].split('_')[1]+' '+lc_list[i].split('/')[-1].split('_')[-1].split('.csv')[0]
			df = pd.read_csv(lc_list[i])
			times = np.array(df['BJD TDB'])
			breakpoint()
			rel_targ_flux = np.array(df['Target Post-Processed Normalized Flux'])
			rel_targ_flux_err = np.array(df['Target Post-Processed Normalized Flux Error'])
			
			#Trim any NaNs
			use_inds = ~np.isnan(rel_targ_flux)
			times = times[use_inds]
			rel_targ_flux = rel_targ_flux[use_inds]
			rel_targ_flux_err = rel_targ_flux_err[use_inds] 
			
			#Sigmaclip
			v,l,h = sigmaclip(rel_targ_flux)
			use_inds = np.where((rel_targ_flux>l)&(rel_targ_flux<h))[0]
			times = times[use_inds]
			rel_targ_flux = rel_targ_flux[use_inds]
			rel_targ_flux_err = rel_targ_flux_err[use_inds]
			
			norm = np.mean(rel_targ_flux)
			rel_targ_flux /= norm 
			rel_targ_flux_err /= norm
			
			#Allow user to specify start/stop times over which to evaluate light curve (i.e., to ignore transits)
			if (start_time != 0) and (stop_time != 0):
				eval_inds = np.where((times>=start_time)&(times<=stop_time))[0]
			else:
				eval_inds = np.arange(len(times))

			times_eval = times[eval_inds]
			flux_eval = rel_targ_flux[eval_inds]

			#Option 1: Evaluate the median standard deviation over 5-minute intervals 
			bin_inds = tierras_binner_inds(times_eval, bin_mins=5)
			stddevs = np.zeros(len(bin_inds))
			for j in range(len(bin_inds)):
				stddevs[j] = np.nanstd(flux_eval[bin_inds[j]])
			med_stddev = np.nanmedian(stddevs)

			breakpoint()
			# #Option 2: just evaluate stddev
			# med_stddev = np.std(flux_eval)

			#moving_avg = moving_average(rel_targ_flux,int(len(times)/50))
			if plot:
				ax[i].errorbar(times, rel_targ_flux, rel_targ_flux_err, marker='.',color='#b0b0b0',ls='')
				#ax[i].plot(times, moving_avg,color='tab:orange',lw=2,zorder=3)
				ax2 = ax[i].twinx()
				ax2.set_ylabel(lc_list[i].split('_')[-1].split('.csv')[0],rotation=270,labelpad=12)
				ax2.set_yticks([])
				
			#stddev = np.std(rel_targ_flux)
			print(f'{type} median 5-min stddev: {med_stddev*1e6:.1f} ppm')
			if med_stddev < best_stddev:
				best_ind = i
				best_lc_path = lc_list[i]
				best_stddev = med_stddev
				#weights_save = weights
		
		if plot:
			ax[-1].set_xlabel('Time (BJD$_{TDB}$)')
			plt.tight_layout()
			optimized_lc_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/{date}_{target}_optimized_lc.png'
			plt.savefig(optimized_lc_path,dpi=300)
			set_tierras_permissions(optimized_lc_path)
		
		#Write out the path of the optimum light curve file
		with open (f'/data/tierras/lightcurves/{date}/{target}/{ffname}/optimal_lc.txt','w') as f:
			f.write(best_lc_path)
		set_tierras_permissions(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/optimal_lc.txt')
		
		# #Save a .csv of the reference weights for the optimum light curve
		# ref_labels = [f'Ref {i+1}' for i in range(len(weights_save))]
		# weight_strs = [f'{val:.7f}' for val in weights_save]
		# weight_df = pd.DataFrame({'Reference':ref_labels,'Weight':weight_strs})
		# output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/night_weights.csv'
		# weight_df.to_csv(output_path,index=0)
		# set_tierras_permissions(output_path)
		

	return Path(best_lc_path)

def mearth_style_pat_weighted_flux(data_dict):
	""" Use the comparison stars to derive a frame-by-frame zero-point magnitude. Also filter and mask bad cadences """
	""" it's called "mearth_style" because it's inspired by the mearth pipeline """
	""" this version works with fluxes bc I hate the magnitude system with a burning passion"""
	
	bjds = data_dict['BJD']
	flux = data_dict['Flux']
	flux_err = data_dict['Flux Error']
	airmasses = data_dict['Airmass']
	exptimes = data_dict['Exptime']
	D = 130 #cm 
	H = 2306 # m 

	sigma_s = 0.09*D**(-2/3)*airmasses**(7/4)*(2*exptimes)**(-1/2)*np.exp(-H/8000)

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
		# target_source_id = cluster_ids[i] # this represents the ID of the "target" *in the photometry files
		regressor_inds = [j for j in np.arange(len(flux)) if i != j] # get the indices of the stars to use as the zero point calibrators; these represent the indices of the calibrators *in the data_dict arrays*
		# regressor_source_ids = cluster_ids[regressor_inds] # these represent the IDs of the calibrators *in the photometry files*  

		# grab target and source fluxes and apply initial mask 
		target_flux = data_dict['Flux'][i]
		target_flux_err = data_dict['Flux Error'][i]
		regressors = data_dict['Flux'][regressor_inds]
		regressors_err = data_dict['Flux Error'][regressor_inds]

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 

		tot_regressor = np.sum(regressors, axis=0)  # the total regressor flux at each time point = sum of comp star fluxes in each exposure
		tot_regressor[~mask] = np.nan

		# identify cadences with "low" flux by looking at normalized summed reference star fluxes
		zp0s = tot_regressor/np.nanmedian(tot_regressor) 	
		mask = np.ones_like(zp0s, dtype='bool')  # initialize another bad data mask
		mask[np.where(zp0s < 0.8)[0]] = 0  # if regressor flux is decremented by 20% or more, this cadence is bad
		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 

		# repeat the cs estimate now that we've masked out the bad cadences
		# phot_regressor = np.nanpercentile(regressors, 90, axis=1)  # estimate photometric flux level for each star
		norms = np.nanmedian(regressors, axis=1)
		regressors_norm = regressors / norms[:, None]
		regressors_err_norm = regressors_err / norms[:, None]

		# mask out any exposures where any reference star is significantly discrepant 
		mask = np.ones_like(target_flux, dtype='bool')
		for j in range(len(regressors_norm)):
			v, l, h = sigmaclip(regressors_norm[j][~np.isnan(regressors_norm[j])])
			mask[np.where((regressors_norm[j] < l) | (regressors_norm[j] > h))[0]] = 0

		target_flux[~mask] = np.nan 
		target_flux_err[~mask] = np.nan 
		for j in range(len(regressors)):
			regressors[j][~mask] = np.nan 
			regressors_err[j][~mask] = np.nan
			regressors_norm[j][~mask] = np.nan
			regressors_err_norm[j][~mask] = np.nan

		# now calculate the weights for each regressor
		# give stars weights equal to 1/(median photometric errors)**2
		# weights_init = np.ones(len(regressors))/len(regressors)
		weights_init = 1/(np.nanmedian(regressors_err_norm,axis=1)**2)
		weights_init /= np.nansum(weights_init)
	
		# do a 'crude' weighting loop to figure out which regressors, if any, should be totally discarded	
		delta_weights = np.zeros(len(regressors))+999 # initialize
		threshold = 1e-4 # delta_weights must converge to this value for the loop to stop
		weights_old = weights_init
		full_ref_inds = np.arange(len(regressors))
		while len(np.where(delta_weights>threshold)[0]) > 0:
			stddevs_measured = np.zeros(len(regressors))		
			stddevs_expected = np.zeros(len(regressors))

			# loop over each regressor
			for jj in range(len(regressors)):
				F_t = regressors[jj]
				N_t = regressors_err[jj]

				# make its zeropoint correction using the flux of all the *other* regressors
				use_inds = np.delete(full_ref_inds, jj)

				# re-normalize the weights to sum to one
				weights_wo_jj = weights_old[use_inds]
				weights_wo_jj /= np.nansum(weights_wo_jj)
				
				# create a zeropoint correction using those weights 
				F_e = np.matmul(weights_wo_jj, regressors[use_inds])
				N_e = np.sqrt(np.matmul(weights_wo_jj**2, regressors_err[use_inds]**2))

				# calculate the relative flux 
				F_rel_flux = F_t / F_e
				sigma_rel_flux = np.sqrt((N_t/F_e)**2 + (F_t*N_e/(F_e**2))**2)

				# calculate total error on F_rel flux from sigma_rel_flux and sigma_scint				
				sigma_scint = 1.5*sigma_s*np.sqrt(1 + 1/(len(use_inds)))
				sigma_tot = np.sqrt(sigma_rel_flux**2 + sigma_scint**2)

				# renormalize
				norm = np.nanmedian(F_rel_flux)
				F_corr = F_rel_flux/norm 
				sigma_tot_corr = sigma_tot/norm 	
	
				# record the standard deviation of the corrected flux
				stddevs_measured[jj] = np.nanstd(F_corr)
				stddevs_expected[jj] = np.nanmean(sigma_tot_corr)

			# update the weights using the measured standard deviations
			weights_new = 1/stddevs_measured**2
			weights_new /= np.nansum(weights_new)
			delta_weights = abs(weights_new-weights_old)
			weights_old = weights_new

		weights = weights_new

		# determine if any references should be totally thrown out based on the ratio of their measured/expected noise
		noise_ratios = stddevs_measured/stddevs_expected
		# the noise ratio threshold will depend on how many bad/variable reference stars were used in the ALC
		# sigmaclip the noise ratios and set the upper limit to the n-sigma upper bound 
		# v, l, h = sigmaclip(noise_ratios, 2, 2)
		# weights[np.where(noise_ratios>h)[0]] = 0

		weights[np.where(noise_ratios>5)[0]] = 0
		weights /= sum(weights)
		
		if len(np.where(weights == 0)[0]) > 0:
			# now repeat the weighting loop with the bad refs removed 
			delta_weights = np.zeros(len(regressors))+999 # initialize
			threshold = 1e-6 # delta_weights must converge to this value for the loop to stop
			weights_old = weights
			full_ref_inds = np.arange(len(regressors))
			count = 0
			while len(np.where(delta_weights>threshold)[0]) > 0:
				stddevs_measured = np.zeros(len(regressors))
				stddevs_expected = np.zeros(len(regressors))

				for jj in range(len(regressors)):
					if weights_old[jj] == 0:
						continue
					F_t = regressors[jj]
					N_t = regressors_err[jj]
		
					use_inds = np.delete(full_ref_inds, jj)
					weights_wo_jj = weights_old[use_inds]
					weights_wo_jj /= np.nansum(weights_wo_jj)
					
					# create a zeropoint correction using those weights 
					F_e = np.matmul(weights_wo_jj, regressors[use_inds])
					N_e = np.sqrt(np.matmul(weights_wo_jj**2, regressors_err[use_inds]**2))

					# calculate the relative flux 
					F_rel_flux = F_t / F_e
					sigma_rel_flux = np.sqrt((N_t/F_e)**2 + (F_t*N_e/(F_e**2))**2)

					# calculate total error on F_rel flux from sigma_rel_flux and sigma_scint				
					sigma_scint = 1.5*sigma_s*np.sqrt(1 + 1/(len(use_inds)))
					sigma_tot = np.sqrt(sigma_rel_flux**2 + sigma_scint**2)

					# renormalize
					norm = np.nanmedian(F_rel_flux)
					F_corr = F_rel_flux/norm 
					sigma_tot_corr = sigma_tot/norm 	
		
					# record the standard deviation of the corrected flux
					stddevs_measured[jj] = np.nanstd(F_corr)
					stddevs_expected[jj] = np.nanmean(sigma_tot_corr)
				
				weights_new = 1/(stddevs_measured**2)
				weights_new /= np.sum(weights_new[~np.isinf(weights_new)])
				weights_new[np.isinf(weights_new)] = 0
				delta_weights = abs(weights_new-weights_old)
				weights_old = weights_new
				count += 1

		weights = weights_new

		
		# calculate the zero-point correction

		F_e = np.matmul(weights, regressors)
		N_e = np.sqrt(np.matmul(weights**2, regressors_err**2))	
		
		flux_corr = target_flux / F_e
		err_corr = np.sqrt((target_flux_err/F_e)**2 + (target_flux*N_e/(F_e**2))**2)

		# renormalize
		norm = np.nanmedian(flux_corr)
		flux_corr /= norm 
		err_corr /= norm 

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

def ap_range(file_list, targ_and_refs, overwrite=False, plots=False):
	'''Measures the average FWHM of the target across a set of images to determine a range of apertures for performing photometry.
	'''
	ffname = file_list[0].parent.name
	target = file_list[0].parent.parent.name
	date = file_list[0].parent.parent.parent.name


	output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/aperture_range.csv'
	output_path_2 = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/fwhm.csv'

	if not (os.path.exists(output_path)) or not (os.path.exists(output_path_2)) or (overwrite==True):

		print('Determining optimal aperture range...')
		time.sleep(2)

		PLATE_SCALE = 0.43 #arcsec pix^-1, from Juliana's dissertation Table 1.1

		#bpm = load_bad_pixel_mask()

		#load in the reference image 
		reference_image_hdu = fits.open('/data/tierras/targets/'+target+'/'+target+'_stacked_image.fits')[0] #TODO: should match image from target/reference csv file, and that should be loaded automatically.

		reference_image_header = reference_image_hdu.header
		reference_wcs = WCS(reference_image_header)
		try:
			reference_world_coordinates = reference_wcs.pixel_to_world(targ_and_refs['X pix'][0], targ_and_refs['Y pix'][0])
		except:
			reference_world_coordinates = reference_wcs.pixel_to_world(targ_and_refs['x'][0],targ_and_refs['y'][0]) #Get world coordinates of target in the reference image.

		fwhm_x = np.zeros(len(file_list))
		fwhm_y = np.zeros(len(file_list))
		theta = np.zeros(len(file_list))

		if plots:
			fig = plt.figure(figsize=(10,7))
			gs = gridspec.GridSpec(2,3)
			ax1 = fig.add_subplot(gs[0,0])
			ax2 = fig.add_subplot(gs[0,1])
			ax3 = fig.add_subplot(gs[0,2])
			ax4 = fig.add_subplot(gs[1,:])

		for i in range(len(file_list)):
			hdu = fits.open(file_list[i])[0]
			header = hdu.header
			data = hdu.data
			wcs = WCS(header)
			transformed_pixel_coordinates = wcs.world_to_pixel(reference_world_coordinates) 
			x_pos_image = transformed_pixel_coordinates[0]
			y_pos_image = transformed_pixel_coordinates[1]

			#Set up the target cutout
			cutout_y_start = int(y_pos_image-15)
			if cutout_y_start < 0:
				cutout_y_start = 0
			cutout_y_end = int(y_pos_image+15)
			if cutout_y_end > 2047:
				cutout_y_end = 2047
			cutout_x_start = int(x_pos_image-15)
			if cutout_x_start < 0:
				cutout_x_start = 0
			cutout_x_end = int(x_pos_image+15)
			if cutout_x_end > 4095:
				cutout_x_end = 4095

			cutout = data[cutout_y_start:cutout_y_end+1,cutout_x_start:cutout_x_end+1]

			#Fit a 2D gaussian to the cutout
			xx,yy = np.meshgrid(np.arange(cutout.shape[1]),np.arange(cutout.shape[0]))
			g_init = models.Gaussian2D(amplitude=cutout[int(cutout.shape[1]/2), int(cutout.shape[0]/2)]-np.median(cutout),x_mean=cutout.shape[1]/2,y_mean=cutout.shape[0]/2, x_stddev=5, y_stddev=5)
			# g_init.theta.bounds = (-np.pi/2, np.pi/2)
			fit_g = fitting.LevMarLSQFitter()
			g = fit_g(g_init,xx,yy,cutout-np.median(cutout))

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
			theta[i] = theta_rad
			fwhm_x[i] = x_fwhm_arcsec
			fwhm_y[i] = y_fwhm_arcsec

			print(f'{i+1} of {len(file_list)}')

			if plots:
				norm = simple_norm(cutout, 'linear', min_percent=1, max_percent=95)
				im1 = ax1.imshow(cutout,origin='lower',interpolation='none', norm=norm)
				divider = make_axes_locatable(ax1)
				cax = divider.append_axes('right',size='5%',pad=0.05)
				cb1 = fig.colorbar(im1, cax=cax, orientation='vertical')
				ax1.set_title('Target Cutout')

				im2 = ax2.imshow(g(xx,yy),origin='lower',interpolation='none', norm=norm)			
				divider = make_axes_locatable(ax2)
				cax = divider.append_axes('right',size='5%',pad=0.05)
				cb2 = fig.colorbar(im2, cax=cax, orientation='vertical')
				ax2.set_title('2D Gaussian Model')

				im3 = ax3.imshow(cutout-g(xx,yy),origin='lower',interpolation='none')
				divider = make_axes_locatable(ax3)
				cax = divider.append_axes('right',size='5%',pad=0.05)
				cb3 = fig.colorbar(im3, cax=cax, orientation='vertical')
				ax3.set_title('Residuals')

				ax4.plot(fwhm_x[0:i+1],lw=2,label='FWHM X')
				ax4.plot(fwhm_y[0:i+1],lw=2,label='FWHM Y')
				ax4.legend(loc='lower right')
				ax4.set_ylabel('FWHM (")')

				plt.subplots_adjust(wspace=0.4,left=0.1,right=0.9)
				plt.suptitle(f'{i+1} of {len(file_list)}')
				plt.pause(0.2)
				if not os.path.exists(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/fwhm_measurements/'):
					os.mkdir(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/fwhm_measurements/')
					set_tierras_permissions(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/fwhm_measurements/')
				output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/fwhm_measurements/'+str(i).zfill(4)+'.jpg'
				plt.savefig(output_path,dpi=100)
				set_tierras_permissions(output_path)
				ax1.cla()
				cb1.remove()
				ax2.cla()
				cb2.remove()
				ax3.cla()
				cb3.remove()
				ax4.cla()
		
		fwhm_x_save = copy.deepcopy(fwhm_x)
		fwhm_y_save = copy.deepcopy(fwhm_y)
		theta_save = copy.deepcopy(theta)

		#Sigma clip the results
		v1,l1,h1 = sigmaclip(fwhm_x)
		v2,l2,h2 = sigmaclip(fwhm_y) 
		use_inds = np.where((fwhm_x>l1)&(fwhm_x<h1)&(fwhm_y>l2)&(fwhm_y<h2))[0]
		fwhm_x = fwhm_x[use_inds]
		fwhm_y = fwhm_y[use_inds]

		#Use the lower of the 75th percentiles of fwhm_x/y to set the lower aperture radius bound 
		fwhm_x_75_pix = np.percentile(fwhm_x/PLATE_SCALE,75)
		fwhm_y_75_pix = np.percentile(fwhm_y/PLATE_SCALE,75)
		stddev_x_75_pix = fwhm_x_75_pix/2.355
		stddev_y_75_pix = fwhm_y_75_pix/2.355
		# lower_pix_bound = int(np.floor(np.min([fwhm_x_75_pix,fwhm_y_75_pix])*1))
		# if lower_pix_bound < 1:
		# 	lower_pix_bound = 1
		# upper_pix_bound = int(np.ceil(np.max([fwhm_x_75_pix,fwhm_y_75_pix])*2))

		lower_pix_bound = int(np.floor(np.min([stddev_x_75_pix,stddev_y_75_pix])*1.25))
		if lower_pix_bound < 4: #ain't no way it's smaller than 4
			lower_pix_bound = 4
		upper_pix_bound = int(np.ceil(np.max([stddev_x_75_pix,stddev_y_75_pix])*4.0))

		aps_to_use = np.arange(lower_pix_bound, upper_pix_bound+1)

		an_in = int(np.ceil(6*np.max([fwhm_x_75_pix,fwhm_y_75_pix])))
		an_ins_to_use = np.zeros(len(aps_to_use),dtype=int) + an_in
		an_outs_to_use = an_ins_to_use + 20

		output_dict = {'Aperture radii':aps_to_use, 'Inner annulus radii':an_ins_to_use, 'Outer annulus radii':an_outs_to_use}
		output_df = pd.DataFrame(output_dict)
		output_df.to_csv(output_path,index=False)
		set_tierras_permissions(output_path)

		#Write out fwhm measurements
		output_dict_2 = {'FWHM X':fwhm_x_save, 'FWHM Y':fwhm_y_save, 'Theta':theta_save}
		output_df_2 = pd.DataFrame(output_dict_2)
		output_df_2.to_csv(output_path_2,index=False)
		set_tierras_permissions(output_path_2)
	else:
		print(f'Restoring aperture range output from {output_path}.')
		output_df = pd.read_csv(output_path)
		aps_to_use = np.array(output_df['Aperture radii'])
		an_ins_to_use = np.array(output_df['Inner annulus radii'])
		an_outs_to_use = np.array(output_df['Outer annulus radii'])

	return aps_to_use, an_ins_to_use[0], an_outs_to_use[0]

def exclude_files(date, target, ffname,stdcrms_clip_threshold=6):
	#TODO: This will not work on files for which you don't have rwx permissions...
	file_list = get_flattened_files(date,target,ffname)
	stdcrms = np.zeros(len(file_list))
	for i in range(len(file_list)):
		print(f'{i+1} of {len(file_list)}')
		hdu = fits.open(file_list[i])[0]
		data = hdu.data
		header = hdu.header
		stdcrms[i] = header['STDCRMS']
		
	v,l,h = sigmaclip(stdcrms,stdcrms_clip_threshold,stdcrms_clip_threshold)
	bad_inds = np.where((stdcrms<l)|(stdcrms>h))[0]
	if len(bad_inds) > 0:
		if not (file_list[0].parent/'excluded').exists():
			os.mkdir(file_list[0].parent/'excluded')
			set_tierras_permissions(file_list[0].parent/'excluded')
		for i in range(len(bad_inds)):
			file_to_move = file_list[bad_inds[i]]
			print(f'Moving {file_to_move} to {file_to_move.parent}/excluded/')
			breakpoint()
			os.replace(file_to_move, file_list[0].parent/('excluded')/(file_to_move.name))

	return

def set_tierras_permissions(path):
	try:
		os.chmod(path, stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR|stat.S_IRGRP|stat.S_IWGRP|stat.S_IXGRP|stat.S_IROTH|stat.S_IXOTH)
		shutil.chown(path, user=None, group='exoplanet')
	except:
		print(f'Could not change permissions on {path}, returning.')
	return 

def get_lunar_distance(ra, dec, time, loc=coord.EarthLocation.of_site('Whipple')):
	#TODO: NOT POSITIVE THIS IS RIGHT!
	astropy_time = Time(time, format='jd', scale='tdb')
	field_coord = SkyCoord(ra,dec, unit=(u.hourangle, u.deg))
	moon_coord = get_body('moon', astropy_time, location=loc)
	field_coord_alt_az = field_coord.transform_to(AltAz(obstime=astropy_time, location=loc))
	moon_alt_az = moon_coord.transform_to(AltAz(obstime=astropy_time, location=loc))
	moon_radec = moon_alt_az.transform_to('icrs')
	return field_coord_alt_az.separation(moon_alt_az).value

def t_or_f(arg):
	ua = str(arg).upper()
	if 'TRUE'.startswith(ua):
		return True
	elif 'FALSE'.startswith(ua):
		return False
	else:
		print(f'ERROR: check passed argument for {arg}.')

def tierras_ref_weighting(df, crude_convergence=1e-4, fine_convergence=1e-6, bad_ref_threshold=10, iteration_limit=100, plots=False):
	'''Based off the PINES algorithm, but entirely de-weights references if they have measured noise that is bad_ref_threshold times higher than their expected noise.
	'''
	#times = np.array(df['BJD TDB'])
	x = np.array(df['Target X'])
	y = np.array(df['Target Y'])
	fwhm_x = np.array(df['Target X FWHM Arcsec'])
	fwhm_y = np.array(df['Target Y FWHM Arcsec'])
	airmass = np.array(df['Airmass'])
	
	n_refs = int(df.keys()[-1].split('Ref ')[1].split(' ')[0])
	ref_inds = np.arange(n_refs)
	n_ims = len(df)
	raw_fluxes = np.zeros((n_ims,n_refs))
	raw_flux_errs = np.zeros((n_ims,n_refs))
	for i in range(n_refs):
		raw_fluxes[:,i] = np.array(df[f'Ref {i+1} Source-Sky ADU'])
		raw_flux_errs[:,i] = np.array(df[f'Ref {i+1} Source-Sky Error ADU'])
	
	w_var = np.nanmean(raw_flux_errs,axis=0)**2 #Set initial weights using calculated uncertainties
	w_var /= np.nansum(w_var)

	#Do a 'crude' loop to first figure out which refs should be totally tossed out
	corr_fluxes = np.zeros((n_ims,n_refs))
	w_old = copy.deepcopy(w_var)
	delta_weights = np.ones(n_refs)
	count = 0 
	while sum(delta_weights>crude_convergence)>0:
		#print(f'{count+1}')
		w_new = np.zeros(n_refs)
		for i in range(n_refs):
			salc_inds = np.delete(ref_inds, i) #Create a "special" ALC for this reference, using the fluxes of all OTHER reference stars. This is the main difference between the SPECULOOS and PINES algorithms. 
			salc_fluxes = raw_fluxes[:,salc_inds]
			w_salc = w_old[salc_inds]
			w_salc /= np.nansum(w_salc) #renormalize
			salc = np.nansum(w_salc*salc_fluxes,axis=1)

			corr_flux = raw_fluxes[:,i]/salc #Correct using the salc
			corr_flux /= np.nanmean(corr_flux) 
			#corr_fluxes[:,i] = corr_flux
			v, l, h = sigmaclip(corr_flux[~np.isnan(corr_flux)])
			use_inds = np.where((corr_flux>l)&(corr_flux<h))[0]
			
			#NEW BIT: do a regression against ancillary variables and THEN measure weight
			ancillary_dict = {'X':x[use_inds], 'Y':y[use_inds], 'FWHM X':fwhm_x[use_inds], 'FWHM Y':fwhm_y[use_inds], 'Airmass':airmass[use_inds]}
			reg_flux, intercept, coeffs, regress_dict = regression(corr_flux[use_inds],ancillary_dict)
			corr_fluxes[use_inds,i] = reg_flux
			w_new[i] = 1/(np.nanstd(reg_flux)**2)
			#w_new[i] = 1/(np.nanstd(corr_flux[use_inds])**2) #Set new weight using measured standard deviation of corrected flux
		
		w_new /= np.nansum(w_new)
		delta_weights = abs(w_old - w_new)
		w_old = w_new
		count += 1
		if count == iteration_limit:
			break 

	w_crude = w_new 
	#Now determine which refs should be totally excluded based on the ratio of their measured/expected noise. 
	use_ref_inds = np.ones(n_refs,dtype='int')
	for i in range(n_refs):
		corr_flux = corr_fluxes[:,i]
		raw_flux = raw_fluxes[:,i]
		raw_flux_err = raw_flux_errs[:,i]

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

	#Now do a more intensive loop with bad references given 0 weight. 
	w_old *= use_ref_inds
	w_old /= np.nansum(w_old)
	corr_fluxes = np.zeros((n_ims,n_refs))
	delta_weights = np.ones(n_refs)
	count = 0 
	while sum(delta_weights>fine_convergence)>0:
		#print(f'{count+1}')
		w_new = np.zeros(n_refs)
		for i in range(n_refs):
			if use_ref_inds[i] == 0: #Don't bother optimizing on bad refs 
			   continue 
			salc_inds = np.delete(ref_inds, i) #Create a "special" ALC for this reference, using the fluxes of all OTHER reference stars. This is the main difference between the SPECULOOS and PINES algorithms. 
			salc_fluxes = raw_fluxes[:,salc_inds]
			w_salc = w_old[salc_inds]
			w_salc /= np.nansum(w_salc) #renormalize
			salc = np.nansum(w_salc*salc_fluxes,axis=1)

			corr_flux = raw_fluxes[:,i]/salc #Correct using the salc
			corr_flux /= np.nanmean(corr_flux) 
			#corr_fluxes[:,i] = corr_flux
			v, l, h = sigmaclip(corr_flux[~np.isnan(corr_flux)])
			use_inds = np.where((corr_flux>l)&(corr_flux<h))[0]
			
			#NEW BIT: do a regression against ancillary variables and THEN measure weight
			ancillary_dict = {'X':x[use_inds], 'Y':y[use_inds], 'FWHM X':fwhm_x[use_inds], 'FWHM Y':fwhm_y[use_inds], 'Airmass':airmass[use_inds]}
			reg_flux, intercept, coeffs, regress_dict = regression(corr_flux[use_inds],ancillary_dict)
			w_new[i] = 1/(np.nanstd(reg_flux)**2)
			#w_new[i] = 1/(np.nanstd(corr_flux[use_inds])**2) #Set new weight using measured standard deviation of corrected flux
		
		w_new /= np.nansum(w_new)
		delta_weights = abs(w_old - w_new)
		w_old = w_new
		count += 1
		if count == iteration_limit:
			break
	alc = np.nansum(w_new*raw_fluxes,axis=1)
	alc_err = np.sqrt(np.nansum((w_new*raw_flux_errs)**2,axis=1))
	if len(np.where(alc == 0)[0]) == len(alc):
		breakpoint()
	return w_new, alc, alc_err

def weighted_alc(lc_path):
	lc_path = Path(lc_path)
	ffname = lc_path.parents[0].name
	target = lc_path.parents[1].name
	date = lc_path.parents[2].name
	try:
		weight_df = pd.read_csv(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/night_weights.csv')
	except:
		raise RuntimeError(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/night_weights.csv does not exist, run optimal_lc_chooser first!')
	weights = np.array(weight_df['Weight'])
	flux_df = pd.read_csv(lc_path)
	n_refs = int(flux_df.keys()[-1].split('Ref ')[1].split(' ')[0])
	n_ims = len(flux_df)
	if n_refs != len(weight_df):
		raise RuntimeError()
	ref_fluxes = np.zeros((n_ims, n_refs))
	ref_flux_errs = np.zeros((n_ims, n_refs))
	for i in range(n_refs):
		ref_fluxes[:,i] = np.array(flux_df[f'Ref {i+1} Source-Sky ADU'])
		ref_flux_errs[:,i] = np.array(flux_df[f'Ref {i+1} Source-Sky Error ADU'])
	alc = np.sum(weights*ref_fluxes,axis=1)
	alc_err = np.sqrt(np.sum((weights*ref_flux_errs)**2,axis=1))
	return alc, alc_err

def regression(flux, ancillary_dict, pval_threshold=1e-3, verbose=False):
	'''
		PURPOSE: 
			Identifies data vectors that are significantly correlated with a flux array and performs a linear regression with those vectors to correct the flux.
		INPUTS:
			flux (array): The flux that you wish to correct.
			ancillary_dict (dict): A dictionary containing the data vectors that you would like to test in the regression. E.g., {'X':x_position,'Y':y_position,'FWHM X':fwhm_x,'FWHM Y':fwhm_y, 'Airmass':airmass}
			pval_threshold (float): The P-value of the null hypothesis that a data vector and flux are not correlated. If the measured P-value is less than pval_threshold, the associated vector is used in the regression.
		OUTPUTS:
			regressed_flux (array): Array of fluxes corrected by the regression model
			regression_model (array): Array of regression model fluxes

	'''
	regr = linear_model.LinearRegression()
	regress_dict = {}
	
	#Check for significant correlations between ancillary data and corrected target flux.
	#Any significantly correlated vectors get added to regress_dict
	for key in ancillary_dict:
		try:
			corr, pvalue = pearsonr(flux,ancillary_dict[key])
		except:
			continue
		if pvalue < pval_threshold:
			regress_dict[key] = ancillary_dict[key]
			if verbose:
				print(f'{key}, corr:{corr:.2f}, P-value: {pvalue:.2E}')

	if len(regress_dict.keys()) == 0:
		if verbose:
			print('No significantly correlated data vectors, returning.')
		return flux, 0, [0], {}

	regress_dict['flux'] = flux
	keylist = list(regress_dict.keys())
	
	regress_df = pd.DataFrame(regress_dict, columns=list(regress_dict.keys()))
	x = regress_df[keylist[0:len(keylist)-1]]
	y = regress_df['flux']

	#Perform regression and unpack the model
	regr.fit(x,y)
	regression_model = regr.intercept_
	regress_dict_return = {}
	for i in range(len(keylist[:-1])):
		if verbose:
			print(f'{regr.coef_[i]:.4E}*{keylist[i]}')
		regression_model += regr.coef_[i]*regress_dict[keylist[i]]
		regress_dict_return[keylist[i]] = regress_dict[keylist[i]]

	#Correct the flux with the regression model 
	regressed_flux = flux/regression_model
	
	intercept = regr.intercept_
	coeffs = regr.coef_
	return regressed_flux, intercept, coeffs, regress_dict_return

def quotient_uncertainty(a,a_err,b,b_err):
	return np.sqrt((a_err/b)**2+(a*b_err/(b**2))**2)

def lc_post_processing(date, target, ffname,overwrite=False):
	optimum_lc_file = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/optimal_lc.txt'
	weight_file = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/night_weights.csv'

	if (os.path.exists(optimum_lc_file)) and (os.path.exists(weight_file)) and not overwrite:
		with open(optimum_lc_file) as f:
			best_lc_path = f.readline()
	else:
		GAIN = 5.9 #e- ADU^-1, does this ever change? 
		lc_list = np.array(glob(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/*phot*.csv'))
		sort_inds = np.argsort([float(i.split('/')[-1].split('_')[-1].split('.csv')[0]) for i in lc_list])
		lc_list = lc_list[sort_inds]
		best_stddev = 9999. #Initialize
		for i in range(len(lc_list)):
			print(f'Post-processing {lc_list[i]}')
			type = lc_list[i].split('/')[-1].split('_')[1]+' '+lc_list[i].split('/')[-1].split('_')[-1].split('.csv')[0]
			df = pd.read_csv(lc_list[i])
			n_ims = len(df)
			n_refs = int(df.keys()[-1].split('Ref ')[1].split(' ')[0])

			raw_fluxes = np.zeros((n_ims, n_refs+1))
			raw_flux_errors = np.zeros((n_ims,n_refs+1))
			#The post-processed fluxes are the fluxes constructed with a weighted ALC and run through a regression
			post_processed_fluxes = np.zeros((n_ims,n_refs+1))
			post_processed_flux_errors = np.zeros((n_ims,n_refs+1))
			regression_models = np.zeros((n_ims, n_refs+1))
			weighted_alcs = np.zeros((n_ims, n_refs+1))
			weighted_alc_errs = np.zeros((n_ims, n_refs+1))

			times = np.array(df['BJD TDB'])
			x = np.array(df['Target X'])
			y = np.array(df['Target Y'])
			fwhm_x = np.array(df['Target X FWHM Arcsec'])
			fwhm_y = np.array(df['Target Y FWHM Arcsec'])
			airmass = np.array(df['Airmass'])

			#NEW: Use Tierras weighting to create the ALC and generate relative target flux
			weights, alc, alc_err = tierras_ref_weighting(df)
			
			#Read in raw fluxes 
			for j in range(n_refs+1):
				if j == 0:
					targ = 'Target'
				else:
					targ = f'Ref {j}'
				raw_fluxes[:,j] = np.array(df[f'{targ} Source-Sky ADU'])
				raw_flux_errors[:,j] = np.array(df[f'{targ} Source-Sky Error ADU'])
			
			#Creat post-processed light curves using weights and performing regression
			for j in range(n_refs+1):

				# create a dictionary that can be used in the regression 
				ancillary_dict = {'X':x,'Y':y,'FWHM X':fwhm_x,'FWHM Y':fwhm_y,'Airmass':airmass}
				ancillary_dict_sc = copy.deepcopy(ancillary_dict)

				#
				use_ref_inds = np.arange(0,n_refs)
				if j == 0:
					targ = 'Target'
				else:
					targ = f'Ref {j}'
					use_ref_inds = np.delete(use_ref_inds, j-1)
				
				weights_loop = weights[use_ref_inds]
				weights_loop /= np.sum(weights_loop) #Renormalize

				#Generate this target's ALC using the weights
				raw_flux = raw_fluxes[:,j]
				raw_flux_err = raw_flux_errors[:,j]
				alc = np.nansum(weights_loop*raw_fluxes[:,use_ref_inds+1],axis=1)
				alc_err = np.sqrt(np.nansum((weights_loop*raw_flux_errors[:,use_ref_inds+1])**2,axis=1))
				
				#Correct with the ALC
				rel_flux = raw_flux/alc
				rel_flux_err = quotient_uncertainty(raw_flux,raw_flux_err,alc,alc_err)

				#Trim any NaNs
				use_inds = ~np.isnan(rel_flux)
				rel_flux_sc = rel_flux[use_inds]
				rel_flux_err_sc = rel_flux_err[use_inds] 

				#Sigmaclip
				v,l,h = sigmaclip(rel_flux_sc)
				use_inds = np.where((rel_flux_sc>l)&(rel_flux_sc<h))[0]
				rel_flux_sc = rel_flux_sc[use_inds]
				rel_flux_sc_err = rel_flux_err_sc[use_inds]
				for key in ancillary_dict.keys():
					ancillary_dict_sc[key] = ancillary_dict_sc[key][use_inds]

				norm = np.mean(rel_flux_sc)
				rel_flux_sc /= norm 
				rel_flux_sc_err /= norm

				#NEW: DO REGRESSION
				reg_flux, intercept, coeffs, regress_dict = regression(rel_flux_sc,ancillary_dict_sc,verbose=False)

				#Construct the model on the FULL flux (not sigmaclipped/NaN'd)
				if intercept == 0:
					#If no regression was performed (no variables were significantly correlated with the input flux) the regression model should be treated as an array of ones
					reg_model = np.ones(len(rel_flux))
				else:
					reg_model = intercept
					reg_keys = regress_dict.keys()
					k = 0  
					reg_dict_full = {}
					for key in regress_dict.keys():
						reg_dict_full[key] = ancillary_dict[key]
						reg_model += coeffs[k]*reg_dict_full[key]
						k+=1
				reg_flux = rel_flux / reg_model 
				reg_flux_err = rel_flux_err /reg_model
				
				norm = np.nanmean(reg_flux[use_inds])
				reg_flux /= norm 
				reg_flux_err /= norm 

				post_processed_fluxes[:,j] = reg_flux
				post_processed_flux_errors[:,j] = reg_flux_err
				regression_models[:,j] = reg_model		
				weighted_alcs[:,j] = alc
				weighted_alc_errs[:,j] = alc_err
				
				#Insert the new columns into the dataframe.
				ind1 = int(np.where(df.keys() == f'{targ} Ensemble ALC Error ADU')[0][0]+1)
				try:
					df.insert(ind1, f'{targ} Weighted ALC ADU',  weighted_alcs[:,j])
				except:
					df[f'{targ} Weighted ALC ADU'] = weighted_alcs[:,j]
				try:
					df.insert(ind1+1, f'{targ} Weighted ALC Error ADU', weighted_alc_errs[:,j])
				except:
					df[f'{targ} Weighted ALC Error ADU'] = weighted_alc_errs[:,j]
				try:
					df.insert(ind1+2, f'{targ} Weighted ALC e', weighted_alcs[:,j]*GAIN)
				except:
					df[f'{targ} Weighted ALC e'] = weighted_alcs[:,j]*GAIN
				try:
					df.insert(ind1+3, f'{targ} Weighted ALC Error e', weighted_alc_errs[:,j]*GAIN)
				except:
					df[f'{targ} Weighted ALC Error e'] = weighted_alc_errs[:,j]*GAIN

				ind2 = int(np.where(df.keys() == f'{targ} Relative Flux Error')[0][0]+1)
				try:
					df.insert(ind2, f'{targ} Regression Model',regression_models[:,j])
				except:
					df[f'{targ} Regression Model'] = regression_models[:,j]
				try:
					df.insert(ind2+1, f'{targ} Post-Processed Normalized Flux', post_processed_fluxes[:,j])
				except:
					df[f'{targ} Post-Processed Normalized Flux'] = post_processed_fluxes[:,j]
				try:
					df.insert(ind2+2, f'{targ} Post-Processed Normalized Flux Error', post_processed_flux_errors[:,j])
				except:
					df[f'{targ} Post-Processed Normalized Flux Error'] = post_processed_flux_errors[:,j]
				
				#Measure the median standard deviation over 5-min bins to select the best aperture 
				if j == 0:
					rel_targ_flux = post_processed_fluxes[:,j]
					rel_targ_flux_err = post_processed_flux_errors[:,j]
					#Trim any NaNs
					use_inds = ~np.isnan(rel_targ_flux)
					times = times[use_inds]
					rel_targ_flux = rel_targ_flux[use_inds]
					rel_targ_flux_err = rel_targ_flux_err[use_inds] 
					
					#Sigmaclip
					v,l,h = sigmaclip(rel_targ_flux)
					use_inds = np.where((rel_targ_flux>l)&(rel_targ_flux<h))[0]
					times = times[use_inds]
					rel_targ_flux = rel_targ_flux[use_inds]
					rel_targ_flux_err = rel_targ_flux_err[use_inds]
					
					norm = np.mean(rel_targ_flux)
					rel_targ_flux /= norm 
					rel_targ_flux_err /= norm

					#Option 1: Evaluate the median standard deviation over 5-minute intervals 
					bin_inds = tierras_binner_inds(times, bin_mins=5)
					stddevs = np.zeros(len(bin_inds))
					for jj in range(len(bin_inds)):
						stddevs[jj] = np.nanstd(rel_targ_flux[bin_inds[jj]])
					med_stddev = np.nanmedian(stddevs)

						
					#stddev = np.std(rel_targ_flux)
					print(f'Median 5-min stddev: {med_stddev*1e6:.1f} ppm\n')
					if med_stddev < best_stddev:
						best_ind = i
						best_lc_path = lc_list[i]
						best_stddev = med_stddev
						weights_save = weights
					

			#Write out the updated dataframe
			df.to_csv(lc_list[i],index=0)
		
		#Write out the path of the optimum light curve file
		with open (f'/data/tierras/lightcurves/{date}/{target}/{ffname}/optimal_lc.txt','w') as f:
			f.write(best_lc_path)
		set_tierras_permissions(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/optimal_lc.txt')
		
		#Save a .csv of the reference weights for the optimum light curve
		ref_labels = [f'Ref {i+1}' for i in range(len(weights_save))]
		weight_strs = [f'{val:.7f}' for val in weights_save]
		weight_df = pd.DataFrame({'Reference':ref_labels,'Weight':weight_strs})
		output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/night_weights.csv'
		weight_df.to_csv(output_path,index=0)
		set_tierras_permissions(output_path)

	return Path(best_lc_path)

def make_data_dirs(date, target, ffname):
	#Define base paths
	global fpath, lcpath
	lcpath = '/data/tierras/lightcurves'
	photpath = '/data/tierras/photometry'

	if not os.path.exists(lcpath+f'/{date}'):
		os.mkdir(lcpath+f'/{date}')
		set_tierras_permissions(lcpath+f'/{date}')
	if not os.path.exists(lcpath+f'/{date}/{target}'):
		os.mkdir(lcpath+f'/{date}/{target}')
		set_tierras_permissions(lcpath+f'/{date}/{target}')
	if not os.path.exists(lcpath+f'/{date}/{target}/{ffname}'):
		os.mkdir(lcpath+f'/{date}/{target}/{ffname}')
		set_tierras_permissions(lcpath+f'/{date}/{target}/{ffname}')
	
	if not os.path.exists(photpath+f'/{date}'):
		os.mkdir(photpath+f'/{date}')
		set_tierras_permissions(photpath+f'/{date}')
	if not os.path.exists(photpath+f'/{date}/{target}'):
		os.mkdir(photpath+f'/{date}/{target}')
		set_tierras_permissions(photpath+f'/{date}/{target}')
	if not os.path.exists(photpath+f'/{date}/{target}/{ffname}'):
		os.mkdir(photpath+f'/{date}/{target}/{ffname}')
		set_tierras_permissions(photpath+f'/{date}/{target}/{ffname}')
	return 

def main(raw_args=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
	ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")
	ap.add_argument("-ffname", required=False, default='flat0000', help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
	ap.add_argument("-ap_rad_lo", required=False, default=5, help="Lower bound of aperture radii in pixels. Apertures with sizes between ap_rad_lo and ap_rad_hi will be used.", type=float)
	ap.add_argument("-ap_rad_hi", required=False, default=20, help="Upper bound of aperture radii in pixels. Apertures with sizes between ap_rad_lo and ap_rad_hi will be used.", type=float)
	ap.add_argument("-an_in", required=False, default=35, help='Inner background annulus radius in pixels.', type=float)
	ap.add_argument("-an_out", required=False, default=55, help='Outer background annulus radius in pixels.', type=float)
	ap.add_argument("-edge_limit",required=False,default=20,help="Minimum separation a source has from the detector edge to be considered as a reference star.",type=float)
	ap.add_argument("-centroid",required=False,default=False,help="Whether or not to centroid during aperture photometry.",type=str)
	ap.add_argument("-centroid_type",required=False,default='centroid_2dg',help="Photutils centroid function. Can be 'centroid_1dg', 'centroid_2dg', 'centroid_com', or 'centroid_quadratic'.",type=str)
	args = ap.parse_args(raw_args)

	#Access observation info
	date = args.date
	target = args.target
	ffname = args.ffname
	ap_rad_lo = args.ap_rad_lo 
	ap_rad_hi = args.ap_rad_hi 
	ap_radii = np.arange(ap_rad_lo, ap_rad_hi + 1)
	an_in = args.an_in 
	an_out = args.an_out 
	edge_limit = args.edge_limit
	centroid = t_or_f(args.centroid)
	centroid_type = args.centroid_type

	# set up the directories for storing photometry data 
	make_data_dirs(date, target, ffname)

	# set up logger
	# anything at level DEBUG will be saved to the log file, while anything INFO or higher will be logged to the file and also printed to the console
	log_path = Path('/data/tierras/photometry/'+date+'/'+target+'/'+ffname+f'/{date}_{target}_ap_phot.log')
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.DEBUG)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
	ch.setFormatter(formatter)
	logger.addHandler(ch)	
	fh = logging.FileHandler(log_path, mode='w')
	fh.setFormatter(formatter)
	logger.addHandler(fh)
	set_tierras_permissions(log_path)

	logger.info(f'Running field {target} on {date}')

	# log the input parameters 
	input_params = vars(ap.parse_args(raw_args))
	if not centroid:
		input_params['centroid_type'] = 'n/a'
	for arg in input_params:
		logger.debug(f'{arg}: {input_params[arg]}')

	#TODO: Remove any bad images from the analysis that were missed by sort_and_red
	#exclude_files(date, target, ffname)

	#Get paths to the reduced data for this night/target/ffname
	flattened_files = get_flattened_files(date, target, ffname)

	# identify sources in the field 
	sources = source_selection(flattened_files, logger, edge_limit=edge_limit)

	circular_aperture_photometry(flattened_files, sources, ap_radii, logger, an_in=an_in, an_out=an_out, centroid=centroid, centroid_type=centroid_type, interpolate_cosmics=True)

	#Determine the optimal aperture size
	# optimal_lc_path = lc_post_processing(date, target, ffname, overwrite=True)
	# optimal_lc_path = optimal_lc_chooser(date, target, ffname, plot=False,start_time=0, stop_time=0)
	# print(f'Optimal light curve: {optimal_lc_path}')
	
	#Use the optimal aperture to plot the target light curve
	# plot_target_summary(optimal_lc_path)
	# plot_target_lightcurve(optimal_lc_path)
	# plot_ref_lightcurves(optimal_lc_path)
	# plot_raw_fluxes(optimal_lc_path)

if __name__ == '__main__':
	main()
	

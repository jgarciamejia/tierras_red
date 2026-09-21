#!/usr/bin/env python

import logging
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.time import Time
from astropy.table import Table,  join
import astropy.units as u 
from astropy import log as astropy_log
astropy_log.setLevel('ERROR') # ignore esa status messages
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'
from photutils.aperture import CircularAperture, aperture_photometry
from scipy.stats import sigmaclip
import os
import stat
from pathlib import Path
import copy
import shutil
import warnings
from astroquery.simbad import Simbad
from astropy_healpix import HEALPix  
from scipy.interpolate import RectBivariateSpline
from glob import glob
from fitsutil import *

# Suppress all Astropy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="astropy")

def get_median_field_pointing(target):
	file_paths = sorted(glob(f'/data/tierras/flattened/*/{target}/flat*/*_red.fit'))[::-1]
	dates = np.unique(sorted([i.split('/')[4] for i in file_paths]))
	
	# user can specify dates to ignore for this calculation in /data/tierras/fields/TARGET/ignore_dates.txt
	if os.path.exists(f'/data/tierras/fields/{target}/ignore_dates.txt'):
		with open(f'/data/tierras/fields/{target}/ignore_dates.txt') as f:
			ignore_dates = f.readlines()
		ignore_dates = [i.strip() for i in ignore_dates]
		file_paths = np.array(file_paths)
		delete_inds = []
		for i in range(len(file_paths)):
			file_path = file_paths[i]
			file_date = file_path.split('/')[4]
			if file_date in ignore_dates:
				delete_inds.append(i)

		file_paths = np.delete(file_paths, delete_inds)

	ras, decs = [], []
	im_shape = (2048, 4096)
	median_ra = 0
	median_dec = 0 
	pscale = 0.432
	pscale_deg = pscale/3600
	for i in range(len(file_paths)):
		with fits.open(file_paths[i]) as hdul:
			header = hdul[0].header
			# ignore files with AGOFFX/Y = 0; these correspond to images where the acquire sequence failed
			if header['AGOFFX'] != 0 and header['AGOFFY'] != 0:
				wcs = WCS(header)
				sc = wcs.pixel_to_world(im_shape[1]/2-1, im_shape[0]/2-1)
				ras.append(sc.ra.value)
				decs.append(sc.dec.value)
				median_ra_loop = np.median(ras)
				median_dec_loop = np.median(decs)
				# allow the calculation to terminate early if the median ra and dec have converged to within a tenth of a pixel from their values the previous loop AND we've looked at at least 100 files
				if abs(median_ra_loop - median_ra) < pscale_deg/10 and abs(median_dec_loop - median_dec) < pscale_deg/10 and i >= 100:
					median_ra = median_ra_loop
					median_dec = median_dec_loop
					break
				median_ra = median_ra_loop
				median_dec = median_dec_loop
	return median_ra, median_dec

def source_selection(file_list, logger=None, ra=None, dec=None, edge_limit=20, plot=False, plate_scale=0.432, overwrite=False, rp_mag_limit=17, is_thwomp=False, targ_distance_cut=150, thwomp_contamination_limit=1.01):
	'''
		PURPOSE: identify sources in a Tierras field over a night
		INPUTS: 
			file_list (array): list of paths to Tierras images. If no existing stacked image exists in the /data/tierras/fields/{field} directory, one will be made using images from this list

			target_position (tuple, optional): the user-specified target pixel position. If (0,0), the code will use the RA/Dec coordinates of the target in the header of the stacked field image and the associated WCS to estimate its position. 

			plot (bool, optional): whether or not to produce/save plots of the selected reference stars in the field and a color-magnitude diagram

			overwrite (bool, optional): whether or not to restore previously saved output from the /data/tierras/fields/{field}/ directory

			dimness_limit (float, optional): the minimum mean flux ratio in Gaia Rp-band that a candidate reference can have to the target and still be retained as a reference

			nearness_limit (float, optional): the minimum distance (in pixels) that a candidate reference star must be from all other sources to still be retained as a reference

			edge_limit (float, optional): the minimum distance (in pixels) that a candidate reference star can be from the edge of the detector and still be retained as a reference

			targ_distance_limit (float, optional): the maximum distance (in pixels) that a candidate reference star can be from the target and still be retained as a reference
		
		OUTPUTS:
			output_df (pandas DataFrame): a data frame containing the target and reference stars, with the target as the first entry

	'''
	
	if len(file_list) == 0:
		return None

	# if file_list is list of strings, convert to pathlib objects
	if type(file_list[0]) is str:
		file_list = [Path(i) for i in file_list]

	date = file_list[0].parent.parent.parent.name 
	target = file_list[0].parent.parent.name 
	ffname = file_list[0].parent.name 
	source_path = f'/data/tierras/photometry/{date}/{target}/{ffname}/{date}_{target}_sources.csv'

	if os.path.exists(source_path) and not overwrite:
		if logger is not None:
			logger.info(f'Restoring existing sources from {source_path}.')
		source_df = pd.read_csv(source_path)
		return source_df

	if ra is None and dec is None:
		# If no field ra/dec were passed, use the wcs to evaluate the coordinates of the central pixel in images over the night to determine average pointing
		central_ras = []
		central_decs = []
		ag_files = []
		bad_ag_files = 0
		for ii in range(len(file_list)):
			with fits.open(file_list[ii]) as hdul:
				header = hdul[0].header
				wcs = WCS(header)
			# EXCLUDE any images that have AGOFFX = AGOFFY = 0. This indicates that acquisition failed and we don't want these to bias the average pointing calculation.
			if header['AGOFFX'] == 0 and header['AGOFFY'] == 0:
				bad_ag_files += 1
				continue
			im_shape = hdul[0].shape
			sc = wcs.pixel_to_world(im_shape[1]/2-1, im_shape[0]/2-1)
			central_ras.append(sc.ra.value)
			central_decs.append(sc.dec.value)
			ag_files.append(file_list[ii])

		# do a sigma clipping and take the median of the ra/dec lists to represent the average field center over the night 	
		v1, l1, h1 = sigmaclip(central_ras, 1, 1)
		avg_central_ra = np.median(v1)
		v2, l2, h2 = sigmaclip(central_decs, 1, 1)
		avg_central_dec = np.median(v2)

	else: 
		avg_central_ra = ra 
		avg_central_dec = dec 
		ag_files = []
		central_ras = []
		central_decs = []
		bad_ag_files = 0 
		for ii in range(len(file_list)):
			with fits.open(file_list[ii]) as hdul:
				header = hdul[0].header
			# EXCLUDE any images that have AGOFFX = AGOFFY = 0. This indicates that acquisition failed and we don't want these to bias the average pointing calculation.

			wcs = WCS(header)
			im_shape = hdul[0].shape
			sc = wcs.pixel_to_world(im_shape[1]/2-1, im_shape[0]/2-1)
			central_ras.append(sc.ra.value)
			central_decs.append(sc.dec.value)
			ag_files.append(file_list[ii])

	# some nights are full of only bad guiding images; skip them by returning None here
	if len(file_list) == bad_ag_files:
		if logger is not None:
			logging.info('No exposures with successful acquisition! Returning.')
		return None
	
	# identify the image closest to the average position; if it's off by more than 100 pix from the average pointing, skip
	im_distances = np.sqrt((avg_central_ra-np.array(central_ras))**2 + (avg_central_dec-np.array(central_decs))**2)
	if min(im_distances*60*60/plate_scale) > 100:
		if logger is not None:
			logger.info(f'Image closest to field center is off by more than 100 pixels, returning.')
		return None

	if logger is not None:
		logger.debug(f'Average central RA/Dec: {avg_central_ra:.6f}, {avg_central_dec:.6f}')	

	central_im_file = ag_files[np.argmin(im_distances)]

	with fits.open(central_im_file) as hdul:
		central_im = hdul[0].data
		header = hdul[0].header
		wcs = WCS(header)

	if plot:
		fig, ax = plot_image(central_im)

	# get the epoch of these observations 
	tierras_epoch = Time(header['TELDATE'],format='decimalyear')

	if logger is not None:
		logger.debug(f'Epoch of Tierras observations: {tierras_epoch.value:.6f}')

	# set up the region on sky that we'll query in Gaia
	# to be safe, set the width/height to be a bit larger than the estimates from plate scale alone, and cut to sources that actually fall on the chip after the query is complete
	#	after the query is complete

	coord = SkyCoord(avg_central_ra*u.deg, avg_central_dec*u.deg)
	width = u.Quantity(plate_scale*im_shape[0],u.arcsec)/np.cos(np.radians(avg_central_dec))
	height = u.Quantity(plate_scale*im_shape[1],u.arcsec)
	if logger is not None:	
		logger.debug(f'Using a Gaia RP mag limit of {rp_mag_limit:.1f}.')

	# query gaia for sources
	try: # try doing this locally first
		res = query_gaia_source_local(coord, wcs, im_shape, rp_mag_limit, logger=logger)
	except: # if that fails, try querying the gaia archive
		if logger is not None:
			logger.debug(f'Local Gaia query failed! Trying web query.')
		res = query_gaia_source(coord, width, height, rp_mag_limit)

	# Do a separate search for objects in the Bailer-Jones 'photogeo' catalog
	try:
		res2 = query_bailer_jones_local(wcs, im_shape)
	except:
		res2 = query_bailer_jones(coord, width, height, rp_mag_limit)
	
	# add the Bailer-Jones data into the main table 
	for key in res2.keys()[1:]:
		res[key] = np.zeros(len(res))

	# join the gaia source and bailer jones tables
	for i in range(len(res)):
		if res['source_id'][i] in res2['source_id']:
			ind = np.where(res['source_id'][i] == res2['source_id'])[0][0]
			for key in res2.keys()[1:]:
				res[key][i] = res2[key][ind]
		else:
			for key in res2.keys()[1:]:
				res[key][i] = np.nan

	# add absolute magnitude calculations 
	res['gq_geo'] = res['phot_g_mean_mag'] - 5*np.log10(res['r_med_geo']) + 5
	res['gq_photogeo'] = res['phot_g_mean_mag'] - 5*np.log10(res['r_med_photogeo']) + 5

	# cut to entries without masked pmra values; otherwise the crossmatch will break
	try:
		problem_inds = np.where(np.isnan(res['pmra']))[0]
	except:
		problem_inds = np.where(res['pmra'].mask)[0]

	# set the pmra, pmdec, and parallax of those indices to 0
	res['pmra'][problem_inds] = 0
	res['pmdec'][problem_inds] = 0
	res['parallax'][problem_inds] = 0

	try:
		gaia_coords = SkyCoord(ra=res['ra']*u.deg, dec=res['dec']*u.deg, pm_ra_cosdec=res['pmra']*u.mas/u.yr, pm_dec=res['pmdec']*u.mas/u.yr, obstime=Time(res['ref_epoch'],format='decimalyear'))
	except:
		# TODO: why is this except clause needed sometimes? 
		# and NOTE that the forced ref epoch of '2016.0' is only valid for Gaia DR3 coordinates
		try:
			gaia_coords = SkyCoord(ra=res['ra']*u.deg, dec=res['dec']*u.deg, pm_ra_cosdec=res['pmra'], pm_dec=res['pmdec'], obstime=Time('2016.0',format='decimalyear'))
		except:
			# sometimes it also breaks when you multiply by deg, resulting in deg^2 units??? I have no idea why that would happen.
			gaia_coords = SkyCoord(ra=res['ra'], dec=res['dec'], pm_ra_cosdec=res['pmra'], pm_dec=res['pmdec'], obstime=Time('2016.0',format='decimalyear'))
	
	gaia_coords_tierras_epoch = gaia_coords.apply_space_motion(tierras_epoch)

	#Now set problem indices back to NaNs
	res['pmra'][problem_inds] = np.nan
	res['pmdec'][problem_inds] = np.nan
	res['parallax'][problem_inds] = np.nan
	
	# figure out source positions in the Tierras epoch 
	tierras_pixel_coords = wcs.world_to_pixel(gaia_coords_tierras_epoch)

	# # add 2MASS data and pixel positions to the source table
	
	res.add_column(tierras_pixel_coords[0],name='X pix', index=2)
	res.add_column(tierras_pixel_coords[1],name='Y pix', index=3)
	res.add_column(gaia_coords_tierras_epoch.ra, name='ra_tierras', index=4)
	res.add_column(gaia_coords_tierras_epoch.dec, name='dec_tierras', index=5)

	# check on the target and make sure it has a proper motion from Gaia 
	hdr = fits.open(file_list[-1])[0].header
	targ_x = hdr['CAT-X']
	targ_y = hdr['CAT-Y']
	closest_source = np.nanargmin(np.sqrt((res['X pix']-targ_x)**2 + (res['Y pix']-targ_y)**2))
	if np.isnan(res['pmra'][closest_source]) or np.isnan(res['pmdec'][closest_source]):
		if logger is not None:
			logger.info('WARNING: The closest source to the CAT-X/Y position lacks proper motion measurements in Gaia DR3. Attempting to find them on Simbad.')
		simbad = Simbad()
		simbad.add_votable_fields("mespm", "otype")
		try:
			simbad_res = simbad.query_object(f'Gaia DR3 {res["source_id"][closest_source]}')
			try:
				res['pmra'][closest_source] = simbad_res['PM_pmra'][0]
				res['pmdec'][closest_source] = simbad_res['PM_pmde'][0]
			except:
				# sometimes the table gets returned with different keywords for proper motions...
				res['pmra'][closest_source] = simbad_res['mespm.pmra'][0]
				res['pmdec'][closest_source] = simbad_res['mespm.pmde'][0]

			gaia_coord = SkyCoord(ra=res['ra'][closest_source]*u.deg, dec=res['dec'][closest_source]*u.deg, pm_ra_cosdec=res['pmra'][closest_source]*u.mas/u.yr, pm_dec=res['pmdec'][closest_source]*u.mas/u.yr, obstime=Time('2016',format='decimalyear'))
			
			gaia_coord_tierras_epoch = gaia_coord.apply_space_motion(tierras_epoch)
			tierras_pixel_coord = wcs.world_to_pixel(gaia_coord_tierras_epoch)
			res['X pix'][closest_source] = tierras_pixel_coord[0]
			res['Y pix'][closest_source] = tierras_pixel_coord[1]
			res['ra_tierras'][closest_source] = gaia_coord_tierras_epoch.ra.value
			res['dec_tierras'][closest_source] = gaia_coord_tierras_epoch.dec.value
		except:
			if logger is not None:
				logger.info('ERROR: Simbad query failed. Expected source coordinates in Tierras data may be innacurate.')

	# determine which chip the sources fall on 
	# 0 = bottom, 1 = top 
	chip_inds = np.zeros(len(res),dtype='int')
	chip_inds[np.where(res['Y pix'] >= 1023)] = 1
	res.add_column(chip_inds, name='Chip')

	#Cut to sources that actually fall in the image
	use_inds = np.where((tierras_pixel_coords[0]>0)&(tierras_pixel_coords[0]<im_shape[1]-1)&(tierras_pixel_coords[1]>0)&(tierras_pixel_coords[1]<im_shape[0]-1))[0]
	res = res[use_inds]
	res_full = copy.deepcopy(res)	

	if logger is not None:
		logger.debug(f'Found {len(res)} sources in Gaia query.')

	# for THWOMP (defocused) fields, model the defocused psfs and cut any with high contamination
	if is_thwomp:
		targ_ind = np.nanargmin(np.sqrt((res['X pix']-targ_x)**2 + (res['Y pix']-targ_y)**2))
	
		# read in and normalize pre-generated ePSF for defocused THWOMP images

		epsf = epsf_interp(load_epsf_fits(f'/data/tierras/psfs/defocused_psf.fits'))

		exptime = header['EXPTIME'] # is this general?
		contaminated_inds = []
		
		res['contamination'] = np.zeros(len(res))

		sim_img_shape = (200,200)


		for i in range(len(res)):

			if logger is not None:
				logger.debug(f'Estimating contamination for source {i+1} of {len(res)}')

			source_x = res['X pix'][i]
			source_y = res['Y pix'][i]
			source_rp = res['phot_rp_mean_mag'][i] 

			source_dists = np.sqrt((res['X pix']-source_x)**2 + (res['Y pix']-source_y)**2)
			
			# remove any sources too close to the target
			near_inds = np.where((source_dists <= targ_distance_cut) & (np.arange(len(res)) != targ_ind))[0]

			if len(near_inds) > 0:
				nearby_rp = np.array(res['phot_rp_mean_mag'][near_inds])
				nearby_x = np.array(res['X pix'][near_inds] - source_x) + sim_img_shape[1]/2
				nearby_y = np.array(res['Y pix'][near_inds] - source_y) + sim_img_shape[0]/2

				# sometimes the rp mag is nan, remove these entries
				use_inds = np.where(~np.isnan(nearby_rp))[0]
				nearby_rp = nearby_rp[use_inds]
				nearby_x = nearby_x[use_inds]
				nearby_y = nearby_y[use_inds]

				# enforce that a nearby source cannot have the same rp magnitude as the source in question, that's almost certainly a duplicate
				use_inds = np.where(nearby_rp != source_rp)
				nearby_rp = nearby_rp[use_inds]
				nearby_x = nearby_x[use_inds]
				nearby_y = nearby_y[use_inds]


				# add this source to a simulated image and place a circular aperture to measure its expected flux without any contamination 

				sim_img = generate_defocused_psf(sim_img_shape[1]/2, sim_img_shape[0]/2, source_rp, sim_img_shape, epsf, exptime=exptime)
				ap = CircularAperture((sim_img_shape[1]/2, sim_img_shape[0]/2), r=60)
				source_flux = aperture_photometry(sim_img, ap)['aperture_sum'][0]

				# now add in nearby sources
				for jj in range(len(nearby_rp)): 
					sim_img += generate_defocused_psf(nearby_x[jj], nearby_y[jj], nearby_rp[jj], sim_img.shape, epsf, exptime=exptime)
				
				# now measure the source flux again with the contaminating sources added in
				source_flux_contaminated = aperture_photometry(sim_img, ap)['aperture_sum'][0]
				contamination = source_flux_contaminated / source_flux
				res['contamination'][i] = contamination

				if (contamination > thwomp_contamination_limit) and (i != targ_ind): # never remove the target!
					contaminated_inds.append(i)

		res.remove_rows(np.array(contaminated_inds))
		logger.info(f'Removed {len(contaminated_inds)} contaminated THWOMP sources above contamination limit of {thwomp_contamination_limit}.')

		# do an additional cut based on the brightness of sources
		# G_RP = 13 seems reasonable
		faint_inds = np.where(res['phot_rp_mean_mag'] > 13)[0]
		res.remove_rows(np.array(faint_inds))
		logger.info(f'Removed {len(faint_inds)} THWOMP sources fainter than G_RP = 13 mag.')

	#Cut to sources that are away from the edges
	if is_thwomp:
		edge_limit = 70 # defocused psfs need more edge padding 
	use_inds = np.where((res['Y pix'] > edge_limit) & (res['Y pix']<im_shape[0]
	-edge_limit-1) & (res['X pix'] > edge_limit) & (res['X pix'] < im_shape[1]-edge_limit-1))[0]
	
	if logger is not None:
		logger.debug(f'Removed {len(res)-len(use_inds)} sources that are within {edge_limit} pixels of the detector edges.')
	res = res[use_inds]
	
	# remove ref stars that are too close to the bad columns or the divide between the detector halves
	if not is_thwomp:
		bad_inds_col_1 = np.where((res['X pix'] >= 1431) & (res['X pix'] <= 1472) & (res['Y pix'] <= 1032))[0]
	else: # need larger tolerance for defocused thwomp images
		bad_inds_col_1 = np.where((res['X pix'] >= 1380) & (res['X pix'] <= 1520) & (res['Y pix'] <= 1032))[0]

	res.remove_rows(bad_inds_col_1)
	if logger is not None:
		logger.debug(f'Removed {len(bad_inds_col_1)} sources that were too near the bad pixel column in the lower detector half.')

	if not is_thwomp:
		bad_inds_col_2 = np.where((res['X pix'] >= 1700) & (res['X pix'] <= 1860) & (res['Y pix'] >= 1023))[0]
	else:
		bad_inds_col_2 = np.where((res['X pix'] >= 1771) & (res['X pix'] <= 1813) & (res['Y pix'] >= 1023))[0]
	res.remove_rows(bad_inds_col_2)
	if logger is not None:
		logger.debug(f'Removed {len(bad_inds_col_2)} sources that were too near the bad pixel column in the upper detector half.')

	bad_inds_half = np.where((res['Y pix'] >= 1019) & (res['Y pix'] <= 1032))[0]
	res.remove_rows(bad_inds_half)
	if logger is not None:
		logger.debug(f'Removed {len(bad_inds_half)} sources that were too near the divide between the upper and lower detector halves.')

	if logger is not None:
		logger.info(f'Found {len(res)} sources!')

	if plot:
		ax.plot(res['X pix'], res['Y pix'], marker='x', ls='', color='tab:red')

		fig1, ax1 = plt.subplots(1,1,figsize=(6,5))
		ax1.scatter(res['bp_rp'], res['gq_photogeo'], marker='x', color='tab:red')
		ax1.invert_yaxis()
		ax1.set_xlabel('B$_{p}-$R$_p$', fontsize=14)
		ax1.set_ylabel('M$_{G}$', fontsize=14)
		ax1.tick_params(labelsize=12)
		plt.tight_layout()
		print('plot')
		breakpoint()	

	# create the output dataframe consisting of the target as the 0th entry and the reference stars
	# try:
	# 	output_table = copy.deepcopy(res)
	# except:
	# 	breakpoint()

	output_df = res.to_pandas()
	output_df.to_csv(source_path, index=0)
	set_tierras_permissions(source_path)

	if logger is not None:
		logger.debug(f'Saved source csv to {source_path}')
	return output_df

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
	cmap = matplotlib.colormaps[cmap_name]
	im_scale = 2
	
	
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

def set_tierras_permissions(path):
	try:
		os.chmod(path, stat.S_IRUSR|stat.S_IWUSR|stat.S_IXUSR|stat.S_IRGRP|stat.S_IWGRP|stat.S_IXGRP|stat.S_IROTH|stat.S_IXOTH)
		shutil.chown(path, user=None, group='exoplanet')
	except:
		print(f'Could not change permissions on {path}, returning.')
	return 

def query_gaia_source(coord, width, height, rp_mag_limit):
	# query Gaia DR3 for all the sources in the field brighter than the calculated magnitude limit	
	job = Gaia.launch_job_async("""
									SELECT source_id, ra, ra_error, dec, dec_error, ref_epoch, pmra, pmra_error, pmdec, pmdec_error, parallax, parallax_error, parallax_over_error, ruwe, phot_bp_mean_mag, phot_g_mean_mag, phot_rp_mean_mag, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_g_mean_flux, phot_g_mean_flux_error, phot_rp_mean_flux, phot_rp_mean_flux_error, bp_rp, bp_g, g_rp, grvs_mag, grvs_mag_error, phot_variable_flag,radial_velocity, radial_velocity_error, non_single_star, teff_gspphot, logg_gspphot, mh_gspphot, rvs_spec_sig_to_noise
									FROM gaiadr3.gaia_source as gaia
									WHERE gaia.ra BETWEEN {} AND {} AND
											gaia.dec BETWEEN {} AND {} AND
											gaia.phot_rp_mean_mag <= {}
									ORDER BY phot_rp_mean_mag ASC
								""".format(coord.ra.value-width.to(u.deg).value/2, coord.ra.value+width.to(u.deg).value/2, coord.dec.value-height.to(u.deg).value/2, coord.dec.value+height.to(u.deg).value/2, rp_mag_limit)
								)

	res = job.get_results()
	try:
		res['SOURCE_ID'].name = 'source_id' # why does this sometimes get returned in all caps? 
	except:
		pass
	return res 

def query_gaia_source_local(coord, wcs, im_shape, rp_mag_limit, logger=None):
	gaia_path = '/data/tierras/gaia_dr3/gaia_source/'
	hpx_level = 6

	# determine the ra/dec limits over which sources need to be retained
	ra_min = np.min([wcs.pixel_to_world(0,0).ra.value, wcs.pixel_to_world(im_shape[1],0).ra.value])
	ra_max = np.max([wcs.pixel_to_world(0,im_shape[0]).ra.value, wcs.pixel_to_world(im_shape[1],im_shape[0]).ra.value])
	dec_min = np.min([wcs.pixel_to_world(im_shape[1],0).dec.value, wcs.pixel_to_world(im_shape[1],im_shape[0]).dec.value])
	dec_max = np.min([wcs.pixel_to_world(0,0).dec.value, wcs.pixel_to_world(0,im_shape[0]).dec.value])

	height = 3600*(dec_max-dec_min) * u.arcsec

	# figure out which gaia files correspond to the desired sky query 
	md5sum_file     = pd.read_csv(gaia_path+'_MD5SUM.txt', header=None, sep='\s+', names=['md5Sum', 'file'])
	md5sum_file.drop(md5sum_file.tail(1).index,inplace=True) # The last row in the "_MD5SUM.txt" file in the DR3 directories includes the md5Sum value of the _MD5SUM.txt file

	# Extract HEALPix level-8 from file name ======================================
	healpix_8_min  = [int(file[file.find('_')+1:file.rfind('-')])     for file in md5sum_file['file']]
	healpix_8_max  = [int(file[file.rfind('-')+1:file.rfind('.csv')]) for file in md5sum_file['file']]
	reference_file = pd.DataFrame({'file':md5sum_file['file'], 'healpix8_min':healpix_8_min, 'healpix8_max':healpix_8_max}).reset_index(drop=True)

	# Compute HEALPix levels 6,7, and 9 ===========================================
	reference_file['healpix7_min'] = [inp >> 2 for inp in reference_file['healpix8_min']]
	reference_file['healpix7_max'] = [inp >> 2 for inp in reference_file['healpix8_max']]

	reference_file['healpix6_min'] = [inp >> 2 for inp in reference_file['healpix7_min']]
	reference_file['healpix6_max'] = [inp >> 2 for inp in reference_file['healpix7_max']]

	reference_file['healpix9_min'] = [inp << 2       for inp in reference_file['healpix8_min']]
	reference_file['healpix9_max'] = [(inp << 2) + 3 for inp in reference_file['healpix8_max']]

	# Generate reference file =====================================================
	ncols          = ['file', 'healpix6_min', 'healpix6_max', 'healpix7_min', 'healpix7_max', 'healpix8_min', 'healpix8_max', 'healpix9_min', 'healpix9_max']
	reference_file = reference_file[ncols]

	hp             = HEALPix(nside=2**hpx_level, order='nested')
	hp_cone_search = hp.cone_search_lonlat(coord.ra, coord.dec, radius=height.to(u.degree)) # i don't think a rectangular search is implemented

	subset     = []
	for index in reference_file.index:
		row = reference_file.iloc[index]
		hp_min, hp_max = row[f'healpix{hpx_level}_min'], row[f'healpix{hpx_level}_max']
		if np.any(np.logical_and(hp_min <= hp_cone_search, hp_cone_search <= hp_max)):
			bulk_file = row['file'].split('.csv')[0]+'_sub.fits'
			subset.append(bulk_file)
	
	sources = []
	n = 0 	
	for i in range(len(subset)):
		try:
			hdul = fits.open(gaia_path+subset[i])
		except:
			warn = f'WARNING: {subset[i]} not on disk skipping.\n You should download it from https://cdn.gea.esac.esa.int/?prefix=Gaia/gdr3/gaia_source/, extract it, place it in /data/tierras/gaia_dr3/gaia_source/, then run make_subset.py and convert_to_fits.py on it.'
			if logger is not None:
				logger.info(warn)
			else:
				print(warn)
			continue

		tab = Table(hdul[1].data)
		
		if ra_min > ra_max: # this happens when the field RA is near 360 degrees and you get wraparound to 0 for ra_max. 
			ra_max += 360
			tab['ra'][np.where(tab['ra'] < 180)[0]] += 360
		
		source_inds = np.where((tab['ra'] > ra_min) & (tab['ra'] < ra_max) & (tab['dec'] > dec_min) & (tab['dec'] < dec_max) & (tab['phot_rp_mean_mag'] <= rp_mag_limit))[0]

		if len(source_inds) > 0:
			sources.append(tab[source_inds])

			if n == 0:
				try:
					res = sources[n]
					n += 1
				except:
					breakpoint()
			else:
				# if sources were found spanning multiple gaia files, we need to stitch them toghether
				res = join(res, tab[source_inds], join_type='outer')
	
	try:
		res['SOURCE_ID'].name = 'source_id' # why does this sometimes get returned in all caps? 
	except:
		pass
	
	# sort on rp mag
	try:
		res.sort(keys='phot_rp_mean_mag')
	except:
		print('local query failed???')
	return res 

def query_bailer_jones(coord, width, height, rp_mag_limit):
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
								""".format(coord.ra.value-width.to(u.deg).value/2, coord.ra.value+width.to(u.deg).value/2, coord.dec.value-height.to(u.deg).value/2, coord.dec.value+height.to(u.deg).value/2, rp_mag_limit)
								)
	res2 = job.get_results()	
	return res2

def query_bailer_jones_local(wcs, im_shape):
	bailerjones_path = '/data/tierras/gaia_dr3/bailer_jones/'

	ra_min = np.min([wcs.pixel_to_world(0,0).ra.value, wcs.pixel_to_world(im_shape[1],0).ra.value])
	ra_max = np.max([wcs.pixel_to_world(0,im_shape[0]).ra.value, wcs.pixel_to_world(im_shape[1],im_shape[0]).ra.value])
	dec_min = np.min([wcs.pixel_to_world(im_shape[1],0).dec.value, wcs.pixel_to_world(im_shape[1],im_shape[0]).dec.value])
	dec_max = np.min([wcs.pixel_to_world(0,0).dec.value, wcs.pixel_to_world(0,im_shape[0]).dec.value])

	# now do the same thing for the bailer-jones data
	bj_ra_start = np.floor(ra_min*10)/10
	bj_ra_end = np.ceil(ra_max*10)/10
	bj_file_ras = np.arange(bj_ra_start, bj_ra_end, 0.1)
	bj_sources = []
	for i in range(len(bj_file_ras)):
		hdul = fits.open(bailerjones_path+f'gedr3dist_RA_{bj_file_ras[i]:.1f}.fits')
		tab = Table(hdul[1].data, names=['source_id', 'ra', 'dec', 'r_med_geo','r_lo_geo','r_hi_geo','r_med_photogeo','r_lo_photogeo','r_hi_photogeo','flag'])
		source_inds = np.where((tab['ra'] > ra_min) & (tab['ra'] < ra_max) & (tab['dec'] > dec_min) & (tab['dec'] < dec_max))[0]
		if len(source_inds) > 0:
			bj_sources.append(tab[source_inds])

			if i == 0:
				res2 = bj_sources[i]
			else:
				# if sources were found spanning multiple gaia files, we need to stitch them toghether
				res2 = join(res2, tab[source_inds], join_type='outer')
	res2.remove_columns(['ra', 'dec']) # do not want these, use ra/dec from main gaia query
	return res2 

def load_epsf_fits(filepath):
	"""
	Load an EPSFModel from a FITS file saved by save_epsf_fits().

	Returns an EPSFModel instance ready for PSFPhotometry /
	IterativePSFPhotometry, regardless of photutils version.
	"""
	# Version-aware import (photutils API changed across versions)
	try:
		from photutils.psf import EPSFModel             # photutils < 2.0
	except ImportError:
		try:
			from photutils.psf import FittableImageModel as EPSFModel
		except ImportError:
			from photutils.psf import ImagePSF as EPSFModel  # photutils >= 1.9

	# from photutils.psf import ImagePSF
	with fits.open(filepath) as hdul:
		data = hdul[0].data.astype(np.float64)
		hdr  = hdul[0].header

		os_x   = int(hdr.get('OVERSMPX', 1))
		os_y   = int(hdr.get('OVERSMPY', os_x))
		orig_x = float(hdr.get('ORIG_X',  (data.shape[1] - 1) / 2.0))
		orig_y = float(hdr.get('ORIG_Y',  (data.shape[0] - 1) / 2.0))

	oversampling = os_x if (os_x == os_y) else (os_x, os_y)
	origin       = (orig_x, orig_y)

	epsf = EPSFModel(data=data, oversampling=oversampling, origin=origin)
	# print(f"Loaded ePSF  shape={data.shape}  "
	# 	  f"oversampling={oversampling}  ← {filepath}")
	return epsf

def epsf_interp(epsf):
	# first diagnose the normalization of the ePSF
	psf_data = epsf.data.copy()
	os       = float(np.atleast_1d(epsf.oversampling)[0])
	ny_os, nx_os = psf_data.shape

	x_ax = (np.arange(nx_os) - (nx_os - 1) / 2.0) / os
	y_ax = (np.arange(ny_os) - (ny_os - 1) / 2.0) / os
	interp_raw = RectBivariateSpline(y_ax, x_ax, psf_data, kx=3, ky=3)

	# Evaluate on a native pixel grid large enough to capture all flux
	half_eval = int(max(nx_os, ny_os) / (2 * os)) + 10
	yy_e, xx_e = np.mgrid[-half_eval:half_eval+1,
						-half_eval:half_eval+1].astype(float)
	psf_native = interp_raw(yy_e.ravel(), xx_e.ravel(),
							grid=False).reshape(yy_e.shape)
	
	# now renormalize
	norm_factor = psf_native.sum()
	psf_data_norm = psf_data / norm_factor

	interp = RectBivariateSpline(y_ax, x_ax, psf_data_norm, kx=3, ky=3)
	return interp

def generate_defocused_psf(x0, y0, rp_mag, shape, epsf_interp, gain=5.9, exptime=1):
	"Will generate a defocused PSF for a given G_RP mag and exosure time in Tierras images in ADU"
	def flux_model(x, A):
		"""
			fittable model of flux (e-/s) as a function of magnitude
		"""
		return A*10**(-x/2.5)
	
	yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)

	A = 1.21437174e+09 # e-/s, determined from fit of defocused sources in HIP107350 field on 20260621
	flux = flux_model(rp_mag, A)
	
	model = flux * epsf_interp((yy - y0).ravel(),
				(xx - x0).ravel(),
				grid=False).reshape(shape) * exptime / gain # model in units of ADU
	
	return model

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


def t_or_f(arg):
	ua = str(arg).upper()
	if 'TRUE'.startswith(ua):
		return True
	elif 'FALSE'.startswith(ua):
		return False
	else:
		print(f'ERROR: check passed argument for {arg}.')

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
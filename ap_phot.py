import numpy as np 
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import ImageNormalize, ZScaleInterval
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.modeling import models, fitting
from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from photutils import make_source_mask
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from scipy.stats import sigmaclip
from scipy.spatial.distance import cdist
from scipy.signal import correlate2d, fftconvolve
from copy import deepcopy
import argparse
import pdb 
import os 
import sys
import lfa
import time
import astroalign as aa
import reproject as rp
import sep 
from fitsutil import *
from pathlib import Path
import csv 


def get_flattened_files():
	#Get a list of data files sorted by exposure number
	ffolder = fpath+'/'+date+'/'+target+'/'+ffname
	red_files = []
	for file in os.listdir(ffolder): 
		if '_red.fit' in file:
			red_files.append(ffolder+'/'+file)
	sorted_files = np.array(sorted(red_files, key=lambda x: int(x.split('.')[1])))
	return sorted_files 

def plot_image(data,use_wcs=False,scale='zscale',cmap_name='viridis'):
	#Do a quick plot of a Tierras image

	#TODO: Do we want the image orientation to match the orientation on-sky? 
	#TODO: WCS and pixel coordinates simultaneously? 

	#TODO: support for other image scalings.
	if scale == 'zscale':
		interval=ZScaleInterval()
	
	#if use_wcs:
	#	wcs = WCS(header)
	
	norm = ImageNormalize(data[4:2042,:], interval=interval) #Ignore a few rows near the top/bottom for the purpose of getting a good colormap
	cmap = get_cmap(cmap_name)
	im_scale = 2.5
	
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

def orient_on_sky(data,header):
	#Flip a Tierras image so that it matches its on-sky orientation.
	#TODO: how do you also flip the WCS information????
	fig, ax = plot_image(data, header, use_wcs=True)

	data2 = np.flipud(data)
	fig2, ax2 = plot_image(data2, header, use_wcs=True)

	data3 = np.rot90(data2, k=1)
	fig3, ax3 = plot_image(data3, header, use_wcs=True)

	pdb.set_trace()
	return

def detect_sources(data,header,mode='auto',model_bkg=False,plot=False):
	#Detect stars in a Tierras image. 
	#mode can be 'auto' or 'user'
	#TODO: Self-calibrate to set properties of the starfinding algorithm

	if mode == 'user':
		print('User mode not yet implemented!')
		return

	wcs = WCS(header)

	if model_bkg:
		#Do a 2D MedianBackground model and remove it to facilitate source detection.
		print('Modeling background...')
		sigma_clip = SigmaClip(sigma=3.0)
		bkg_estimator = MedianBackground()
		bkg = Background2D(data, (50, 50), filter_size=(9, 9),
					sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
	
		data = deepcopy(data-bkg.background)

	#Get sigma-clipped median and stddev on background-subtracted data
	vals, lo, hi = sigmaclip(data,low=4,high=4)
	med = np.median(vals)
	std = np.std(vals)

	print('Detecting sources...')
	#finder = DAOStarFinder(fwhm=4.0, threshold=5.0*std,ratio=0.5,theta=135,roundhi=0.7,sharphi=0.8)
	finder = IRAFStarFinder(fwhm=4.0, threshold=8.0*std,roundhi=0.7)
	sources = finder(data-med)

	#Deal with any sources in two bad regions on the detector.
	bad_inds_1 = np.where((1449 <= sources['xcentroid']) & (sources['xcentroid'] < 1453))[0]
	bad_inds_2 = np.where((1791 <= sources['xcentroid']) & (sources['xcentroid'] < 1794) & (sources['ycentroid'] >= 1023))[0]
	bad_inds_3 = np.where((sources['xcentroid']>= 4089))[0] #Remove some near x edges
	bad_inds = np.concatenate((bad_inds_1,bad_inds_2,bad_inds_3))
	sources.remove_rows(bad_inds)
	
	#Convert to a pandas dataframe 
	source_df = sources.to_pandas()

	#Resort based on flux 
	source_df.sort_values('flux',ascending=False).reset_index()
	
	positions = np.transpose((source_df['xcentroid'], source_df['ycentroid']))  

	#Identify target from header
	#TODO: Where do CAT-RA and CAT-DEC come from? Do we need to apply proper motion?
	#TODO: What happens if no source is found near the target position?
	catra = header['CAT-RA']
	catde = header['CAT-DEC']
	cateq = float(header['CAT-EQUI'])
	if cateq == 0:
		sys.exit(1)
	ratarg, rv = lfa.base60_to_10(catra, ":", lfa.UNIT_HR, lfa.UNIT_RAD)
	detarg, rv = lfa.base60_to_10(catde, ":", lfa.UNIT_DEG, lfa.UNIT_RAD)

	xtarg, ytarg = wcs.all_world2pix(ratarg * lfa.RAD_TO_DEG, detarg * lfa.RAD_TO_DEG, 1)
	target_id = np.argmin(cdist(np.array((xtarg,ytarg)).reshape(1,-1),positions))

	#Move the target to the first entry in the dataframe
	source_df = pd.concat([source_df.iloc[[target_id],:], source_df.drop(target_id, axis=0)], axis=0).reset_index()

	if plot:
		fig, ax = plot_image(data,header,use_wcs=False)

		#Plot source indicators 
		for i in range(len(sources)):
			if i == target_id:
				indicator_color = 'b'
			else:
				indicator_color = 'r'
			circ_rad = 14
			circle = plt.Circle(positions[i], 14, color=indicator_color, fill=False, lw=1.5, alpha=0.5)
			ax.add_patch(circle)
			ax.plot([sources['xcentroid'][i]-circ_rad,sources['xcentroid'][i]+circ_rad],[sources['ycentroid'][i],sources['ycentroid'][i]],color=indicator_color,lw=1.5,alpha=0.5)
			ax.plot([sources['xcentroid'][i],sources['xcentroid'][i]],[sources['ycentroid'][i]-circ_rad,sources['ycentroid'][i]+circ_rad],color=indicator_color,lw=1.5,alpha=0.5)

	return source_df

def reference_star_chooser(file_list, nonlinear_limit=40000, dimness_limit=0.05, nrefs=1000):
	
	#Start by checking for existing csv file about target/reference positions
	reference_file_path = Path('/data/tierras/targets/'+target+'/'+target+'_target_and_ref_stars.csv')
	if not reference_file_path.exists():
		print('No saved target/reference star positions found!\n')
		if not reference_file_path.parent.exists():
			os.mkdir(reference_file_path.parent)
		
		stacked_image_path = reference_file_path.parent/(target+'_stacked_image.fits')
		if not stacked_image_path.exists():
			print('No stacked field image found!')
			stacked_hdu = align_and_stack_images(file_list)
			stacked_hdu.writeto(Path('/data/tierras/targets/'+target+'/'+target+'_stacked_image.fits'), overwrite=True)
			print(f"Saved stacked field to {'/data/tierras/targets/'+target+'/'+target+'_stacked_image.fits'}")
		else:
			print(f'Restoring stacked field image from {stacked_image_path}.')
		
		stacked_image_hdu = fits.open(stacked_image_path)[0]
		stacked_image = stacked_image_hdu.data
		stacked_image_header = stacked_image_hdu.header
		#Get the number of images that were stacked in the stacked image
		n_stacked_images = int(stacked_image_header['COMMENT'][-1].split('image ')[1].split(':')[0])+1
		wcs = WCS(stacked_image_header)

		#Do SEP detection on stacked image 
		bpm = load_bad_pixel_mask()
		#Add some extra masking to deal with image stacking effects. 
		
		#Extra masking on bad columns
		bpm[0:1032, 1447:1464] = True
		bpm[1023:, 1788:1801]  = True

		#25-pixel mask on all edges
		bpm[:, 0:25+1] = True
		bpm[:,4096-1-25:] = True
		bpm[0:25+1,:] = True
		bpm[2048-1-25:,:] = True
		# plot_image(bpm)
		# breakpoint()

		#Do SEP source detection on background-subtracted stacked image.
		thresh = 2.0
		minpix = 4
		try:
			bkg_stack = sep.Background(stacked_image)
		except:
			bkg_stack = sep.Background(stacked_image.byteswap().newbyteorder())

		objs_stack = sep.extract(stacked_image-bkg_stack, thresh, err=bkg_stack.globalrms, mask=bpm, minarea=minpix)
		
		#Write all object detections out to csv file
		df = pd.DataFrame(objs_stack)
		output_path = reference_file_path.parent/(target+'_stacked_source_detections.csv')
		df.to_csv(output_path, index=False)


		#Figure out where the target is 
		catra = stacked_image_header['CAT-RA']
		catde = stacked_image_header['CAT-DEC']
		cateq = float(stacked_image_header['CAT-EQUI'])
		if cateq == 0:
			raise RuntimeError('Target position is not set!')

		ratarg, rv = lfa.base60_to_10(catra, ":", lfa.UNIT_HR, lfa.UNIT_RAD)
		detarg, rv = lfa.base60_to_10(catde, ":", lfa.UNIT_DEG, lfa.UNIT_RAD)

		xtarg, ytarg = wcs.all_world2pix(ratarg * lfa.RAD_TO_DEG,detarg * lfa.RAD_TO_DEG, 1)
		
		itarg = np.argmin(np.hypot(objs_stack["x"]+1-xtarg, objs_stack["y"]+1-ytarg))
		targ = objs_stack[[itarg]]

		#Identify suitable reference stars
		#TODO: 40000*number of stacked images is an estimate, can we be more exact?
		possible_ref_inds = np.where((objs_stack['peak']<nonlinear_limit*n_stacked_images)&(objs_stack['flux']>targ['flux']*dimness_limit))[0]
		possible_ref_inds = np.delete(possible_ref_inds, np.where(possible_ref_inds == itarg)[0][0])
		
		#Remove refs that are too close to other sources (dist < 12 pix)
		refs_to_remove = []
		for i in range(len(possible_ref_inds)):
			dists = np.hypot(objs_stack['x']-objs_stack[possible_ref_inds[i]]['x'], objs_stack['y']-objs_stack[possible_ref_inds[i]]['y'])
			dists = np.delete(dists,possible_ref_inds[i]) #Remove the source itself from the distance calculation
			close_sources = np.where(dists < 12)[0]
			if len(close_sources) > 0:
				refs_to_remove.append(possible_ref_inds[i])
		for i in range(len(refs_to_remove)):
			possible_ref_inds = np.delete(possible_ref_inds, np.where(possible_ref_inds == refs_to_remove[i])[0][0])
		possible_refs = objs_stack[possible_ref_inds]
		
		#Select up to nrefs of the remaining reference stars sorted by flux
		bydecflux = np.argsort(-possible_refs["flux"])
		if len(bydecflux) > nrefs:
			refs = possible_refs[bydecflux[0:nrefs]]
		else:
			refs = possible_refs[bydecflux]
    
		print("Selected {0:d} reference stars".format(len(refs)))

		targ_and_refs = np.concatenate((targ, refs))
		df = pd.DataFrame(targ_and_refs)
		df.to_csv(reference_file_path, index=False)
	else:
		print(f'Restoring target/reference star positions from {reference_file_path}')
		df = pd.read_csv(reference_file_path)
	return df

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
	amplist = np.array(amplist, dtype=np.int)
	sectlist = np.array(sectlist, dtype=np.int)
	vallist = np.array(vallist, dtype=np.int)

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

def align_and_stack_images(file_list, ref_image_num=0, n_ims_to_stack=20):
	#TODO: by default, will treat first image in the file list as the reference image, and stack the next 20 images to get a high snr image of the field.
	#	Not sure how to choose which should be the reference exposure programatically.
	#	Also not sure how many images we want to stack. 
	target = file_list[0].split('.')[2].split('_')[0]

	bpm = load_bad_pixel_mask()

	reference_hdu = fits.open(file_list[ref_image_num])[0] #TODO: how to choose programmatically?
	reference_image = reference_hdu.data
	reference_header = reference_hdu.header
	reference_header.append(('COMMENT',f'Reference image: {Path(file_list[ref_image_num]).name}'), end=True)

	bkg = sep.Background(reference_image.byteswap().newbyteorder()) #TODO: why are the byteswap().newbyteorder() commands necessary?
	stacked_image_aa = np.zeros(reference_image.shape, dtype='float32')
	stacked_image_rp = np.zeros(reference_image.shape, dtype='float32')
	stacked_image_aa += reference_image - bkg.back()
	stacked_image_rp += reference_image - bkg.back()

	#Do a loop over n_ims, aligning and stacking them. 
	all_inds = np.arange(len(file_list))
	inds_excluding_reference = np.delete(all_inds, ref_image_num) #Remove the reference image from the index list
	inds_to_stack = inds_excluding_reference[:n_ims_to_stack]
	print('Aligning and stacking images...')
	counter = 0 
	for i in inds_to_stack:
		source_hdu = fits.open(file_list[i])[0]
		source_image = source_hdu.data 
		
		#METHOD 1: using astroalign
		#Astroalign does image *REGISTRATION*, i.e., does not rely on header WCS.
		masked_source_image = np.ma.array(source_image,mask=bpm) #aa requires the use of numpy masked arrays to do bad pixel masking
		registered_image, footprint = aa.register(masked_source_image,reference_image)
		bkg_aa = sep.Background(registered_image)
		stacked_image_aa += registered_image-bkg_aa.back()
		
		# #METHOD 2: using reproject
		# #reproject uses WCS information in the fits headers to align images.
		# #It is much slower than aa and gives comparable results.
		# reprojected_image, footprint = rp.reproject_interp(source_hdu, reference_hdu.header)
		# print(f'rp time: {time.time()-t1:.1f} s')
		# bkg_rp = sep.Background(reprojected_image)
		# stacked_image_rp += reprojected_image - bkg_rp.back()

		print(f'{counter+1} of {n_ims_to_stack}.')
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

def fixed_circular_aperture_photometry(file_list, targ_and_refs, ap_radii, an_in=40,an_out=60):
	#file_list = file_list[0:10]
	
	DARK_CURRENT = 0.19 #e- pix^-1 s^-1
	NONLINEAR_THRESHOLD = 40000. #ADU
	SATURATION_TRESHOLD = 55000. #ADU
	
	#Set up arrays for doing photometry 

	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH FILE
	filenames = []
	mjd_utc = np.zeros(len(file_list),dtype='float')
	jd_utc = np.zeros(len(file_list),dtype='float')
	bjd_tdb = np.zeros(len(file_list),dtype='float')
	airmasses = np.zeros(len(file_list),dtype='float')
	ccd_temps = np.zeros(len(file_list),dtype='float')
	exp_times = np.zeros(len(file_list),dtype='float')
	dome_temps = np.zeros(len(file_list),dtype='float')
	focuses = np.zeros(len(file_list),dtype='float')
	dome_humidities = np.zeros(len(file_list),dtype='float')
	sec_temps = np.zeros(len(file_list),dtype='float')
	ret_temps = np.zeros(len(file_list),dtype='float')
	pri_temps = np.zeros(len(file_list),dtype='float')
	rod_temps = np.zeros(len(file_list),dtype='float')
	cab_temps = np.zeros(len(file_list),dtype='float')
	inst_temps = np.zeros(len(file_list),dtype='float')
	temps = np.zeros(len(file_list),dtype='float')
	humidities = np.zeros(len(file_list),dtype='float')
	dewpoints = np.zeros(len(file_list),dtype='float')
	sky_temps = np.zeros(len(file_list),dtype='float')
	
	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH SOURCE IN EACH FILE
	source_x = np.zeros((len(targ_and_refs),len(file_list)),dtype='float')
	source_y = np.zeros((len(targ_and_refs),len(file_list)),dtype='float')
	source_sky_ADU = np.zeros((len(targ_and_refs),len(file_list)),dtype='float')
	source_sky_e = np.zeros((len(targ_and_refs),len(file_list)),dtype='float')


	#ARRAYS THAT CONTAIN DATA PERTAININING TO EACH APERTURE RADIUS FOR EACH SOURCE FOR EACH FILE
	source_minus_sky_ADU = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float')
	source_minus_sky_e = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float')
	source_minus_sky_err_ADU = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float')
	source_minus_sky_err_e = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float')
	non_linear_flags = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='int')
	saturated_flags = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='int')
	
	total_ref_ADU = np.zeros((len(ap_radii),len(file_list)),dtype='float')
	total_ref_err_ADU = np.zeros((len(ap_radii),len(file_list)),dtype='float')
	total_ref_e = np.zeros((len(ap_radii),len(file_list)),dtype='float')
	total_ref_err_e = np.zeros((len(ap_radii),len(file_list)),dtype='float')


	source_radii = np.zeros((len(ap_radii),len(file_list)),dtype='float')
	an_in_radii = np.zeros((len(ap_radii),len(file_list)),dtype='float')
	an_out_radii = np.zeros((len(ap_radii),len(file_list)),dtype='float')

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
	plt.figure()
	print(f'Doing fixed-radius circular aperture photometry on {n_files} images with aperture radii of {ap_radii} pixels, an inner annulus radius of {an_in} pixels, and an outer annulus radius of {an_out} pixels.\n')
	time.sleep(2)
	for i in range(n_files):
		# #TESTING
		#i = 1349
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
		filenames.append(file_list[i].split('/')[-1])
		mjd_utc[i] = source_header['MJD-OBS'] + (EXPTIME/2)/(24*60*60) #MJD-OBS is the modified julian date at the start of the exposure. Add on half the exposure time in days to get the time at mid-exposure. 
		jd_utc[i] = mjd_utc[i]+2400000.5 #Convert MJD_UTC to JD_UTC
		bjd_tdb[i] = jd_utc_to_bjd_tdb(jd_utc[i], RA, DEC)
		airmasses[i] = source_header['AIRMASS']
		ccd_temps[i] = source_header['CCDTEMP']
		exp_times[i] = source_header['EXPTIME']
		dome_temps[i] = source_header['DOMETEMP']
		focuses[i] = source_header['FOCUS']
		dome_humidities[i] = source_header['DOMEHUMI']
		#SECTEMP keyword is sometimes missing
		try:
			sec_temps[i] = source_header['SECTEMP']
		except:
			sec_temps[i] = np.nan
		ret_temps[i] = source_header['RETTEMP']
		pri_temps[i] = source_header['PRITEMP']
		rod_temps[i] = source_header['RODTEMP']
		#CABTEMP keyword is sometimes missing
		try:
			cab_temps[i] = source_header['CABTEMP']
		except:
			cab_temps[i] = np.nan
		#INSTTEMP keyword is sometimes missing
		try:
			inst_temps[i] = source_header['INSTTEMP']
		except:
			inst_temps[i] = np.nan
		temps[i] = source_header['TEMPERAT']
		humidities[i] = source_header['HUMIDITY']
		dewpoints[i] = source_header['DEWPOINT']
		sky_temps[i] = source_header['SKYTEMP']


		#UPDATE SOURCE POSITIONS
		#METHOD 1: WCS
		source_wcs = WCS(source_header)
		try:
			transformed_pixel_coordinates = [source_wcs.world_to_pixel(reference_world_coordinates[i]) for i in range(len(reference_world_coordinates))]
			
		except:
			#METHOD 2: astroalign 
			#Seems to give iffy results when it doesn't find matching sources spread across the whole chip, but could be used as a backup method.
			#TODO: When should we fall back on this? Just put it in a try/except clause for now
			source_data[np.where(bpm == 1)] = np.nan #Mask out pixels in the BPM
			#Background-subtract the data for figuring out shifts
			try:
				bkg = sep.Background(source_data)
			except:
				bkg = sep.Background(source_data.byteswap().newbyteorder())
			bkg_subtracted_source_data -= bkg.back()
			#Find the transform between the source and reference images and use that to update the target/reference star positions
			transform, (source_list, target_list) = aa.find_transform(reference_image_data,bkg_subtracted_source_data,detection_sigma=3)
			ref_positions = np.array([(targ_and_refs['x'][i], targ_and_refs['y'][i]) for i in range(len(targ_and_refs))])
			transformed_pixel_coordinates = aa.matrix_transform(ref_positions, transform.params)
		
		#Save transformed pixel coordinates of sources
		for j in range(len(targ_and_refs)):
			source_x[j,i] = transformed_pixel_coordinates[j][0]
			source_y[j,i] = transformed_pixel_coordinates[j][1]

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

			x_pos_cutout = x_pos_image-int(x_pos_image)+an_out
			y_pos_cutout = y_pos_image-int(y_pos_image)+an_out

			# norm = ImageNormalize(cutout,interval=ZScaleInterval())
			# plt.imshow(cutout,origin='lower',norm=norm,interpolation='none')
			# plt.plot(x_pos_cutout,y_pos_cutout,'kx',mew=2,ms=8)
			
			# #Verify that the cutout position matches the image position
			# plt.figure()
			# plt.imshow(source_data, origin='lower', norm=norm, interpolation='none')
			# plt.ylim(int(y_pos_image-an_out),int(y_pos_image+an_out))
			# plt.xlim(int(x_pos_image-an_out),int(x_pos_image+an_out))
			# plt.plot(x_pos_image,y_pos_image,'rx',mew=2,ms=8)
			# ap = CircularAperture((x_pos_image,y_pos_image),r=ap_radii[0])
			# ap.plot(color='r',lw=2.5)

			for k in range(len(ap_radii)):
				ap = CircularAperture((x_pos_cutout,y_pos_cutout),r=ap_radii[k])
				an = CircularAnnulus((x_pos_cutout,y_pos_cutout),r_in=an_in,r_out=an_out)

				source_radii[k,i] = ap_radii[k]
				an_in_radii[k,i] = an_in
				an_out_radii[k,i] = an_out

				# ap.plot(color='r',lw=2.5)
				# an.plot(color='r',lw=2.5)

				#DO PHOTOMETRY
				#t1 = time.time()
				phot_table = aperture_photometry(cutout, ap)

				#Check for non-linear/saturated pixels in the aperture
				max_pix = np.max(ap.to_mask().multiply(cutout))
				if max_pix >= SATURATION_TRESHOLD:
					saturated_flags[k,j,i] = 1
				if max_pix >= NONLINEAR_THRESHOLD:
					non_linear_flags[k,j,i] = 1

				#Estimate background 
				#TODO: Mask sources? Current implementation is breaking because the size of the source mask does not match the size of the input image (which makes no sense)
				annulus_mask = an.to_mask(method='center').to_image(np.shape(cutout))
				an_data = annulus_mask * cutout
				
				source_mask = make_source_mask(cutout, nsigma=3, npixels=5, dilate_size=4)

				# #Plot the source mask
				# fig, ax = plt.subplots(1,2,figsize=(10,7))
				# ax[0].imshow(an_data, origin='lower',norm=ImageNormalize(cutout,interval=ZScaleInterval()),interpolation='none')
				# ax[1].imshow(source_mask, origin='lower',norm=ImageNormalize(cutout,interval=ZScaleInterval()),interpolation='none')
				# breakpoint()

				an_data *= ~source_mask 

				an_data_1d = an_data[an_data != 0] #unwrap into 1d array

				an_vals, hi, lo = sigmaclip(an_data_1d) #toss outliers
				bkg = np.median(an_vals) #take median of remaining values as per-pixel background estimate
				
				source_sky_ADU[j,i] = bkg
				source_sky_e[j,i] = bkg*GAIN

				# if j == 19:
				# 	plt.figure()
				# 	plt.imshow(an_data, origin='lower',norm=ImageNormalize(an_data,interval=ZScaleInterval()),interpolation='none')
				# 	breakpoint()

				#Plot histogram of sigma-clipped, source-masked annulus values. 
				# plt.figure()
				# plt.hist(an_vals,bins=25)
				# plt.axvline(bkg, color='tab:orange')

				source_minus_sky_ADU[k,j,i] = phot_table['aperture_sum'][0]-bkg*ap.area 
				source_minus_sky_e[k,j,i] = source_minus_sky_ADU[k,j,i]*GAIN
				source_minus_sky_err_e[k,j,i] = np.sqrt(phot_table['aperture_sum'][0]*GAIN + bkg*ap.area*GAIN + DARK_CURRENT*source_header['EXPTIME']*ap.area + ap.area*READ_NOISE**2)
				source_minus_sky_err_ADU[k,j,i] = source_minus_sky_err_e[k,j,i]/GAIN

				#Plot normalized target source-sky as you go along
				if j == 0 and k == 0:
					target_renorm_factor = np.mean(source_minus_sky_ADU[k,j,0:i+1])
					targ_norm = source_minus_sky_ADU[k,j,0:i+1]/target_renorm_factor
					targ_norm_err = source_minus_sky_err_ADU[k,j,0:i+1]/target_renorm_factor
					
					plt.errorbar(mjd_utc[0:i+1],targ_norm,targ_norm_err,color='k',marker='.',ls='',ecolor='k')
					#plt.ylim(380000,440000)
					plt.xlabel('Time (MJD)')
					plt.ylabel('ADU')
					

				#Create first-order ALC by summing all reference counts (by convention, positions 1: in our arrays)
				total_ref_ADU[k,i] = sum(source_minus_sky_ADU[k,1:,i]) #Sum up all the reference star counts
				total_ref_err_ADU[k,i] = np.sqrt(np.sum(source_minus_sky_err_ADU[k,1:,i]**2))
				total_ref_e[k,i] = sum(source_minus_sky_e[k,1:,i])
				total_ref_err_e[k,i] = np.sqrt(np.sum(source_minus_sky_err_e[k,1:,i]**2))

		if k == 0:
			alc_renorm_factor = np.mean(total_ref_ADU[k,0:i+1])
			alc_norm = total_ref_ADU[k,0:i+1]/alc_renorm_factor
			alc_norm_err = total_ref_err_ADU[k,0:i+1]/alc_renorm_factor
			plt.errorbar(mjd_utc[0:i+1],alc_norm, alc_norm_err,color='r',marker='.',ls='',ecolor='r')
			plt.ylim(0.98,1.02)
			plt.pause(0.2)
			plt.clf()

	#Write out photometry. 
	for i in range(len(ap_radii)):
		output_path = '/data/tierras/lightcurves/'+date+'/'+target+'/'+ffname+f'/circular_fixed_ap_phot_{ap_radii[i]}.csv'

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

		for j in range(len(targ_and_refs)):
			if j == 0:
				source_name = 'Target'
			else:
				source_name = f'Ref {j}'
			output_list.append([f'{val:.4f}' for val in source_x[j]])
			output_header.append(source_name+' X')
			output_list.append([f'{val:.4f}' for val in source_y[j]])
			output_header.append(source_name+' Y')
			output_list.append([f'{val:.4f}' for val in source_minus_sky_ADU[i,j]])
			output_header.append(source_name+' Source-Sky ADU')
			output_list.append([f'{val:.4f}' for val in source_minus_sky_err_ADU[i,j]])
			output_header.append(source_name+' Source-Sky Error ADU')
			output_list.append([f'{val:.4f}' for val in source_minus_sky_e[i,j]])
			output_header.append(source_name+' Source-Sky e')
			output_list.append([f'{val:.4f}' for val in source_minus_sky_err_e[i,j]])
			output_header.append(source_name+' Source-Sky Error e')

			output_list.append([f'{val:.4f}' for val in source_sky_ADU[j]])
			output_header.append(source_name+' Sky ADU')
			output_list.append([f'{val:.4f}' for val in source_sky_e[j]])
			output_header.append(source_name+' Sky e')

			output_list.append([f'{val:d}' for val in non_linear_flags[i,j]])
			output_header.append(source_name+' Non-Linear Flag')
			output_list.append([f'{val:d}' for val in saturated_flags[i,j]])
			output_header.append(source_name+' Saturated Flag')

		output_list.append([f'{val:.4f}' for val in total_ref_ADU[i]])
		output_header.append('Total Reference ADU')
		output_list.append([f'{val:.4f}' for val in total_ref_err_ADU[i]])
		output_header.append('Total Reference Error ADU')
		output_list.append([f'{val:.4f}' for val in total_ref_e[i]])
		output_header.append('Total Reference e')
		output_list.append([f'{val:.4f}' for val in total_ref_err_e[i]])
		output_header.append('Total Reference Error e')

		output_df = pd.DataFrame(np.transpose(output_list),columns=output_header)
		output_df.to_csv(output_path,index=False)

	breakpoint()
	return 

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
	ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")
	ap.add_argument("-ffname", required=True, help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat     was used.")
	args = ap.parse_args()

	#Access observation info
	date = args.date
	target = args.target
	ffname = args.ffname

	#Define base paths
	fpath = '/data/tierras/flattened'
	lcpath = '/data/tierras/lightcurves'
	
	#hdu = fits.open('/data/tierras/flattened/20230919/2MASSJ23373601+/flat0000/20230919.0284.2MASSJ23373601+_red.fit')[0]
	#data = hdu.data
	#header = hdu.header
	#detect_sources(data, header, mode='auto', model_bkg=True, plot=True)
	#breakpoint()

	#Get paths to the reduced data for this night/target/ffname
	flattened_files = get_flattened_files()

	#Stacked image demo
	# stacked_hdu = align_and_stack_images(flattened_files)
	# fig,ax = plt.subplots(1,2,figsize=(15,5),sharex=True,sharey=True)
	# unstacked_image = fits.open(flattened_files[0])[0].data
	# norm = ImageNormalize(unstacked_image, interval=ZScaleInterval())
	# ax[0].imshow(unstacked_image, origin='lower', interpolation='none', norm=norm)
	# ax[0].set_title('Reference image')

	# norm = ImageNormalize(stacked_hdu.data, interval=ZScaleInterval())
	# ax[1].imshow(stacked_hdu.data, origin='lower', interpolation='none', norm=norm)
	# ax[1].set_title('20 images stacked on reference image')
	# plt.tight_layout()
	
	targ_and_refs = reference_star_chooser(flattened_files)

	ap_radii = np.arange(13,14)
	fixed_circular_aperture_photometry(flattened_files, targ_and_refs, ap_radii, an_in=40, an_out=60)

	fig, ax = plot_image(fits.open(flattened_files[0])[0].data)
	ax.plot(targ_and_refs['x'][0], targ_and_refs['y'][0],'bx')
	ax.plot(targ_and_refs['x'][1:], targ_and_refs['y'][1:],'rx')
	for i in range(1, len(targ_and_refs)):
		ax.text(targ_and_refs['x'][i]+5,targ_and_refs['y'][i]+5,f'R{i}',color='r',fontsize=14,)
	breakpoint()
	

import numpy as np 
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
plt.ion()
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import ImageNormalize, ZScaleInterval, simple_norm
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
from photutils.aperture import CircularAperture, EllipticalAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_1dg, centroid_2dg
from scipy.stats import sigmaclip, pearsonr, linregress
from scipy.spatial.distance import cdist
from scipy.signal import correlate2d, fftconvolve, savgol_filter
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
from sklearn import linear_model
import copy
import batman
from glob import glob

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
	
	#norm = ImageNormalize(data[4:2042,:], interval=interval) #Ignore a few rows near the top/bottom for the purpose of getting a good colormap

	norm = simple_norm(data, stretch='linear', min_percent=1,max_percent=99.9)
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

def reference_star_chooser(file_list, mode='automatic', plot=False, overwrite=False, nonlinear_limit=40000, dimness_limit=0.05, nearness_limit=15, edge_limit=50, targ_distance_limit=4000, nrefs=1000):
	
	#Start by checking for existing csv file about target/reference positions
	reference_file_path = Path('/data/tierras/targets/'+target+'/'+target+'_target_and_ref_stars.csv')
	if (reference_file_path.exists() == False) or (overwrite==True):
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
		
		#Remove refs that are too close to other sources (dist < nearness_limit pix)
		refs_to_remove = []
		for i in range(len(possible_ref_inds)):
			dists = np.hypot(objs_stack['x']-objs_stack[possible_ref_inds[i]]['x'], objs_stack['y']-objs_stack[possible_ref_inds[i]]['y'])
			dists = np.delete(dists,possible_ref_inds[i]) #Remove the source itself from the distance calculation
			close_sources = np.where(dists < nearness_limit)[0]
			if len(close_sources) > 0:
				refs_to_remove.append(possible_ref_inds[i])
		for i in range(len(refs_to_remove)):
			possible_ref_inds = np.delete(possible_ref_inds, np.where(possible_ref_inds == refs_to_remove[i])[0][0])	
		possible_refs = objs_stack[possible_ref_inds]
		
		#Remove refs that are within edge_limit pixels of the detector edge 
		refs_to_remove = []
		for i in range(len(possible_ref_inds)):
			possible_ref_x = objs_stack[possible_ref_inds[i]]['x']
			possible_ref_y = objs_stack[possible_ref_inds[i]]['y']
			if (possible_ref_x < edge_limit) | (possible_ref_y < edge_limit) | (possible_ref_x > np.shape(stacked_image)[1]-edge_limit) | (possible_ref_y > np.shape(stacked_image)[0]-edge_limit):
				refs_to_remove.append(possible_ref_inds[i])
		for i in range(len(refs_to_remove)):
			possible_ref_inds = np.delete(possible_ref_inds, np.where(possible_ref_inds == refs_to_remove[i])[0][0])
		possible_refs = objs_stack[possible_ref_inds]

		#Remove refs that are more than targ_distance_limit away from the target
		dists = np.sqrt((objs_stack['x'][possible_ref_inds]-xtarg)**2+(objs_stack['y'][possible_ref_inds]-ytarg)**2)
		refs_to_remove = possible_ref_inds[np.where(dists>targ_distance_limit)[0]]
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

		#Determine the reference image from the stacked image header
		stack_comments = stacked_image_header['COMMENT']
		for i in range(len(stack_comments)):
			if 'Reference image' in stack_comments[i]:
				reference_filename = stack_comments[i].split(': ')[1]
				break
		#Identify that image in the list of flattened files.
		#NOTE this will only work if you're using the data from the same night as the stacked image
		for i in range(len(flattened_files)):
			if reference_filename in flattened_files[i]:
				reference_filepath = flattened_files[i]
				break

		if plot or mode=='manual':
			fig, ax = plot_image(fits.open(reference_filepath)[0].data)
			#ax.plot(objs_stack['x'],objs_stack['y'],'kx')
			ax.plot(targ_and_refs['x'][0], targ_and_refs['y'][0],'bx')
			ax.plot(targ_and_refs['x'][1:], targ_and_refs['y'][1:],'rx')
			ax.text(targ_and_refs['x'][0]+5, targ_and_refs['y'][0]+5,'T',color='b',fontsize=14)
			for i in range(1, len(targ_and_refs)):
				ax.text(targ_and_refs['x'][i]+5,targ_and_refs['y'][i]+5,f'R{i}',color='r',fontsize=14,)
			
			if mode == 'manual':
				ans = input('Enter IDs of reference stars to remove separated by commas (e.g. 2,4,15): ')
				if len(ans) > 0:
					split_ans = ans.replace(' ','').split(',')
					refs_to_remove = np.sort([int(i) for i in split_ans])[::-1]
					for i in refs_to_remove:
						targ_and_refs = np.delete(targ_and_refs,i)
					
					plt.close()
					
					fig, ax = plot_image(fits.open(flattened_files[0])[0].data)
					#ax.plot(objs_stack['x'],objs_stack['y'],'kx')
					ax.plot(targ_and_refs['x'][0], targ_and_refs['y'][0],'bx')
					ax.plot(targ_and_refs['x'][1:], targ_and_refs['y'][1:],'rx')
					ax.text(targ_and_refs['x'][0]+5, targ_and_refs['y'][0]+5,'T',color='b',fontsize=14)
					for i in range(1, len(targ_and_refs)):
						ax.text(targ_and_refs['x'][i]+5,targ_and_refs['y'][i]+5,f'R{i}',color='r',fontsize=14,)

				plt.savefig(reference_file_path.parent/(f'{target}_target_and_refs.png'),dpi=300)
				plt.close()
		
		#df = pd.DataFrame(targ_and_refs)
		output_dict = {}
		output_dict['x'] = [f'{val:0.4f}' for val in targ_and_refs['x']]
		output_dict['y'] = [f'{val:0.4f}' for val in targ_and_refs['y']]
		coords = [wcs.pixel_to_world(targ_and_refs['x'][i],targ_and_refs['y'][i]) for i in range(len(targ_and_refs))]
		ras_deg = [coords[i].ra.value for i in range(len(coords))]
		decs_deg = [coords[i].dec.value for i in range(len(coords))]
		output_dict['ra'] = [f'{val:.5f}' for val in ras_deg]
		output_dict['dec'] = [f'{val:.5f}' for val in decs_deg]
		df = pd.DataFrame(output_dict)
		df.to_csv(reference_file_path, index=False)

		#Query Gaia for information about sources. 

		breakpoint()
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

def align_and_stack_images(file_list, ref_image_num=10, n_ims_to_stack=20):
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
	#inds_to_stack = inds_excluding_reference[:n_ims_to_stack] #Count from 0...
	inds_to_stack = inds_excluding_reference[ref_image_num:ref_image_num+n_ims_to_stack] #Count from ref_image_num...
	print('Aligning and stacking images...')
	counter = 0 
	for i in inds_to_stack:
		print(f'{file_list[i]}, {counter+1} of {n_ims_to_stack}.')
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

def fixed_circular_aperture_photometry(file_list, targ_and_refs, ap_radii, an_in=40, an_out=60, centroid=False, live_plot=False):
	#file_list = file_list[-5:] #TESTING!!!
	
	DARK_CURRENT = 0.19 #e- pix^-1 s^-1
	NONLINEAR_THRESHOLD = 40000. #ADU
	SATURATION_TRESHOLD = 55000. #ADU
	PLATE_SCALE = 0.43 #arcsec pix^-1, from Juliana's dissertation Table 1.1
	
	# cutout_output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/target_cutouts/'
	# if not os.path.exists(cutout_output_path):
	# 	os.mkdir(cutout_output_path)
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
	
	#ARRAYS THAT CONTAIN DATA PERTAINING TO EACH SOURCE IN EACH FILE
	source_x = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_y = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_sky_ADU = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_sky_e = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_x_fwhm_arcsec = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_y_fwhm_arcsec = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')
	source_theta_radians = np.zeros((len(targ_and_refs),len(file_list)),dtype='float32')


	#ARRAYS THAT CONTAIN DATA PERTAININING TO EACH APERTURE RADIUS FOR EACH SOURCE FOR EACH FILE
	source_minus_sky_ADU = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	source_minus_sky_e = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	source_minus_sky_err_ADU = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	source_minus_sky_err_e = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	non_linear_flags = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='bool')
	saturated_flags = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='bool')
	ensemble_alc_ADU = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	ensemble_alc_e = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	ensemble_alc_err_ADU = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	ensemble_alc_err_e = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	relative_flux = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	relative_flux_err = np.zeros((len(ap_radii),len(targ_and_refs),len(file_list)),dtype='float32')
	

	# total_ref_ADU = np.zeros((len(ap_radii),len(file_list)),dtype='float32')
	# total_ref_err_ADU = np.zeros((len(ap_radii),len(file_list)),dtype='float32')
	# total_ref_e = np.zeros((len(ap_radii),len(file_list)),dtype='float32')
	# total_ref_err_e = np.zeros((len(ap_radii),len(file_list)),dtype='float32')

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
		fig, ax = plt.subplots(2,2,figsize=(16,9))

	print(f'Doing fixed-radius circular aperture photometry on {n_files} images with aperture radii of {ap_radii} pixels, an inner annulus radius of {an_in} pixels, and an outer annulus radius of {an_out} pixels.\n')
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
		#These keywords are sometimes missing
		try:
			sec_temps[i] = source_header['SECTEMP']
			rod_temps[i] = source_header['RODTEMP']
			cab_temps[i] = source_header['CABTEMP']
			inst_temps[i] = source_header['INSTTEMP']
		except:
			sec_temps[i] = np.nan
			rod_temps[i] = np.nan
			cab_temps[i] = np.nan
			inst_temps[i] = np.nan

		ret_temps[i] = source_header['RETTEMP']
		pri_temps[i] = source_header['PRITEMP']
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
				source_minus_sky_ADU[k,j,i] = np.nan
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
			xx,yy = np.meshgrid(np.arange(cutout.shape[1]),np.arange(cutout.shape[0]))

			x_pos_cutout = x_pos_image-int(x_pos_image)+an_out
			y_pos_cutout = y_pos_image-int(y_pos_image)+an_out

			# #Verify that the cutout position matches the image position
			# fig, ax = plt.subplots(1,2,figsize=(10,8))
			# #norm = ImageNormalize(cutout,interval=ZScaleInterval())
			# norm = simple_norm(cutout, 'linear', min_percent=1, max_percent=99)
			# ax[0].imshow(source_data, origin='lower', norm=norm, interpolation='none')
			# ax[0].set_ylim(int(y_pos_image-an_out),int(y_pos_image+an_out))
			# ax[0].set_xlim(int(x_pos_image-an_out),int(x_pos_image+an_out))
			# ax[0].plot(x_pos_image,y_pos_image,'rx',mew=2,ms=8)
			# ap_circ = plt.Circle((x_pos_image,y_pos_image),13,fill=False,color='r',lw=2)
			# ax[0].add_patch(ap_circ)
			# ax[0].set_ylim(y_pos_image-15,y_pos_image+15)
			# ax[0].set_xlim(x_pos_image-15,x_pos_image+15)
			# #ap = CircularAperture((x_pos_image,y_pos_image),r=ap_radii[0])
			# #ap.plot(color='r',lw=2.5)

			# ax[1].imshow(cutout,origin='lower',norm=norm,interpolation='none')
			# ax[1].plot(x_pos_cutout,y_pos_cutout,'kx',mew=2,ms=8)
			# ap_circ = plt.Circle((x_pos_cutout,y_pos_cutout),13,fill=False,color='k',lw=2)
			# ax[1].add_patch(ap_circ)
			# ax[1].set_ylim(y_pos_cutout-15,y_pos_cutout+15)
			# ax[1].set_xlim(x_pos_cutout-15,x_pos_cutout+15)
			# breakpoint()

			#Optionally recompute the centroid
			if centroid:
				centroid_mask = np.zeros(cutout.shape, dtype='bool')
				centroid_mask[0:int(an_out/2),:] = True
				centroid_mask[:,0:int(an_out/2)] = True
				centroid_mask[cutout.shape[0]-int(an_out/2):,:] = True
				centroid_mask[:,cutout.shape[1]-int(an_out/2):] = True
				x_pos_cutout_centroid, y_pos_cutout_centroid = centroid_1dg(cutout-np.median(cutout),mask=centroid_mask)

				#Make sure the measured centroid is actually in the cutout
				if (x_pos_cutout_centroid > 0) and (x_pos_cutout_centroid < cutout.shape[1]) and (y_pos_cutout_centroid > 0) and (y_pos_cutout_centroid < cutout.shape[0]):
					x_pos_cutout = x_pos_cutout_centroid
					y_pos_cutout = y_pos_cutout_centroid


			for k in range(len(ap_radii)):
				ap = CircularAperture((x_pos_cutout,y_pos_cutout),r=ap_radii[k])
				#ap = EllipticalAperture((x_pos_cutout,y_pos_cutout),a=15,b=9, theta=90*np.pi/180)
				an = CircularAnnulus((x_pos_cutout,y_pos_cutout),r_in=an_in,r_out=an_out)


				if j == 0 and k == 0 and live_plot:
					norm = simple_norm(cutout,'linear',min_percent=0,max_percent=99.5)
					ax[1,0].imshow(cutout,origin='lower',interpolation='none',norm=norm,cmap='Greys_r')
					#ax[1,0].imshow(cutout,origin='lower',interpolation='none',norm=norm)
					ax[1,0].plot(x_pos_cutout,y_pos_cutout, color='m', marker='x',mew=1.5,ms=8)
					ap_circle = plt.Circle((x_pos_cutout,y_pos_cutout),ap_radii[k],fill=False,color='m',lw=2)
					an_in_circle = plt.Circle((x_pos_cutout,y_pos_cutout),an_in,fill=False,color='m',lw=2)
					an_out_circle = plt.Circle((x_pos_cutout,y_pos_cutout),an_out,fill=False,color='m',lw=2)
					ax[1,0].add_patch(ap_circle)
					ax[1,0].add_patch(an_in_circle)
					ax[1,0].add_patch(an_out_circle)
					ax[1,0].set_xlim(0,cutout.shape[1])
					ax[1,0].set_ylim(0,cutout.shape[0])
					ax[1,0].grid(False)
					ax[1,0].set_title('Target')

					ax[0,0].imshow(source_data,origin='lower',interpolation='none',norm=simple_norm(source_data,'linear',min_percent=1,max_percent=99.9), cmap='Greys_r')
					ax[0,0].grid(False)
					ax[0,0].set_title(file_list[i].split('/')[-1])
					for l in range(len(source_x)):
						if l == 0:
							color = 'm'
							name = 'T'
						else:
							color = 'tab:red'
							name = f'R{l}'
						ap_circle = plt.Circle((source_x[l,i],source_y[l,i]),30,fill=False,color=color,lw=1)
						ax[0,0].add_patch(ap_circle)
						ax[0,0].text(source_x[l,i]+15,source_y[l,i]+15,name,color=color,fontsize=14)


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
				
				#source_mask = make_source_mask(cutout, nsigma=3, npixels=5, dilate_size=4) #source mask eats up a LOT of memory, why???
				source_mask = np.zeros(np.shape(cutout),dtype='bool')

				# #Plot the source mask
				# fig, ax = plt.subplots(1,2,figsize=(10,7))
				# ax[0].imshow(an_data, origin='lower',norm=ImageNormalize(an_data[an_data!=0],interval=ZScaleInterval()),interpolation='none')
				# ax[1].imshow(source_mask, origin='lower',norm=ImageNormalize(cutout,interval=ZScaleInterval()),interpolation='none')

				an_data_masked = an_data*~source_mask 
				an_data_1d = an_data_masked[an_data_masked != 0] #unwrap into 1d array


				#Discard any negative values in the annulus
				an_data_1d = an_data_1d[np.where(an_data_1d>0)[0]]

				an_vals, hi, lo = sigmaclip(an_data_1d,3,3) #toss outliers
				bkg = np.mean(an_vals) #take median of remaining values as per-pixel background estimate

				# plt.figure()
				# plt.hist(an_vals,bins=15)
				# plt.axvline(bkg, color='tab:orange')
				# breakpoint()
				
				source_sky_ADU[j,i] = bkg
				source_sky_e[j,i] = bkg*GAIN

				source_minus_sky_ADU[k,j,i] = phot_table['aperture_sum'][0]-bkg*ap.area 
				source_minus_sky_e[k,j,i] = source_minus_sky_ADU[k,j,i]*GAIN
				source_minus_sky_err_e[k,j,i] = np.sqrt(phot_table['aperture_sum'][0]*GAIN + bkg*ap.area*GAIN + DARK_CURRENT*source_header['EXPTIME']*ap.area + ap.area*READ_NOISE**2)
				source_minus_sky_err_ADU[k,j,i] = source_minus_sky_err_e[k,j,i]/GAIN

				#Measure shape by fitting a 2D Gaussian to the cutout.
				#Don't do for every aperture size, just do it once. 
				if k == 0:
					g_init = models.Gaussian2D(amplitude=cutout[int(cutout.shape[1]/2), int(cutout.shape[0]/2)]-bkg,x_mean=cutout.shape[1]/2,y_mean=cutout.shape[0]/2, x_stddev=5, y_stddev=5)
					fit_g = fitting.LevMarLSQFitter()
					g = fit_g(g_init,xx,yy,cutout-bkg)
					
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

					# fig, ax = plt.subplots(1,2,figsize=(12,8),sharex=True,sharey=True)
					# norm = ImageNormalize(cutout-bkg,interval=ZScaleInterval())
					# ax[0].imshow(cutout-bkg,origin='lower',interpolation='none',norm=norm)
					# ax[1].imshow(g(xx,yy),origin='lower',interpolation='none',norm=norm)
					# plt.tight_layout()

				#Plot normalized target source-sky as you go along
				if live_plot and j == 0 and k == 0:
					target_renorm_factor = np.mean(source_minus_sky_ADU[k,j,0:i+1])
					targ_norm = source_minus_sky_ADU[k,j,0:i+1]/target_renorm_factor
					targ_norm_err = source_minus_sky_err_ADU[k,j,0:i+1]/target_renorm_factor
					
					ax[0,1].errorbar(bjd_tdb[0:i+1],targ_norm,targ_norm_err,color='k',marker='.',ls='',ecolor='k',label='Normalized target flux')
					#plt.ylim(380000,440000)
					ax[0,1].set_ylabel('Normalized Flux')
					

				# #Create first-order ALC by summing all reference counts (by convention, positions 1: in our arrays)
				# total_ref_ADU[k,i] = sum(source_minus_sky_ADU[k,1:,i]) #Sum up all the reference star counts
				# total_ref_err_ADU[k,i] = np.sqrt(np.sum(source_minus_sky_err_ADU[k,1:,i]**2))
				# total_ref_e[k,i] = sum(source_minus_sky_e[k,1:,i])
				# total_ref_err_e[k,i] = np.sqrt(np.sum(source_minus_sky_err_e[k,1:,i]**2))
		
		#Create ensemble ALCs (summed reference fluxes with no weighting) for each source
		for l in range(len(targ_and_refs)):
			#For the target, use all reference stars
			ref_inds = np.arange(1,len(targ_and_refs))
			#For the reference stars, use all other references and NOT the target
			if l != 0:
				ref_inds = np.delete(ref_inds,l-1)
			for m in range(len(ap_radii)):
				ensemble_alc_ADU[m,l,i] = sum(source_minus_sky_ADU[m,ref_inds,i])
				ensemble_alc_err_ADU[m,l,i] = np.sqrt(np.sum(source_minus_sky_err_ADU[m,ref_inds,i]**2))
				ensemble_alc_e[m,l,i] = sum(source_minus_sky_e[m,ref_inds,i])
				ensemble_alc_err_e[m,l,i] = np.sqrt(np.sum(source_minus_sky_err_e[m,ref_inds,i]**2))

				relative_flux[m,l,i] = source_minus_sky_ADU[m,l,i]/ensemble_alc_ADU[m,l,i]
				relative_flux_err[m,l,i] = np.sqrt((source_minus_sky_err_ADU[m,l,i]/ensemble_alc_ADU[m,l,i])**2+(source_minus_sky_ADU[m,l,i]*ensemble_alc_err_ADU[m,l,i]/(ensemble_alc_ADU[m,l,i]**2))**2)

		if live_plot:
			alc_renorm_factor = np.mean(ensemble_alc_ADU[0,0,0:i+1]) #This means, grab the ALC associated with the 0th aperture for the 0th source (the target) in all images up to and including this one.
			alc_norm = ensemble_alc_ADU[0,0,0:i+1]/alc_renorm_factor
			alc_norm_err = ensemble_alc_err_ADU[0,0,0:i+1]/alc_renorm_factor
			v,l,h=sigmaclip(alc_norm)
			ax[0,1].errorbar(bjd_tdb[0:i+1],alc_norm, alc_norm_err,color='r',marker='.',ls='',ecolor='r', label='Normalized ALC flux')
			ax[0,1].set_ylim(l,h)
			ax[0,1].legend() 

			corrected_flux = targ_norm/alc_norm
			corrected_flux_err = np.sqrt((targ_norm_err/alc_norm)**2+(targ_norm*alc_norm_err/(alc_norm**2))**2)
			v,l,h=sigmaclip(corrected_flux)
			ax[1,1].errorbar(bjd_tdb[0:i+1],corrected_flux, corrected_flux_err, color='k', marker='.', ls='', ecolor='k', label='Corrected target flux')
			ax[1,1].set_ylim(l,h)
			ax[1,1].legend()
			ax[1,1].set_ylabel('Normalized Flux')
			ax[1,1].set_xlabel('Time (BJD$_{TDB}$)')
			#plt.tight_layout()
			plt.pause(0.01)
			ax[0,0].cla()
			ax[1,0].cla()
			ax[0,1].cla()
			ax[1,1].cla()

	#Write out photometry. 
	for i in range(len(ap_radii)):
		output_path = Path('/data/tierras/lightcurves/'+date+'/'+target+'/'+ffname+f'/circular_fixed_ap_phot_{ap_radii[i]}.csv')

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

			output_list.append([f'{val:.4f}' for val in ensemble_alc_ADU[i,j]])
			output_header.append(source_name+' Ensemble ALC ADU')
			output_list.append([f'{val:.4f}' for val in ensemble_alc_err_ADU[i,j]])
			output_header.append(source_name+' Ensemble ALC Error ADU')
			output_list.append([f'{val:.4f}' for val in ensemble_alc_e[i,j]])
			output_header.append(source_name+' Ensemble ALC e')
			output_list.append([f'{val:.4f}' for val in ensemble_alc_err_e[i,j]])
			output_header.append(source_name+' Ensemble ALC Error e')
			output_list.append([f'{val:.4f}' for val in relative_flux[i,j]])
			output_header.append(source_name+' Relative Flux')
			output_list.append([f'{val:.4f}' for val in relative_flux_err[i,j]])
			output_header.append(source_name+' Relative Flux Error')

			output_list.append([f'{val:.4f}' for val in source_sky_ADU[j]])
			output_header.append(source_name+' Sky ADU')
			output_list.append([f'{val:.4f}' for val in source_sky_e[j]])
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

		# output_list.append([f'{val:.4f}' for val in total_ref_ADU[i]])
		# output_header.append('Total Reference ADU')
		# output_list.append([f'{val:.4f}' for val in total_ref_err_ADU[i]])
		# output_header.append('Total Reference Error ADU')
		# output_list.append([f'{val:.4f}' for val in total_ref_e[i]])
		# output_header.append('Total Reference e')
		# output_list.append([f'{val:.4f}' for val in total_ref_err_e[i]])
		# output_header.append('Total Reference Error e')

		output_df = pd.DataFrame(np.transpose(output_list),columns=output_header)
		if not os.path.exists(output_path.parent.parent):
			os.mkdir(output_path.parent.parent)
		if not os.path.exists(output_path.parent):
			os.mkdir(output_path.parent)
		output_df.to_csv(output_path,index=False)

	return 

def plot_target_lightcurve(file_path,regression=False,pval_threshold=0.001):
	df = pd.read_csv(file_path)
	times = np.array(df['BJD TDB'])
	x_offset =  int(np.floor(times[0]))
	times -= x_offset

	targ_flux = np.array(df['Target Source-Sky ADU'])
	targ_flux_err = np.array(df['Target Source-Sky Error ADU'])
	alc_flux = np.array(df['Target Ensemble ALC ADU'])
	alc_flux_err = np.array(df['Target Ensemble ALC Error ADU'])

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
	alc_flux = alc_flux[use_inds]
	alc_flux_err = alc_flux_err[use_inds]

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

	fig, ax = plt.subplots(6,1,figsize=(8,10),sharex=True)
	ax[0].errorbar(times, targ_flux_norm, targ_flux_err_norm, marker='.', color='k',ls='', ecolor='k', label='Normalized target flux')
	ax[0].errorbar(times, alc_flux_norm, alc_flux_err_norm, marker='.', color='r',ls='', ecolor='r', label='Normalized ALC flux')
	ax[0].tick_params(labelsize=14)
	ax[0].legend()
	ax[0].grid(alpha=0.8)
	ax[0].set_ylabel('Norm. Flux',fontsize=14)

	ax[1].errorbar(times, corrected_targ_flux, corrected_targ_flux_err,marker='.',color='k',ecolor='k',ls='',label='Corrected target flux')
	ax[1].legend()
	ax[1].tick_params(labelsize=14)
	ax[1].set_ylabel('Norm. Flux',fontsize=14)
	
	ax[2].plot(times,ancillary_dict['Airmass'], color='tab:blue',lw=2)
	ax[2].tick_params(labelsize=14)
	ax[2].set_ylabel('Airmass',fontsize=14)

	ax[3].plot(times,ancillary_dict['Target Sky ADU'],color='tab:orange',lw=2)
	ax[3].tick_params(labelsize=14)
	ax[3].set_ylabel('Sky (ADU/pix)',fontsize=14)

	ax[4].plot(times,ancillary_dict['Target X'],color='tab:green',lw=2,label='X-med(X)')
	ax[4].plot(times,ancillary_dict['Target Y'],color='tab:red',lw=2,label='Y-med(Y)')
	ax[4].tick_params(labelsize=14)
	ax[4].set_ylabel('Pos.',fontsize=14)
	ax[4].legend()
	v1,l1,h1 = sigmaclip(ancillary_dict['Target X'],5,5)
	v2,l2,h2 = sigmaclip(ancillary_dict['Target X'],5,5)
	ax[4].set_ylim(np.min([l1,l2]),np.max([h1,h2]))

	ax[5].plot(times,ancillary_dict['Target X FWHM Arcsec'],color='tab:pink',lw=2,label='X')
	ax[5].plot(times, ancillary_dict['Target Y FWHM Arcsec'], color='tab:purple', lw=2,label='Y')
	ax[5].legend()
	ax[5].tick_params(labelsize=14)
	ax[5].set_ylabel('FWHM (")',fontsize=14)
	v1,l1,h1 = sigmaclip(ancillary_dict['Target X FWHM Arcsec'],5,5)
	v2,l2,h2 = sigmaclip(ancillary_dict['Target Y FWHM Arcsec'],5,5)
	ax[5].set_ylim(np.min([l1,l2]),np.max([h1,h2]))

	ax[-1].set_xlabel(f'Time - {x_offset}'+'(BJD$_{TDB}$)',fontsize=16)

	plt.tight_layout()

	if regression:
		regr = linear_model.LinearRegression()
		regress_dict = {}
		#Check for significant correlations between ancillary data and corrected target flux
		for key in ancillary_dict:
			try:
				corr, pvalue = pearsonr(corrected_targ_flux,ancillary_dict[key])
			except:
				continue
			if pvalue < pval_threshold:
				regress_dict[key] = ancillary_dict[key]
				print(f'{key}, corr:{corr:.2f}, P-value: {pvalue:.2E}')
		  
		regress_dict['flux'] = corrected_targ_flux
		keylist = list(regress_dict.keys())
		
		regress_df = pd.DataFrame(regress_dict, columns=list(regress_dict.keys()))
		x = regress_df[keylist[0:len(keylist)-1]]
		y = regress_df['flux']
		regr.fit(x,y)
		regression_model = regr.intercept_
		for i in range(len(keylist[:-1])):
			regression_model += regr.coef_[i]*regress_dict[keylist[i]]

		ax[1].plot(times, regression_model, lw=2, zorder=4, label='Linear regression model')
		ax[1].legend()

		plt.figure()
		regressed_flux = corrected_targ_flux/regression_model
		regressed_flux /= np.mean(regressed_flux[np.where(times>0.72)[0]])	
		# coeffs = np.polyfit(times[np.where(times>0.72)[0]],regressed_flux[np.where(times>0.72)[0]],1)
		# fit = times*coeffs[0]+coeffs[1]
		# regressed_flux /= fit
		points_to_bin = 200
		n_bins = int(np.ceil(len(times)/points_to_bin))
		bx = np.zeros(n_bins)
		by = np.zeros(n_bins)
		bye = np.zeros(n_bins)
		for i in range(n_bins):
			if i == n_bins-1:
				bin_inds = np.arange(i*points_to_bin,len(times))
			else:
				bin_inds = np.arange(i*points_to_bin,(i+1)*points_to_bin)
			bx[i] = np.mean(times[bin_inds])
			by[i] = np.mean(regressed_flux[bin_inds])
			bye[i] = np.std(regressed_flux[bin_inds])/np.sqrt(len(bin_inds))
		plt.plot(times, regressed_flux, marker='.',color='#b0b0b0',ls='')
		plt.errorbar(bx, by, bye, marker='o', color='none', mec='k', ecolor='k', mew=2, ls='', zorder=3)
	
	planet_model = transit_model(times, 2459510.82759-x_offset, 22.09341, 0.0424+.0009, 21.53, 87.93, 0, 0, 0.209, 0.314)
	plt.plot(times, planet_model, lw=2)
	plt.ylim(0.991,1.009)
	plt.xlim(times[0],times[-1])

	#Do bin plot
	bins = np.arange(0.5,20.5,0.5)
	std, theo = juliana_binning(bins, times, corrected_targ_flux, corrected_targ_flux_err)

	plt.figure()
	plt.plot(bins, std[1:]*1e6, lw=2,label='Measured')
	plt.plot(bins, theo[1:]*1e6,lw=2,label='Theoretical')
	plt.xlabel('Bin size (min)',fontsize=14)
	plt.ylabel('$\sigma$ (ppm)',fontsize=14)
	plt.legend()
	breakpoint()

def plot_ref_positions(file_list, targ_and_refs):
	im = fits.open(file_list[5])[0].data
	fig, ax = plot_image(im)
	ax.plot(targ_and_refs['x'][0], targ_and_refs['y'][0],'bx')
	for i in range(1,len(targ_and_refs)):
		ax.plot(targ_and_refs['x'][i], targ_and_refs['y'][i],'rx')
		ax.text(targ_and_refs['x'][i]+5, targ_and_refs['y'][i]+5, f'R{i}',fontsize=14,color='r')
	return 

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

def optimal_lc_chooser(date, target, ffname, plot=False):
	lc_list = np.array(glob(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/*phot*.csv'))
	sort_inds = np.argsort([float(i.split('/')[-1].split('_')[-1].split('.')[0]) for i in lc_list])
	lc_list = lc_list[sort_inds]

	if plot:
		fig, ax = plt.subplots(len(lc_list),1,figsize=(10,1.25*len(lc_list)),sharex=True,sharey=True)

	best_stddev = 9999.
	for i in range(len(lc_list)):
		df = pd.read_csv(lc_list[i])
		times = np.array(df['BJD TDB'])
		rel_targ_flux = np.array(df['Target Relative Flux'])
		rel_targ_flux_err = np.array(df['Target Relative Flux Error'])
		v,l,h = sigmaclip(rel_targ_flux)
		use_inds = np.where((rel_targ_flux>l)&(rel_targ_flux<h))[0]
		times = times[use_inds]
		rel_targ_flux = rel_targ_flux[use_inds]
		rel_targ_flux_err = rel_targ_flux_err[use_inds]
		
		norm = np.mean(rel_targ_flux)
		rel_targ_flux /= norm 
		rel_targ_flux_err /= norm

		#moving_avg = moving_average(rel_targ_flux,int(len(times)/50))
		if plot:
			ax[i].errorbar(times, rel_targ_flux, rel_targ_flux_err, marker='.',color='#b0b0b0',ls='')
			#ax[i].plot(times, moving_avg,color='tab:orange',lw=2,zorder=3)
			ax2 = ax[i].twinx()
			ax2.set_ylabel(lc_list[i].split('_')[-1].split('.csv')[0],rotation=270,labelpad=12)
			ax2.set_yticks([])
		stddev = np.std(rel_targ_flux)
		print(np.std(rel_targ_flux[100:110])*1e6)
		if stddev < best_stddev:
			best_ind = i
			best_lc_path = lc_list[i]
			best_stddev = stddev

	if plot:
		ax[-1].set_xlabel('Time (BJD$_{TDB}$)')
		plt.tight_layout()
		plt.savefig(f'/data/tierras/lightcurves/{date}/{target}/{ffname}/{target}_{date}_optimized_lc_.png',dpi=300)
	return best_lc_path

def ap_range(file_list, targ_and_refs, overwrite=False):
	'''Measures the average FWHM of the target across a set of images to determine a range of apertures for performing photometry.
	'''

	output_path = f'/data/tierras/lightcurves/{date}/{target}/{ffname}/aperture_range.csv'

	if not (os.path.exists(output_path)) or (overwrite==True):

		print('Determining optimal aperture range...')
		time.sleep(2)

		PLATE_SCALE = 0.43 #arcsec pix^-1, from Juliana's dissertation Table 1.1

		bpm = load_bad_pixel_mask()

		#load in the reference image 
		reference_image_hdu = fits.open('/data/tierras/targets/'+target+'/'+target+'_stacked_image.fits')[0] #TODO: should match image from target/reference csv file, and that should be loaded automatically.

		reference_image_header = reference_image_hdu.header
		reference_wcs = WCS(reference_image_header)
		reference_world_coordinates = reference_wcs.pixel_to_world(targ_and_refs['x'][0],targ_and_refs['y'][0]) #Get world coordinates of target in the reference image.

		fwhm_x = np.zeros(len(file_list))
		fwhm_y = np.zeros(len(file_list))
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
			fit_g = fitting.LevMarLSQFitter()
			g = fit_g(g_init,xx,yy,cutout-np.median(cutout))
			
			x_stddev_pix = g.x_stddev.value
			y_stddev_pix = g.y_stddev.value 
			x_fwhm_pix = x_stddev_pix * 2*np.sqrt(2*np.log(2))
			y_fwhm_pix = y_stddev_pix * 2*np.sqrt(2*np.log(2))
			x_fwhm_arcsec = x_fwhm_pix * PLATE_SCALE
			y_fwhm_arcsec = y_fwhm_pix * PLATE_SCALE
			theta_rad = g.theta.value
			fwhm_x[i] = x_fwhm_arcsec
			fwhm_y[i] = y_fwhm_arcsec
			print(f'{i+1} of {len(file_list)}')
		#Sigma clip the results
		v1,l1,h1 = sigmaclip(fwhm_x)
		v2,l2,h2 = sigmaclip(fwhm_y) 
		use_inds = np.where((fwhm_x>l1)&(fwhm_x<h1)&(fwhm_y>l2)&(fwhm_y<h2))[0]
		fwhm_x = fwhm_x[use_inds]
		fwhm_y = fwhm_y[use_inds]

		#Use the lower of the 75th percentiles of fwhm_x/y to set the lower aperture radius bound 
		#Good choice of aperture is between 1.5-2.0 FWHM
		fwhm_x_75_pix = np.percentile(fwhm_x/PLATE_SCALE,75)
		fwhm_y_75_pix = np.percentile(fwhm_y/PLATE_SCALE,75)
		lower_pix_bound = int(np.floor(np.min([fwhm_x_75_pix,fwhm_y_75_pix])*1.5))-1 #Subtract one for some tolerance
		upper_pix_bound = int(np.ceil(np.max([fwhm_x_75_pix,fwhm_y_75_pix])*2))+1 #Add one for some tolerance
		aps_to_use = np.arange(lower_pix_bound, upper_pix_bound+1)

		output_dict = {'Aperture radii':aps_to_use}
		output_df = pd.DataFrame(output_dict)
		output_df.to_csv(output_path,index=False)
	else:
		print(f'Restoring aperture range output from {output_path}.')
		output_df = pd.read_csv(output_path)
		aps_to_use = np.array(output_df['Aperture radii'])
	
	return aps_to_use

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
	if not os.path.exists(lcpath+f'/{date}'):
		os.mkdir(lcpath+f'/{date}')
	if not os.path.exists(lcpath+f'/{date}/{target}'):
		os.mkdir(lcpath+f'/{date}/{target}')
	if not os.path.exists(lcpath+f'/{date}/{target}/{ffname}'):
		os.mkdir(lcpath+f'/{date}/{target}/{ffname}')

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
	
	targ_and_refs = reference_star_chooser(flattened_files, mode='manual', plot=True, nearness_limit=12, edge_limit=40,dimness_limit=0.01, targ_distance_limit=3000, overwrite=True)

	# ap_radii = ap_range(flattened_files, targ_and_refs)

	# fixed_circular_aperture_photometry(flattened_files, targ_and_refs, ap_radii, an_in=40, an_out=80, centroid=True, live_plot=True)

	optimal_lc_path = optimal_lc_chooser(date,target,ffname,plot=True)

	plot_target_lightcurve(optimal_lc_path, regression=True)
	
	breakpoint()
	
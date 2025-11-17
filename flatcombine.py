#!/opt/cfpython/anaconda3.7/bin/python

from __future__ import print_function

import argparse
import logging
import math
import re
import sys
import warnings
from glob import glob
import matplotlib.pyplot as plt 
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ion()
from astropy.io import fits 
from ap_phot import set_tierras_permissions
import numpy as np
from scipy.stats import sigmaclip
import matplotlib.cm as cm 
import matplotlib.colors as colors 
from ap_phot import t_or_f
try:
  import astropy.utils.exceptions
  import astropy.io.fits as pyfits
except ImportError:
  import pyfits

import lfa

from fitsutil import *

def process_extension(imp, iext):
	hdr = imp.header

	deg = 8 # degree of polynomial fit to overscan for bias subtraction
	x = np.arange(1024) # row indices for computing bias subtraction fit
  
	if "BIASSEC" in hdr:
		biassec = fits_section(hdr["BIASSEC"])
	else:
		biassec = None
	  
	if "TRIMSEC" in hdr:
		trimsec = fits_section(hdr["TRIMSEC"])
	else:
		trimsec = None
	  
	raw = np.float32(imp.data)

	if biassec is not None:
		# biaslev, biassig = lfa.skylevel_image(raw[biassec[2]:biassec[3],biassec[0]:biassec[1]])
		bias_img = raw[biassec[2]:biassec[3],biassec[0]:biassec[1]]
		# blocks = np.median(bias_img,axis=1).reshape(8,128) 

		# means = np.mean(blocks, axis=1)
		# biaslev = np.repeat(means, 128)
		coeffs = np.polyfit(x, np.median(bias_img, axis=1), deg)
		biaslev = np.polyval(coeffs, x) 
	else:
		biaslev = 0

	if trimsec is not None:
		procimg = raw[trimsec[2]:trimsec[3],trimsec[0]:trimsec[1]] - biaslev[:,None]
	else:
		procimg = raw - biaslev

	if iext == 2:
		procimg = np.flipud(procimg) # top half of image is read in upside down under the current read-in scheme, so flip it here

	return procimg.astype(np.float32)

def shutter_model(im, exptime, read_speed=7e4):
	# model the extra exposure time for the rows toward the chip center using the read speed
	# NOTE that the speed of 7e4 Hz was MEASURED using several nights of flat data in 202509 and 202510
	lower_section = np.ones((1024, 4096)) 
	lower_section *= np.arange(0, 1024)[..., None] / read_speed # [rows] * [rows / s] = [s]
	upper_section = lower_section[::-1]
	model = np.concatenate([lower_section, upper_section])

	return exptime / (exptime + model) * np.median(im)

if __name__ == "__main__":
	# Deal with the command line.
	ap = argparse.ArgumentParser()
	# ap.add_argument("filelist", metavar="file", nargs="+", help="input files")
	# ap.add_argument("-o", help="output file")
	ap.add_argument("-date", required=True, help="YYYYMMDD of the date on which flats were taken.")
	ap.add_argument("-shutter_correction", required=False, default='True', help='Whether or not to correct flats for shutter effect')


  
	args = ap.parse_args()

	date = args.date
	shutter_correction = t_or_f(args.shutter_correction)

	filepath = f'/data/tierras/incoming/{date}/'
	filelist = glob(filepath+'*FLAT[0-9][0-9][0-9].fit')
	filelist = np.array(sorted(filelist, key=lambda x:int(x.split('FLAT')[-1].split('.')[0]))) # make sure the files are sorted 
	nf = len(filelist)

	if len(filelist) < 60: # should get ~43 flats in a single evening flat sequence. If the count is way over, something has gone wrong. 

		fplist = [ pyfits.open(filename) for filename in filelist ]

		outhdus = []

		combined_images = []
		exptimes = []
		for i, filename in enumerate(filelist):
			print(f'Doing {filename} ({i+1} of {nf})')
			with fits.open(filename) as ifp:
				proc_imgs = [process_extension(ifp[j], j) for j in range(1, 3)]
				exptimes.append(ifp[0].header['EXPTIME'])
			combined_img = np.concatenate(proc_imgs, axis=0)
			combined_images.append(combined_img)

		combined_images = np.array(combined_images)
		# loop over the combined flats and reject any that are significant flux outliers 
		flat_medians = np.zeros(len(combined_images))
		for i in range(len(combined_images)):
			flat_medians[i] = np.median(combined_images[i])

		v, l, h = sigmaclip(flat_medians, 5, 5) # do a 5-sigma outlier rejection

		# plt.axvline(l, color='tab:orange', ls='--')
		# plt.axvline(h, color='tab:orange', ls='--')
		use_inds = np.where((flat_medians > l) & (flat_medians < h))[0]
		print(f'Discarded {len(combined_images) - len(use_inds)} flats with 5-sigma median flux clipping. {len(use_inds)} flats will be combined.')

		# update arrays with use_inds 
		n_files = len(use_inds)
		flat_files = filelist[use_inds]
		combined_images = combined_images[use_inds]
		flat_medians = flat_medians[use_inds] 

		# fig, ax = plt.subplots(1,2,figsize=(16,6), sharex=True, sharey=True)
		# shutter_ratio = combined_images[-2] / combined_images[1] # without any shutter corrections
		# norm = simple_norm(shutter_ratio, min_percent=1, max_percent=99)
		# ax[0].imshow(shutter_ratio, origin='lower', norm=norm)

		# correct the images for the shutter closing effect 
		if shutter_correction:
			for i in range(len(combined_images)):
				shutter = shutter_model(combined_images[i], exptimes[i])
				combined_images[i] /= shutter

		# median-normalize each individual flat 
		for i in range(len(combined_images)):
			norm = np.nanmedian(combined_images[i])
			combined_images[i] /= norm 

		# shutter_ratio = combined_images[-2] / combined_images[1] # after shutter corrections have been applied
		# norm = simple_norm(shutter_ratio, min_percent=1, max_percent=99)

		# ax[1].imshow(shutter_ratio, origin='lower', norm=norm)

		# # examine the behavior of slices of flats 
		# n_central_rows = 1
		# n_central_cols = 1
		# median_central_rows = np.median(combined_images[:,1024-int(n_central_rows/2):1024+int(n_central_rows/2),:],axis=1)
		# median_central_cols = np.median(combined_images[:,:,2048-int(n_central_cols/2):2048+int(n_central_cols/2)],axis=2)

		# median_central_rows = combined_images[:,760,:]
		# median_central_cols = combined_images[:,:,2048]
		# # breakpoint()
		# cmap = cm.viridis
		
		# # plot central rows
		# fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True, gridspec_kw={'height_ratios':[2,1]})
		# norm = simple_norm(combined_images[0], min_percent=1, max_percent=99)
		# im_ = ax[0].imshow(combined_images[0], origin='lower', interpolation='none', norm=norm, aspect='auto')
		# save_ylim = ax[0].get_ylim()
		# ax[0].fill_between(np.arange(0,4096), 760-int(n_central_rows/2), 760+int(n_central_rows/2), color='m', alpha=0.6)
		# ax[0].set_ylim(save_ylim)

		# # add a hidden colorbar to match size of bottom plot
		# divider = make_axes_locatable(ax[0])
		# cax = divider.append_axes('right', size='5%', pad=0.1)
		# cb = fig.colorbar(im_, cax=cax, orientation='vertical')
		# cb.remove()
		# cb.outline.set_visible(False)

		# # now plot the rows data
		# for i in range(combined_images.shape[0]):
		# 	pl = ax[1].plot(median_central_rows[i], color=cmap(int(255*i/(combined_images.shape[0]-1))), label=f'FLAT {str(i+1).zfill(2)}')
		# ax[1].set_xlabel('Column Number', fontsize=14)
		# ax[1].set_ylabel('Normalized Flux', fontsize=14)
		# ax[1].grid(alpha=0.5)
		# v, l, h = sigmaclip(np.median(median_central_rows,axis=0))
		# ax[1].set_ylim(l, h)

		# divider = make_axes_locatable(ax[1])
		# cax = divider.append_axes('right', size='5%', pad=0.1)
		# cb_norm = colors.Normalize(vmin=0, vmax=combined_images.shape[0])
		# mappable = cm.ScalarMappable(cmap=cmap, norm=cb_norm)
		# cb = fig.colorbar(mappable, cax=cax, orientation='vertical')
		# ticks = np.arange(0,combined_images.shape[0],10)
		# cb.ax.yaxis.set_ticks(ticks)
		# cb.set_label('Flat Number')
		# fig.tight_layout()

		# # now do columns
		# fig, ax = plt.subplots(1,2,figsize=(16,5), sharey=True, gridspec_kw={'width_ratios':[2,1]})

		# ax[0].imshow(combined_images[0], origin='lower', interpolation='none', norm=norm, aspect='auto')
		# save_ylim = ax[0].get_ylim()
		# ax[0].fill_between(np.arange(2048-int(n_central_cols/2), 2048+int(n_central_cols/2)), 0, 2048, color='m', alpha=0.6)
		# ax[0].set_ylim(save_ylim)

		# for i in range(combined_images.shape[0]):
		# 	ax[1].plot(median_central_cols[i], np.arange(0, 2048), color=cmap(int(255*i/(combined_images.shape[0]-1))))
		# ax[1].set_ylabel('Row Number', fontsize=14)
		# ax[1].set_xlabel('Normalized Flux', fontsize=14)
		# ax[1].grid(alpha=0.5)
		# v, l, h = sigmaclip(np.median(median_central_cols,axis=0))
		# ax[1].set_xlim(l, h)
		
		# divider = make_axes_locatable(ax[1])
		# cax = divider.append_axes('right', size='5%', pad=0.1)
		# cb_norm = colors.Normalize(vmin=0, vmax=combined_images.shape[0])
		# mappable = cm.ScalarMappable(cmap=cmap, norm=cb_norm)
		# cb = fig.colorbar(mappable, cax=cax, orientation='vertical')
		# ticks = np.arange(0,combined_images.shape[0],10)
		# cb.ax.yaxis.set_ticks(ticks)
		# cb.set_label('Flat Number')

		# fig.tight_layout()

		median_image = np.median(combined_images, axis=0) # the median-combined image (across the flat sequence)

		# make a header to save some info about the flats that went into the combined flat
		hdr = fits.Header()
		hdr['COMMENT'] = f'N_flats = {n_files}'
		hdr['COMMENT'] = f'Median illumiation = {np.median(flat_medians)} ADU'
		hdr['COMMENT'] = 'Combined the following flats:'
		for i in range(n_files):
			hdr['COMMENT'] = flat_files[i]
		
		output_hdul = fits.HDUList([fits.PrimaryHDU(data=median_image, header=hdr)])


		# record: n_flats, median pixel illumination, filepaths of flat files 
		output_path = f'/data/tierras/flats/{date}_FLAT.fit'
		output_hdul.writeto(output_path, overwrite=True)
		set_tierras_permissions(output_path)

		# plt.close('all')
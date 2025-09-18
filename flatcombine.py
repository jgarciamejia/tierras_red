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

try:
  import astropy.utils.exceptions
  import astropy.io.fits as pyfits
except ImportError:
  import pyfits

import lfa

from fitsutil import *

def process_extension(imp, iext):
	hdr = imp.header
  
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
		blocks = np.median(bias_img,axis=1).reshape(8,128) 

		means = np.mean(blocks, axis=1)
		biaslev = np.repeat(means, 128)
	else:
		biaslev = 0

	if trimsec is not None:
		procimg = raw[trimsec[2]:trimsec[3],trimsec[0]:trimsec[1]] - biaslev[:,None]
	else:
		procimg = raw - biaslev

	if iext == 2:
		procimg = np.flipud(procimg) # top half of image is read in upside down under the current read-in scheme, so flip it here

	return procimg

if __name__ == "__main__":
	# Deal with the command line.
	ap = argparse.ArgumentParser()
	# ap.add_argument("filelist", metavar="file", nargs="+", help="input files")
	# ap.add_argument("-o", help="output file")
	ap.add_argument("-date", required=True, help="YYYYMMDD of the date on which flats were taken.")

  
	args = ap.parse_args()
  
	date = args.date

	filepath = f'/data/tierras/incoming/{date}/'
	filelist = glob(filepath+'*FLAT[0-9][0-9][0-9].fit')
	filelist = np.array(sorted(filelist, key=lambda x:int(x.split('FLAT')[-1].split('.')[0]))) # make sure the files are sorted 
	nf = len(filelist)

	fplist = [ pyfits.open(filename) for filename in filelist ]

	outhdus = []

	combined_images = []
	for i, filename in enumerate(filelist):
		print(f'Doing {filename} ({i+1} of {nf})')
		with fits.open(filename) as ifp:
			proc_imgs = [process_extension(ifp[j], j) for j in range(1, 3)]
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

	# median-normalize each individual flat 
	for i in range(len(combined_images)):
		norm = np.nanmedian(combined_images[i])
		combined_images[i] /= norm 

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

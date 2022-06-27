"""
Script to plot cumulative photometry as a function of Julian Date. 
Path, texp, and binsize currently hard-coded for TOI2013. 
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import glob
import re

import load_data as ld 
import bin_lc as bl
import detrending as dt
from jgmmedsig import *

# Only thing hard-coded right now nis the sizes and colors of the plot points

#mainpath = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output_Ryan_fixedaperture/'
def plot_cum_phot(mainpath,targetname,threshold=3,normalize='nightly',binsize=10,
				exclude_dates = None, show_plot=True,fig_name=False,*exclude_comps):
	# Initialize array to hold global rel fluxes
	glob_bjds = np.array([])
	glob_rel_flux = np.array([])

	# Initialize Figure 
	fig, ax = plt.subplots(figsize=(15,5))

	# Find all full filenames (including path) 
	filenames = np.sort(glob.glob(mainpath+'/**/'+targetname+'**xls',recursive=True))
	dfs = [None] * len(filenames)
	flags = [None] * len(filenames)
	# For each date
	for indf,filename in enumerate(filenames):
		if exclude_dates is not None:
			for date in exclude_dates:
				if date in filename:
					continue
		# Load lightcurve data
		date_pattern = '(\\d{8})' #not super robust
		obsdate = re.search(date_pattern,filename).group()
		print (obsdate)
		#break
		df,bjds,rel_flux,airmasses,widths,flag = ld.return_data_onedate(mainpath,targetname,obsdate,threshold,*exclude_comps)
		texp = df[' EXPTIME'][0]
		dfs[indf] = df
		flags[indf] = flag
		# Bin the light curve 
		xbin, ybin, _ = bl.bin_lc_binsize(bjds[flag], rel_flux[flag], binsize)

		# Normalize the light curve, if desired
		if normalize == 'none':
			ax.scatter(bjds[flag], rel_flux[flag], s=20, color='seagreen')
			ax.scatter(xbin, ybin, s=60, color='darkgreen', alpha = 0.9)
		elif normalize == 'nightly': 
			rel_flux /= np.median(rel_flux)
			ax.scatter(bjds[flag], rel_flux[flag], s=20, color='seagreen')
			ax.scatter(xbin, ybin, s=60, color='darkgreen', alpha = 0.9)
		elif normalize == 'global':
			glob_rel_flux = np.append(glob_rel_flux, rel_flux)
			glob_bjds = np.append(glob_bjds, bjds)

	if normalize == 'global':
			glob_rel_flux /= np.median(glob_rel_flux)
			ax.scatter(glob_bjds, glob_rel_flux, s=20, color='seagreen')
			glob_xbin, glob_ybin, _ = bl.bin_lc_binsize(glob_bjds, glob_rel_flux, binsize)
			ax.scatter(glob_xbin, glob_ybin, s=60, color='darkgreen', alpha = 0.9)

	# Config plot labels
	ax.set_xlabel("Time (BJD)")
	ax.set_ylabel("Normalized flux")
	
	#if normalize == 'none':
		#ax.set_title('No nightly or global norm., texp = {} min, Bin size = {} min'.format(texp/60,binsize))
	#elif normalize == 'nightly':
		#ax.set_title('Nightly norm., texp = {} min, Bin size = {} min'.format(texp/60,binsize))
	#elif normalize == 'global':
		#ax.set_title('Global norm., texp = {} min, Bin size = {} min'.format(texp/60,binsize))
	
	if show_plot:
		plt.show()
	if fig_name:
		fig.save_fig(fig_name)

	return dfs,filenames,flags



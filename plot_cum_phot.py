"""
Script to plot cumulative photometry as a function of Julian Date.
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

def plot_cum_phot(mainpath,targetname,threshold=3,normalize='nightly',binsize=10,
				exclude_dates = None, show_plot=True,fig_name=False,*exclude_comps):
	
	# Initialize array to hold global rel fluxes
	glob_bjds = np.array([])
	glob_rel_flux = np.array([])

	# Initialize Figure 
	fig, ax = plt.subplots(figsize=(15,5))

	# Find all full filenames (including path) 
	#filenames = np.sort(glob.glob(mainpath+'/**/'+targetname+'**xls',recursive=True))
	filenames = np.sort(glob.glob(mainpath+'/**/'+targetname+'**measurements.xls',recursive=True))
	#filenames = np.sort(glob.glob(mainpath+'/**/'+targetname+'**_r.xls',recursive=True)) #to read in corrected data
	dfs = [None] * len(filenames)
	flags = [None] * len(filenames)
	compss_used = [None] * len(filenames)
	allobsdates = []
	plotteddates = np.array([])
	# Load each file
	for indf,filename in enumerate(filenames):
		# Figure out date of file
		date_pattern = '(\\d{8})'
		possible_obsdates = re.findall(date_pattern,filename)
		for possible_obsdate in possible_obsdates:
			if is_plausible_date(possible_obsdate):
				obsdate = possible_obsdate 
		print (obsdate)
		allobsdates.append(obsdate)
		if np.any(exclude_dates == obsdate):
		#if exclude_dates and obsdate in exclude_dates:
			print ('excluded')
			continue
		else:
			df,bjds,rel_flux,airmasses,widths,flag,comps_used = ld.return_data_onedate(mainpath,targetname,obsdate,threshold,*exclude_comps,flag_output = False)
			texp = df[' EXPTIME'][0]
			dfs[indf] = df
			flags[indf] = flag 
			compss_used[indf] = comps_used
			plotteddates = np.append(plotteddates,obsdate)
			# Bin and normalize (if desired) the light curve
			if normalize == 'none':
				xbin, ybin, _ = bl.bin_lc_binsize(bjds[flag], rel_flux[flag], binsize)
				ax.scatter(bjds[flag], rel_flux[flag], s=20, color='seagreen')
				ax.scatter(xbin, ybin, s=60, color='darkgreen', alpha = 0.9)
			elif normalize == 'nightly': 
				rel_flux /= np.median(rel_flux)
				xbin, ybin, _ = bl.bin_lc_binsize(bjds[flag], rel_flux[flag], binsize)
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
	
	if normalize == 'none':
		ax.set_title('No nightly or global norm., texp = {} min, Bin size = {} min'.format(texp/60,binsize))
	elif normalize == 'nightly':
		ax.set_title('Nightly norm., texp = {} min, Bin size = {} min'.format(texp/60,binsize))
	elif normalize == 'global':
		ax.set_title('Global norm., texp = {} min, Bin size = {} min'.format(texp/60,binsize))
	
	if show_plot:
		plt.show()
	if fig_name:
		fig.save_fig(fig_name)

	return dfs,allobsdates,plotteddates,filenames,flags, compss_used

# Code below is from ChatGPT, with minor adaptations for years 

def is_plausible_date(date):
  # Convert the date to a string
  date_str = str(date)

  # Check if the string is the correct length
  if len(date_str) != 8:
    return False

  # Extract the year, month, and day from the date
  year = int(date_str[:4])
  month = int(date_str[4:6])
  day = int(date_str[6:])

  # JGM: Check if the year is a year Tierras has been active
  Tierras_years = [2020,2021,2022,2023]
  if year not in Tierras_years:
  	return False

  # Check if the month is between 1 and 12
  if not 1 <= month <= 12:
    return False

  # Check if the day is between 1 and the number of days in the month
  if not 1 <= day <= days_in_month(year, month):
    return False

  return True

def days_in_month(year, month):
  # This function returns the number of days in the given month of the given year

  # February has 29 days in a leap year
  if month == 2 and is_leap_year(year):
    return 29

  # The other months have a fixed number of days
  return [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month]

def is_leap_year(year):
  # This function returns True if the given year is a leap year, and False otherwise
  return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)





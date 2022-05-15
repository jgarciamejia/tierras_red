"""
Script to plot cumulative photometry as a function of Julian Date. 
Path, texp, and binsize currently hard-coded for TOI2013. 
"""

import csv 
import pdb#; pdb.set_trace()
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np
import math
from bin_lc import bin_lc_binsize

# Normalization per night
# if true, data median-normalized per night. 
# If false, all fluxes saved and global normalization carried out 
norm_each_night = False 
# Global normalization: can only be done if norm_each_night = False.
# However, if both are set to false, code prints unnormalized data
norm_globally = False

# Date array 
obsdates = ['20220504','20220505','20220506', '20220507', '20220508',
'20220509', '20220510', '20220511', '20220512']

# Transit day
#obsdates = ['20220509']

# Handy function
def medsig(a):
  median = np.median(a)
  sigma = 1.482602218505601*np.median(np.absolute(a-median))
  return(median, sigma)

# Scintillation
# Calculate scintillation and add to error bars
DIAM = 1300
HEIGHT = 2345
texp = 30 #seconds

# Set up plot
plt.rcParams.update({'font.size':12})
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.axisbelow'] = True

fig, (ax1,ax2) = plt.subplots(2,1, sharex = True, figsize=(15,5), gridspec_kw={'height_ratios': [2, 1]})
# Light Curve
ax1.grid(linestyle='dashed', linewidth=2, alpha=0.6)

# Initialize array to hold all rel fluxes
glob_jds = np.array([])
glob_rel_flux = np.array([])

for date in obsdates:
	path = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output_Ryan/TOI2013_'+date+'/'
	print (path)
	try:
		df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m2-I_measurements.xls')
	except FileNotFoundError:
		df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m3-I_measurements.xls')
	#pdb.set_trace()
	jds = df['J.D.-2400000'].to_numpy() 
	jds -= (2457000-2400000) 
	rel_flux = df['rel_flux_T1'].to_numpy() 
	rel_flux_err = df['rel_flux_err_T1'].to_numpy()
	airmass = df['AIRMASS'].to_numpy()

	# Flag outliers 
	medflux, sigflux = medsig (rel_flux)
	thisflag = np.absolute(rel_flux - medflux) < 5*sigflux
	jds = jds[thisflag]
	rel_flux = rel_flux[thisflag]
	rel_flux_err = rel_flux_err[thisflag]
	airmass = airmass[thisflag]

	# Add scintillation to flux err bars
	ssc = 0.09 * (DIAM/10.0) ** (-2.0/3.0) * np.power(airmass,3.0/2.0) * math.exp(-HEIGHT/8000.0) / np.sqrt(2*texp)
	ssc = ssc.astype(float)
	rel_flux_err = rel_flux_err.astype(float)
	rel_flux_err = np.hypot(rel_flux_err, ssc)

	# Bin the light curve 
	binsize = 10.0 #mins
	xbin, binned_flux, bins = bin_lc_binsize(jds, rel_flux, binsize)

	if norm_each_night: 
		# Median-normalize relative flux per night
		rel_flux /= np.median(rel_flux)
		# Plot 
		for i,line in enumerate(bins):
			ax1.scatter(jds, rel_flux, s=2, color='seagreen')
			#ax1.errorbar(jds, rel_flux, rel_flux_err, fmt='none',capsize = 3.5, color='seagreen', alpha = 0.8)
			ax2.scatter(xbin, binned_flux, s=20, color='darkgreen', alpha = 0.9)
	elif not norm_each_night:
		glob_rel_flux = np.append(glob_rel_flux, rel_flux)
		glob_jds = np.append(glob_jds, jds)

if not norm_each_night:
		if norm_globally:
			glob_rel_flux /= np.median(glob_rel_flux)
		ax1.scatter(glob_jds, glob_rel_flux, s=2, color='seagreen')
		binsize = 10.0 #mins
		glob_xbin, glob_binned_flux, glob_bins = bin_lc_binsize(glob_jds, glob_rel_flux, binsize)
		ax2.scatter(glob_xbin, glob_binned_flux, s=20, color='darkgreen', alpha = 0.9)


# Config plot grid+ticks
ax1.tick_params(direction='in', length=4, width=2)
ax2.grid(linestyle='dashed', linewidth=2, alpha=0.6)
#ax2.set_yticks([0.9985, 0.9990, 0.9995, 1.0000, 1.0005, 1.0010, 1.0015])
#ax2.set_yticklabels(['0.9985','0.9990','0.9995', '1.0000', '1.0005', '1.0010','1.0015'])
#ax2.set_ylim(.9985,1.0015)
ax2.tick_params(direction='in', length=4, width=2)
ax2.get_xaxis().get_major_formatter().set_useOffset(False)

# Config plot 
if norm_each_night:
	ax1.set_title('Separate Norm. Per Night, Bin size = {} min'.format(texp/60))
elif not norm_each_night:
	if norm_globally:
		ax1.set_title('Global Normalization (All Nights Together), Bin size = {} min'.format(texp/60))
	else:
		ax1.set_title('No nightly or global norm., Bin size = {} min'.format(texp/60))
ax2.set_title('Bin size = {} min'.format(binsize))
ax2.set_xlabel("Time (JD - 2457000)", size=15, color = 'black')
fig.text(0.07,0.48, "Normalized flux", size=15, ha='center', va = 'center', rotation = 'vertical')

plt.show()
#fig.savefig('TOI2013_plot_cum_phot.pdf')

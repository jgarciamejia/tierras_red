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

# Date array 
obsdates = ['20220504','20220505','20220506', '20220507']
for date in obsdates:
	path = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output_Ryan/TOI2013_'+date+'/'
	print (path)
	try:
		data = pd.read_table(path+'toi2013_'+date+'-Tierras_1m2-I_measurements.xls')
	except FileNotFoundError:
		data = pd.read_table(path+'toi2013_'+date+'-Tierras_1m3-I_measurements.xls')
	#pdb.set_trace()
	data_np = pd.DataFrame.to_numpy(data)
	jds = data_np[:,4] # - data_np[0,4]
	jds -= (2457000-2400000) 
	rel_flux = data_np[:,20]
	rel_flux_err = data_np[:,42]
	airmass = data_np[:,9]

	# Median-normalize relative flux
	rel_flux /= np.median(rel_flux)
	#rel_flux_err /= np.median(rel_flux_err)

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
	nbin = (jds[-1] - jds[0])*24*60 / binsize 
	bins = jds[0] + binsize * np.arange(nbin+1) / (24*60) #*
	wt = 1.0 / (np.square(rel_flux_err))
	ybn = np.histogram(jds, bins=bins, weights = rel_flux*wt)[0]
	ybd = np.histogram(jds, bins=bins, weights = wt)[0]
	wb = ybd > 0 
	binned_flux = ybn[wb] / ybd[wb]
	x = 0.5*(bins[0:-1] + bins[1:])
	x = x[wb]

	ax1.scatter(jds, rel_flux, s=2, color='seagreen')
	#ax1.errorbar(jds, rel_flux, rel_flux_err, fmt='none',capsize = 3.5, color='seagreen', alpha = 0.8)
	ax2.scatter(x[0:-1], binned_flux[0:-1], s=20, color='darkgreen', alpha = 0.9)


# Config grid+ticks
ax1.tick_params(direction='in', length=4, width=2)
ax2.grid(linestyle='dashed', linewidth=2, alpha=0.6)
#ax2.set_yticks([0.9985, 0.9990, 0.9995, 1.0000, 1.0005, 1.0010, 1.0015])
#ax2.set_yticklabels(['0.9985','0.9990','0.9995', '1.0000', '1.0005', '1.0010','1.0015'])
#ax2.set_ylim(.9985,1.0015)
ax2.tick_params(direction='in', length=4, width=2)
ax2.get_xaxis().get_major_formatter().set_useOffset(False)

# Config plot labels
ax1.set_title('Bin size = {} min'.format(texp/60))
ax2.set_title('Bin size = {} min'.format(binsize))
ax2.set_xlabel("Time (JD - 2457000)", size=15, color = 'black')
fig.text(0.07,0.48, "Normalized flux", size=15, ha='center', va = 'center', rotation = 'vertical')

plt.show()


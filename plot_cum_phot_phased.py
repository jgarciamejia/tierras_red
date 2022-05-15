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
import sys
from astropy.time import Time
import time
from datetime import datetime
import get_tess_jgm
from bin_lc import bin_lc_binsize
from medsig import medsig

def phase_fold(times, reftime, P):
	phased = (((times - reftime)) % P) / P
	phased[phased > 0.5] -= 1
	return phased


# Present time
utcnow = datetime.utcnow()
utcnow = utcnow.strftime("%Y-%m-%dT%H:%M:%S")
JDnow = Time(utcnow, format='isot').jd

# Observed dates to include
texp = 30 #seconds
obsdates = ['20220504','20220505','20220506', '20220507', '20220508',
						'20220509', '20220510', '20220511']

# Transit day
#obsdates = ['20220509']

# TOI2013b
t0_b = 2459320.05808 # transit center, BJD
P_b = 2.6162745 # Period, days

# TOI2013c (candidate)
t0_c = 2459072.44 # transit center, BJD
P_c = 14.303 # Period, days

###### TOI 2013 TESS Data
ticid_TOI2013 = 188589164

tess_BJD_24, tess_flux_24, tess_fluxerr_24, tess_sectors_24 = get_tess_jgm.get_tess_LC(ticid_TOI2013,24)
tess_BJD_25, tess_flux_25, tess_fluxerr_25, tess_sectors_25 = get_tess_jgm.get_tess_LC(ticid_TOI2013,25)

tess_BJD_24 += 2457000
tess_BJD_25 += 2457000
# bjd - 2457000

# Phase fold to planet b
#t0_b = 0.5
t0_b = 2458956.3959244997
tess_BJD_24 = phase_fold(tess_BJD_24, t0_b, P_b)
tess_BJD_25 = phase_fold(tess_BJD_25, t0_b, P_b)

#tess_BJD_24 = (((tess_BJD_24 - t0_b)) % P_b) / P_b
#tess_BJD_25 = ((tess_BJD_25 - t0_b) % P_b) / P_b
#pdb.set_trace()

###### TOI 2013 Tierras Data
#### PLOT Phase-folded light curve

# Define global figure params
plt.rcParams.update({'font.size':12})
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.axisbelow'] = True

# Define figure
#fig, (ax1,ax2) = plt.subplots(2,1, sharex = True, figsize=(15,5), gridspec_kw={'height_ratios': [2, 1]})
fig, ax1 = plt.subplots(figsize=(15,5))
ax2 = ax1

for date in obsdates:
	path = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output_Ryan/TOI2013_'+date+'/'
	try:
		df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m2-I_measurements.xls')
	except FileNotFoundError:
		df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m3-I_measurements.xls')
	jds = df['J.D.-2400000'].to_numpy() +2400000 
	rel_flux = df['rel_flux_T1'].to_numpy()
	rel_flux_err = df['rel_flux_err_T1'].to_numpy()
	airmass = df['AIRMASS'].to_numpy()

	#pdb.set_trace()
	# bin jds to 2 min
	binsize = 2.0
	x2min, y2min,_ = bin_lc_binsize(jds,rel_flux, binsize)
	# phase fold jds
	jds = phase_fold(jds, t0_b, P_b)
	#(((jds - t0_b)) % P_b) / P_b
	# phase fold binned data 
	#x2min = (((x2min - t0_b)) % P_b) / P_b
	x2min = phase_fold(x2min, t0_b, P_b)
	# 

	# Median-normalize relative flux
	rel_flux /= np.median(rel_flux)
	y2min /= np.median(y2min)
	#rel_flux_err /= np.median(rel_flux_err)

	# Flag outliers - commented out for now 
	# medflux, sigflux = medsig (rel_flux)
	# thisflag = np.absolute(rel_flux - medflux) < 5*sigflux
	# jds = jds[thisflag]
	# rel_flux = rel_flux[thisflag]
	# rel_flux_err = rel_flux_err[thisflag]
	# airmass = airmass[thisflag]
	
	print (jds)
	#pdb.set_trace()
	

	if date == obsdates[0]:
		#ax1.scatter(jds, rel_flux, s=3, color='seagreen', zorder = 3, alpha = 0.2, label = 'Tierras, 0.5 min bin')
		ax1.scatter(x2min, y2min, s=7, color='darkgreen', zorder = 4, label = 'Tierras, 2 min bin')
		ax1.scatter(tess_BJD_24, tess_flux_24, s=1, color='royalblue', zorder = 1, alpha = 0.5, label = 'Sector 24, 2 min bin')
		ax1.scatter(tess_BJD_25, tess_flux_25, s=1, color='indianred', zorder = 2, alpha = 0.5, label = 'Sector 25, 2 min bin')
		#ax1.errorbar(jds, rel_flux, rel_flux_err, fmt='none',capsize = 3.5, color='seagreen', alpha = 0.8)
		#ax2.scatter(x[0:-1], binned_flux[0:-1], s=20, color='darkgreen', alpha = 0.9)
	else: 
		ax1.scatter(jds, rel_flux, s=3, color='seagreen', zorder = 3)
		ax1.scatter(x2min, y2min, s=7, color='darkgreen', zorder = 4)
		ax1.scatter(tess_BJD_24, tess_flux_24, s=1, color='royalblue', zorder = 1, alpha = 0.3)
		ax1.scatter(tess_BJD_25, tess_flux_25, s=1, color='indianred', zorder = 2, alpha = 0.3)

# Plot global normalized rel_flux 
#ax1.scatter(jds, rel_flux, s=3, color='seagreen', zorder = 3, alpha = 0.2, label = 'Tierras, 0.5 min bin')

# Config grid+ticks
ax1.tick_params(direction='in', length=4, width=2)
ax1.grid(linestyle='dashed', linewidth=2, alpha=0.6)
ax1.get_xaxis().get_major_formatter().set_useOffset(False)

# Config plot labels
#ax1.set_title('Bin size = {} min'.format(binsize/60))
ax1.set_xlabel("Orbital Phase", size=15, color = 'black')
ax1.set_ylabel("Normalized flux", size=15, ha='center', va = 'center', rotation = 'vertical')
ax1.set_ylim(0.995, 1.005)
ax1.legend()
plt.show()
#pdb.set_trace()
#fig.savefig('TOI2013_plot_cum_phot_phased_planetc.pdf')




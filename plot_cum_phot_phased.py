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

# Present time
utcnow = datetime.utcnow()
utcnow = utcnow.strftime("%Y-%m-%dT%H:%M:%S")
JDnow = Time(utcnow, format='isot').jd

# Medsig function
def medsig(a):
  median = np.median(a)
  sigma = 1.482602218505601*np.median(np.absolute(a-median))
  return(median, sigma)

# Scintillation constants
DIAM = 1300
HEIGHT = 2345
texp = 30 #seconds

####### PLANET C (CANDIDATE)
# Planet c candidate transit parameters
t0_c = 2459072.44 # transit center, Barycentric Julian date in barycentric dynamical time standard 
t0_c_perr = 0.41 
t0_c_merr = 0.41
P_c = 14.303 # Period, days
P_c_perr = 0.034
P_c_merr = 0.035

# Planet c transit window and cumulative uncertainties
n = 300 # number of future transits to predict
t0s_c = t0_c + (np.arange(n)*P_c) #BJD
t0s_c_perr = t0_c_perr + (np.arange(n)*P_c_perr)
t0s_c_merr = t0_c_merr + (np.arange(n)*P_c_merr)
# convert to isot format
t0s_c_isot = Time(t0s_c, format='jd').isot

# Isolate transit times for future obs
future_ind = np.argwhere(t0s_c-JDnow >=0)
t0s_c_future = t0s_c_isot[future_ind]

# Calculate cumulative uncertainty in the present
print ('Cumulative uncertainties, today')
print (t0s_c_perr[future_ind[0]], t0s_c_merr[future_ind[0]]) #days
#print (t0s_c_perr[future_ind[0]]*24*60, t0s_c_merr[future_ind[0]]*24*60) #minutes

###### TOI 2013 TESS Data
ticid_TOI2013 = 188589164
tess_BJD_24, tess_flux_24, tess_fluxerr_24, tess_sectors_24 = get_tess_jgm.get_tess_LC(ticid_TOI2013,24)
tess_BJD_25, tess_flux_25, tess_fluxerr_25, tess_sectors_25 = get_tess_jgm.get_tess_LC(ticid_TOI2013,25)
# Phase fold
tess_BJD_24 = ((tess_BJD_24 - t0_c) / P_c) % 1
tess_BJD_25 = ((tess_BJD_25 - t0_c) / P_c) % 1

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
# Date array 
obsdates = ['20220504','20220505','20220506', '20220507', '20220508']
for date in obsdates:
	path = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output_Ryan/TOI2013_'+date+'/'
	try:
		df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m2-I_measurements.xls')
	except FileNotFoundError:
		df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m3-I_measurements.xls')
	jds = df['J.D.-2400000'].to_numpy() 
	jds = ((jds - t0_c) / P_c) % 1 # phase-folded 
	rel_flux = df['rel_flux_T1'].to_numpy() 
	rel_flux_err = df['rel_flux_err_T1'].to_numpy()
	airmass = df['AIRMASS'].to_numpy()

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

	if date == obsdates[0]:
		ax1.scatter(jds, rel_flux, s=3, color='seagreen', zorder = 3, label = 'Tierras')
		ax1.scatter(tess_BJD_24, tess_flux_24, s=1, color='royalblue', zorder = 1, alpha = 0.2, label = 'Sector 24')
		ax1.scatter(tess_BJD_25, tess_flux_25, s=1, color='indianred', zorder = 2, alpha = 0.2, label = 'Sector 25')
		#ax1.errorbar(jds, rel_flux, rel_flux_err, fmt='none',capsize = 3.5, color='seagreen', alpha = 0.8)
		#ax2.scatter(x[0:-1], binned_flux[0:-1], s=20, color='darkgreen', alpha = 0.9)
	else: 
		ax1.scatter(jds, rel_flux, s=3, color='seagreen', zorder = 3)
		ax1.scatter(tess_BJD_24, tess_flux_24, s=1, color='royalblue', zorder = 1, alpha = 0.2)
		ax1.scatter(tess_BJD_25, tess_flux_25, s=1, color='indianred', zorder = 2, alpha = 0.2)

# Config grid+ticks
ax1.tick_params(direction='in', length=4, width=2)
ax1.grid(linestyle='dashed', linewidth=2, alpha=0.6)
ax1.get_xaxis().get_major_formatter().set_useOffset(False)

# Config plot labels
ax1.set_title('Bin size = {} min'.format(texp/60))
ax1.set_xlabel("Orbital Phase", size=15, color = 'black')
ax1.set_ylabel("Normalized flux", size=15, ha='center', va = 'center', rotation = 'vertical')
ax1.legend()
plt.show()
fig.savefig('TOI2013_plot_cum_phot_phased.pdf')


# Sanity check: 
# ###### PLANET B 
# # Values from Kemmer et al., 2022
# # Planet b transit parameters
# t0_b = 2459320.05808 # transit center, Barycentric Julian date in barycentric dynamical time standard 
# t0_b_perr = 0.00018 
# t0_b_merr = 0.00019
# P_b = 2.6162745 # Period, days
# P_b_perr = 2.9e-6
# P_b_merr = 3e-6

# # Planet b transit window and cumulative uncertainties
# n = 300 # number of future transits to predict
# t0s_b = t0_b + (np.arange(n)*P_b) #BJD
# t0s_b_perr = t0_b_perr + (np.arange(n)*P_b_perr)
# t0s_b_merr = t0_b_merr + (np.arange(n)*P_b_merr)
# # convert to isot format
# t0s_b_isot = Time(t0s_b, format='jd').isot

# # Isolate transit times for future obs
# utcnow = datetime.utcnow()
# utcnow = utcnow.strftime("%Y-%m-%dT%H:%M:%S")
# JDnow = Time(utcnow, format='isot').jd
# future_ind = np.argwhere(t0s_b-JDnow >=0)
# t0s_b_future = t0s_b[future_ind]

# # Calculate cumulative uncertainty in the present
# print (t0s_b_perr[future_ind[0]]*24*60, t0s_b_merr[future_ind[0]]*24*60) #minutes 


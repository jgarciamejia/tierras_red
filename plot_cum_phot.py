import csv 
import pdb#; pdb.set_trace()
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import numpy as np
import math
from medsig import *

# Scintillation
# Calculate scintillation and add to error bars
DIAM = 1300
HEIGHT = 2345
texp = 25 # seconds
ssc = 0.09 * (DIAM/10.0) ** (-2.0/3.0) * np.power(airmass,3.0/2.0) * math.exp(-HEIGHT/8000.0) / np.sqrt(2*texp)
ssc = ssc.astype(float)

# Set up plot
plt.rcParams.update({'font.size':12})
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.axisbelow'] = True

fig, (ax1,ax2) = plt.subplots(2,1, sharex = True, figsize=(15,5), gridspec_kw={'height_ratios': [2, 1]})
# Light Curve
ax1.grid(linestyle='dashed', linewidth=2, alpha=0.6)

# Date array 
obsdates = ['20220504','20220505','20220506']
for date in obsdates:
	path = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output_Ryan/TOI2013_'+date
	print (path)
	data = pd.read_table(path+'toi2013_'+date+'20220504-Tierras_1m2-I_measurements.xls')
	pdb.set_trace()
	data_np = pd.DataFrame.to_numpy(data)
	jds = data_np[:,4] - data_np[0,4]
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

	ax1.errorbar(jds*24, rel_flux, rel_flux_err, fmt='o', capsize = 3.5, color='seagreen', alpha = 0.8, label = '0.55 min bin')
	ax2.scatter(x[0:-1]*24, binned_flux[0:-1], color='darkgreen', alpha = 0.9, label = '10 min bin')


# Config plot 
ax1.tick_params(direction='in', length=4, width=2)
#ax1.set_ylabel("Normalized flux", size=15)
fig.text(0.07,0.48, "Normalized flux", size=15, ha='center', va = 'center', rotation = 'vertical')
ax1.legend(loc = 'lower left')
ax2.grid(linestyle='dashed', linewidth=2, alpha=0.6)
#ax2.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax2.set_yticks([0.9990,0.9995, 1.0000, 1.0005, 1.0010])
ax2.set_yticklabels(['0.9990','0.9995', '1.0000', '1.0005', '1.0010'])
ax2.tick_params(direction='in', length=4, width=2)
ax2.set_xlabel("Time (hours)", size=15, color = 'black')
ax2.set_ylim(.9985,1.0015)
ax2.legend(loc = 'lower left')
plt.show()


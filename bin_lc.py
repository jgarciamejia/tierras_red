import numpy as np 
import pdb
import pandas as pd

"""
Functions to bin light curve. 
Take:
jds - Julian Date
rel_flux - Relative target flux. Can also be raw counts. 
nbins - desired numebr of bins
binsize - desired bin size, in minutes

Return: 
xbin, ybin - binned jds with bin at center of jd range 
covered in each bin, binned fluxes weighted by sum of rel_fluxes 
inside the bin divided by number of points in the bin. 
"""

def bin_lc_nbins(jds,rel_flux, nbins):
	bins = jds[0] + binsize * np.arange(nbins) / (24*60) 
	ybin_num = np.histogram(jds, bins=bins, weights = rel_flux)[0]
	ybin_denom = np.histogram(jds, bins=bins)[0]
	ybin_denom[ybin_num == 0] = 1.0
	ybin = ybin_num / ybin_denom
	xbin = 0.5*(bins[1:]+bins[:-1])
	return xbin, ybin

def bin_lc_binsize(jds,rel_flux, binsize):
	nbins = (jds[-1] - jds[0])*24*60 / binsize 
	bins = jds[0] + binsize * np.arange(nbins) / (24*60) 
	ybin_num = np.histogram(jds, bins=bins, weights = rel_flux)[0]
	ybin_denom = np.histogram(jds, bins=bins)[0]
	ybin_denom[ybin_num == 0] = 1.0
	ybin = ybin_num / ybin_denom
	xbin = 0.5*(bins[1:]+bins[:-1])
	return xbin, ybin

date = '20220509'
path = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output_Ryan/TOI2013_'+date+'/'
print (path)
try:
	df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m2-I_measurements.xls')
except FileNotFoundError:
	df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m3-I_measurements.xls')
#pdb.set_trace()
jds = df['J.D.-2400000'].to_numpy() 
jds -= (2459709-2400000) 
rel_flux = df['rel_flux_T1'].to_numpy()

# Bin light curve by number of bins 
# Bin light curve by time 
# plot and compare results

pdb.set_trace()

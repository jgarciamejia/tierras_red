import numpy as np 
import pdb
import pandas as pd
import matplotlib.pyplot as plt

"""
Functions to bin light curve. 
Take:
jds - Julian Date
rel_flux - Relative target flux. Can also be raw counts. 
nbins - desired numebr of bins
binsize - desired bin size, in minutes

Return: 
xbin, ybin, bins - binned jds with bin at center of jd range 
covered in each bin, binned fluxes weighted by sum of rel_fluxes 
inside the bin divided by number of points in the bin, left edge of bins 
"""

def bin_lc_nbins(jds,rel_flux, nbins):
	bins = jds[0] + binsize * np.arange(nbins+1) / (24*60)# nbins+1 needed to ensure enough bins are created. otherwise points will be lost 
	ybin_num = np.histogram(jds, bins=bins, weights = rel_flux)[0]
	ybin_denom = np.histogram(jds, bins=bins)[0]
	#ybin_denom[ybin_num == 0] = 1.0
	validbin = ybin_denom > 0 # if bin is not occupied, set to False
	ybin = ybin_num[validbin] / ybin_denom[validbin] # Mask to only include occupied bins
	xbin = 0.5*(bins[1:]+bins[:-1])
	return xbin[validbin], ybin, bins

def bin_lc_binsize(jds,rel_flux, binsize):
	nbins = (jds[-1] - jds[0])*24*60 / binsize 
	bins = jds[0] + binsize * np.arange(nbins+1) / (24*60) 
	ybin_num = np.histogram(jds, bins=bins, weights = rel_flux)[0]
	ybin_denom = np.histogram(jds, bins=bins)[0]
	#ybin_denom[ybin_num == 0] = 1.0
	validbin = ybin_denom > 0 # if bin is not occupied, set to False
	ybin = ybin_num[validbin] / ybin_denom[validbin] # Mask to only include occupied bins
	xbin = 0.5*(bins[1:]+bins[:-1])
	return xbin[validbin], ybin, bins


# # TEST CODE
# date = '20220504'
# path = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output_Ryan/TOI2013_'+date+'/'
# print (path)
# try:
# 	df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m2-I_measurements.xls')
# except FileNotFoundError:
# 	df = pd.read_table(path+'toi2013_'+date+'-Tierras_1m3-I_measurements.xls')
# #pdb.set_trace()
# jds = df['J.D.-2400000'].to_numpy() 
# jds += 2400000 
# rel_flux = df['rel_flux_T1'].to_numpy()
# # Bin light curve by time, plot contours of bins superimposed on 
# # light curve.  
# x,y,xlines = bin_lc_binsize(jds,rel_flux, 10.0)
# plt.scatter(jds, rel_flux)
# plt.scatter(x,y)
# for i,line in enumerate(xlines):
# 	plt.axvline(line, color='blue', alpha=0.2, linestyle='--') # bins defined on leftedge of bin
# 	try:
# 		plt.axvline(x[i], color='orange', alpha=0.2, linestyle='--')# bins defined in the center of the bin 
# 	except IndexError:
# 		continue
# plt.show()
# plot and compare results


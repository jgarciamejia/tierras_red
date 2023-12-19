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

def bin_lc_nbins(jds,rel_flux, nbins): #not finished
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

def rms_vs_binsize(jds,rel_flux,binsizes):
	# Compute rms vs bin size for rel flux
	rmss = np.array([])
	for binsize in binsizes:
		xbin,ybin,_ = bin_lc_binsize(jds,rel_flux,binsize)
		rmss = np.append(rmss, np.std(ybin))
	return rmss

def ep_bin(x, y, tbin): # Written by Emily Pass
    """ Basic binning routine, assumes tbin shares the units of x """
    bins = np.arange(np.min(x), np.max(x), tbin)
    binned = []
    binned_e = []
    if len(bins) < 2:
        return [np.nan], [np.nan], [np.nan]
    for ii in range(len(bins)-1):
        use_inds = np.where((x < bins[ii + 1]) & (x > bins[ii]))[0]
        if len(use_inds) < 1:
            binned.append(np.nan)
            binned_e.append(np.nan)
        else:
            binned.append(np.median(y[use_inds]))
            binned_e.append((np.percentile(y[use_inds], 84)-np.percentile(y[use_inds], 16))/2.)

    return bins[:-1] + (bins[1] - bins[0]) / 2., np.array(binned), np.array(binned_e)


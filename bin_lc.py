import numpy as np 
import pdb

"""
Function to bin light curve 
"""

# Bin light curve by number of bins 

# Bin light curve by time 
date = '20220509'
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

pdb.set_trace()

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

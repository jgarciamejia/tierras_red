#!/usr/bin/env python
import numpy as np
import pdb
import pandas as pd 
from astropy import units as u
import matplotlib.pyplot as plt
import sys
from funcs import mag_to_flux_ratio
#plt.rcParams['text.usetex'] = True


# # Load csv table from NASA Exo Archive 
# filename = sys.argv[0] #'PS_2022.04.19_11.38.20.csv'
# df = pd.read_csv(sys.argv[0], skiprows=61) #known planets around M dwarfs
# planet_radii = df['pl_rade'].to_numpy()*u.Rearth # R_Earth
# planet_period = df['pl_']
# stellar_radii = df['st_rad'].to_numpy()*u.Rsun #R_Sun
# stellar_mass = df['st_mass'].to_numpy()*u.Msun #M_Sun

#pdb.set_trace()
# Load csv table from TESS Exo Archive 
#filename = sys.argv[0] #'PS_2022.04.19_11.38.20.csv'
df = pd.read_csv('TOI_2022.05.04_15.42.25.csv',skiprows=51)
#df = pd.read_csv(filename, skiprows=51) #known planets around M dwarfs
planet_radii = df['pl_rade'].to_numpy()*u.Rearth # R_Earth
planet_period = df['pl_orbper'].to_numpy()*u.d #days
stellar_radii = df['st_rad'].to_numpy()*u.Rsun #R_Sun
#stellar_mass = df['st_mass'].to_numpy()*u.Msun #M_Sun
TESS_mags = df['st_tmag'].to_numpy()
TIDs = df['tid'].to_numpy()

# Calculate Signal of Earth & Ganymede around stars
Rganymede = 0.41*u.Rearth
signal_Earth = (1*u.Rearth / stellar_radii.to(u.Rearth))**2 #*1e6 #ppm
signal_Ganymede = (Rganymede / stellar_radii.to(u.Rearth))**2 #*1e6 #ppm

# Calculate Noise 
ref_noise = 250/1e6 #ppm
TESS_mag_LHS2913 = 9.90035
flux_ratios_targs = mag_to_flux_ratio(TESS_mags,TESS_mag_LHS2913)
noise = ref_noise / (np.sqrt(flux_ratios_targs))

pdb.set_trace()
# SNR 
SNR = signal_Earth / noise 

# Search SNR list 
np.where(TIDs == 44313455)
print (noise[np.where(TIDs == 44313455)])
# add new value to data base, so you are able to query for all stars with 
# this declination, this RA range, stars smaller than 0.4 radii 
# give me all those and rank them by signal to noise 
# there will be 10 

# 1 03 2 nearby m dwarfs with transiting planet and only one sector of tess data 
# we dont win at 0.5 day, we win at 7 days 
# TOI 2013 188589164()  1 ppt -> SAFE but with two transits.  
# TOI 4438 22233480 7 day, 3 ppt -> 
# TOI 2142 44313455 10 day -> not great airmass 
# TOI 2136 336128819 


#SNR 

#2136 1.22
#2142 1.48
#4438 1.27 
#2013 2.67


# Looked at MAST for TESS data 

# 2142 - clean and weak 
# 4438 - clean and strong  

# Then looked at the Spectra to make sure it is inactive in H Alpha 

# no H alpha in emission 
# sodium is nice and broad. Pressure broadening confirms it is a dwarf objects
# very narrow for giant stars 

# checked TESS transit data by eye

# https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html

# Take TESS data and inspect 
# can you do better than they can?

# 2013 
# get PDCSAP light curve and compare them 
# 

# filter calibration: Dittman 
# 






# plt.scatter(stellar_radii, signal_Earth, label = 'R_p = R_Earth')
# plt.scatter(stellar_radii, signal_Ganymede, label = 'R_p = R_Ganymede')
# plt.axhline(200, color='red')
# plt.xlabel('Stellar Radius (Rsun)')
# plt.ylabel('$(R_p/R_s)^2 (ppm)')
# plt.show()

#pdb.set_trace()

# Noise 
# noise in unit of time (ten minutes)
# for star in poster, you get RMS of 250 ppm. 
# scale it: take magnitude, calculate the flux that it is brighter

# magnitude in I band or TESS mag (closest). 
# calculate flux from magnitude 
# S/N is sqrt of flux ratio 
# assuming photon noise dominated 
# 


# Calculate Hill Sphere 
















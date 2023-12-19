#!/usr/bin/env python

import numpy as np
from astropy.io import fits 
import sys
import re 
print ('Checking that all files in the light curve have the same exposure time')
import matplotlib.pyplot as plt

sample_filename = sys.argv[1]
DATE = re.split('\.',sample_filename)[0]
TARGET = re.split('\.', sample_filename)[2]

exptimes = np.array([])
filenums = np.array([])

for ifile,fits_filename in enumerate(sys.argv[1:]):
    FILENUM = int(re.split('\.', fits_filename)[1])
    filenums = np.append(filenums, FILENUM)
    #load primary header, containing exptime key 
    hdr = fits.getheader(fits_filename,0)
    file_exptime = hdr['EXPTIME']
    exptimes = np.append(exptimes, file_exptime)
    #if file_exptime != desired_exptime:
    #    print ('{0} has exp time = {1} sec'.format(fits_filename,file_exptime))
if np.all(exptimes == exptimes[0]):
    print ('All files in the stack have the same exposure time')
elif not np.all(exptimes == exptimes[0]):
    print ('Files in the stack have different exposure times')
    
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,.6,.75])

ax.plot(filenums, exptimes, '.', color='black')
ax.set_xlabel('File Number (in the name)')
ax.set_ylabel('Exposure Time (seconds)')
fig.savefig("{0}.{1}_exptimes.pdf".format(DATE,TARGET))
plt.show()

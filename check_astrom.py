#!/usr/bin/env python

# To use, pass the script: maxSTDCRMS minNUMBRMS filenames 

import numpy as np
import sys
import os
import re
import pdb
import glob

from scipy import stats
from astropy.io import fits 
import matplotlib.pyplot as plt

print ('Checking astrometric solution on plate solved files...')

sample_filename = sys.argv[3]
DATE = re.split('\.',sample_filename)[0]
TARGET = re.split('\.', sample_filename)[2]

nfiles = len(sys.argv[3:])
nstdcrms, nnumbrms = 0,0
maxrms, minnum = float(sys.argv[1]),float(sys.argv[2])

stdcrms_lst = np.array([])
numbrms_lst = np.array([])
exptimes = np.array([])
flagged_files = np.array([])

for ifile,fits_filename in enumerate(sys.argv[3:]):
    if '_red' in fits_filename:
        hdr_ind = 0
    elif '_red' not in fits_filename:
        hdr_ind = 1
    hdr = fits.getheader(fits_filename,hdr_ind)
    # read out astrometric fit coordinate rms (arcsec)
    # and number of astrometric standards used. 
    try:
        stdcrms, numbrms, exptime = hdr['STDCRMS'], hdr['NUMBRMS'], hdr['EXPTIME']
    except KeyError:
        print ('Astrom failed for:')
        print (fits_filename)
        flagged_files = np.append(flagged_files, fits_filename)
        continue
    if stdcrms > maxrms:
        print ('stdcrms > {} for:'.format(maxrms))
        print (fits_filename,stdcrms)
        flagged_files = np.append(flagged_files, fits_filename)
    if numbrms < minnum:
        print ('numbrms < {} for:'.format(minnum))
        print (fits_filename,numbrms)
        flagged_files = np.append(flagged_files, fits_filename)
    if stdcrms <= maxrms and numbrms >= minnum:
        stdcrms_lst = np.append(stdcrms_lst,stdcrms)
        numbrms_lst = np.append(numbrms_lst, numbrms)
    exptimes = np.append(exptimes, exptime)

# Add any files with different exposure time to majority to flagged list
texp = stats.mode(exptimes)[0][0]
print ('Stack Exposure Time: {} s'.format(texp))
for ifile, fits_filename in enumerate(sys.argv[3:]):
    if exptimes[ifile] != texp:
        print ('texp = {} s for:'.format(exptimes[ifile]))
        print (fits_filename)
        flagged_files = np.append(flagged_files,fits_filename)
print ('Exposure Time Checks Done')

# Save flagged file list for future ref
if len(flagged_files) >= 1:
    print ('saved a list of flagged files')
elif len(flagged_files) == 0:
    print ('no flagged files')
np.savetxt('{}.{}.flagged_files.txt'.format(DATE,TARGET),np.unique(flagged_files),fmt='%s')

#Make excluded directory and move flagged files there
os.system('mkdir excluded')
path = os.getcwd()
for flaggedfile in flagged_files:
    if not os.path.exists(os.path.join(path,"excluded/",flaggedfile)):
        os.system('mv '+ flaggedfile +"excluded/")

print ('Astrometry Checks Done')


fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,7))
fig.suptitle('Astrometry Evaluation')

ax1.hist(stdcrms_lst, 20)
ax1.set_xlabel('Astrometric fit coord rms (arcsec)')
ax1.set_ylabel('Number of Exposures')

ax2.hist(numbrms_lst, 20)
ax2.set_ylabel('Number of Exposures')
ax2.set_xlabel('Number of astrometric standards used')
fig.savefig("{0}.{1}_astrom_hist.pdf".format(DATE,TARGET))
plt.show()

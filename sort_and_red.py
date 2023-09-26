#!/usr/bin/env python

from __future__ import print_function

import logging
import argparse 
import numpy 
import sys
import os
import re
import glob
import pdb

import numpy as np
from scipy import stats
from astropy.io import fits 
#import matplotlib.pyplot as plt

from imred import *

import logging

def create_directories(basepath,date,target,folder1):
    datepath = os.path.join(basepath,date)
    targetpath = os.path.join(datepath,target)
    folder1path = os.path.join(targetpath,folder1)

    if not os.path.exists(datepath):
        logging.info('Creating {} Directory'.format(datepath))
        os.system('mkdir {}'.format(datepath))
    
    if not os.path.exists(targetpath):
        logging.info('Creating {} Directory'.format(targetpath))
        os.system('mkdir {}'.format(targetpath))
        os.system('mkdir {}'.format(folder1path))

    return folder1path

def make_filelist(basepath,date,target):
    fullpath = os.path.join(basepath,date)
    return sorted([os.path.join(fullpath,f) for f in os.listdir(fullpath) if target+'.fit' in f])


# Deal with command line
ap = argparse.ArgumentParser()
ap.add_argument("-f", help="Flat file with which to reduce data.")
ap.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")
ap.add_argument("-ffname", required=True, help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
args = ap.parse_args()

# Access observation info
date = args.date
target = args.target
ffname = args.ffname

# Define base paths    
ipath = '/data/tierras/incoming'
fpath = '/data/tierras/flattened'
lcpath = '/data/tierras/lightcurves'

# Create flattened file and light curve directories 
ffolder = create_directories(fpath,date,target,ffname)
lcfolder = create_directories(lcpath,date,target,ffname)

# Set up logger
logfile = os.path.join(ffolder,'{}.{}.redlog.txt'.format(date,target))
#logfile = '{}.{}.redlog.txt'.format(date,target)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=logfile, level=logging.INFO)

# Get list of files to be reduced
filelist = make_filelist(ipath,date,target)

# Reduce each FITS file from date and target
logging.info('Reducing {} FITS files...'.format(len(filelist)))
irobj = imred(args.f)
rfilelist = []
for ifile,filename in enumerate(filelist):
    logging.info(filename)
    ohl = irobj.read_and_reduce(filename,stitch=True)
    basename = os.path.basename(filename)
    rfilename = re.sub('\.fit','',basename)+'_red.fit'
    rfilename = os.path.join(ffolder,rfilename)
    ohl.writeto(rfilename,overwrite=True)
    rfilelist.append(rfilename)
    logging.info(filename+ " -> " +rfilename)

# Exclude files where ASTROM solution fails and exptime diff. to mode of stack
logging.info('Checking astrometric solution on plate solved files...')

exptimes = np.array([])
badfiles = np.array([])
stdcrms_lst = np.array([])
numbrms_lst = np.array([])

for irfile,rfilename in enumerate(rfilelist):
    if '_red' in rfilename:
        hdr_ind = 0
    elif '_red' not in rfilename:
        hdr_ind = 1
    hdr = fits.getheader(rfilename,hdr_ind)

    #read out exp time BEFORE astrom params
    exptime = hdr['EXPTIME']
    exptimes = np.append(exptimes, exptime)

    # read out astrometric fit coordinate rms (arcsec)
    # and number of astrometric standards used. 
    try:
        stdcrms, numbrms = hdr['STDCRMS'], hdr['NUMBRMS']
    except KeyError:
        logging.info('Astrom failed for:')
        logging.info(rfilename)
        badfiles = np.append(badfiles,rfilename)
        continue
    stdcrms_lst = np.append(stdcrms_lst,stdcrms)
    numbrms_lst = np.append(numbrms_lst, numbrms)

logging.info('Astrometry checks: Done.')

# Add any files with different exposure time to flagged list
texp = stats.mode(exptimes)[0][0]
logging.info('Stack texp = {} s.'.format(texp))
for irfile,rfilename in enumerate(rfilelist):
    if exptimes[ifile] != texp:
        logging.info('texp = {} s for:'.format(exptimes[ifile]))
        logging.info(rfilename)
        badfiles = np.append(badfiles,rfilename)
logging.info('Exposure time checks: Done.')

# Save flagged file list for future ref
if len(badfiles) >= 1:
    logging.info('Saved a list of flagged files.')
elif len(badfiles) == 0:
    logging.info('No files were flagged.')
np.savetxt(os.path.join(ffolder,'{}.{}.flagged_files.txt'.format(date,target)),np.unique(badfiles),fmt='%s')

# Make excluded directory and move flagged files there
excfolder = os.path.join(ffolder,'excluded/')
logging.info(excfolder)
os.system('mkdir '+excfolder)
for badfile in badfiles:
    os.system('mv ' + badfile + ' '+ excfolder)

# Save histogram with astrom solution stdv and number of stars used
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,7))
fig.suptitle('Astrometry Evaluation: {} {}'.format(date,target))

ax1.hist(stdcrms_lst, 20)
ax1.set_xlabel('Astrometric fit coord rms (arcsec)')
ax1.set_ylabel('Number of Exposures')

ax2.hist(numbrms_lst, 20)
ax2.set_ylabel('Number of Exposures')
ax2.set_xlabel('Number of astrometric standards used')
histogram = os.path.join(ffolder,"{}.{}_astrom_hist.pdf".format(date,target))
fig.savefig(histogram)
logging.info('Saved two histograms summarizing the astrometric solution.')

# Send log file and STDRMS/NUMBRMS .pdf file to email
subject = '[Tierras]_Data_Reduction_Report:{}_{}'.format(date,target)
append = '{} {}'.format(logfile,histogram)
#append = '{}'.format(histogram)
email = 'juliana.garcia-mejia@cfa.harvard.edu'
os.system('echo | mutt {} -s {} -a {}'.format(email,subject,append))

print ('Data reduction done.')
print ('Data reduction report sent to {}'.format(email))

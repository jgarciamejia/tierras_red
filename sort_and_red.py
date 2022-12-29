#!/usr/bin/env python

from __future__ import print_function

import logging
import argparse 
import numpy 
import sys
import os
import re
import glob

import numpy as np
from scipy import stats
from astropy.io import fits 
import matplotlib.pyplot as plt

from imred import *

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

# Set up logger with FileHandler and StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create FileHandler
file_handler = logging.FileHandler('{}.{}.redlog.txt'.format(date,target))
file_handler.setLevel(logging.INFO)

#Create StreamHandler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

#Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - (levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add both handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

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

# Create image reduction object
irobj = imred(args.f)

# Make list of files for imred
filelist = make_filelist(ipath,date,target)

# Reduce each FITS file from date and target
print ('Reducing {} FITS files...'.format(len(filelist)))
rfilelist = []
for ifile,filename in enumerate(filelist):
    print (filename)
    ohl = irobj.read_and_reduce(filename,stitch=True)
    basename = os.path.basename(filename)
    rfilename = re.sub('\.fit','',basename)+'_red.fit'
    rfilename = os.path.join(ffolder,rfilename)
    ohl.writeto(rfilename,overwrite=True)
    rfilelist.append(rfilename)
    print (filename+ " -> " +rfilename)

# Exclude files where ASTROM solution fails and exptime diff. to mode of stack
print ('Checking astrometric solution on plate solved files...')

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
        print ('Astrom failed for:')
        print (rfilename)
        badfiles = np.append(badfiles,rfilename)
        continue
    stdcrms_lst = np.append(stdcrms_lst,stdcrms)
    numbrms_lst = np.append(numbrms_lst, numbrms)

print ('Astrometry checks: Done.')

# Add any files with different exposure time to majority to flagged list
texp = stats.mode(exptimes)[0][0]
print ('Stack texp = {} s.'.format(texp))
for irfile,rfilename in enumerate(rfilelist):
    if exptimes[ifile] != texp:
        print ('texp = {} s for:'.format(exptimes[ifile]))
        print (rfilename)
        badfiles = np.append(badfiles,rfilename)
print ('Exposure time checks: Done.')

# Save flagged file list for future ref
if len(badfiles) >= 1:
    print ('Saved a list of flagged files.')
elif len(badfiles) == 0:
    print ('No files were flagged.')
np.savetxt(os.path.join(ffolder,'{}.{}.flagged_files.txt'.format(date,target)),np.unique(badfiles),fmt='%s')

# Make excluded directory and move flagged files there
excfolder = os.path.join(ffolder,'excluded/')
print (excfolder)
os.system('mkdir '+excfolder)
for badfile in badfiles:
    #print ('mv ' + badfile + ' '+ excfolder)
    #if not os.path.exists(os.path.join(excfolder,badfile)):
    os.system('mv ' + badfile + ' '+ excfolder)

# Save histogram with astrom solution stdv and number of stars used
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,7))
fig.suptitle('Astrometry Evaluation')

ax1.hist(stdcrms_lst, 20)
ax1.set_xlabel('Astrometric fit coord rms (arcsec)')
ax1.set_ylabel('Number of Exposures')

ax2.hist(numbrms_lst, 20)
ax2.set_ylabel('Number of Exposures')
ax2.set_xlabel('Number of astrometric standards used')
histogram = os.path.join(ffolder,"{}.{}_astrom_hist.pdf".format(date,target))
fig.savefig(histogram)

# Send log file and STDRMS/NUMBRMS .pdf file to email
#logfile = os.path.join(ffolder,'log.txt')
#subject = '[Tierras]_Data_Reduction_Report:{}_{}'.format(date,target)
#append = '{} {}'.format(logfile,histogram)
#append = '{}'.format(histogram)
#email = 'juliana.garcia-mejia@cfa.harvard.edu'
#os.system('echo | mutt {} -s {} -a {}'.format(email,subject,append))

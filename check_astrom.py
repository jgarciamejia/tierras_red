#!/usr/bin/env python

import numpy as np
from astropy.io import fits 
import sys
import matplotlib.pyplot as plt
import re

print ('Checking astrometric solution on plate solved files...')

sample_filename = sys.argv[1]
DATE = re.split('\.',sample_filename)[0]
TARGET = re.split('\.', sample_filename)[2]

nfiles = len(sys.argv[1:])
nstdcrms, nnumbrms = 0,0
maxrms, minnum = 0.14,45

stdcrms_lst = np.array([])
numbrms_lst = np.array([])

for ifile,fits_filename in enumerate(sys.argv[1:]):
    #print (fits_filename)
    hdr = fits.getheader(fits_filename,1)
    #print (fits_filename)
    # read out astrometric fit coordinate rms (arcsec)
    # and number of astrometric standards used. 
    stdcrms, numbrms = hdr['STDCRMS'], hdr['NUMBRMS']
    if stdcrms > maxrms:
        print ('stdcrms > {}'.format(maxrms))
        print (fits_filename,stdcrms)
    if numbrms < minnum:
        print ('numbrms < {}'.format(minnum))
        print (fits_filename,numbrms)
    stdcrms_lst = np.append(stdcrms_lst,stdcrms)
    numbrms_lst = np.append(numbrms_lst, numbrms)
    #print ('done')

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,7))
fig.suptitle('Astrometry Evaluation')

ax1.hist(stdcrms_lst, 20)
ax1.set_xlabel('Astrometric fit coord rms (arcsec)')
ax1.set_ylabel('Number of Exposures')

ax2.hist(numbrms_lst, 20)
ax2.set_ylabel('Number of Exposures')
ax2.set_xlabel('Number of astrometric standards used')
fig.savefig("{0}.{1}_astrom_hist.pdf".format(DATE,TARGET))

'''
# Deprecated version of the loop above, to print how many files 
# there are above or below a certain threshold value for stdcrms and numbrms

for ifile,fits_filename in enumerate(sys.argv[1:]):
    # print (fits_filename)
    hdr = fits.getheader(fits_filename,1)
    # print (hdr['HISTORY']) # as a check to confirm that headers were updated by astrom
    # read out astrometric fit coordinate rms (arcsec)
    # and number of astrometric standards used. 
    stdcrms, numbrms = hdr['STDCRMS'], hdr['NUMBRMS']
    if stdcrms >= maxrms:
        #print (fits_filename)
        #print ("STDCRMS is too high")
        #print ("STDCRMS = {0}".format(hdr['STDCRMS']))
        nstdcrms += 1
    if numbrms <= minnum:
        #print (fits_filename)
        #print ("NUMBRMS is too low")
        #print ("NUMBRMS = {1}".format(hdr['NUMBRMS']))
        nnumbrms += 1 

print ("STDCRMS > {0} for {1} files".format(maxrms,nstdcrms))
print ("NUMBRMS < {0} for {1} files".format(minnum,nnumbrms))
#    with fits.open(fits_filename,'update') as f:
#        for hdu in f:
#            hdu.header['CAT-RA']='04:12:58.00'
#            hdu.header['CAT-DEC']='+52:36:23.9'
#        print ('After Modification')
#        print (hdu.header['CAT-RA'],hdu.header['CAT-DEC'])
'''

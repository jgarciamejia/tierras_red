#!/usr/bin/env python

# to use, pass command line <script.py> astra astdec filenames 
# astra 00:00:00, astdec +/-00:00:00
import numpy as np
from astropy.io import fits 
import sys
from datetime import date
import time
import pdb#; pdb.set_trace()

#pdb.set_trace()
todayy = date.today()
todaysdate = todayy.strftime("%d/%m/%Y")

# assumes data reduced on same day or day after they were taken
currentepoch = 2000.0 + (time.time() / 86400.0 - 10957.5) / 365.25

astra = sys.argv[1]
astdec = sys.argv[2]

#sys.argv[1] = astra # in h,min,sec
#sys.argv[2] = astde # in deg,min,sec

nfiles = len(sys.argv[1:])
for ifile,fits_filename in enumerate(sys.argv[3:]):
    #fits.info(fits_filename)
    #pdb.set_trace()
    print (fits_filename)
    print ('Before Modification')
    hdr = fits.getheader(fits_filename,0)
    print (hdr['CAT-RA'], hdr['CAT-DEC'])

    with fits.open(fits_filename,'update') as f:
        for hdu in f:
            hdu.header['CAT-RA']=astra
            hdu.header['CAT-DEC']=astdec
            hdu.header['HISTORY']='{}: updated CAT-RA, CAT-DEC from epoch 2000 to epoch {} coords.'.format(todaysdate,currentepoch)
        print ('After Modification')
        print (hdu.header['CAT-RA'],hdu.header['CAT-DEC'])

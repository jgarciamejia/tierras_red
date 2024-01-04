#!/usr/bin/env python

import numpy as np
from astropy.io import fits 
import sys
import matplotlib.pyplot as plt
import re
import pdb
from fitsutil import *
from flat_field import _trim_edges

fits_filename = sys.argv[1]
hdu = fits.open(fits_filename)
d1 = hdu[1].data
d2 = hdu[2].data
img1 = _trim_edges(d1)
img2 = _trim_edges(d2)

# mask bad pixels 
maskfile = '/home/jmejia/tierras/git/sicamd/config/badpix.mask'

amplist = []
sectlist = []
vallist = []

with open(maskfile, "r") as mfp:
    for line in mfp:
        ls = line.strip()
        lc = ls.split("#", 1)
        ln = lc[0]
        if ln == "":
            continue

        amp, sect, value = ln.split()
        xl, xh, yl, yh = fits_section(sect)

        amplist.append(int(amp))
        sectlist.append([xl, xh, yl, yh])
        vallist.append(int(value))

    amplist = np.array(amplist, dtype=np.int)
    sectlist = np.array(sectlist, dtype=np.int)
    vallist = np.array(vallist, dtype=np.int)

    allamps = np.unique(amplist)
    namps = len(allamps)

    masks = [None] * namps

    for amp in allamps:
        ww = amplist == amp
        thissect = sectlist[ww,:]
        thisval = vallist[ww]

        nx = np.max(thissect[:,1])
        ny = np.max(thissect[:,3])

        img = np.ones([ny, nx], dtype=np.uint8)

        nsect = thissect.shape[0]

        for isect in range(nsect):
          xl, xh, yl, yh = thissect[isect,:]
          img[yl:yh,xl:xh] = thisval[isect]

        masks[amp-1] = img


f1 = masks[0] == 1 
f2 = masks[1] == 1

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,7))
fig.suptitle('Flat Counts')

ax1.hist(img1[f1].flatten(), 100)
ax1.set_title('CCD Half 1')
ax1.set_xlabel('Counts')
ax1.set_ylabel('Number of Pixels')
ax1.set_yscale('log',nonpositive='clip')

ax2.hist(img2[f2].flatten(), 100)
ax2.set_title('CCD Half 2')
ax2.set_ylabel('Number of Pixels')
ax2.set_xlabel('Counts')
#fig.savefig("{0}.{1}_astrom_hist.pdf".format(DATE,TARGET))
ax2.set_yscale('log',nonpositive='clip')
plt.show()

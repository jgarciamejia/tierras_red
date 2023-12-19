#!/usr/bin/env python

import numpy as np
from fitsutil import *

#maskfile = '/home/jmejia/tierras/git/sicamd/config/badpix.mask'

def _gen_masks(maskfile):
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
    return masks


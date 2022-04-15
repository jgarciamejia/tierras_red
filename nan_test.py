#!/usr/bin/env python
from __future__ import print_function

import argparse
import math
import numpy as np
import sys

import lfa
import sep
from imred import *
import pdb
from fitsutil import *
import flat_field

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

####  MASTER FLAT 

# Open file, create pyfits object
ffilename = sys.argv[1]
filename = re.split('/',ffilename)[5]
pobj = pyfits.open(ffilename)
pobj.info()

# Read trim, overscan, and smear section info from header (only one half of chip)
hdr = pobj[1].header
biassec = fits_section(hdr["BIASSEC"])
smearsec = fits_section(hdr["SMEARSEC"])
trimsec = fits_section(hdr["TRIMSEC"])

# Open image data and assign NaN to overscan and smear sections via Ryan's _edges_to_nan; i.e., only illuminated pixels should remains as numbers
img = pobj[1].data
img_nanedges = flat_field._edges_to_nan(pobj,img,fill_value=np.nan) 

# Plot image. Zoom into edges and note that x=50-70 range has weird some nan, some non-nan values.
# Can you please fix this?  
plt.imshow(img_nanedges,cmap='gray',interpolation='none')
plt.colorbar()
plt.show()

# Trim image via trimsec
img_trim = img[1:1024,51:4146] #shape (1023,4095)

# Plot image. Nans persist interior to the trimmed image. 
plt.imshow(img_nanedges,cmap='gray',interpolation='none')
plt.colorbar()
plt.show()

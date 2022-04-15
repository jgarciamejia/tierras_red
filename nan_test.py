#!/usr/bin/env python
import numpy as np
import sys
import re
import flat_field
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt

####  MASTER FLAT 

# Open file, create pyfits object
ffilename = sys.argv[1]
filename = re.split('/',ffilename)[5]
pobj = pyfits.open(ffilename)
pobj.info()

# Open image data and assign NaN to overscan and smear sections via Ryan's _edges_to_nan; i.e., only illuminated pixels should remains as numbers
img = pobj[1].data
img_nanedges = flat_field._edges_to_nan(pobj,img,fill_value=np.nan) 

# Plot image. Zoom into edges and note that x=50-70 range has weird some nan, some non-nan values.
# Can you please fix this?  
plt.imshow(img_nanedges,cmap='gray',interpolation='none')
plt.colorbar()
plt.show()

# Trim image to only include illuminated pixels. These values come straight from trimsec as defined in the image header.  
img_trim = img[1:1024,51:4146] #shape (1023,4095)

# Plot image. Nans persist interior to the trimmed image, towards the edges. THAT IS THE PROBLEM! 
plt.imshow(img_trim,cmap='gray',interpolation='none')
plt.colorbar()
plt.show()

#hdr = pobj[1].header

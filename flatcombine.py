#!/usr/bin/env python

from __future__ import print_function

import argparse
import logging
import math
import re
import sys
import warnings
from glob import glob
import matplotlib.pyplot as plt 
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ion()

import numpy as np

try:
  import astropy.utils.exceptions
  import astropy.io.fits as pyfits
except ImportError:
  import pyfits

import lfa

from fitsutil import *

def process_extension(imp):
  hdr = imp.header
  
  if "BIASSEC" in hdr:
    biassec = fits_section(hdr["BIASSEC"])
  else:
    biassec = None
      
  if "TRIMSEC" in hdr:
    trimsec = fits_section(hdr["TRIMSEC"])
  else:
    trimsec = None
      
  raw = np.float32(imp.data)

  if biassec is not None:
    biaslev, biassig = lfa.skylevel_image(raw[biassec[2]:biassec[3],biassec[0]:biassec[1]])
  else:
    biaslev = 0

  if trimsec is not None:
    procimg = raw[trimsec[2]:trimsec[3],trimsec[0]:trimsec[1]] - biaslev
  else:
    procimg = raw - biaslev

  skylev, skynoise = lfa.skylevel_image(procimg)

  if iext == 2:
    procimg = np.flipud(procimg) # top half of image is read in upside down under the current read-in scheme, so flip it here

  return procimg / skylev, skylev, skynoise / skylev

if __name__ == "__main__":
  try:
    warnings.simplefilter("ignore",
                          astropy.utils.exceptions.AstropyDeprecationWarning)
  except NameError:
    pass
  
  # Deal with the command line.
  ap = argparse.ArgumentParser()
  # ap.add_argument("filelist", metavar="file", nargs="+", help="input files")
  # ap.add_argument("-o", help="output file")
  ap.add_argument("-date", required=True, help="YYYYMMDD of the date on which flats were taken.")

  
  args = ap.parse_args()
  
  date = args.date

  filepath = f'/data/tierras/incoming/{date}/'
  filelist = glob(filepath+'*FLAT[0-9][0-9][0-9].fit')
  filelist = np.array(sorted(filelist, key=lambda x:int(x.split('FLAT')[-1].split('.')[0]))) # make sure the files are sorted 
  nf = len(filelist)

  fplist = [ pyfits.open(filename) for filename in filelist ]
  
  outhdus = []

  for iext, rmp in enumerate(fplist[0]):
    if hasimg(rmp):
      hdr = rmp.header
      hdrout = hdr.copy(strip=True)

      for key in ("BIASSEC", "SMEARSEC"):
        if key in hdrout:
          del hdrout[key]

      allimg = [None] * nf
      allsky = [None] * nf
      allrms = [None] * nf

      for ifile in range(nf):
        print(f'Reading {filelist[ifile]}, {ifile+1} of {len(filelist)}, extension {iext}')
        # Open file.
        ifp = fplist[ifile]
        imp = ifp[iext]
        allimg[ifile], allsky[ifile], allrms[ifile] = process_extension(imp)

      allimg = np.array(allimg)
      allsky = np.array(allsky)
      allrms = np.array(allrms)

      # Median combine normalized images.
      print(f'Median combining the flats, extension {iext}')
      medimg = np.median(allimg, axis=0)

      ny, nx = medimg.shape
      
      # Simple upper envelope clipping.
      resid = allimg - medimg

      # This trick broadcasts the comparison to rms over the first dimension
      # of "resid" (the input file) by transposing temporarily so it's the
      # last dimension during the calculation.
      ww = (resid.T > 3.0 * allrms).T
      
      nclipped = np.sum(ww)

      if nclipped > 0:
        allimg[ww] = np.nan
        medimg[:] = np.nanmedian(allimg, axis=0)

      logging.info("Clipped {0:d}".format(nclipped))

      # Scale output back to original level.
      outimg = medimg * allsky[0]

      # Adjust headers to account for trim.
      newsect = "[1:{0:d},1:{1:d}]".format(nx, ny)
      hdrout["DATASEC"] = newsect
      hdrout["TRIMSEC"] = newsect

      if iext == 0:
        outhdu = pyfits.PrimaryHDU(outimg, header=hdrout)
      else:
        outhdu = pyfits.ImageHDU(outimg, header=hdrout)
    else:
      outhdu = rmp.copy()

    outhdus.append(outhdu)

  # stitch into a single image 
  outimg_stitch = np.zeros((ny*2, nx), dtype=np.float32)
  outimg_stitch[0:ny,:] = outhdus[1].data
  outimg_stitch[ny:, :] = outhdus[2].data
  
  norm = simple_norm(outimg_stitch, min_percent=1, max_percent=99)

  fig, ax = plt.subplots(1, 1, figsize=(12,6))
  im = ax.imshow(outimg_stitch, origin='lower', norm=norm, interpolation='none')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes('right', size='5%', pad=0.05)
  cb = fig.colorbar(im, cax=cax, orientation='vertical')
  cb.set_label(label='ADU')
  ax.set_title(f'Combined Flat, {date}')

  plt.tight_layout()

  breakpoint()
  # Write output file.
  if args.o is not None:
    ohl = pyfits.HDUList(outhdus)
    ohl.writeto(args.o, clobber=True)

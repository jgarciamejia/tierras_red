#!/usr/bin/env python

from __future__ import print_function

import argparse
import math
import numpy
import sys

import lfa
import sep

try:
  import astropy.wcs as pywcs
except ImportError:
  import pywcs

from imred import *
#import imred
#import pdb
#pdb.set_trace()


# Deal with the command line.
ap = argparse.ArgumentParser()
ap.add_argument("-f", help="flat file")
ap.add_argument("filelist", metavar="file", nargs="*", help="input files")

if len(sys.argv) == 1:
  ap.print_help(sys.stderr)
  sys.exit(1)

args = ap.parse_args()

irnobj = imred()
irobj = imred(args.f)

# Detection parameters.
minpix = 4
thresh = 2.0

# Half chip mode?
halfchip = False

# Position measurement is best done using a smaller FWHM matched aperture
# so we measure mostly the core of the image.
rcent = 10.0

# Photometry needs a larger aperture typically 2*FWHM.
rap = numpy.arange(5, 26, dtype=numpy.double)

# Sky annulus size.  The usual rule of thumb from 2MASS is 6 to 8 * FWHM.
rinn = 40.0
rout = 60.0

# Number of reference stars.  This should be as large as possible,
# but you don't want to go too faint.
nrefs = 1000

# Output files.
# target centric light curve per aperture
ofp = [None] * len(rap)
for irap, thisrap in enumerate(rap):
  ofp[irap] = open("lc{0:02.0f}.txt".format(thisrap), "w")

# Output files
# reference exposure star positions
refexpfile = open("reffile_starpos.txt", "w")


# Process reference without flat so counts are accurate.
if True:
  filename = args.filelist[0]

  hl = irnobj.read_and_reduce(filename, stitch=True)
  mp = hl[0]
  if halfchip:
    img = mp.data[0:1024,:]  # throw away the other half for now
  else:
    img = mp.data
  hdr = mp.header
  if halfchip:
    mask = hl[1].data[0:1024,:]  # throw away the other half for now
  else:
    mask = hl[1].data

  wcs = pywcs.WCS(hdr)

  mjd = float(hdr["MJD-OBS"])
  exptime = float(hdr["EXPTIME"])
  airmass = float(hdr["AIRMASS"])
  gain = float(hdr["GAIN"])

  if True:  # reference file
    # Figure out where the target star is.
    catra = hdr["CAT-RA"]
    catde = hdr["CAT-DEC"]
    cateq = float(hdr["CAT-EQUI"])
    if cateq == 0:
      logging.error("Target position is not set")
      sys.exit(1)

    ratarg, rv = lfa.base60_to_10(catra, ":", lfa.UNIT_HR, lfa.UNIT_RAD)
    detarg, rv = lfa.base60_to_10(catde, ":", lfa.UNIT_DEG, lfa.UNIT_RAD)

    xtarg, ytarg = wcs.all_world2pix(ratarg * lfa.RAD_TO_DEG,
                                     detarg * lfa.RAD_TO_DEG,
                                     1)

    # Extract sources.
    notmask = numpy.logical_not(mask)
    
    bkg = sep.Background(img, mask=notmask)
    skylev = bkg.globalback
    skynoise = bkg.globalrms
    objs = sep.extract(img-bkg, thresh,
                       err=bkg.globalrms,
                       mask=notmask,
                       minarea=minpix)
    
    # Find the target.
    # Note: sep numbers from 0.  Everything else here follows FITS in
    # numbering from 1 so we modify where needed.
    itarg = numpy.argmin(numpy.hypot(objs["x"]+1-xtarg, objs["y"]+1-ytarg))
    targ = objs[itarg]
    
    # Select reference stars.
    ww = numpy.logical_and(objs["peak"] < 50000,
                           objs["flux"] > targ["flux"]/20)
    ww[itarg] = 0  # deselect target
    
    possible_refs = objs[ww]

    bydecflux = numpy.argsort(-possible_refs["flux"])
    if len(bydecflux) > nrefs:
      refs = possible_refs[bydecflux[0:nrefs]]
    else:
      refs = possible_refs[bydecflux]
    
    print("Selected {0:d} reference stars".format(len(refs)))
    # For now, exclude variable refs by hand
    oldnrefs = len(refs)
    #refs = numpy.delete(refs,[5])
    #pdb.set_trace()
    #print("Excluded {0:d} reference stars by hand".format(oldnrefs-len(refs)))
    
    # Array of positions to do photometry at.
    xphot = numpy.concatenate(([targ["x"]], refs["x"]))+1
    yphot = numpy.concatenate(([targ["y"]], refs["y"]))+1

    raphot, dephot = wcs.all_pix2world(xphot, yphot, 1)

    # Save ref file information
    for i in range(len(xphot)):
      print (i, xphot[i], yphot[i], raphot[i], dephot[i])
      refexpfile.write("{0} {1} {2} {3} \n".format(xphot[i], yphot[i], raphot[i],dephot[i]))
      #pdb.set_trace()

# Output files 
# data per comparison(=ref) per aperture.
ofpc = [[None]*len(refs) for i in range(len(rap))]
for irap,thisrap in enumerate(rap):
  for iref,ref in enumerate(refs):
    ofpc[irap][iref] = open("lc{0:02.0f}.ref{1}".format(int(thisrap),iref), "w")

for ifile, filename in enumerate(args.filelist):
  print(filename)
  filenamenum = re.split('\.',filename)[1]
  hl = irobj.read_and_reduce(filename, stitch=True)
  mp = hl[0]
  if halfchip:
    img = mp.data[0:1024,:]  # throw away the other half for now
  else:
    img = mp.data
  hdr = mp.header
  if halfchip:
    mask = hl[1].data[0:1024,:]  # throw away the other half for now
  else:
    mask = hl[1].data

  wcs = pywcs.WCS(hdr)

  mjd = float(hdr["MJD-OBS"])
  exptime = float(hdr["EXPTIME"])
  airmass = float(hdr["AIRMASS"])
  gain = float(hdr["GAIN"])
  temperat = float(hdr["TEMPERAT"])
  humid = float(hdr["HUMIDITY"])
  focus = float(hdr["FOCUS"])
  dometemp = float(hdr["DOMETEMP"])
  domehumid = float(hdr["DOMEHUMI"])

  stdcrms = float(hdr["STDCRMS"])
  numbrms = int(hdr["NUMBRMS"])

  if stdcrms > 1.0 or numbrms < 5:
    logging.error("{0:s} astrometry is bad".format(filename))

  # Calculate pixel coordinates on this image by applying transformation.
  xap, yap = wcs.all_world2pix(raphot, dephot, 1)

  # Determine positions by measuring this image.
  skylev, skynoise = lfa.ap_skyann(img, xap, yap, rinn, rout)
  xnew, ynew, flux = lfa.ap_phot(img, skylev, xphot, yphot, rcent)

  # Calculate half flux diameter
  # target only
  #hfr, flags = sep.flux_radius(img-skylev[0], [xnew[0]-1], [ynew[0]-1], [rinn], 0.5)
  #hfd = 2*hfr[0]
  
  # target and comps
  #pdb.set_trace()
  hfr, flags = sep.flux_radius(img-skylev[0], numpy.array(xnew-1), numpy.array(ynew-1), rinn*numpy.ones(len(xnew)), 0.5)
  hfd = 2*hfr
  
  for irap, thisrap in enumerate(rap):
    
    # Aperture photometry.
    # There's a choice to be made here when doing the photometry, we
    # can either use the measured x,y ("new") or the predicted ("ap").
    # Each has its own advantages and disadvantages.
    flux, fluxerr, flags = sep.sum_circle(img, xap-1, yap-1, thisrap,
                                          bkgann=(rinn, rout),
                                          err=skynoise[0],  # target, not ideal
                                          gain=gain)


    # Comparison star sum and noise in the sum.
    norm = numpy.sum(flux[1:])
    e_norm = numpy.sqrt(numpy.sum(numpy.square(fluxerr[1:])))
    
    # Differential photometry.
    diff_flux = flux / norm
    e_diff_flux = diff_flux * numpy.hypot(fluxerr/flux, e_norm/norm)
    
    # Write target star and observatory output.
    ofp[irap].write("{0:14.6f} {1:10.8f} {2:10.8f} {3:12.6f} {4:12.6f} {5:12.6f} {6:12.6f} {7:10.8f} {8:10.8f} {9:10.5f} {10:10.1f} {11:10.8f} {12:10.8f} {13:10.8f} {14:8.3f} {15:10.5f} {16:10.3f} {17:10.3f} {18:8.1f} {19:10.3f} {20:10.3f}\n".format(2400000.5+mjd, flux[0], fluxerr[0],xnew[0], ynew[0], xap[0], yap[0], skylev[0], skynoise[0], hfd[0], norm, e_norm, diff_flux[0], e_diff_flux[0], exptime, airmass, temperat, humid,focus, dometemp, domehumid))

    # Save comp star lc output. Each comp star gets its own file per aperture.
    for iref in range(len(refs)):
    #for iref,ref in enumerate(refs):
      ofpc[irap][iref].write("{0:14.6f} {1:10.8f} {2:10.8f} {3:12.6f} {4:12.6f} {5:12.6f} {6:12.6f} {7:10.8f} {8:10.8f} {9:10.5f}\n".format(2400000.5+mjd, flux[iref+1], fluxerr[iref+1], xnew[iref+1], ynew[iref+1], xap[iref+1], yap[iref+1], skylev[iref+1], skynoise[iref+1], hfd[iref+1]))


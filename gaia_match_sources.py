#!/usr/bin/env python
#
# match_gaia: extract photometry from images and match to GAIA reference
#             sources.  Emit ascii catalogue for later processing.
#

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

try:
  import astropy.io.fits as pyfits
except ImportError:
  import pyfits

import load_data as ld
from fpcoord import *
from getgaia import *
from imred import *

# Correct proper motion from epoch 1 to epoch 2
def pmcorr(ra, de, pmra, pmde, ep1, ep2):
  sa = numpy.sin(ra)
  ca = numpy.cos(ra)
  sd = numpy.sin(de)
  cd = numpy.cos(de)
  
  x = ca * cd
  y = sa * cd
  z = sd

  dxdt = -pmra * sa - pmde * ca*sd
  dydt =  pmra * ca - pmde * sa*sd
  dzdt =  pmde * cd

  dep = ep2 - ep1

  x += dep * dxdt
  y += dep * dydt
  z += dep * dzdt
  
  a = numpy.arctan2(y, x)
  a = numpy.where(a < 0, 2*math.pi+a, a)

  d = numpy.arctan2(z, numpy.hypot(x, y))

  return a, d

# Utility to convert MJD to Julian epoch for above.
def mjd2ep(mjd):
  return(2000.0 + (mjd - 51544.5) / 365.25)


# load x and y positions for comp stars in file. JGM Dec 2023: needs work to generalize. 
lcpath = '/data/tierras/lightcurves/'
rdpath = '/data/tierras/flattened/'
#targetname = '/2MASSJ03304890+/flat0000/'
#fname = '20221211.0047.2MASSJ03304890+_red.fit'

exclude_comps = []
#date = '20221211'
sig_threshold = 5
#df,bjds,relfluxes,airmasses,widths,flag, comps_used = ld.return_data_onedate(lcpath,
                                        targetname,date,sig_threshold,exclude_comps,
                                        flag_output = True)

# Extract comp star locations from fname 
aij_comps = ld.get_AIJ_star_numbers(df,'Source-Sky_C')
aij_targs = ld.get_AIJ_star_numbers(df,'Source-Sky_T') 
aij_allstars = numpy.sort(numpy.concatenate((aij_targs,aij_comps)))
xs, ys = [], []

# get RA and Dec x,y pixel positions in FITS file for each comp
for comp_num in aij_allstars:
  print (comp_num)
  if comp_num in aij_targs:
    x_hdr,y_hdr = ld.find_all_cols_w_keywords(df,'X(FITS)_T'+str(comp_num))[0],ld.find_all_cols_w_keywords(df,'Y(FITS)_T'+str(comp_num))[0]
  elif comp_num in aij_comps:
    x_hdr,y_hdr = ld.find_all_cols_w_keywords(df,'X(FITS)_C'+str(comp_num))[0],ld.find_all_cols_w_keywords(df,'Y(FITS)_C'+str(comp_num))[0]
  ix,iy = df[x_hdr].to_numpy()[8], df[y_hdr].to_numpy()[8] # exposure index hardcoded to match fname

  xs.append(ix)
  ys.append(iy)

# Read image 
fullfname = rdpath+date+targetname+fname
print (fullfname)

hl = pyfits.open(fullfname)
mp = hl[0]
img = mp.data
hdr = mp.header
mask = hl[1].data

# Image size, for later.
ny, nx = img.shape

# WCS.
wcs = pywcs.WCS(hdr)

# Calculate approximate centre and size of image for catalogue cutout.
det = numpy.linalg.det(wcs.wcs.cd)
scl = math.radians(math.sqrt(abs(det)))

# How big do we need to cut out?
boxsize = 1.5 * scl * max(ny, nx)

# Get other headers.
utc = float(hdr["MJD-OBS"])
exptime = float(hdr["EXPTIME"])
airmass = float(hdr["AIRMASS"])
gain = float(hdr["GAIN"])

# Print astrometric soln. is stdcrms > 1.0 or numbrms < 5 dont use this exp.
stdcrms = float(hdr["STDCRMS"])
numbrms = int(hdr["NUMBRMS"])
print (stdcrms,numbrms)

x = numpy.array(xs) 
y = numpy.array(ys) 

row = numpy.arange(len(xs)) + 1

ra, de = wcs.all_pix2world(x, y, 1)
tpa, tpd = wcs.all_pix2world(nx/2, ny/2, 1)  # middle of img

# Convert to sensible units.
ra = numpy.radians(ra)
de = numpy.radians(de)

tpa = numpy.radians(tpa)
tpd = numpy.radians(tpd)

xi, xn = standc(ra, de, tpa, tpd)

# Get GAIA.
all_gaia = boxgaia(tpa, tpd, boxsize, subset=True)

ngaia = len(all_gaia)

# Extract columns.
all_gaia_ra = all_gaia["ra"]
all_gaia_de = all_gaia["dec"]
all_gaia_pmra = all_gaia["pmra"]
all_gaia_pmde = all_gaia["pmdec"]
all_gaia_plx = all_gaia["parallax"]
all_gaia_gmag = all_gaia["G"]
all_gaia_bp = all_gaia["BP"]
all_gaia_rp = all_gaia["RP"]
catepoch = 2016.0

# Figure out what proper motion and parallax to use.
use_pmra = numpy.where(numpy.isfinite(all_gaia_pmra), all_gaia_pmra, 0)
use_pmde = numpy.where(numpy.isfinite(all_gaia_pmde), all_gaia_pmde, 0)
use_plx = numpy.where(numpy.isfinite(all_gaia_plx), all_gaia_plx, 0)

# Compute astrometric place.
imgepoch = mjd2ep(utc)

all_gaia_ranow, all_gaia_denow = pmcorr(all_gaia_ra, all_gaia_de,
                                        use_pmra*lfa.MAS_TO_RAD,
                                        use_pmde*lfa.MAS_TO_RAD,
                                        catepoch, imgepoch)

all_gaia_xi, all_gaia_xn = standc(all_gaia_ranow, all_gaia_denow,
                                  tpa, tpd)

# Rough cubic relation for predicting MEarth-G from Bp-Rp from
# python/mearth/makegaia (could do with improvement, see there).
# This is only used for likelihood ratio calculation for match
# where it's important to have a magnitude for everything.
coef_gaia_mearth = numpy.array([ 0.06347146, -0.85234365,
                                 0.1178988, -0.00490911 ])

bp_rp = all_gaia_bp - all_gaia_rp
corr = numpy.polynomial.polynomial.polyval(bp_rp, coef_gaia_mearth)

all_gaia_imag = all_gaia_gmag + numpy.where(numpy.isfinite(corr), corr, 0.0)

# Catalogue must be sorted.
sort_gaia = numpy.argsort(all_gaia_xi)

all_gaia = all_gaia[sort_gaia]
all_gaia_xi = all_gaia_xi[sort_gaia]
all_gaia_xn = all_gaia_xn[sort_gaia]
all_gaia_imag = all_gaia_imag[sort_gaia]

# Perform match.
err = numpy.empty_like(xi)
err.fill(1.0 * lfa.AS_TO_RAD)

best_gaia_for_img, best_img_for_gaia = lfa.lrmatch(xi,
                                                   xn,
                                                   None,
                                                   err,
                                                   all_gaia_xi,
                                                   all_gaia_xn,
                                                   all_gaia_imag,
                                                   None,
                                                   1.0*lfa.AS_TO_RAD,
                                                   False)

# Extract matched sources.
ismatched = best_gaia_for_img >= 0

matched_row = row[ismatched]
matched_x = x[ismatched]
matched_y = y[ismatched]
gaia_indices = best_gaia_for_img[ismatched]
matched_gaia = all_gaia[gaia_indices]

all_gaia_id = all_gaia["source_id"]
all_gaia_ra = all_gaia["ra"]
all_gaia_de = all_gaia["dec"]
matched_ids = all_gaia_id[gaia_indices]
matched_ras =  all_gaia_ra[gaia_indices]*(180/numpy.pi)
matched_decs = all_gaia_de[gaia_indices]*(180/numpy.pi)

rerr = numpy.hypot(xi[ismatched] - all_gaia_xi[gaia_indices],
                   xn[ismatched] - all_gaia_xn[gaia_indices])
rerr = numpy.degrees(rerr) * 3600  # arcsec

# Save information for later processing.
basefile = lcpath+date+targetname
outfile = basefile +"_gaia_info.txt"

numpy.savetxt(outfile,
              numpy.column_stack((matched_row,
                                  matched_ids,
                                  matched_x,
                                  matched_y,
                                  matched_ras,
                                  matched_decs,
                                  matched_gaia["G"],
                                  matched_gaia["BP"],
                                  matched_gaia["RP"],
                                  rerr)),
              fmt="%-5d %12.0f %12.6f %12.6f %12.12f %12.12f %6.3f %6.3f %6.3f %6.3f")

# Save information for later processing:ids only
basefile = lcpath+date+targetname
outfile = basefile + "_gaia_ids_only.txt"

with open(outfile,"w") as file:
  file.writelines('Gaia DR2'+ ' '+ str(s) + '\n' for s in matched_ids)

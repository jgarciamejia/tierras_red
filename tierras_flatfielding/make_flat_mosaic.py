#!/usr/bin/env python
#

from __future__ import print_function

import argparse
import math
import numpy
import os
import re
import subprocess
import sys
import pdb

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

try:
  import astropy.io.fits as pyfits
except ImportError:
  import pyfits

from imred import *
import lfa

# Minimum exposure time of flats to be considered. 
# Used to mnimize non-uniform exp time issues with Tierras CCD. 
MIN_FLAT_EXPTIME = 9.0  #sec

# Max allowed lunar illumination fraction
MAX_MOON_ILLUM = 0.1

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("fitsfile", nargs="+", help="fits file")
  ap.add_argument("-o", default="compare.png", type=str, help="output image file")
  ap.add_argument("-z", default=0.01, type=float, help="zscale range")

  if len(sys.argv) == 1:
    ap.print_help(sys.stderr)
    sys.exit(1)

  args = ap.parse_args()

  # Locate monospace font using fontconfig.
  argv = [ "fc-match", "monospace", "file" ]

  p = subprocess.Popen(argv, stdout=subprocess.PIPE)
  pout, perr = p.communicate()

  mm = re.match(r'^:file=(.*)$', pout.decode("utf-8"))
  gg = mm.groups()
  fontfile = gg[0]

  # Load it.
  fnt = PIL.ImageFont.truetype(font=fontfile,
                               size=16)

  nf = len(args.fitsfile)

  npy = int(math.ceil(math.sqrt(nf)))
  npx = npy

  panelgap = 8

  im = None

  irnobj = imred()
  for ifile, fitsfile in enumerate(args.fitsfile):
    print (fitsfile)
    if 'medfilt_kernel' in fitsfile:
      continue
    if 'MASTERFLAT' in fitsfile:
      continue

    obsdate = re.findall(r'\d{8}',fitsfile)[0]
    fnum = re.findall(r'\.\d{4}',fitsfile)[0].lstrip('.')
    hl = irnobj.read_and_reduce(fitsfile, stitch=True)
    mp = hl[0]
    hdr = mp.header

    lat = math.radians(float(hdr["LATITUDE"]))
    lon = math.radians(float(hdr["LONGITUD"]))
    height = float(hdr["HEIGHT"])
    utc = float(hdr["MJD-OBS"])
    exptime = float(hdr["EXPTIME"]) 
    if exptime < MIN_FLAT_EXPTIME:
      continue

    # Set up observer.
    obs = lfa.observer(lon, lat, height)
    
    # Figure out TT-UTC from given UTC.
    iutc = int(utc)
    ttmutc = obs.dtai(iutc, utc-iutc) + lfa.DTT
    
    # Compute time-dependent quantities.
    obs.update(utc, ttmutc, lfa.OBSERVER_UPDATE_ALL)
    
    # Compute topocentric Sun and Moon position.
    sun = lfa.source_planet(lfa.JPLEPH_SUN)
    ssun, dsdtsun, prsun = obs.place(sun, lfa.TR_TO_TOPO_AZ)
    
    moon = lfa.source_planet(lfa.JPLEPH_MOON)
    smoon, dsdtmoon, prmoon = obs.place(moon, lfa.TR_TO_TOPO_AZ)
    
    # Moon elevation.
    moonaz = math.atan2(smoon[1], -smoon[0])
    moonaz = lfa.ranorm(moonaz)
    moonel = math.atan2(smoon[2], math.hypot(smoon[0], smoon[1]))
    
    # Cosine of elongation of moon from sun = dot product.
    cosphi = numpy.dot(ssun, smoon)
    
    # Fraction of lunar surface illuminated.
    moonillum = 0.5*(1.0 - cosphi)
    if moonillum > MAX_MOON_ILLUM:
      continue

    domeaz = float(hdr["DOMEAZ"])
    
    catra = hdr["CAT-RA"]
    catde = hdr["CAT-DEC"]
    ratarg, rv = lfa.base60_to_10(catra, ":", lfa.UNIT_HR, lfa.UNIT_RAD)
    detarg, rv = lfa.base60_to_10(catde, ":", lfa.UNIT_DEG, lfa.UNIT_RAD)

    targ = lfa.source(ratarg, detarg)
    starg, dsdttarg, prtarg = obs.place(targ, lfa.TR_TO_TOPO_AZ)
    
    targaz = math.atan2(starg[1], -starg[0])
    targaz = lfa.ranorm(targaz)
    targel = math.atan2(starg[2], math.hypot(starg[0], starg[1]))

    moonsep = lfa.v_angle_v(starg, smoon)

    exptime = hdr["EXPTIME"]

    img = mp.data

    skylev, skynoise = lfa.skylevel_image(img)

    # Binned image.
    nyin, nxin = img.shape

    binning = 8

    nyout, nxout = nyin//binning, nxin//binning

    sh = nyout, binning, nxout, binning

    imgbin = img[0:binning*nyout,0:binning*nxout].reshape(sh).sum(-1).sum(1) / (binning*binning)

    # Save preview.
    z1 = (1.0 - args.z) * skylev
    z2 = (1.0 + args.z) * skylev

    print("z1={0:.1f} z2={1:.1f}".format(z1, z2))

    tmp = (imgbin - z1) / (z2 - z1)
    tmp[tmp < 0] = 0
    tmp[tmp > 1] = 1

    out = 255.0 * tmp
    out[out < 0] = 0
    out[out > 255.0] = 255.0

    imgout = numpy.round(out).astype(numpy.uint8)

    ix = ifile % npx
    iy = ifile // npx

    ny, nx = imgout.shape

    if im is None:
      outimx = nx*npx+panelgap*(npx-1)
      outimy = ny*npy+panelgap*(npy-1)

      print("Creating {0:d}x{1:d} image".format(outimx, outimy))
      im = PIL.Image.new(mode="L", size=(outimx, outimy))

    x = (nx+panelgap) * ix
    y = (ny+panelgap) * iy

    tmpim = PIL.Image.fromarray(imgout[::-1,:])
    im.paste(tmpim, box=(x, y))

    s1 = "{0} file num={1} targ_az={2:.0f} targ_el={3:.0f}".format(obsdate, fnum, math.degrees(targaz), math.degrees(targel))
    s2 = "moon_az={0:.0f} moon_el={1:.0f} ill={2:.2f} sep={3:.0f}".format(math.degrees(moonaz), math.degrees(moonel), moonillum, math.degrees(moonsep))

    drw = PIL.ImageDraw.Draw(im)
    sx, sy = drw.textsize(s1, font=fnt)
    drw.text((x+panelgap, y+panelgap), s1, font=fnt, fill="white")

    drw = PIL.ImageDraw.Draw(im)
    drw.text((x+panelgap, y+2*panelgap+sy), s2, font=fnt, fill="white")

  drw = PIL.ImageDraw.Draw(im)
  s = "Range: +/- {0:.2f}".format(args.z)
  sx, sy = drw.textsize(s, font=fnt)
  drw.text((outimx-panelgap-sx, outimy-panelgap-sy), s, font=fnt, fill="white")

  im.save(args.o)

if __name__ == "__main__":
  main()

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

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

try:
  import astropy.io.fits as pyfits
except ImportError:
  import pyfits

import lfa

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
                               size=24)

  nf = len(args.fitsfile)

  npy = int(math.ceil(math.sqrt(nf)))
  npx = npy

  panelgap = 8

  im = None

  for ifile, fitsfile in enumerate(args.fitsfile):
    fp = pyfits.open(fitsfile)
    mp = fp[0]
    hdr = mp.header

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

    drw = PIL.ImageDraw.Draw(im)
    drw.text((x+panelgap, y+panelgap), "{0:.3f}s".format(exptime), font=fnt, fill="white")

  drw = PIL.ImageDraw.Draw(im)
  s = "Range: +/- {0:.2f}".format(args.z)
  sx, sy = drw.textsize(s, font=fnt)
  drw.text((outimx-panelgap-sx, outimy-panelgap-sy), s, font=fnt, fill="white")

  im.save(args.o)

if __name__ == "__main__":
  main()

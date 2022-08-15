#!/usr/bin/env python 

import glob 
import lfa 
import palpy 
import numpy 
import math 
import argparse
import sys
import palutil 
import re
import os

# Minimum exposure time of flats to be considered. 
# Used to mnimize non-uniform exp time issues with Tierras CCD. 
MIN_FLAT_EXPTIME = 9  #sec

# Max allowed lunar illumination fraction
MAX_MOON_ILLUM = .1

# Path to INCOMING folder
PATH = '/data/tierras/incoming/'
try:
  import astropy.io.fits as pyfits
except ImportError:
  import pyfits

def main():
  
  obsdates = os.listdir(PATH)
  for obsdate  in obsdates:
    filenames = np.sort(glob.glob(PATH+'obsdate'+'/*FLAT*fit'))
  # iterate through files, and rank from least (Moon) bright to most

  fnums = []
  exptimes = []
  elevs = []
  illums = []

  for ifile, fitsfile in enumerate(args.filenames):
    #print (fitsfile)
    if 'medfilt_kernel' in fitsfile:
      continue
    if 'MASTERFLAT' in fitsfile:
      continue
    
    fp = pyfits.open(fitsfile)
    mp = fp[0]
    hdr = mp.header

    lat = math.radians(float(hdr["LATITUDE"]))
    lon = math.radians(float(hdr["LONGITUD"]))
    height = float(hdr["HEIGHT"])
    utc = float(hdr["MJD-OBS"])
    exptime = float(hdr["EXPTIME"])
    if exptime < MIN_FLAT_EXPTIME:
      continue

    # MOON CALCS USING LFA 
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
    lfa_moonaz = lfa.ranorm(moonaz)
    lfa_moonel = math.atan2(smoon[2], math.hypot(smoon[0], smoon[1]))

    # Cosine of elongation of moon from sun = dot product.
    cosphi = numpy.dot(ssun, smoon)

    # Fraction of lunar surface illuminated.
    lfa_moonillum = 0.5*(1.0 - cosphi)
    if lfa_moonillum > MAX_MOON_ILLUM:
      continue
    
    domeaz = float(hdr["DOMEAZ"])

    catra = hdr["CAT-RA"]
    catde = hdr["CAT-DEC"]
    ratarg, rv = lfa.base60_to_10(catra, ":", lfa.UNIT_HR, lfa.UNIT_RAD)
    detarg, rv = lfa.base60_to_10(catde, ":", lfa.UNIT_DEG, lfa.UNIT_RAD)

    targ = lfa.source(ratarg, detarg)
    starg, dsdttarg, prtarg = obs.place(targ, lfa.TR_TO_TOPO_AZ)

    targaz = math.atan2(starg[1], -starg[0])
    lfa_targaz = lfa.ranorm(targaz)
    lfa_targel = math.atan2(starg[2], math.hypot(starg[0], starg[1]))

    lfa_moonsep = lfa.v_angle_v(starg, smoon)


    fnum = int(re.findall(r'\.\d{4}',fitsfile)[0].lstrip('.'))
    fnums.append(fnum)
    exptimes.append(exptime)
    elevs.append(lfa_moonel*lfa.RAD_TO_DEG)
    illums.append(lfa_moonillum)
  
  data = [fnums,exptimes,elevs,illums]
  for i,fnum in enumerate(fnums):
    if i == 0:
      print ("{: >20} {: >10} {: >20} {: >20}".format('File No.', 'Exp. Time', 'Moon El.', 'Moon Illum.'))
    print ("{: >20} {: >10} {: >20} {: >20}".format(fnums[i],exptimes[i],elevs[i],illums[i]))

if __name__ == "__main__":
    main()

#!/usr/bin/env python 

import glob 
import lfa 
import palpy 
import numpy 
import math 
import argparse
import sys

try:
  import astropy.io.fits as pyfits
except ImportError:
  import pyfits

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("filenames", nargs="+", help="fits filenames")

  if len(sys.argv) == 1:
    ap.print_help(sys.stderr)
    sys.exit(1)

  args = ap.parse_args()
  
  for ifile, fitsfile in enumerate(args.filenames):
    print (fitsfile)
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

    print (lfa_targaz, lfa_targel, lfa_moonaz, lfa_moonillum, lfa_moonsep)

    # MOON CALCS USING PALPY 

    # Calculate TT from UTC.
    tt = utc + palpy.dtt(utc) / lfa.DAY

    # Local Sidereal Time.
    lst = palutil.get_lst(utc, tt, lon)

    # Get Sun and Moon apparent place.
    sunrap, sundap, sundiam = palpy.rdplan(tt, 0, lon, lat)
    moonrap, moondap, moondiam = palpy.rdplan(tt, 3, lon, lat)

    # Convert to geocentric apparent azimuth and elevation.
    sunaz, sunel = palpy.de2h(lst - sunrap, sundap, lat)
    palpy_moonaz, palpy_moonel = palpy.de2h(lst - moonrap, moondap, lat)

    # Desired pointing position, antisolar azimuth and ZD = FLAT_ZD.
    palpy_targaz = palpy.dranrm(sunaz + math.pi)
    palpy_targel = 0.5*math.pi - FLAT_ZD
    
    vt = palpy.dcs2c(az, el)
    vm = palpy.dcs2c(moonaz, moonel)

    palpy_moonsep = palpy.dsepv(vt, vm)

    # # Check proximity to the moon.
    # MOON_SEP_MIN = 45.0 * lfa.DEG_TO_RAD
    # if moonsep < MOON_SEP_MIN:
    #   # Angle by which we need to rotate.
    #   theta = MOON_SEP_MIN - moonsep
      
    #   st = numpy.sin(theta)
    #   ct = numpy.cos(theta)

    #   # Rotate using Rodrigues' rotation formula about unit vector
    #   # k = (vm x vt) / |vm x vt| to rotate directly away from the moon.
    #   # vrot = ct * vt + st * k x vt + (1-ct) * (k.vt) k
    #   # last term is always zero.
    #   tmp = numpy.cross(vm, vt)
    #   k = tmp / numpy.linalg.norm(tmp)
      
    #   vrot = ct * vt + st * numpy.cross(k, vt)

    #   # Recompute angles.
    #   az, el = palpy.dcc2s(vrot)

    #   # Check elevation limit.
    #   if el < MIN_EL:
    #     logging.info("Elevation {0:.1f} deg beyond limit, clamping".format(el * lfa.RAD_TO_DEG))
    #     el = MIN_EL
    #     vt = palpy.dcs2c(az, el)
    #   else:
    #     vt = vrot

    #   moonsep = palpy.dsepv(vt, vm)
      
    #   logging.info("Final separation from moon {0:.1f} deg".format(moonsep * lfa.RAD_TO_DEG))
    
    # Transform to astrometric.
    apha, apde = palpy.dh2e(az, el, lat)
    apra = palpy.dranrm(lst - apha)

    centra, centde = palpy.amp(apra, apde, tt, 2000.0)
    
    print (palpy_targaz, palpy_targel, palpy_moonaz, lfa_moonillum, palpy_moonsep)


if __name__ == "__main__":
    main()

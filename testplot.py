#!/usr/bin/env python

'''
To use, pass code targetname, obsdate, savefig=True/False, and lcx.txt files
'''

from __future__ import print_function

import math
import numpy
import re
import sys

import matplotlib.gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker

from medsig import *
from poly import *
import pdb

decimal_to_ppm = (10)**6

#User-input 
idealap = 10.0 #pixel, change as desired
idealbinsize = 10.0 #mins, change as desired

#nap = len(sys.argv[1:])
#targname = 'GJ1105_short'
#obsdate = '20220308'

# Input name and date directly at command line
targname = sys.argv[1]
obsdate = sys.argv[2]
savefig = sys.argv[3] #must be True or False
n = 4
nap = len(sys.argv[n:])
apsize = numpy.empty([nap])
binsize = numpy.linspace(0.2, 10, 50)

std = numpy.empty([nap, len(binsize)+1])
decstd = numpy.empty([nap, len(binsize)+1])
theo = numpy.empty([nap, len(binsize)+1])

plt.figure(figsize=(10, 15))

gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=(2, 1))

#for iap, filename in enumerate(sys.argv[1:]):
for iap, filename in enumerate(sys.argv[n:]):
  mm = re.search(r'(\d+)', filename)
  gg = mm.groups()
  apsize[iap] = float(gg[0])

  lc = numpy.loadtxt(filename)

  mjd = lc[:,0] - lc[0,0]
  flux = lc[:,12]
  e_flux = lc[:,13]
  texp = lc[:,14]
  hfd = lc[:,9]
  airmass = lc[:,15]
  skylev = lc[:,7]

  x = skylev - numpy.median(skylev)

  coef = clippolyfit(x, flux, 1)

  decflux = flux - coef[1] * x

  cad = numpy.median(mjd[1:]-mjd[0:-1]) * 1440

  norm = numpy.median(flux)

  flux /= norm
  e_flux /= norm

  decnorm = numpy.median(decflux)

  decflux /= norm

  medflux, sigflux = medsig(flux)

  thisflag = numpy.absolute(flux-medflux) < 5*sigflux

  mjd = mjd[thisflag]
  flux = flux[thisflag]
  e_flux = e_flux[thisflag]
  texp = texp[thisflag]
  hfd = hfd[thisflag]
  airmass = airmass[thisflag]
  skylev = skylev[thisflag]
  decflux = decflux[thisflag]

  # Scintillation, Young formula.
  DIAM = 1300
  HEIGHT = 2345

  ssc = 0.09 * (DIAM/10.0)**(-2.0/3.0) * numpy.power(airmass, 3.0/2.0) * math.exp(-HEIGHT/8000.0) / numpy.sqrt(2*texp)

  e_flux = numpy.hypot(e_flux, ssc)

  std[iap,0] = numpy.std(flux)

  for ibinsize, thisbinsize in enumerate(binsize):
    nbin = (mjd[-1] - mjd[0]) * 1440.0 / thisbinsize

    bins = mjd[0] + thisbinsize * numpy.arange(nbin+1) / 1440.0
    
    wt = 1.0 / numpy.square(e_flux)
    
    ybn = numpy.histogram(mjd, bins=bins, weights=flux*wt)[0]
    ybnd = numpy.histogram(mjd, bins=bins, weights=decflux*wt)[0]
    ybd = numpy.histogram(mjd, bins=bins, weights=wt)[0]
    
    wb = ybd > 0
    
    binned_flux = ybn[wb] / ybd[wb]
    binned_decflux = ybnd[wb] / ybd[wb]

    std[iap,ibinsize+1] = numpy.std(binned_flux)
    decstd[iap,ibinsize+1] = numpy.std(binned_decflux)
    theo[iap,ibinsize+1] = numpy.sqrt(numpy.mean(1.0 / ybd[wb]))
#    if abs(thisbinsize-0.2) < 0.1:
    if abs(thisbinsize-idealbinsize) < 0.1:
      x = 0.5*(bins[0:-1] + bins[1:])  # bin centres
      x = x[wb]

      print(apsize[iap], std[iap,ibinsize+1])

      gslc = matplotlib.gridspec.GridSpecFromSubplotSpec(4, 1, height_ratios=(3, 1, 1, 1), hspace=0, subplot_spec=gs[0])

      figlc = plt.subplot(gslc[0])

      plt.title("{0} test photometry, {1}, {2}pix ap".format(targname,obsdate,float(idealap)))
#      plt.title("Gl 905 test photometry, 2021-10-27 UT")
#      plt.title("GJ 1288 test photometry, 2021-10-28 UT")

      plt.errorbar(mjd*24, flux, e_flux, fmt="none", color="grey", alpha=0.25, label="{0:.2f} min original data".format(cad))
      plt.plot(x*24, binned_flux, "o", color="black", label="{0:.1f} min bin".format(thisbinsize))
#      plt.plot(x*24, binned_decflux, "o", color="black", label="{0:.1f} min bin".format(thisbinsize))

      plt.gca().xaxis.set_visible(False)

      plt.ylabel("Normalized flux")

      plt.xlim(mjd[0]*24, mjd[-1]*24)
      plt.ylim(0.991, 1.009)

      plt.legend()

      plt.subplot(gslc[1], sharex=figlc)

      plt.plot(mjd*24, hfd*0.4316, ".", color="black")

      plt.gca().xaxis.set_visible(False)
      plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

      plt.ylabel("Seeing (\")")

      plt.subplot(gslc[2], sharex=figlc)

      plt.plot(mjd*24, skylev*5.9, ".", color="black")

      plt.gca().xaxis.set_visible(False)
      plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.0f"))

      plt.ylabel("Sky (e-)")

      plt.subplot(gslc[3], sharex=figlc)

      plt.plot(mjd*24, airmass, ".", color="black")

      plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
      plt.gca().invert_yaxis()

      plt.ylabel("Airmass")

      plt.xlim(mjd[0]*24, mjd[-1]*24)

      plt.xlabel("Time (hours)")

ax = plt.subplot(gs[1])

#for iap, filename in enumerate(sys.argv[1:]):
for iap, filename in enumerate(sys.argv[n:]):
  if apsize[iap] == idealap:
    idealap_ind = iap # Added JGM Jan192022
    plt.plot(binsize, std[iap,1:]*1e6, label="Binned rms".format(apsize[iap]))
    plt.plot(binsize, decstd[iap,1:]*1e6, label="Decorrelated (sky)".format(apsize[iap]))
    plt.plot(binsize, theo[iap,1:]*1e6, label="Theoretical rms".format(apsize[iap]))

plt.grid(color="grey", linestyle="dotted", alpha=0.5)

plt.legend()

#
# add secondary axis
#Rp_per_Rsun = 109.0763707
#M3_radius = 0.3 # estimate to improve
#x_sigma = 1
# Convert y1 to y2
#pdb.set_trace()
#P_REarth = lambda P_ppm: numpy.sqrt(x_sigma*P_ppm/decimal_to_ppm) * M3_radius * Rp_per_Rsun
# Convert y2 to y1
#P_ppm = lambda P_REarth: (1/x_sigma)*(P_REarth/(M3_radius * Rp_per_Rsun))**2
#ax2 = ax.secondary_yaxis("right", functions=(P_REarth, P_ppm))
#ax2.set_ylabel(r"Photometric Precision Required for ${}\sigma$ Detection (Earth radii)".format(x_sigma), wrap=True)
#ax2.set_ylabel(r"Earth radii".format(x_sigma), wrap=True)
#ax2.tick_params(direction='in', length=6, width=3, colors='black')
# 

#plt.xlim(0, numpy.max(binsize))
#plt.ylim(0, numpy.max(std[idealap_ind])*1000*1.05)

plt.xlabel("Bin size (minutes)")
plt.ylabel("RMS (ppt)")
plt.ylabel("RMS (ppm)")
if savefig== 'True':
  plt.savefig('{0}_{1}_{2}pix_jgm.pdf'.format(obsdate,targname,idealap), format='pdf')
  plt.show()
elif savefig == 'False':
  plt.show()
else:
  #plt.show()
  #print ('user forgot arg specifying whether to save figure')
  print ('pass code targetname, obsdate, savefig=True/False, and lcx.txt files')

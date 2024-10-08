import argparse
from imred import *
from glob import glob 
import numpy as np 
import os 
import re 
import lfa
import matplotlib.pyplot as plt 
plt.ion()

def get_bias_regions(date, field, ffname='flat0000'):

    # Define base paths    
    ipath = '/data/tierras/incoming'
    fpath = '/data/tierras/flattened'
    lcpath = '/data/tierras/lightcurves'
    ffolder = fpath+f'/{date}/{field}/{ffname}'
    
    # get list of incoming files for this target on this night
    filelist = glob(ipath+f'/{date}/*{field}*.fit')
    filelist = np.array(sorted(filelist, key=lambda x:int(x.split('.')[1])))
    n_files = len(filelist)

    # Reduce each FITS file from date and target
    irobj = imred()

    # declare an array to hold the stiched-together bias regions 

    bias_regions = np.zeros((n_files, 2048, 48))

    for ifile,filename in enumerate(filelist):
        print(f'{date}: {ifile + 1} of {n_files}')
        # Open file.
        ifp = pyfits.open(filename)

        # Save primary header.
        prihdr = ifp[0].header

        iimg = 0
        for iext, imp in enumerate(ifp):
            if hasimg(imp):
                hdr = imp.header
                biassec = fits_section(hdr["BIASSEC"])
                raw = numpy.float32(imp.data)
                # ny, nx = raw.shape

                bias_region = raw[biassec[2]:biassec[3],biassec[0]:biassec[1]]
                # biaslev, biassig = lfa.skylevel_image(bias_region)

                if iext == 1:
                    row_inds = np.arange(0,1024)
                elif iext == 2:
                    row_inds = np.arange(1024, 2048)
                bias_regions[ifile, row_inds, :] = bias_region

    # calculate the median across the columns of the bias regions and transpose
    # this leaves you with a 2048 x n_files array 
    med_bias = np.median(bias_regions, axis=2).T

    # take the median across the exposures 
    # this leaves you with a 2048-element array representing the median bias level across all exposures on this night as a function of row number 
    med_bias_night = np.median(med_bias, axis=1)

    # plt.plot(med_bias)
    # plt.plot(med_bias_night, color='k', lw=2)

    return bias_regions, med_bias, med_bias_night

if __name__ == '__main__':
    dates = ['20240429', '20240430', '20240501', '20240502', '20240503', '20240504', '20240508', '20240509', '20240510', '20240511', '20240512', '20240527', '20240529', '20240530', '20240601', '20240602', '20240603', '20240709', '20240712']
    target = 'TIC362144730'

    plt.figure(figsize=(15,10))
    plt.ylabel('Bias Level (ADU)', fontsize=14)
    plt.xlabel('Row Number', fontsize=14)

    cmap = plt.get_cmap('viridis')

    color_inds = np.array(255*np.arange(len(dates))/(len(dates)-1), dtype='int')

    bias_regions = [] 
    med_biases = []
    med_bias_nights = []
    for i in range(len(dates)):
        date = dates[i]
        b_, med_b_, med_b_night_ = get_bias_regions(date, target)
        bias_regions.append(b_)
        med_biases.append(med_b_)
        med_bias_nights.append(med_b_night_)
    
        plt.plot(med_bias_nights[i], label=date, color=cmap(color_inds[i]))
        plt.pause(0.1)

    plt.legend()
    plt.tight_layout()
    breakpoint()

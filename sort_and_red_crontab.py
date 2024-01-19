import logging
import argparse 
import numpy 
import sys
import os
import re
import glob
import pdb

import numpy as np
from scipy import stats
from astropy.io import fits 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from imred import *

def create_directories(basepath,date,target,folder1):
    datepath = os.path.join(basepath,date)
    targetpath = os.path.join(datepath,target)
    folder1path = os.path.join(targetpath,folder1)

    if not os.path.exists(datepath):
        logging.info('Creating {} Directory'.format(datepath))
        os.system('mkdir {}'.format(datepath))
        os.system('chmod g+rwx {}'.format(datepath))
        os.system('chmod g+rwx {}/*'.format(datepath))
        
    if not os.path.exists(targetpath):
        logging.info('Creating {} Directory'.format(targetpath))
        os.system('mkdir {}'.format(targetpath))
        os.system('chmod g+rwx {}'.format(targetpath))
        os.system('mkdir {}'.format(folder1path))
        os.system('chmod g+rwx {}'.format(folder1path))
        os.system('chmod g+rwx {}/*'.format(folder1path))

    return folder1path

def make_filelist(basepath,date,target):
    fullpath = os.path.join(basepath,date)
    return sorted([os.path.join(fullpath,f) for f in os.listdir(fullpath) if target+'.fit' in f])

def get_target_list(datepath):
    flist = os.listdir(datepath)
    target_list = set([flist[ind].split('.')[2] for ind in range(len(flist))])
    if 'FLAT001' in target_list:
        target_list = [targetname for targetname in target_list if not targetname.startswith('FLAT')]
        #print (target_list)
    return target_list

def get_yday_date():
    # Get today's date
    today = datetime.now()
    # Calculate yesterday's date
    yesterday = today - timedelta(days=1)
    # Format yesterday's date as "YYYYMMDD"
    formatted_date = yesterday.strftime("%Y%m%d")
    # Print the result
    return formatted_date

def main():

    # Argparser for flatfielding (but note ffname is fixed)
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", help="flat file with which to reduce data")
    args = ap.parse_args()

    # Access observation info
    ffname = "flat0000" #eventually will have to upgrade to actually pass a flat file
    date = get_yday_date()

    # Define base paths    
    ipath = '/data/tierras/incoming'
    fpath = '/data/tierras/flattened'
    lcpath = '/data/tierras/lightcurves'

    # Get target names observed
    targets = get_target_list(os.path.join(ipath,date))

    for target in targets:
        if os.path.exists(os.path.join(fpath,date,target,ffname))
            continue
        # Create flattened file and light curve directories 
        ffolder = create_directories(fpath,date,target,ffname)

        # Set up logger
        logfile = os.path.join(ffolder,'{}.{}.redlog.txt'.format(date,target))
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=logfile, level=logging.INFO)

        # Get list of files to be reduced
        filelist = make_filelist(ipath,date,target)

        # Reduce each FITS file from date and target
        logging.info('Reducing {} FITS files...'.format(len(filelist)))
        irobj = imred(args.f)
        rfilelist = []
        for ifile,filename in enumerate(filelist):
            logging.info(filename)
            ohl = irobj.read_and_reduce(filename,stitch=True)
            basename = os.path.basename(filename)
            rfilename = re.sub('\.fit','',basename)+'_red.fit'
            rfilename = os.path.join(ffolder,rfilename)
            ohl.writeto(rfilename,overwrite=True)
            rfilelist.append(rfilename)
            logging.info(filename+ " -> " +rfilename)

        # Exclude files where ASTROM solution fails and exptime diff. to mode of stack
        logging.info('Checking astrometric solution on plate solved files...')

        exptimes = np.array([])
        badfiles = np.array([])
        stdcrms_lst = np.array([])
        numbrms_lst = np.array([])

        for irfile,rfilename in enumerate(rfilelist):
            if '_red' in rfilename:
                hdr_ind = 0
            elif '_red' not in rfilename:
                hdr_ind = 1
            hdr = fits.getheader(rfilename,hdr_ind)

            #read out exp time BEFORE astrom params
            exptime = hdr['EXPTIME']
            exptimes = np.append(exptimes, exptime)

            # read out astrometric fit coordinate rms (arcsec)
            # and number of astrometric standards used. 
            try:
                stdcrms, numbrms = hdr['STDCRMS'], hdr['NUMBRMS']
            except KeyError:
                logging.info('Astrom failed for:')
                logging.info(rfilename)
                badfiles = np.append(badfiles,rfilename)
                continue
            stdcrms_lst = np.append(stdcrms_lst,stdcrms)
            numbrms_lst = np.append(numbrms_lst, numbrms)

        logging.info('Astrometry checks: Done.')

        # Add any files with different exposure time to flagged list
        texp = stats.mode(exptimes)[0][0]
        logging.info('Stack texp = {} s.'.format(texp))
        for irfile,rfilename in enumerate(rfilelist):
            if exptimes[ifile] != texp:
                logging.info('texp = {} s for:'.format(exptimes[ifile]))
                logging.info(rfilename)
                badfiles = np.append(badfiles,rfilename)
        logging.info('Exposure time checks: Done.')

        # Save flagged file list for future ref
        if len(badfiles) >= 1:
            logging.info('Saved a list of flagged files.')
        elif len(badfiles) == 0:
            logging.info('No files were flagged.')
        np.savetxt(os.path.join(ffolder,'{}.{}.flagged_files.txt'.format(date,target)),np.unique(badfiles),fmt='%s')

        # Make excluded directory and move flagged files there
        excfolder = os.path.join(ffolder,'excluded/')
        logging.info(excfolder)
        os.system('mkdir '+excfolder)
        for badfile in badfiles:
            os.system('mv ' + badfile + ' '+ excfolder)

        # Save histogram with astrom solution stdv and number of stars used
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,7))
        fig.suptitle('Astrometry Evaluation: {} {}'.format(date,target))

        ax1.hist(stdcrms_lst, 20)
        ax1.set_xlabel('Astrometric fit coord rms (arcsec)')
        ax1.set_ylabel('Number of Exposures')

        ax2.hist(numbrms_lst, 20)
        ax2.set_ylabel('Number of Exposures')
        ax2.set_xlabel('Number of astrometric standards used')
        histogram = os.path.join(ffolder,"{}.{}_astrom_hist.pdf".format(date,target))
        fig.savefig(histogram)
        logging.info('Saved two histograms summarizing the astrometric solution.')
        logging.info('Data reduction for {} on {} done.'.format(target,date))
        logging.info('Data reduction report emailed')

        # Send log file and STDRMS/NUMBRMS .pdf file to email
        subject = '[Tierras]_Data_Reduction_Report:{}_{}'.format(date,target)
        append = '{} {}'.format(logfile,histogram)
        #append = '{}'.format(histogram)
        emails = 'juliana.garcia-mejia@cfa.harvard.edu patrick.tamburo@cfa.harvard.edu'
        os.system('echo | mutt {} -s {} -a {}'.format(emails,subject,append))

if __name__ == '__main__':
    main()
    

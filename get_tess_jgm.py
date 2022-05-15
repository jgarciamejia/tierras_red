"""Ryan Cloutier's code to read TESS data.
JGM modified path to be compatible with local machine"""

from imports import *
import os, requests
from bs4 import BeautifulSoup

# add path to use as outdir in JGM Mac
path = '/Users/jgarciamejia/Downloads'


def tic_data_url(ticid, sector):
    url = 'https://archive.stsci.edu/missions/tess/tid/s%.4d/'%sector
    ticstr = '%.16d'%ticid
    url += '%s/'%ticstr[:4]
    url += '%s/'%ticstr[4:8]
    url += '%s/'%ticstr[8:12]
    url += '%s/'%ticstr[12:]
    return url, listFD(url)


def listFD(url, ext=''):
    '''List the contents of an https directory.'''
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return np.array([url + '/' + node.get('href') \
                     for node in soup.find_all('a') \
                     if node.get('href').endswith(ext)])


def download_dvs(ticid, sector, ext='dvm.pdf',
                 outdir=path, opendvs=True):
    '''Download the data summary report for a given TIC.'''
    url,_ = tic_data_url(ticid, sector)
    fs = listFD(url, ext=ext)
    print('URL (%s) contains %i DVS reports.'%(url, len(fs)))
    for f in fs:
        os.system('wget %s'%f)
        os.system('mv %s %s'%(f.split('//')[-1], outdir))

    if opendvs:
        for f in fs:
            os.system('open %s/%s'%(outdir, f.split('//')[-1]))


def get_tess_LC(ticid, sector, outdir=path,
                pltt=True, SAP=False):
    '''Download and return the light curve.'''
    url,_ = tic_data_url(ticid, sector)
    fs = listFD(url, ext='lc.fits')
    print('URL (%s) contains %i 0light curve files.'%(url, len(fs)))
    bjd, f, ef, sectors = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    for fi in fs:
        os.system('wget %s'%fi)
        os.system('mv %s %s'%(fi.split('//')[-1], outdir))

        # read-in data
        hdr = fits.open('%s/%s'%(outdir, fi.split('//')[-1]))
        bjd = np.append(bjd, hdr[1].data['TIME'])

        # normalize the data
        label = 'SAP' if SAP else 'PDCSAP'
        ftmp = hdr[1].data['%s_FLUX'%label]
        ef = np.append(ef, hdr[1].data['%s_FLUX_ERR'%label] / np.nanmedian(ftmp))
        f = np.append(f, ftmp / np.nanmedian(ftmp))

        # save sector
        sectors = np.append(sectors, np.repeat(int(fi.split('/s')[1].split('/')[0]), ftmp.size))        
        
    # clean and sort the data
    g = np.isfinite(bjd) & np.isfinite(f) & np.isfinite(ef)
    s = np.argsort(bjd[g])
    bjd, f, ef, sectors = bjd[g][s], f[g][s], ef[g][s], sectors[g][s]

    # plot if desired
    if pltt:
        plt.errorbar(bjd, f, ef, fmt='k.', alpha=.2)
        plt.xlabel('BJD - 2457000', fontsize=12)
        plt.ylabel('Normalized flux', fontsize=12)
        
    return bjd, f, ef, sectors
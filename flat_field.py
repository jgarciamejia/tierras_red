#!/usr/bin/env python
# TO run, apss the following to the command line: 
# script path N 1(0)
# path: full path to flats 
# N: number of flats to median filter and combine
# 1(0): type 1 to use evening set of flats, 0 to use mornign set of flats 

from imports import *

MAD = lambda x: np.median(abs(x-np.median(x)))

def construct_master_flat(path, kernel_size=49, redo_medfilt=False, Nflats=0, evening=True):
    '''
    Construct median filtered flat frames, bias subtract them, and then 
    combine them to create a master flat for each chip on the Tierras detector. 
    '''
    # create and read-in median-filtered flat field images
    print('Median filtering the individual flat frames...')
    kwargs = {'kernel_size': int(kernel_size), 'overwrite': bool(redo_medfilt), 'Nflats':int(Nflats), 'evening': bool(evening)}
    _median_filter_flats(path, **kwargs)

    fs = np.sort(glob.glob('%s/*FLAT*_medfilt_kernel%i*'%(path,kernel_size)))
    assert fs.size > 0
    
    # construct master flat after normalizing the individual flat frames
    print('Constructing the master flat...')
    master_flat = np.zeros((fs.size, 2, cs.xpix, cs.ypix))
    for i,f in enumerate(fs):
        # get flat frame and median-filtered+bias-subtracted flat frame
        frame_str = fs[0].split('.FLAT')[0].split('.')[-1]
        fsv2 = np.sort(glob.glob('%s/*%s.FLAT*fit'%(path,frame_str)))
        assert fsv2.size == 2
        hdu = fits.open(fsv2[0])
        medhdu = fits.open(fsv2[1])

        # median-normalize the flat frame
        master_flat[i,0] = hdu[1].data / medhdu[1].data
        master_flat[i,1] = hdu[2].data / medhdu[2].data
    
    # median combine each median-flattened flat to create the master flat
    hdu = fits.open(fs[0])
    hdu0 = fits.PrimaryHDU(header=hdu[0].header)
    hdu1 = fits.ImageHDU(np.nanmedian(master_flat[:,0],0), header=hdu[1].header)
    hdu2 = fits.ImageHDU(np.nanmedian(master_flat[:,1],0), header=hdu[2].header)
    hdu = fits.HDUList([hdu0, hdu1, hdu2])
    hdu.writeto('%s/MASTERFLAT_mednorm.fit'%path, overwrite=True)
    
    
    
def _median_filter_flats(path, kernel_size=49, Nflats=0, evening=True, overwrite=False):
    '''
    Median filter each individual sky flat and save to a new fits file.
    '''
    # get each chip's scalar bias value
    _,bias1,bias2 = _derive_bias_value(path, verbose=False)
    assert np.all(np.isfinite([bias1,bias2]))

    kernel_size = int(kernel_size+1) if kernel_size % 2 == 0 else int(kernel_size)
    
    # median filter each flat frame
    #N = int(Nflats) if Nflats > 0 else 1000
    #fs = np.sort(glob.glob('%s/*FLAT*'%path))[:N]
    inds = _get_flat_indices(path, Nflats, evening)
    fs = np.sort(glob.glob('%s/*FLAT*'%path))[inds]
    assert fs.size > 0
    for i,f in enumerate(fs):
        
        if ('medfilt' in f) | ('MASTER' in f):
            continue

        with fits.open(f) as hdu:
            # trim overscan (i.e. bias) and smear regions
            img1 = hdu[1].data.astype(float)
            img2 = hdu[2].data.astype(float)
            img1_trimmed = _trim_edges(img1)
            img2_trimmed = _trim_edges(img2)
            
            # compute median-filtered flat field if not already done so
            fout = f.replace('.fit','_medfilt_kernel%i.fit'%kernel_size)
            if (not os.path.exists(fout)) | (overwrite):

                print('%i out of %i (%s)'%(i+1,fs.size,f))

                # calculate spatial median + trim edges + bias-subtract
                medfilt1_trimmed = median_filter(img1_trimmed - bias1, size=kernel_size)
                medfilt2_trimmed = median_filter(img2_trimmed - bias2, size=kernel_size)

                # add overscan and smear regions as NaNs
                medfilt1 = _add_nan_edges(medfilt1_trimmed)
                medfilt2 = _add_nan_edges(medfilt2_trimmed)
            
                # save median-filtered and bias-subtracted image to a new file
                hdu0 = fits.PrimaryHDU(header=hdu[0].header)
                hdu1 = fits.ImageHDU(medfilt1, header=hdu[1].header)
                hdu2 = fits.ImageHDU(medfilt2, header=hdu[2].header)
                hdu = fits.HDUList([hdu0, hdu1, hdu2])
                hdu.writeto(fout, overwrite=True)
                


def _get_flat_indices(path, N, evening=True):
    fs = np.sort(glob.glob('%s/*FLAT*'%path))
    file_inds = []
    for i,f in enumerate(fs):
        if ('medfilt' in f) | ('MASTER' in f):
            continue
        file_inds.append(int(f.split('.')[1]))
    
    start = 0 if evening else np.where(np.diff(file_inds) > 1)[0][0]+1
    return range(start, start+N)



def _add_nan_edges(img_trimmed, fill_value=np.nan):
    '''
    Add NaN edges to a trimmed image
    '''
    assert img_trimmed.shape == (cs.xpix-cs.smear_width, cs.ypix-2*cs.overscan_width)
    x,y = np.meshgrid(np.arange(cs.xpix-cs.smear_width),
                      np.arange(cs.overscan_width, cs.ypix-cs.overscan_width))
    img = np.zeros((cs.xpix, cs.ypix)) + np.nan
    img[x,y] = img_trimmed.T
    return img



def _trim_edges(img):
    '''
    Define the overscan and smear regions and trim them so they won't
    contribute to the median operation
    '''
    assert img.shape == (cs.xpix, cs.ypix)
    x,y = np.meshgrid(np.arange(cs.xpix-cs.smear_width),
                      np.arange(cs.overscan_width, cs.ypix-cs.overscan_width))
    return img[x,y].T

    
        
## get bias value from the bias frames (or from the overscan regions) 
# http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/instruments/L12/index.html
def _derive_bias_value(path, verbose=True, bias_frame=False):
    '''
    Calculate the scalar bias value by averaging the nightly BIAS frames. 
    '''
    if bias_frame:
        fs = np.sort(glob.glob('%s/*BIAS*'%path))
    else:
        fstmp = np.sort(glob.glob('%s/*fit'%path))
        fs = []
        for i,f in enumerate(fstmp):
            if 'BIAS' not in f and \
               'FLAT' not in f and \
               'DARK' not in f and \
               len(fits.open(f)) == 3: 
                fs.append(f)
                break

    out = np.zeros((len(fs),2,4))
    for i,f in enumerate(fs):
        hdu = fits.open(f)
                
        # get image
        img1 = hdu[1].data if bias_frame else np.append(hdu[1].data[:,:cs.overscan_width], hdu[1].data[:,-cs.overscan_width:])
        img2 = hdu[2].data if bias_frame else np.append(hdu[2].data[:,:cs.overscan_width], hdu[2].data[:,-cs.overscan_width:])

        # save stats
        out[i,0,:] = img1.mean(), img1.std(), np.median(img1), MAD(img1)
        out[i,1,:] = img2.mean(), img2.std(), np.median(img2), MAD(img2)
        
        if verbose:
            print('File: %s'%f)
            print('\tChip 1: mean = %.2f, sd = %.2f, med = %.2f, mad = %.2f'%tuple(out[i,0,:]))
            print('\tChip 2: mean = %.2f, sd = %.2f, med = %.2f, mad = %.2f\n'%tuple(out[i,1,:]))
    
    # return bias value for each chip
    bias1 = np.median(out[:,0,2])
    bias2 = np.median(out[:,1,2])
    
    return out, bias1, bias2



if __name__ == '__main__':
    path = sys.argv[1]
    Nflats = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    evening = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    kernel_size = int(sys.argv[4]) if len(sys.argv) > 4 else 49

    kwargs = {'kernel_size': kernel_size,
              'Nflats': Nflats,
              'evening': evening,
              'redo_medfilt': False}
     
    construct_master_flat(path, **kwargs)

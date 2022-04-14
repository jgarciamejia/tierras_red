from imports import *


MAD = lambda x: np.median(abs(x-np.median(x)))


def construct_master_flat(path, kernel_size=49):
    '''
    Construct median filtered flat frames, bias subtract them, and then 
    combine them to create a master flat for each chip on the Tierras detector. 
    '''
    # create and read-in median-filtered flat field images
    print('Median filtering the individual flat frames...')
    _median_filter_flats(path, kernel_size=kernel_size)
    fs = np.sort(glob.glob('%s/*FLAT*_medfilt_kernel%i*'%(path,kernel_size)))
    assert fs.size > 0
    
    # construct master flat after normalizing the individual flat frames
    print('Constructing the master flat...')
    master_flat = np.zeros((fs.size, xpix*2, ypix))
    for i,f in enumerate(fs):
        # get flat frame and median-filtered+bias-subtracted flat frame
        frame_str = fs[0].split('.')[1]
        fsv2 = np.sort(glob.glob('%s/*%s.FLAT*fit'%(path,frame_str)))
        assert fsv2.size == 2
        hdu = fits.open(fsv2[0])
        medhdu = fits.open(fsv2[1])

        # median-normalize the flat frame
        master_flat[i,:xpix,:] = hdu[1].data / medhdu[1].data
        master_flat[i,xpix:,:] = hdu[2].data / medhdu[2].data
    
    # modify the output header
    hdu = fits.open(fs[0])
    assert hdu[1].header['NAXIS2'] == xpix
    hdu[1].header['NAXIS2'] = xpix*2
    
    # median combine each median-flattened flat to create the master flat
    hdu0 = fits.PrimaryHDU(header=hdu[0].header)
    hdu1 = fits.ImageHDU(np.nanmedian(master_flat,0), header=hdu[1].header)    
    hdu = fits.HDUList([hdu0, hdu1])
    hdu.writeto('%s/MASTERFLAT_mednorm.fit'%path, overwrite=True)
    
    
    
def _median_filter_flats(path, kernel_size=49):
    '''
    Median filter each individual sky flat and save to a new fits file.
    '''
    # get each chip's scalar bias value
    _,bias1,bias2 = _derive_bias_value(path, verbose=False)

    kernel_size = int(kernel_size+1) if kernel_size % 2 == 0 else int(kernel_size)
    
    # median filter each flat frame
    fs = np.sort(glob.glob('%s/*FLAT*'%path))
    for i,f in enumerate(fs):
        
        if ('medfilt' in f) | ('MASTER' in f):
            continue

        print('%i out of %i (%s)'%(i+1,fs.size,f))
        with fits.open(f) as hdu:
            # convert to float
            img1 = np.copy(hdu[1].data.astype(float))
            img2 = np.copy(hdu[2].data.astype(float))
        
            # trim: set overscan (i.e. bias) and smear regions to NaN
            img1 = _edges_to_nan(hdu, img1)
            img2 = _edges_to_nan(hdu, img2)

            # compute median-filtered flat field if not already done so
            fout = f.replace('.fit','_medfilt_kernel%i.fit'%kernel_size)
            if not os.path.exists(fout):
                # calculate spatial median + trim edges + bias-subtract
                medfilt1 = _edges_to_nan(hdu, median_filter(img1, size=kernel_size)) - bias1
                medfilt2 = _edges_to_nan(hdu, median_filter(img2, size=kernel_size)) - bias2

                # save median-filtered and bias-subtracted image to a new file
                hdu0 = fits.PrimaryHDU(header=hdu[0].header)
                hdu1 = fits.ImageHDU(_edges_to_nan(hdu, medfilt1), header=hdu[1].header)
                hdu2 = fits.ImageHDU(_edges_to_nan(hdu, medfilt2), header=hdu[2].header)
                hdu = fits.HDUList([hdu0, hdu1, hdu2])
                hdu.writeto(fout, overwrite=True)
                


def _edges_to_nan(hdu, img, fill_value=np.nan):
    # mask overscan (i.e. bias) and smear regions
    xind = np.arange(hdu[1].header['NAXIS2'] - smear_width, hdu[1].header['NAXIS2'])
    yind = np.append(np.arange(overscan_width), 
                     np.arange(hdu[1].header['NAXIS1'] - overscan_width, hdu[1].header['NAXIS1']))
    img[xind] = fill_value
    img[:,yind] = fill_value
    return img


        
## get bias value from the bias frames (or from the overscan regions) 
# http://slittlefair.staff.shef.ac.uk/teaching/phy217/lectures/instruments/L12/index.html
def _derive_bias_value(path, verbose=True):
    '''
    Calculate the scalar bias value by averaging the nightly BIAS frames. 
    '''
    
    fs = np.sort(glob.glob('%s/*BIAS*'%path))
    
    out = np.zeros((fs.size,2,4))
    for i,f in enumerate(fs):
        hdu = fits.open(f)
                
        # save stats
        out[i,0,:] = hdu[1].data.mean(), hdu[1].data.std(), np.median(hdu[1].data), MAD(hdu[1].data)
        out[i,1,:] = hdu[2].data.mean(), hdu[2].data.std(), np.median(hdu[2].data), MAD(hdu[2].data)
        
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

    kernel_size = int(sys.argv[2]) if len(sys.argv) > 2 else 49

    kwargs = {'kernel_size': kernel_size}
    construct_master_flat(path, **kwargs)

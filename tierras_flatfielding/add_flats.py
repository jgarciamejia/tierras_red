import os
import numpy as np 
from astropy.io import fits
import flat_field_custom as ffc

bpath = '/data/tierras/incoming'
days = ['20220520','20220528','20220605','20220613','20220706']

sum_flat_half1 = np.zeros((1049,4196))
sum_flat_half2 = np.zeros((1049,4196))
for di,day in enumerate(days):
    hdu = fits.open('/data/tierras/incoming/{}/MASTERFLAT_daily_mednorm.fit'.format(day))
    sum_flat_half1 += hdu[1].data
    sum_flat_half2 += hdu[2].data

hdu0 = fits.PrimaryHDU(header=hdu[0].header)
hdu1 = fits.ImageHDU(sum_flat_half1, header=hdu[1].header)
hdu2 = fits.ImageHDU(sum_flat_half2, header=hdu[2].header)
hdu = fits.HDUList([hdu0,hdu1,hdu2])
fits_fname = '/data/tierras/flat_sum.fit'
#hdu.writeto(fits_fname,overwrite=True)

# addendum to print bias values per date per chip half
bpath = '/data/tierras/incoming'
for d in days: 
    out,b1,b2 = ffc._derive_bias_value(os.path.join(bpath,d))
    


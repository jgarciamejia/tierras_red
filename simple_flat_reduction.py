import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
from astropy.io import fits 
from astropy.visualization import simple_norm 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ap_phot import set_tierras_permissions

file = '/data/tierras/flattened/20250908/LP653-13/flat0000/20250908.0533.LP653-13_red.fit'
# file = '/data/tierras/flattened/20250914/GJ3470/flat0000/20250914.0389.GJ3470_red.fit'
hdul = fits.open(file)
im = hdul[0].data 
hdr = hdul[0].header

flat = fits.open('/data/tierras/flats/SUPERFLAT_20250908_to_20250910.fit')[0].data

norm = np.median(flat)
flat /= norm

fig, ax = plt.subplots(3,1,figsize=(10,10),sharex=True,sharey=True)

im1 = ax[0].imshow(im, origin='lower', interpolation='none', norm=simple_norm(im, min_percent=1, max_percent=99.5))
div = make_axes_locatable(ax[0])
cax = div.append_axes('right', size='5%', pad=0.05)
cb1 = fig.colorbar(im1, cax=cax, orientation='vertical')

im2 = ax[1].imshow(flat, origin='lower', interpolation='none', norm=simple_norm(flat, min_percent=1, max_percent=99.5))
div = make_axes_locatable(ax[1])
cax = div.append_axes('right', size='5%', pad=0.05)
cb2 = fig.colorbar(im2, cax=cax, orientation='vertical')



flattened_im = im / flat 

im3 = ax[2].imshow(flattened_im, origin='lower', interpolation='none', norm=simple_norm(flattened_im, min_percent=1, max_percent=99.5))
div = make_axes_locatable(ax[2])
cax = div.append_axes('right', size='5%', pad=0.05)
cb3 = fig.colorbar(im3, cax=cax, orientation='vertical')

fig.tight_layout()

output_hdul = fits.HDUList([fits.PrimaryHDU(data=flattened_im, header=hdr)])

filename = file.split('/')[-1].split('.fit')[0]+'_flat.fit'
output_path = f'/home/ptamburo/tierras/pat_scripts/flat_tests/flattened_data/{filename}'
output_hdul.writeto(output_path, overwrite=True)
set_tierras_permissions(output_path)
breakpoint()
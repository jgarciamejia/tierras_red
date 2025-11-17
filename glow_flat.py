#!/opt/cfpython/anaconda3.7/bin/python

import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
from astropy.io import fits 
from astropy.visualization import simple_norm
import argparse
from glob import glob 
import lfa 
from fitsutil import * 
from scipy.spatial import KDTree
from ap_phot import set_tierras_permissions
from scipy.stats import sigmaclip
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm 
import matplotlib.colors as colors 
from scipy.ndimage import median_filter

def process_extension(imp, iext):
	hdr = imp.header
  
	if "BIASSEC" in hdr:
		biassec = fits_section(hdr["BIASSEC"])
	else:
		biassec = None
	  
	if "TRIMSEC" in hdr:
		trimsec = fits_section(hdr["TRIMSEC"])
	else:
		trimsec = None
	  
	raw = np.float32(imp.data)

	if biassec is not None:
		biaslev, biassig = lfa.skylevel_image(raw[biassec[2]:biassec[3],biassec[0]:biassec[1]])
	else:
		biaslev = 0

	if trimsec is not None:
		procimg = raw[trimsec[2]:trimsec[3],trimsec[0]:trimsec[1]] - biaslev
	else:
		procimg = raw - biaslev

	if iext == 2:
		procimg = np.flipud(procimg) # top half of image is read in upside down under the current read-in scheme, so flip it here

	return procimg

def main(raw_args=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
	ap.add_argument("-neighborhood_size", required=False, default=20, type=int, help='The neighborhood size for the median filter which will remove the glow.')
	args = ap.parse_args(raw_args)
	date = args.date
	neighborhood_size = args.neighborhood_size

	path = f'/data/tierras/incoming/{date}/'
	flat_files = np.array(sorted(glob(path+'*FLAT*.fit')))
	n_files = len(flat_files)
	if n_files == 0:
		print(f'No flat files found in /data/tierras/incoming/{date}/!')
	else:
		print(f'Found {n_files} flat files in /data/tierras/incoming/{date}/')

	x, y = np.meshgrid(np.arange(4096), np.arange(2048))
	coordinates = np.vstack([x.ravel(), y.ravel()]).T

	combined_images = []
	# loop to read in and combine flats 
	# need to have them in memory so we can compute neighborhood pixel indices just once for each pixel location instead of n_flats times 
	for i, filename in enumerate(flat_files):
		print(f'Doing {filename} ({i+1} of {n_files})')
		with fits.open(filename) as ifp:
			proc_imgs = [process_extension(ifp[j], j) for j in range(1, 3)]
		combined_img = np.concatenate(proc_imgs, axis=0)
		combined_images.append(combined_img)

	combined_images = np.array(combined_images)
	# loop over the combined flats and reject any that are significant flux outliers 
	flat_medians = np.zeros(len(combined_images))
	for i in range(len(combined_images)):
		flat_medians[i] = np.median(combined_images[i])

	# plt.figure() 
	# plt.hist(flat_medians)

	v, l, h = sigmaclip(flat_medians, 5, 5) # do a 5-sigma outlier rejection

	# plt.axvline(l, color='tab:orange', ls='--')
	# plt.axvline(h, color='tab:orange', ls='--')
	use_inds = np.where((flat_medians > l) & (flat_medians < h))[0]
	print(f'Discarded {len(combined_images) - len(use_inds)} flats with 5-sigma median flux clipping. {len(use_inds)} flats will be combined.')

	# update arrays with use_inds 
	n_files = len(use_inds)
	flat_files = flat_files[use_inds]
	combined_images = combined_images[use_inds]
	flat_medians = flat_medians[use_inds] 

	# For each image, compute the local neighborhood median and divide
	avg_images = np.empty_like(combined_images)
	for i in range(len(combined_images)):
		print(f'Correcting flat {i+1} of {len(combined_images)}')
		neighbor_data = median_filter(combined_images[i], size=neighborhood_size)
		avg_images[i] = combined_images[i] / neighbor_data

	# avg_images = np.zeros_like(combined_images)	

	# # construct the images corrected by the median of neighboring pixels
	# n_ops = im_shape[1]*im_shape[0]
	# # norm = simple_norm(combined_images[0], min_percent=1, max_percent=99)
	# for i in range(n_ops):
	# 	coord = coordinates[i]
	# 	coord_data = combined_images[:, coord[1], coord[0]]
	# 	_, nearest_indices = tree.query(coord, k=k)
	# 	neighbor_yx = coordinates[nearest_indices[1:]]
	# 	combined_images[:, neighbor_yx[:,1], neighbor_yx[:,0]]
	# 	neighbor_data = np.median(combined_images[:, neighbor_yx[:,1], neighbor_yx[:,0]], axis=1) # an n_flat array of the median of the neighborhood_size-surrounding pixels 
	# 	avg_images[:, coord[1], coord[0]] = coord_data / neighbor_data

		
	# 	if i % 99999 == 0 and i != 0:
	# 		percent_complete = (i+1)/n_ops * 100
	# 		print(f'{percent_complete:.2f}% complete.')

	# 		# plt.figure(figsize=(12,8))
	# 		# plt.imshow(combined_images[0], origin='lower', interpolation='none', norm=norm)
	# 		# plt.plot(coord[0], coord[1], 'bx', mew=2, ms=8)
	# 		# for j in range(len(neighbor_yx)):
	# 		# 	plt.plot(neighbor_yx[j][1], neighbor_yx[j][0], marker='x', ls='', color='r', mew=2, ms=8)
	# 		# breakpoint()
	# 		# plt.close()

	# examine the behavior of slices of glow-filtered flats 
	n_central_rows = 1 
	n_central_cols = 1 
	# median_central_rows = np.median(avg_images[:,1024-int(n_central_rows/2):1024+int(n_central_rows/2),:],axis=1)
	# median_central_cols = np.median(avg_images[:,:,2048-int(n_central_cols/2):2048+int(n_central_cols/2)],axis=2)

	median_central_rows = avg_images[:,1024,:]
	median_central_cols = avg_images[:,:,2048]

	cmap = cm.viridis

	# plot central rows
	fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True, gridspec_kw={'height_ratios':[2,1]})
	norm = simple_norm(avg_images[0], min_percent=1, max_percent=99)
	im_ = ax[0].imshow(avg_images[0], origin='lower', interpolation='none', norm=norm, aspect='auto')
	save_ylim = ax[0].get_ylim()
	ax[0].fill_between(np.arange(0,4096), 1024-int(n_central_rows/2), 1024+int(n_central_rows/2), color='m', alpha=0.6)
	ax[0].set_ylim(save_ylim)

	# add a hidden colorbar to match size of bottom plot
	divider = make_axes_locatable(ax[0])
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cb = fig.colorbar(im_, cax=cax, orientation='vertical')
	cb.remove()
	cb.outline.set_visible(False)

	# now plot the rows data
	for i in range(avg_images.shape[0]):
		pl = ax[1].plot(median_central_rows[i], color=cmap(int(255*i/(avg_images.shape[0]-1))), label=f'FLAT {str(i+1).zfill(2)}')
	ax[1].set_xlabel('Column Number', fontsize=14)
	ax[1].set_ylabel('Normalized Flux', fontsize=14)
	ax[1].grid(alpha=0.5)
	v, l, h = sigmaclip(np.median(median_central_rows,axis=0))
	ax[1].set_ylim(l, h)

	divider = make_axes_locatable(ax[1])
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cb_norm = colors.Normalize(vmin=0, vmax=avg_images.shape[0])
	mappable = cm.ScalarMappable(cmap=cmap, norm=cb_norm)
	cb = fig.colorbar(mappable, cax=cax, orientation='vertical')
	ticks = np.arange(0,avg_images.shape[0],10)
	cb.ax.yaxis.set_ticks(ticks)
	cb.set_label('Flat Number')
	fig.tight_layout()

	# now do columns
	fig, ax = plt.subplots(1,2,figsize=(16,5), sharey=True, gridspec_kw={'width_ratios':[2,1]})

	ax[0].imshow(avg_images[0], origin='lower', interpolation='none', norm=norm, aspect='auto')
	save_ylim = ax[0].get_ylim()
	ax[0].fill_between(np.arange(2048-int(n_central_cols/2), 2048+int(n_central_cols/2)), 0, 2048, color='m', alpha=0.6)
	ax[0].set_ylim(save_ylim)

	for i in range(avg_images.shape[0]):
		ax[1].plot(median_central_cols[i], np.arange(0, 2048), color=cmap(int(255*i/(avg_images.shape[0]-1))))
	ax[1].set_ylabel('Row Number', fontsize=14)
	ax[1].set_xlabel('Normalized Flux', fontsize=14)
	ax[1].grid(alpha=0.5)
	v, l, h = sigmaclip(np.median(median_central_cols,axis=0))
	ax[1].set_xlim(l, h)

	divider = make_axes_locatable(ax[1])
	cax = divider.append_axes('right', size='5%', pad=0.1)
	cb_norm = colors.Normalize(vmin=0, vmax=avg_images.shape[0])
	mappable = cm.ScalarMappable(cmap=cmap, norm=cb_norm)
	cb = fig.colorbar(mappable, cax=cax, orientation='vertical')
	ticks = np.arange(0,avg_images.shape[0],10)
	cb.ax.yaxis.set_ticks(ticks)
	cb.set_label('Flat Number')

	fig.tight_layout()
	breakpoint()


	print('Median-combining the glow-filtered flats...')
	# save the median image and stddev images to /data/tierras/flats/
	median_image = np.median(avg_images, axis=0) # the median-combined image (across the flat sequence)
	# stddev_image = np.std(avg_images, axis=0) # the standard deviation image (across the flat sequence)

	# let's assumed 250-pixel neighborhood will be our standard. 
	# if the neighborhood is different than 250, indicate it in the filename
	if neighborhood_size == 20:
		output_path = f'/data/tierras/flats/{date}_GLOW_FLAT.fit'
	else:
		output_path = f'/data/tierras/flats/{date}_GLOW_FLAT_{neighborhood_size}.fit'

	# make a header to save some info about the flats that went into the combined flat
	hdr = fits.Header()
	hdr['COMMENT'] = f'Neighborhood size = {neighborhood_size} pixels'
	hdr['COMMENT'] = f'N_flats = {n_files}'
	hdr['COMMENT'] = f'Median illumiation = {np.median(flat_medians)} ADU'
	hdr['COMMENT'] = 'Combined the following flats:'
	for i in range(n_files):
		hdr['COMMENT'] = flat_files[i]

	output_hdul = fits.HDUList([fits.PrimaryHDU(data=median_image, header=hdr)])

	# record: n_flats, median pixel illumination, filepaths of flat files 
	output_hdul.writeto(output_path)
	set_tierras_permissions(output_path)
	return 

if __name__ == '__main__':
	main()

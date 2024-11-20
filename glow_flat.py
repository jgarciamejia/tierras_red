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
import pickle 
import os 
from ap_phot import set_tierras_permissions
from scipy.spatial import distance_matrix
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
	ap.add_argument("-neighborhood_size", required=False, default=250, type=int, help='The number of pixels surrounding a given pixel to consider when correcting for glow.')
	args = ap.parse_args(raw_args)
	date = args.date
	neighborhood_size = args.neighborhood_size
	k = neighborhood_size+1
	path = f'/data/tierras/incoming/{date}/'
	flat_files = sorted(glob(path+'*FLAT*.fit'))
	n_files = len(flat_files)
	if n_files == 0:
		print(f'No flat files found in /data/tierras/incoming/{date}/!')
	else:
		print(f'Found {n_files} flat files in /data/tierras/incoming/{date}/')

	im_shape = (2048, 4096)
	x, y = np.meshgrid(np.arange(4096), np.arange(2048))
	coordinates = np.vstack([x.ravel(), y.ravel()]).T
	tree = KDTree(coordinates)
	
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
	avg_images = np.zeros_like(combined_images)	

	# construct the images corrected by the median of neighboring pixels
	n_ops = im_shape[1]*im_shape[0]
	for i in range(n_ops):
		coord = coordinates[i]
		coord_data = combined_images[:, coord[1], coord[0]]
		_, nearest_indices = tree.query(coord, k=k)
		neighbor_yx = coordinates[nearest_indices[1:]]
		combined_images[:, neighbor_yx[:,1], neighbor_yx[:,0]]
		neighbor_data = np.median(combined_images[:, neighbor_yx[:,1], neighbor_yx[:,0]], axis=1) # an n_flat array of the median of the neighborhood_size-surrounding pixels 
		avg_images[:, coord[1], coord[0]] = coord_data / neighbor_data
		if i % 99999 == 0 and i != 0:
			percent_complete = (i+1)/n_ops * 100
			print(f'{percent_complete:.2f}% complete.')

	print('Median-combining the glow-filtered flats...')
	# save the median image and stddev images to /data/tierras/flats/
	median_image = np.median(avg_images, axis=0) # the median-combined image (across the flat sequence)
	stddev_image = np.std(avg_images, axis=0) # the standard deviation image (across the flat sequence)
	if neighborhood_size == 250:
		output_path = f'/data/tierras/flats/{date}_FLAT.fit'
	else:
		output_path = f'/data/tierras/flats/{date}_FLAT_{neighborhood_size}.fit'
	output_hdul = fits.HDUList([fits.PrimaryHDU(data=median_image), fits.ImageHDU(data=stddev_image)])
	output_hdul.writeto(output_path)
	set_tierras_permissions(output_path)
	return 

if __name__ == '__main__':
	main()
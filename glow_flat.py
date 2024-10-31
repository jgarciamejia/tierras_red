import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
from astropy.io import fits 
from astropy.visualization import simple_norm
import argparse
from glob import glob 
import lfa 
from fitsutil import * 
from scipy.spatial import cKDTree
import pickle 
import os 
from ap_phot import set_tierras_permissions

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
	args = ap.parse_args()
	date = args.date

	path = f'/data/tierras/incoming/{date}/'
	flat_files = sorted(glob(path+'*FLAT*.fit'))
	n_files = len(flat_files)
	if n_files == 0:
		print(f'No flat files found in /data/tierras/incoming/{date}/!')
	else:
		print(f'Found {n_files} flat files in /data/tierras/incoming/{date}/')

	fplist = [ pyfits.open(filename) for filename in flat_files ]

	im_shape = (2048, 4096)
	
	# pre-compute the nearest pixels at each pixel location 
	k = 51
	nearest_indices_file = f'nearest_indices_{k-1}.p'
	if not os.path.exists(nearest_indices_file):
		print(f'Pre-computing nearest {k-1} pixels at each pixel location...')
		pixel_coords = np.array([(y, x) for x in range(im_shape[0]) for y in range(im_shape[1])], dtype='int32')
		tree = cKDTree(pixel_coords)
		distances, indices = tree.query(pixel_coords, k=k)
		nearest_indices = indices[:,1:]
		print(f'Saving indices to {nearest_indices_file}')
		pickle.dump((pixel_coords, nearest_indices.astype('int32')), open(nearest_indices_file, 'wb')) # save indices to disk for faster processing in future
	else:
		# if the pre-computed neighbor indices have already been saved to disk, restore them
		print(f'Restoring pre-computed {k-1}-pixel neighborhoods for each pixel location.')
		pixel_coords, nearest_indices = pickle.load(open(nearest_indices_file, 'rb'))

	combined_images = []
	avg_images = []
	#for i in range(n_files):
	for i, filename in enumerate(flat_files):
		print(f'Doing {filename} ({i+1} of {n_files})')

		with fits.open(filename) as ifp:
			proc_imgs = [process_extension(ifp[j], j) for j in range(1, 3)]
		
		combined_img = np.concatenate(proc_imgs, axis=0)
		combined_images.append(combined_img)

		# divide each pixel in the combined image by the mean of its 50 neighbors 
		neighbor_yx = pixel_coords[nearest_indices]  # This will be a (8388608, 50, 2) array with y, x coordinates
		neighbor_data = combined_img[neighbor_yx[:, :, 1], neighbor_yx[:, :, 0]]  # Now (8388608, 50) array with pixel values
		avg_neighbors = np.median(neighbor_data, axis=1).reshape(im_shape)
		avg_img = np.divide(combined_img, avg_neighbors, out=np.zeros_like(combined_img), where=avg_neighbors != 0)			
		avg_images.append(avg_img)
	
	print('Median-combining the glow-filtered flats...')
	# save the median image and stddev images to /data/tierras/flats/
	median_image = np.median(np.dstack(avg_images), axis=2) # the median-combined image (across the flat sequence)
	stddev_image = np.std(np.dstack(avg_images), axis=2) # the standard deviation image (across the flat sequence)
	output_path = f'/data/tierras/flats/{date}_FLAT.fit'
	output_hdul = fits.HDUList([fits.PrimaryHDU(data=median_image), fits.ImageHDU(data=stddev_image)])
	output_hdul.writeto(output_path)
	set_tierras_permissions(output_path)
	return 

if __name__ == '__main__':
	main()
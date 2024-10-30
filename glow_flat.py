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
		print(f'Restoring pre-computed nearest {k-1} pixels at each pixel location.')
		pixel_coords, nearest_indices = pickle.load(open(nearest_indices_file, 'rb'))

	avg_images = []
	for i in range(n_files):
		print(f'Doing {flat_files[i]} ({i+1} of {n_files})')
		
		ifp = fplist[i]


		proc_imgs = []
		for j in range(1,3):
			imp = ifp[j]
			proc_imgs.append(process_extension(imp, j))

		combined_img = np.zeros((im_shape))
		combined_img[0:1024, :] = proc_imgs[0]
		combined_img[1024:, :] = proc_imgs[1]

		avg_img = np.zeros_like(combined_img)
		fig, ax = plt.subplots(2,1,figsize=(12,7), sharex=True, sharey=True)
		norm = simple_norm(combined_img, min_percent=5, max_percent=98)
		for j in range(len(pixel_coords)):
			target_coord = pixel_coords[j]
			neighbors = pixel_coords[nearest_indices[j]]
		
			avg_neighbors = np.mean(combined_img[neighbors[:,1], neighbors[:,0]])
			avg_img[target_coord[1], target_coord[0]] = combined_img[target_coord[1], target_coord[0]] / avg_neighbors
			
			# if j % 100000 == 0 and j != 0: 
			# 	ax[0].imshow(combined_img, origin='lower', norm=norm, interpolation='none')
			# 	ax[0].plot(target_coord[0], target_coord[1], 'bx')
			# 	for j in range(len(neighbors)):
			# 		ax[0].plot(neighbors[j][0], neighbors[j][1], 'rx')
				
			# 	norm2 = simple_norm(avg_img, min_percent=1, max_percent=98)
			# 	ax[1].imshow(avg_img, origin='lower', norm=norm2, interpolation='none')
			# 	plt.tight_layout()
			# 	breakpoint()
			# 	ax[0].cla()
			# 	ax[1].cla()
			avg_images.append(avg_img)
	
	breakpoint()
	return 

if __name__ == '__main__':
	main()
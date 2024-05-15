import argparse
import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
from ap_phot import get_flattened_files, plot_image, generate_square_cutout
import pandas as pd 
from astropy.io import fits
import time 
from astropy.modeling import models, fitting

def measure_fwhm_grid(files, sources, box_size=512):
	PLATE_SCALE = 0.432
	fit_g = fitting.LevMarLSQFitter() # fitter object for fitting 2D gaussians to measure FWHM

	# establish grid across the image in which to select sources for fwhm measurement 
	im_shape = fits.open(files[0])[0].data.shape
	box_x_starts = np.arange(0, im_shape[1], box_size)
	box_y_starts = np.arange(0, im_shape[0], box_size)

	source_x = np.array(sources['X pix'])
	source_y = np.array(sources['Y pix'])
	fwhm_star_ids = [] 
	for i in range(len(box_x_starts)):
		x_start = box_x_starts[i]
		x_end = x_start + box_size
		for j in range(len(box_y_starts)):
			y_start = box_y_starts[j]
			y_end = y_start + box_size
			
			sources_in_box = sources.iloc[np.where((source_x >= x_start) & (source_x <= x_end) & (source_y >= y_start) & (source_y <= y_end))[0]]
			sources_in_box_x = np.array(sources_in_box['X pix'])
			sources_in_box_y = np.array(sources_in_box['Y pix'])	
			sources_to_keep = []
			for k in range(len(sources_in_box)):
				dists = np.sqrt((sources_in_box_x[k]-sources_in_box_x)**2 + (sources_in_box_y[k]-sources_in_box_y)**2)
				if len(np.where((dists < 29) & (dists != 0))[0]) == 0:
					sources_to_keep.append(k)
			sources_in_box = sources_in_box.iloc[sources_to_keep]
			# take the brightest star in the box as the FWHM source 
			# TODO: worry about saturated sources
			if len(sources_in_box) != 0:
				fwhm_star_ids.append(sources_in_box.index[0])
	
	fwhm_stars = sources.iloc[fwhm_star_ids]
	fig, ax = plot_image(fits.open(files[0])[0].data)
	ax.plot(fwhm_stars['X pix'], fwhm_stars['Y pix'], 'rx')	

	source_x_fwhm_arcsec = np.zeros((len(fwhm_stars), len(files)))
	source_y_fwhm_arcsec = np.zeros_like(source_x_fwhm_arcsec)
	source_theta_radians = np.zeros_like(source_x_fwhm_arcsec)
	for i in range(len(files)):
		print(f'Doing {files[i]} ({i+1} of {len(files)}).')
		data = fits.open(files[i])[0].data

		tfwhm = time.time()
		#Measure FWHM 
		k = 0
		cutout_size = 30
		g_init = models.Gaussian2D(amplitude=1,x_mean=cutout_size/2,y_mean=cutout_size/2, x_stddev=3, y_stddev=3)
		g_init.theta.bounds = (0, np.pi)
		g_init.x_stddev.bounds = (1,10)
		g_init.y_stddev.bounds = (1,10)

		# pre-compute meshgrid of pixel indices for square cutouts (which will be the case for all sources except those near the edges)
		xx2, yy2 = np.meshgrid(np.arange(cutout_size),np.arange(cutout_size))

		
		for j in range(len(fwhm_stars)):
			
			# TODO: this loop appears to slow down over time, WHY!?
			# this is, in general, the slowest loop of the program. How can we make it faster??
			try:
				g_2d_cutout, cutout_pos = generate_square_cutout(data, (fwhm_stars['X pix'][j], fwhm_stars['Y pix'][j]), cutout_size)
			except:
				source_x_fwhm_arcsec[j,i] = np.nan
				source_y_fwhm_arcsec[j,i] = np.nan
				source_theta_radians[j,i] = np.nan 

			cutout_shape = g_2d_cutout.shape

			bkg = np.median(g_2d_cutout)
			# bkg = np.mean(sigmaclip(g_2d_cutout,2,2)[0])

			g_2d_cutout -= bkg 
			
			# recompute the meshgrid only if you get a non-square cutout
			if g_2d_cutout.shape != (cutout_size, cutout_size):
				xx3, yy3 = np.meshgrid(np.arange(cutout_shape[1]),np.arange(cutout_shape[0]))
				xx = xx3
				yy = yy3
			else:
				xx = xx2
				yy = yy2 
			
			# intialize the model 
			g_init.amplitude = g_2d_cutout[int(cutout_pos[1]), int(cutout_pos[0])]

			# use the cutout position returned from generate_square_cutout to predict its location
			g_init.x_mean = cutout_pos[0]
			g_init.y_mean = cutout_pos[1]


			g = fit_g(g_init,xx,yy,g_2d_cutout)
			
			if g.y_stddev.value > g.x_stddev.value:
				x_stddev_save = g.x_stddev.value
				y_stddev_save = g.y_stddev.value
				g.x_stddev = y_stddev_save
				g.y_stddev = x_stddev_save
				g.theta += np.pi/2

			source_x_fwhm_arcsec[j,i] = g.x_stddev.value * 2.35482 * PLATE_SCALE
			source_y_fwhm_arcsec[j,i] = g.y_stddev.value * 2.35482 * PLATE_SCALE
			source_theta_radians[j,i] = g.theta.value
			#print(time.time()-t1)

			# fig, ax = plt.subplots(1,2,figsize=(12,8),sharex=True,sharey=True)
			# norm = ImageNormalize(g_2d_cutout-bkg,interval=ZScaleInterval())
			# ax[0].imshow(g_2d_cutout-bkg,origin='lower',interpolation='none',norm=norm)
			# ax[1].imshow(g(xx2,yy2),origin='lower',interpolation='none',norm=norm)
			# plt.tight_layout()

	breakpoint()

def main(raw_args=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
	ap.add_argument("-field", required=True, help="Name of observed field exactly as shown in raw FITS files.")
	ap.add_argument("-ffname", required=False, default='flat0000', help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")

	args = ap.parse_args(raw_args)
	date = args.date
	field = args.field 
	ffname = args.ffname 

	flattened_files = get_flattened_files(date, field, ffname)
	sources = pd.read_csv(f'/data/tierras/photometry/{date}/{field}/{ffname}/{date}_{field}_sources.csv')

	measure_fwhm_grid(flattened_files, sources)

	breakpoint()



if __name__ == '__main__':
	main()
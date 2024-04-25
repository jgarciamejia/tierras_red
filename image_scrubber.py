import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import argparse 
from ap_phot import get_flattened_files
from astropy.io import fits 
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.backends.backend_tkagg as tkagg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg
from astropy.visualization import simple_norm 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob 
import numpy as np
import pandas as pd 
import os

class ImageScrubber:
	def __init__(self, master, file_list, phot_file_list):
		self.image_folder = file_list[0].parent
		self.file_list = file_list 
		self.phot_file_list = phot_file_list
		self.phot_dfs, self.phot_df_file_names = self.load_photometry(phot_file_list)

		self.date = self.file_list[0].name.split('.')[0]
		self.target = self.file_list[0].name.split('.')[2].split('_')[0]

		self.master = master
		self.master.title(f'Tierras Image Scrubber: {self.target} on {self.date}')

		self.image_label = tk.Label(self.master)
		self.image_label.pack()	

		self.default_min_percent = 1
		self.default_max_percent = 99.9

		# Create entry boxes for min_percent and max_percent
		self.min_percent_label = tk.Label(self.master, text="Min Percent:")
		self.min_percent_label.pack()
		self.min_percent_entry = tk.Entry(self.master)
		self.min_percent_entry.pack()
		self.min_percent_entry.insert(tk.END, str(self.default_min_percent))
		self.min_percent_entry.bind('<FocusOut>', self.update_image)
		
		self.max_percent_label = tk.Label(self.master, text="Max Percent:")
		self.max_percent_label.pack()
		self.max_percent_entry = tk.Entry(self.master)
		self.max_percent_entry.pack()
		self.max_percent_entry.insert(tk.END, str(self.default_max_percent))	
		self.max_percent_entry.bind('<FocusOut>', self.update_image)

		# Create a button to update the displayed image
		self.update_button = tk.Button(self.master, text="Update", command=self.update_image)
		self.update_button.pack()			

		if len(self.phot_dfs) == 0:
			self.n_sources = np.nan
		else:
			self.n_sources = int(self.phot_dfs[0].keys()[-1].split(' ')[0][1:])

		target_list = ['Full frame']
		if not np.isnan(self.n_sources):
			for i in range(self.n_sources):
				target_list.append('S'+str(i))
		self.target_list = target_list	

		self.selected_target= tk.StringVar(self.master, value=self.target_list[0])  # Default to viewing the 'full frame'

		# Create a dropdown menu to select a source
		self.source_dropdown_label = tk.Label(self.master, text="Select Source:")
		self.source_dropdown_label.pack(side=tk.TOP)
		self.source_dropdown = ttk.Combobox(self.master, textvariable=self.selected_target, values=self.target_list, state="readonly")
		self.source_dropdown.pack(side=tk.TOP)
		self.source_dropdown.bind('<<ComboboxSelected>>', self.load_selected_target)
		
		self.fig = plt.Figure(figsize=(14,7))
		self.canvas = tkagg.FigureCanvasTkAgg(self.fig, master = master)
		self.canvas.get_tk_widget().pack()
		self.navbar = NavigationToolbar2TkAgg(self.canvas, self.master)
		self.axes = self.fig.add_subplot(111)

		self.canvas.draw()
		
		self.images, self.file_names = self.load_images(file_list)

		self.selected_image = tk.StringVar(self.master, value=str(self.file_names[0]).split('/')[-1])  # Default to the first image

		self.current_index = 0

		self.slider = tk.Scale(self.master, from_=0, to=len(self.images), orient=tk.HORIZONTAL, command=self.update_image_slider)
		self.slider.pack(fill=tk.X)
		
		self.previous_button = tk.Button(self.master, text="Previous", command=self.previous_image)
		self.previous_button.pack(side=tk.LEFT)

		self.next_button = tk.Button(self.master, text="Next", command=self.next_image)
		self.next_button.pack(side=tk.RIGHT)

		# Create a dropdown menu to select an image
		self.image_dropdown_label = tk.Label(self.master, text="Select Image:")
		self.image_dropdown_label.pack()
		self.image_dropdown = ttk.Combobox(self.master, textvariable=self.selected_image, values=self.file_names, state="readonly")
		self.image_dropdown.pack()
		self.image_dropdown.bind('<<ComboboxSelected>>', self.load_selected_image)

		self.xlim = (0,4096)
		self.ylim = (0,2048)
		self.update_image()

		

	def load_images(self, file_list):
		images = []
		file_names = []
		print(f'Reading in {len(file_list)} images.')
		for filename in file_list:
			image = fits.open(filename)[0].data
			if image is not None:
				images.append(image)
				file_names.append(str(filename).split('/')[-1])
		file_names = np.array(file_names)
		return images, file_names

	def load_photometry(self, file_list):
		phot_dfs = []
		phot_filenames = []
		print(f'Reading in {len(file_list)} photometry files.')
		for filename in file_list:
			df = pd.read_csv(filename)
			if df is not None:
				phot_dfs.append(df)
				phot_filenames.append(str(filename).split('/')[-1])
		return phot_dfs, phot_filenames

	def load_selected_target(self, event=None):
		source_name = self.source_dropdown.get()
		index = self.current_index
		print(index)
		if source_name == 'Full frame':
			self.xlim = (0,4096)
			self.ylim = (0,2048)
		else:
			source_x = self.phot_dfs[0][f'{source_name} X'][index]
			source_y = self.phot_dfs[0][f'{source_name} Y'][index]
			self.source_x = source_x 
			self.source_y = source_y 

			
			self.xlim = (source_x-100, source_x+100)
			self.ylim = (source_y-100, source_y+100)

	def load_selected_image(self, event=None):
		index = np.where(self.file_names == self.selected_image.get().replace("'",""))[0][0]
		if 0 <= index < len(self.images):
			self.current_index = index
			image = self.images[index]
			self.slider.set(index)
			self.display_image(image, self.file_names[index])

	def update_image_slider(self, event=None):
		index = self.slider.get()
		self.current_index = index
		if 0 <= index < len(self.images):
			image = self.images[index]
			self.load_selected_target(self)
			self.display_image(image, self.file_names[index])

	def update_image(self, event=None):
		# index = self.slider.get()
		# self.current_index = index
		index = self.current_index
		if 0 <= index < len(self.images):
			image = self.images[index]
			self.load_selected_target(self)
			self.display_image(image, self.file_names[index])
			self.slider.set(index)
			self.current_index = index

	def display_image(self, image, file_name):
		#self.fig.clear()
		self.axes.cla()
		if hasattr(self, 'cbar'):
			self.cbar.remove()
		self.min_percent = float(self.min_percent_entry.get())
		self.max_percent = float(self.max_percent_entry.get())
		self.im = self.axes.imshow(image, origin='lower', norm=simple_norm(image, min_percent=self.min_percent, max_percent=self.max_percent), interpolation='none')
		self.divider = make_axes_locatable(self.axes)
		self.cax = self.divider.append_axes('right', size='5%', pad=0.05)
		self.cbar = self.fig.colorbar(self.im, cax=self.cax, orientation='vertical')
		self.cbar.set_label('ADU', rotation=270)
		self.axes.grid(False)
		if self.source_dropdown.get() != 'Full frame':
			self.axes.plot(self.source_x, self.source_y, 'rx')
		self.axes.set_title(file_name)
		self.axes.set_xlim(self.xlim)
		self.axes.set_ylim(self.ylim)
		self.fig.tight_layout()
		self.canvas.draw()

	def previous_image(self):
		if self.current_index > 0:
			self.current_index -= 1
			image = self.images[self.current_index]
			self.load_selected_target(self)
			self.display_image(image, self.file_names[self.current_index])
			self.slider.set(self.current_index)

	def next_image(self):
		if self.current_index < len(self.images) - 1:
			self.current_index += 1
			image = self.images[self.current_index]
			self.load_selected_target(self)
			self.display_image(image, self.file_names[self.current_index])
			self.slider.set(self.current_index)

def main():
	root = tk.Tk()
	app = ImageScrubber(root)
	root.mainloop()

def main(raw_args=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
	ap.add_argument("-target", required=True, help="Name of observed target exactly as shown in raw FITS files.")
	ap.add_argument("-ffname", required=False, default='flat0000', help="Name of folder in which to store reduced+flattened data. Convention is flatXXXX. XXXX=0000 means no flat was used.")
	args = ap.parse_args(raw_args)

	date = args.date 
	target = args.target 
	ffname = args.ffname 

	image_files = get_flattened_files(date, target, ffname)
	photometry_files = np.sort(glob(f'/data/tierras/photometry/{date}/{target}/{ffname}/*ap_phot*.csv'))
	
	root = tk.Tk()
	app = ImageScrubber(root, image_files, photometry_files)
	root.mainloop()



if __name__ == "__main__":
	main()
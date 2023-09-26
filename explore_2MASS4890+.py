import numpy as np 
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import pdb
import re

from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"  # Select Data Release 2

import astropy.coordinates as coord
import astropy.units as u

import load_data as ld 
import plot_cum_phot as pcp
import plot_all_comps as pac
import bin_lc as bl

importlib.reload(ld)
importlib.reload(pcp)
importlib.reload(pac)
importlib.reload(bl)

####### User-defined Params #######

# Tests to run
load_single_date = True
query_simbad = True

do_plot_cum_phot = False
plot_comp_counts = False 
print_comp_star_order = False

do_plot_cum_phot_exccomps = False
plot_comp_counts_exccomps = False
check_comps_used = False
check_ap_sizes = False

save_corrected_relflux = False
save_pca_comps = False
save_relflux_comps = False


# Data, target, date
mainpath = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/AIJ_Output/'
targetname = '2MASSJ03304890+'

# Cumulative plotting
sig_threshold = 5
normalize = 'none'
binsize = 10
bad_weather = np.array(['20221220','20230109','20230118','20230120',
						'20230122','20230127','20230131','20230202']) #['20221220']


######## Load photometry for one day #######
if load_single_date:
	exclude_comps = []
	date = '20221211'
	df,bjds,relfluxes,airmasses,widths,flag, comps_used = ld.return_data_onedate(mainpath,
		                                      targetname,date,sig_threshold,exclude_comps,
		                                      flag_output = True)

######## Query SIMBAD for comp information: magnitude and object ID ###### 

# works with load_single_date for now
if query_simbad:
	# make file 
	prefix = targetname+'_'+date
	comp_id_fname = mainpath+prefix+'/'+prefix+'_comps_radec.txt'
	with open(comp_id_fname, "w") as file:
		# get good and bad comp numbers
		aij_comps = ld.get_AIJ_star_numbers(df,'Source-Sky_C')
		aij_targs = ld.get_AIJ_star_numbers(df,'Source-Sky_T') 
		aij_allstars = np.sort(np.concatenate((aij_targs,aij_comps)))
		#aij_allstars = np.array([1,2,3,4,5,6])
		#simbad_ids = []
		#gaia_ids = []
		ras, decs = [], []

		# get RA and Dec in decimal for each comp
		for comp_num in aij_allstars:
			print (comp_num)
			if comp_num in aij_targs:
				ra_hdr,dec_hdr = ld.find_all_cols_w_keywords(df,'RA_T'+str(comp_num))[0],ld.find_all_cols_w_keywords(df,'DEC_T'+str(comp_num))[0]
			elif comp_num in aij_comps:
				ra_hdr,dec_hdr = ld.find_all_cols_w_keywords(df,'RA_C'+str(comp_num))[0],ld.find_all_cols_w_keywords(df,'DEC_C'+str(comp_num))[0]
			ra,dec = np.median(df[ra_hdr].to_numpy()), np.median(df[dec_hdr].to_numpy()) # get ra and dec position from median of all exposures
			#print (ra*15,dec)

			# define coords for Gaia
			this_coord = coord.SkyCoord(ra=ra*15,dec=dec, unit=(u.degree, u.degree), frame='icrs')
			width = u.Quantity(0.05, u.deg)
			height = u.Quantity(0.05, u.deg)
			# query Gaia
			#result_table = Gaia.query_object_async(coordinate=this_coord, width=width, height=height)
			
			result_table = Simbad.query_region(coord.SkyCoord(ra=ra*15, dec=dec,
	                                   unit=(u.deg, u.deg), frame='icrs'),
	                                   radius=.04* u.deg, epoch = 'J2000',
	                                   equinox=2000)
			#gaia_ids.append(result_table[0]['DESIGNATION'])
			#simbad_ids.append(result_table[0]['MAIN_ID'])
			# print (result_table)
			ras.append(ra)
			decs.append(dec)
			#print (str(ra*15)+' '+str(dec)+'\n')
			file.writelines(str(ra*15)+' '+str(dec)+'\n')	
		#file.writelines(s + '\n' for s in simbad_ids)	
	#open and read the file after the appending:
	#with open(comp_id_fname, "r") as file:
	#	print(file.read())



######## Plot cumulative photometry #######
exclude_comps = []
if do_plot_cum_phot:
	dfs, allobsdates,goodwxdates, filenames, flags, compss_used = pcp.plot_cum_phot(mainpath,targetname,
		                              sig_threshold,normalize,binsize,bad_weather,
		                              True,False, exclude_comps)
	# Generate exclude_comps list with stars that 
	# are saturated in any exposure
	print('Printing saturated comp stars per date. Saturation meanscounts >= 40K') 
	exclude_comps = []
	for iobs,obsdate in enumerate(allobsdates):
		print (obsdate)
		if np.any(bad_weather == obsdate):
			continue
		else:
			exclude_comps = []
			comp_nums = ld.get_AIJ_star_numbers(dfs[iobs],'Source-Sky_C')
			badcomp_nums = ld.get_AIJ_star_numbers(dfs[iobs],'Source-Sky_T')

			for comp_num in comp_nums:
				if np.any(dfs[iobs]['Peak_C'+str(comp_num)].to_numpy() >= 40000):
					exclude_comps.append(comp_num)
			for badcomp_num in badcomp_nums:
				if np.any(dfs[iobs]['Peak_T'+str(badcomp_num)].to_numpy() >= 40000):
					exclude_comps.append(badcomp_num)
			print (exclude_comps)


####### Print comparison star order from brightest to dimmest per night #######
if print_comp_star_order:
	prevdate_sorted_comps = np.array([])
	for i,date in enumerate(goodwxdates):
		medians, stars, sortedmeds,this_sorted_comp_nums = pac.rank_comps(mainpath,targetname,date)
		if not np.array_equal(prevdate_sorted_comps,this_sorted_comp_nums):
			prevdate_sorted_comps = this_sorted_comp_nums
			print ('Date: {}, Comp Stars from Brightest to Dimmest: \n {}'
		    .format(date,this_sorted_comp_nums))


######## Plot comparison star behavior per night to ID additional comps to exclude #######

# select which comps to exclude from plotting to examine them closely 

#exclude_comps = this_sorted_comp_nums[30:]  # brightest 30 comps only
#exclude_comps = np.concatenate((this_sorted_comp_nums[30:],np.array([4,20,33,42,82,83,94,28,47,86,12])))  # brightest 30 comps only with saturated and other odd balls removed

#exclude_comps = np.concatenate((this_sorted_comp_nums[:30],this_sorted_comp_nums[60:])) # mid 30 comps only
#exclude_comps = np.concatenate((exclude_comps,np.array([51,64,66,67,68,69,72,75,79,80])))

#exclude_comps = this_sorted_comp_nums[:60]  # faintest 30 comps only
#exclude_comps = np.concatenate((exclude_comps,np.array([30,56,58,60,70])))

#plot_comp_counts = True
if plot_comp_counts:
	for date in allobsdates:
		comp_kws = pac.plot_all_comps_onedate(mainpath,
		           targetname,date,False,exclude_comps)


### Make judgement call about which comps to exclude based on above

exclude_comps = np.array([4,20,33,42,82,83,94]) # from saturation loop 
erratic_comps = np.array([28,47,86,12,51,64,66,67,68,69,72,75,79,80,30,56,58,60,70,90,91]) #by eye erratic + 90 and 91 which 20221221 does not have 
exclude_comps = np.append(exclude_comps,erratic_comps)  


if do_plot_cum_phot_exccomps:
	dfs, allobsdates,goodwxdates, filenames, flags, compss_used = pcp.plot_cum_phot(mainpath,targetname,
		                              sig_threshold,normalize,binsize,bad_weather,
		                              True,False, exclude_comps)


# Check that the same comp stars were used for all dates
if check_comps_used:
	ncomps = []
	print ('Checking Comps Used')
	for idate,comps_used in enumerate(compss_used):
		if np.any(bad_weather == allobsdates[idate]):
			print (allobsdates[idate])
			print ('excluded due to bad weather')
			continue 
		else:
			ncomps.append(len(compss_used[idate]))
			#print (allobsdates[idate])
			#print (len(compss_used[idate]))
			#print (np.array(compss_used[0]==compss_used[idate])) #should print True
			#print ('\n')
	print (ncomps) # all should be same number 

######## Plot comparison star behavior again to inspect selected comps #######

if plot_comp_counts_exccomps:
	for date in allobsdates:
		comp_kws = pac.plot_all_comps_onedate(mainpath,
		           targetname,date,False,exclude_comps)


######## Check aperture sizes of first exposure in every date #######

if check_ap_sizes:
	for date in allobsdates:
		df,bjds,relfluxes,airmasses,widths,flag,comps_used = ld.return_data_onedate(mainpath,
	                                      targetname,date,sig_threshold,
	                                      exclude_comps,flag_output = False)
		rsource = np.mean(df['Source_Radius'].to_numpy())
		rskymin = np.mean(df['Sky_Rad(min)'].to_numpy())
		rskymax = np.mean(df['Sky_Rad(max)'].to_numpy())
		print (rsource,rskymin,rskymax)


######## Save relflux as column Emily can use ####### 

if save_corrected_relflux:
	for idate,date in enumerate(allobsdates):
		print (date)
		df,bjds,relfluxes,airmasses,widths,flag,comps_used = ld.return_data_onedate(mainpath,
			                                      targetname,date,sig_threshold,
			                                      exclude_comps,flag_output = False)
		print (comps_used)
		ind = np.argwhere(df.columns.to_numpy() == 'rel_flux_T1') + 1
		df.insert(int(ind), 'rel_flux_T1_corr',relfluxes)
		oldfname = filenames[idate]
		newfname = oldfname.replace('custom_measurements','custom_measurements_r')
		#df.to_excel(oldfname, index=False)
		with pd.ExcelWriter(newfname, engine='openpyxl') as writer:
			df.to_excel(writer, index=False, encoding = "utf-8")

######## Save comps to use per date for PCA analysis/ to calculate rel flux ###### 

if save_pca_comps:
	pca_comps = np.arange(2,np.max(comp_nums))
	for exc_comp in exclude_comps:
		pca_comps = np.delete(pca_comps, np.argwhere(pca_comps == exc_comp))

	for date in allobsdates:
		prefix = targetname+'_'+date
		pca_comp_fname = mainpath+prefix+'/'+prefix+'_comps_pca.csv'
		pd.DataFrame(pca_comps).to_csv(pca_comp_fname,header=['Comp_Star_Number'],index=False)


if save_relflux_comps:
	relflux_comps = np.arange(2,np.max(comp_nums))
	for exc_comp in exclude_comps:
		relflux_comps = np.delete(relflux_comps, np.argwhere(relflux_comps == exc_comp))

	for date in allobsdates:
		prefix = targetname+'_'+date
		relflux_comp_fname = mainpath+prefix+'/'+prefix+'_comps_relflux.csv'
		pd.DataFrame(relflux_comps).to_csv(relflux_comp_fname,header=['Comp_Star_Number'],index=False)













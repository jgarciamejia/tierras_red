import numpy as np
import pandas as pd
import glob
import re

from jgmmedsig import *

def find_all_cols_w_keywords(df,*keywords):
	cols = []
	for col in df.columns:
		for keyword in keywords:
			if keyword in col:
				cols.append(col)
	return cols

def get_num_in_str(string):
	num = re.findall(r'[0-9]+', string)
	if len(num) > 1:
		print ('get_num_in_str error: more than one number IDd in string')
	elif len(num) == 1:
		return int(num[0]) 

def get_AIJ_star_numbers(df,column_kw):
	comp_keywords = find_all_cols_w_keywords(df,column_kw)
	comp_nums = [get_num_in_str(keyword) for keyword in comp_keywords]
	return np.array(comp_nums)

def calc_rel_flux(df,*exclude_comps):
	source_min_sky_T1 = df['Source-Sky_T1'].to_numpy()
	sum_source_min_sky_Cs = np.zeros(len(source_min_sky_T1))
	aij_comps = get_AIJ_star_numbers(df,'Source-Sky_C')
	aij_targs = get_AIJ_star_numbers(df,'Source-Sky_T') 
	all_stars = np.arange(2,np.max(aij_comps)) # excluding 0 (no AIJ assignment )and 1 (T1)
	if len(exclude_comps) == 0:
		print ('Error: No comps selected by user for exclusion. If this is intentional, use kw: rel_flux_T1 instead of this function')
		return None
	used_compstars = []
	for star in all_stars:
		if not np.any(exclude_comps == star) and np.any(aij_comps == star):
		#if star not in exclude_comps and star in aij_comps:
			source_min_sky_comp = df['Source-Sky_C'+str(star)].to_numpy()
			sum_source_min_sky_Cs += source_min_sky_comp
			used_compstars.append(star) #added March 16/2023
		if not np.any(exclude_comps == star) and not np.any(aij_comps == star):
		#elif star not in exclude_comps and star not in aij_comps:
			source_min_sky_comp = df['Source-Sky_T'+str(star)].to_numpy()
			sum_source_min_sky_Cs += source_min_sky_comp
			used_compstars.append(star) #added March 16/2023
		elif np.any(exclude_comps == star):
		#elif star in exclude_comps:
			#print ('Comparison star No.{} excluded from rel. flux. calculation'.format(star))
			continue

	return source_min_sky_T1 / sum_source_min_sky_Cs, used_compstars

def return_dataframe_onedate(mainpath,targetname,obsdate,printfnames=False):
	filenames = np.sort(glob.glob(mainpath+'/**/'+targetname+'**measurements.xls',recursive=True))
	#filenames = np.sort(glob.glob(mainpath+'/**/'+targetname+'**_r.xls',recursive=True)) #to read in corrected data
	nfiles = 0
	dfs = []
	names = []
	for filename in filenames:
		if obsdate in filename:
			nfiles += 1
			try:
				dfs.append(pd.read_table(filename, encoding="utf-8"))
			except UnicodeDecodeError:
				dfs.append(pd.read_excel(filename))
			names.append(filename)
	if nfiles == 0:
		print ('Error: No data found for {} on {}'.format(targetname,obsdate))
	elif nfiles > 1:
		print ('Warning: {} files found for {} on {}'.format(nfiles,targetname,obsdate))
		if printfnames:
			for name in names:
				print (name)
		return dfs,names
	elif nfiles == 1:
		return dfs[0],names[0]

def return_data_onedate(mainpath,targetname,obsdate,threshold,*exclude_comps,flag_output=False):
	dfs,filenames = return_dataframe_onedate(mainpath,targetname,obsdate)
	if type(dfs) == list:
		print (dfs,len(dfs))
		print ('Error: Attempting to unpack multiple data frames for a single date.') 
		return None
	elif type(dfs) == pd.core.frame.DataFrame:
		df = dfs
		bjds = df['BJD_TDB_MOBS'].to_numpy()
		widths = df['Width_T1'].to_numpy()
		try:
			airmasses = df['AIRMASS'].to_numpy()
		except KeyError:
			airmasses = df[' AIRMASS'].to_numpy()
		if exclude_comps:
			#print ('Comp stars {} selected by user to be excluded from rel flux calc'.format(exclude_comps))
			rel_flux_T1, comps_used = calc_rel_flux(df,*exclude_comps)
		else:
			rel_flux_T1 = df['rel_flux_T1'].to_numpy()
		
		medflux, sigflux = medsig(rel_flux_T1)
		flag = np.absolute(rel_flux_T1 - medflux) < threshold*sigflux
		if not flag_output:
			return (df,bjds,rel_flux_T1,airmasses,widths,flag,comps_used)
		elif flag_output:
			return (df,bjds[flag],rel_flux_T1[flag],airmasses[flag],widths[flag],flag,comps_used)


def return_data_multidate(path,targetname,dates,threshold):
#path: string, where folders with targetname_date are located
#targetname: string, name of target in all caps, no spaces
#date: list of strings, dates of observation, MST
#threshold: int or float, threshold*sigma to produce a
#flag array (T/F) of length glob_data	get_
	glob_dateind = np.array([])
	glob_bjds = np.array([])
	glob_widths = np.array([])
	glob_airmass = np.array([])
	glob_rel_flux_T1 = np.array([])
	basepath = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/'
	dateind = 0
	for date in obsdates:
		fullpath = basepath+'AIJ_Output_Ryan_fixedaperture/TOI2013_'+date+'/'
		df = pd.read_table(fullpath+'toi2013_'+date+'-Tierras_1m3-I_measurements.xls')
		glob_bjds = np.append(glob_bjds, df['BJD_TDB_MOBS'].to_numpy())
		glob_dateind = np.append(glob_dateind,np.repeat(dateind,df.shape[0])) 
		rel_flux_T1 = df['rel_flux_T1'].to_numpy()
		glob_rel_flux_T1 = np.append(glob_rel_flux_T1,rel_flux_T1)
		glob_widths = np.append(glob_widths, df['Width_T1'].to_numpy())
		try:
			glob_airmass = np.append(glob_airmass,df['AIRMASS'].to_numpy())
		except KeyError:
			glob_airmass = np.append(glob_airmass,df[' AIRMASS'].to_numpy())
		dateind += 1
		# flag gen - will fail globally
		medflux, sigflux = medsig(rel_flux_T1)
		flag = np.absolute(rel_flux_T1 - medflux) < threshold*sigflux

	return (glob_dateind.astype(int),glob_bjds,glob_rel_flux_T1,glob_airmass,glob_widths,flag)


def ret_data_arrays(obsdates,threshold):
	glob_dateind = np.array([])
	glob_bjds = np.array([])
	glob_widths = np.array([])
	glob_airmass = np.array([])
	glob_rel_flux_T1 = np.array([])
	basepath = '/Users/jgarciamejia/Documents/TierrasProject/SCIENCE/'
	dateind = 0
	for date in obsdates:
		fullpath = basepath+'AIJ_Output_Ryan_fixedaperture/TOI2013_'+date+'/'
		df = pd.read_table(fullpath+'toi2013_'+date+'-Tierras_1m3-I_measurements.xls')
		glob_bjds = np.append(glob_bjds, df['BJD_TDB_MOBS'].to_numpy())
		glob_dateind = np.append(glob_dateind,np.repeat(dateind,df.shape[0])) 
		rel_flux_T1 = df['rel_flux_T1'].to_numpy()
		glob_rel_flux_T1 = np.append(glob_rel_flux_T1,rel_flux_T1)
		glob_widths = np.append(glob_widths, df['Width_T1'].to_numpy())
		try:
			glob_airmass = np.append(glob_airmass,df['AIRMASS'].to_numpy())
		except KeyError:
			glob_airmass = np.append(glob_airmass,df[' AIRMASS'].to_numpy())
		dateind += 1
		# flag gen - will fail globally
		medflux, sigflux = medsig(rel_flux_T1)
		flag = np.absolute(rel_flux_T1 - medflux) < threshold*sigflux

	return (glob_dateind.astype(int)[flag],glob_bjds[flag],glob_rel_flux_T1[flag],glob_airmass[flag],glob_widths[flag])




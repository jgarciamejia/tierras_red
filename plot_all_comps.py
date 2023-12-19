"""
Functions to plot and compare star fluxes 
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import load_data as ld
 
def plot_all_comps_onedate(mainpath,targetname,obsdate,plot_separate=True,*exclude_comps):
	df,filename = ld.return_dataframe_onedate(mainpath,targetname,obsdate)
	comp_nums = ld.get_AIJ_star_numbers(df,'Source-Sky_C')
	all_stars = np.arange(1,np.max(comp_nums)) # excluding 0 (no AIJ assignment). Note that 1=T1
	bjds = df['BJD_TDB_MOBS'].to_numpy()
	fig, ax = plt.subplots(figsize=(15,10))
	for star in all_stars:
		if np.any(exclude_comps == star):
		#if star in exclude_comps:
			continue
		elif not np.any(exclude_comps == star):
		#elif star not in exclude_comps:
			if star in comp_nums:
				nthcomp_counts = df['Source-Sky_C'+str(star)].to_numpy()
				ax.scatter(bjds,nthcomp_counts,s=4,label='C'+str(star)) 
			elif star not in comp_nums:
				nthcomp_counts = df['Source-Sky_T'+str(star)].to_numpy()
				if star == 1:
					ax.scatter(bjds,nthcomp_counts,s=6,color='black',label='T'+str(star))
				else:
					ax.scatter(bjds,nthcomp_counts,s=4,label='C'+str(star))  
			if plot_separate:
				ax.set_xlabel('BJD')
				ax.set_ylabel('Comp Counts')
				ax.set_yscale('log')
				plt.legend(loc='upper right')
				fig.tight_layout()
				plt.show()
	if not plot_separate:
		ax.set_xlabel('BJD')
		ax.set_ylabel('Comp Counts')
		ax.set_yscale('log')
		ax.set_title(obsdate)
		plt.legend(loc='upper right')
		fig.tight_layout()
		plt.show() 
	return None

def rank_comps(mainpath,targetname,obsdate,*exclude_comps):
	df,filename = ld.return_dataframe_onedate(mainpath,targetname,obsdate)
	comp_nums = ld.get_AIJ_star_numbers(df,'Source-Sky_C')
	all_stars = np.arange(1,np.max(comp_nums)) # excluding 0 (no AIJ assignment). Note that 1=T1
	medians = np.array([])
	print (len(all_stars))
	for star in all_stars:
		if np.any(exclude_comps == star):
			all_stars = np.delete(all_stars, np.argwhere(all_stars == star))
	print (len(all_stars))

	for star in all_stars:
		try:
			nthcomp_counts = df['Source-Sky_C'+str(star)].to_numpy()
		except KeyError:
			nthcomp_counts = df['Source-Sky_T'+str(star)].to_numpy()
		medians = np.append(medians,np.median(nthcomp_counts))

	sorted_inds = np.argsort(medians)[::-1]
	sorted_medians = medians[sorted_inds] 
	sorted_comp_nums = all_stars[sorted_inds] # brightest to dimmest

	return medians,all_stars,sorted_medians, sorted_comp_nums





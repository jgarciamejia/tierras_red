import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.ion()
import argparse
from datetime import datetime, timedelta
from astropy.time import Time 

def determine_night_breakdown(date, plot=False, verbose=False):

	# if no date was passed, assume we're doing last night 
	if date is None:
		date = (datetime.now()- timedelta(1)).strftime('%Y%m%d')
	
	try:
		with open(f'/data/tierras/log/autoobserve_{date}.log') as f:
			log_lines = f.readlines()
	except:
		return 0, 0, 0, 0, 0, 0

	slew_start_times = []
	slew_end_times = []
	exposure_start_times = []
	exposure_end_times = []
	focus_start_times = []
	focus_end_times = []
	acquisition_start_times = []
	acquisition_end_times = []
	readout_start_times = []
	readout_end_times = []
	guide_start_times = []
	guide_end_times = []

	for i in range(len(log_lines)):
		if verbose:
			print(i)
		if 'SLEW' in log_lines[i] :
			if i == len(log_lines)-1:
				continue
			elif 'Acquisition' in log_lines[i+1]:
				slew_start_times.append(datetime.strptime(log_lines[i].split(' INFO')[0], '%Y-%m-%d %H:%M:%S'))
				slew_end_times.append(datetime.strptime(log_lines[i+1].split(' INFO')[0], '%Y-%m-%d %H:%M:%S'))

		if 'Exposure 'in log_lines[i]:
			exptime = float(log_lines[i].split(' ')[-1].split('s')[0])
			exp_start_time = datetime.strptime(log_lines[i].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')
			exp_end_time = exp_start_time + timedelta(seconds=exptime)
			exposure_start_times.append(exp_start_time)	
			exposure_end_times.append(exp_end_time)

			try:
				readout_end_time = datetime.strptime(log_lines[i+1].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')
				readout_end_times.append(readout_end_time)
				readout_start_times.append(exp_end_time)
			except:
				continue
		

			if 'Finished' in log_lines[i+2] or 'Refocusing' in log_lines[i+2] or 'Dome is closed' in log_lines[i+2]:
				continue
			elif 'SLEW' in log_lines[i+3] : # we don't guide when we slew or focus
				continue
			else:
				guide_start_times.append(readout_end_time) # guide starts at end of readout 

				guide_end = False
				for j in np.arange(i+1, len(log_lines)):
					if 'Exposure' in log_lines[j]:
						guide_end_time = datetime.strptime(log_lines[j].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')
						guide_end = True
						break

				if not guide_end:
					removed_ = guide_start_times.pop()
				else:
					guide_end_times.append(guide_end_time)



		if 'Refocusing' in log_lines[i]:
			focus_start_time = datetime.strptime(log_lines[i].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')
			
			focus_end = False
			
			for j in np.arange(i, len(log_lines)):
				if 'Final focus' in log_lines[j]:
					focus_end_time = datetime.strptime(log_lines[j+1].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')
					focus_end = True
					break
			
			if focus_end:
				focus_start_times.append(focus_start_time)
				focus_end_times.append(focus_end_time)
	
		if 'Acquisition' in log_lines[i]:
			acquisition_start_time = datetime.strptime(log_lines[i].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')

			acquisition_end = False 
			# loop forward. Acquisition sequence ends when exposure sequence begins
			for j in np.arange(i, len(log_lines)):
				if 'Exposure' in log_lines[j]:
					acquisition_end_time = datetime.strptime(log_lines[j].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')
					acquisition_end = True
					break
			if acquisition_end:
				acquisition_start_times.append(acquisition_start_time)
				acquisition_end_times.append(acquisition_end_time)


	unique_acquisition_start_times = []
	unique_acquisition_end_times   = []
	for i in range(len(acquisition_end_times)):
		if acquisition_end_times[i] not in unique_acquisition_end_times:
			unique_acquisition_start_times.append(acquisition_start_times[i])
			unique_acquisition_end_times.append(acquisition_end_times[i])
		else:
			continue
		# breakpoint()
	acquisition_start_times = unique_acquisition_start_times
	acquisition_end_times = unique_acquisition_end_times
		
	if plot:
		fig, ax = plt.subplots(1,1,figsize=(16,2))

	total_slew_time = 0
	for i in range(len(slew_start_times)):
		total_slew_time += (slew_end_times[i]-slew_start_times[i]).seconds
		if plot:
			ax.fill_between([slew_start_times[i], slew_end_times[i]], 0, 1, color='m', alpha=0.6, lw=0, interpolate=False)

	total_exp_time = 0 
	for i in range(len(exposure_start_times)):
		total_exp_time += (exposure_end_times[i]-exposure_start_times[i]).seconds
		if plot:
			ax.fill_between([exposure_start_times[i], exposure_end_times[i]], 0, 1, color='g', alpha=0.6, lw=0, interpolate=False)

	total_focus_time = 0 
	for i in range(len(focus_start_times)):
		total_focus_time += (focus_end_times[i]-focus_start_times[i]).seconds
		if plot:
			ax.fill_between([focus_start_times[i], focus_end_times[i]], 0, 1, color='b', alpha=0.6, lw=0, interpolate=False)

	total_acquisition_time = 0 
	for i in range(len(acquisition_start_times)):
		total_acquisition_time += (acquisition_end_times[i]-acquisition_start_times[i]).seconds
		if plot:
			ax.fill_between([acquisition_start_times[i], acquisition_end_times[i]], 0, 1, color='tab:orange', alpha=0.6, lw=0, interpolate=False)

	total_readout_time = 0
	for i in range(len(readout_start_times)):
		total_readout_time += (readout_end_times[i]-readout_start_times[i]).seconds
		if plot:
			ax.fill_between([readout_start_times[i], readout_end_times[i]], 0, 1, color='y', alpha=0.6, lw=0, interpolate=False)

	total_guide_time = 0
	for i in range(len(guide_start_times)):
		total_guide_time += (guide_end_times[i]-guide_start_times[i]).seconds
		if plot:
			ax.fill_between([guide_start_times[i], guide_end_times[i]], 0, 1, color='tab:cyan', alpha=1, lw=0, interpolate=False)
	if plot:
		ax.grid(False)
		ax.set_ylim(0,1)
	
	if verbose:
		print(f'Total slew time (hours): {total_slew_time/3600:.1f}')
		print(f'Total guide time (hours): {total_guide_time/3600:.1f}')
		print(f'Total acquisition time (hours): {total_acquisition_time/3600:.1f}')
		print(f'Total readout time (hours): {total_readout_time/3600:.1f}')
		print(f'Total focus time (hours): {total_focus_time/3600:.1f}')
		print(f'Total exposure time (hours): {total_exp_time/3600:.1f}')

	if plot:
		fig.savefig(f'/data/tierras/night_reports/{date}_report.png', dpi=300)

	breakpoint()
	return total_slew_time, total_guide_time, total_acquisition_time, total_readout_time, total_focus_time, total_exp_time

if __name__ == '__main__':
	plt.ion()
	date = '20260326'
	slew, guide, acq, read, foc, exp = determine_night_breakdown(date, plot=True)
else:
	plt.ioff()
import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.ion()
import argparse
from datetime import datetime, timedelta
from astropy.time import Time 

def main(raw_args=None):
	ap = argparse.ArgumentParser()
	ap.add_argument("-date", required=False, help="Date of observation in YYYYMMDD format.")
	args = ap.parse_args(raw_args)
	
	date = args.date

	# if no date was passed, assume we're doing last night 
	if date is None:
		date = (datetime.now()- timedelta(1)).strftime('%Y%m%d')
	
	with open(f'/data/tierras/log/autoobserve_{date}.log') as f:
		log_lines = f.readlines()

	slew_start_times = []
	slew_end_times = []
	exposure_start_times = []
	exposure_end_times = []
	focus_start_times = []
	focus_end_times = []
	for i in range(len(log_lines)):
		if 'SLEW' in log_lines[i] and 'Acquisition' in log_lines[i+1]:
			slew_start_times.append(datetime.strptime(log_lines[i].split(' INFO')[0], '%Y-%m-%d %H:%M:%S'))
			slew_end_times.append(datetime.strptime(log_lines[i+1].split(' INFO')[0], '%Y-%m-%d %H:%M:%S'))

		if 'Exposure 'in log_lines[i]:
			exptime = float(log_lines[i].split(' ')[-1].split('s')[0])
			exp_start_time = datetime.strptime(log_lines[i].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')
			exp_end_time = exp_start_time + timedelta(seconds=exptime)
			exposure_start_times.append(exp_start_time)	
			exposure_end_times.append(exp_end_time)
		
		if 'Refocusing' in log_lines[i]:
			focus_start_time = datetime.strptime(log_lines[i].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')
			focus_end = False
			while not focus_end:
				for j in np.arange(i, len(log_lines)):
					if 'Final focus' in log_lines[j]:
						focus_end_time = datetime.strptime(log_lines[j+1].split(' INFO')[0], '%Y-%m-%d %H:%M:%S')
						focus_end = True
						break
			focus_start_times.append(focus_start_time)
			focus_end_times.append(focus_end_time)
	

	fig, ax = plt.subplots(1,1,figsize=(16,2))
	total_slew_time = 0
	for i in range(len(slew_start_times)):
		total_slew_time += (slew_end_times[i]-slew_start_times[i]).seconds
		ax.fill_between([slew_start_times[i], slew_end_times[i]], 0, 1, color='r', alpha=0.6, lw=0, interpolate=False)

	total_exp_time = 0 
	for i in range(len(exposure_start_times)):
		total_exp_time += (exposure_end_times[i]-exposure_start_times[i]).seconds
		ax.fill_between([exposure_start_times[i], exposure_end_times[i]], 0, 1, color='g', alpha=0.6, lw=0, interpolate=False)

	total_focus_time = 0 
	for i in range(len(focus_start_times)):
		total_focus_time += (focus_end_times[i]-focus_start_times[i]).seconds
		ax.fill_between([focus_start_times[i], focus_end_times[i]], 0, 1, color='b', alpha=0.6, lw=0, interpolate=False)

	ax.grid(False)
	ax.set_ylim(0,1)
	
	print(f'Total Slew time (minutes): {total_slew_time/60:.1f}')
	breakpoint()
	fig.savefig(f'/data/tierras/night_reports/{date}_report.png', dpi=300)

if __name__ == '__main__':
	plt.ion()
	main()
else:
	plt.ioff()
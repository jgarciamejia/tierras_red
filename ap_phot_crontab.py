import glob
import argparse
import numpy as np
import ap_phot
#from ap_phot import main 

def get_target_list(datepath):
    flist = os.listdir(datepath)
    target_list = set([flist[ind].split('.')[2] for ind in range(len(flist))])
    if 'FLAT001' in target_list:
        target_list = [targetname for targetname in target_list if not targetname.startswith('FLAT')]
        #print (target_list)
    return target_list

def get_yday_date():
    # Get today's date
    today = datetime.now()
    # Calculate yesterday's date
    yesterday = today - timedelta(days=1)
    # Format yesterday's date as "YYYYMMDD"
    formatted_date = yesterday.strftime("%Y%m%d")
    # Print the result
    return formatted_date

# Deal with command line
ap = argparse.ArgumentParser()
ap.add_argument("-dimness_limit",required=False,default=0.05,help="Minimum flux a reference star can have compared to the target to be considered as a reference star.",type=float)
ap.add_argument("-targ_distance_limit",required=False,default=2000,help="Maximum distance a source can be from the target in pixels to be considered as a reference star.",type=float)
args = ap.parse_args()
live_plot = args.live_plot
dimness_limit = args.dimness_limit
targ_distance_limit = args.targ_distance_limit

# Access observation info
ffname = "flat0000" #eventually will have to upgrade to actually pass a flat file
date = get_yday_date()
live_plot = False


#Define base path
fpath = '/data/tierras/flattened'

# Get target names observed
targets = get_target_list(os.path.join(fpath,date))

for target in targets:
	folderlist = np.sort(glob.glob(os.path.join(fpath,date,target))
	arg_str = f'-target {targetname} -date {datelist[i]} -ffname {ffname} -live_plot {live_plot} -dimness_limit {dimness_limit} -targ_distance_limit {targ_distance_limit}'.split()
	try:
		ap_phot.main(arg_str)
	except Exception as e:
		print(f"Error in Iteration {i}: {e}")
		failed_dates.append(datelist[i])
		continue

	print ("Photometric extraction failed on the following dates: {}".format(failed_dates))


# TODOs: add all output of ap_phot to a log file that is sent to user 
# send email to Pat and JGM with summary of how reduction went 
# 
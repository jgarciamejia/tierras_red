import os
import glob
import argparse
import numpy as np
import ap_phot_forcrontab
#from ap_phot import main 

def get_yday_date():
    # Get today's date
    today = datetime.now()
    # Calculate yesterday's date
    yesterday = today - timedelta(days=1)
    # Format yesterday's date as "YYYYMMDD"
    formatted_date = yesterday.strftime("%Y%m%d")
    # Print the result
    return formatted_date

# Access observation info
ffname = "flat0000" # TO DO: upgrade when flats available
date = '20240115'#get_yday_date()
live_plot = False # TO DO: upgrade if needed

#Define base path
fpath = '/data/tierras/flattened'

# Get target names observed
targets = os.listdir(os.path.join(fpath,date)) #implies that only reduced/flattened targets will be extracted

for target in targets:
	folderlist = np.sort(glob.glob(os.path.join(fpath,date,target)))
	arg_str = f'-target {target} -date {date} -ffname {ffname} -live_plot {live_plot}'.split()
	ap_phot_forcrontab.main(arg_str)

# TODOs: 
# add all output of ap_phot to a log file that is sent to user 
# send email to Pat and JGM with summary of how reduction went
# would prefer to have a single ap_phot because the pipeline is continually updated

from ap_phot import main as ap_phot_main
from analyze_global import main as analyze_global_main
from build_tierras_db import main as build_tierras_db_main

import numpy as np 
import os 
from datetime import datetime 
from astropy.time import Time 
import subprocess
import argparse
from ap_phot import t_or_f

'''
    a wrapper function to do photometry 
'''
ap = argparse.ArgumentParser()
ap.add_argument("-single_field", required=False, default='', help="If passed, run pipeline on specified field only.")
ap.add_argument("-start_field", required=False, default='', help="If you pass a name, the code will skip all targets in the target list preceding the passed field. This is just for convenience if you need to stop running the code in-person and start running in a remote session, or if the code crashes halway through.")
ap.add_argument('-date', required=False, default=None, help='Date of data to process. If not passed, will default to last night.' )
ap.add_argument('-ffname', required=False, default='flat0000', help='Name of flattened directory.')

args = ap.parse_args()
single_field = args.single_field
start_field = args.start_field
date = args.date 
ffname = args.ffname

# if no date was passed, grab last night's date
if date is None:
    last_night = (Time(datetime.now()) - 1).value
    date = str(last_night.year)+str(last_night.month).zfill(2)+str(last_night.day).zfill(2)

# specify the date list
phot_type = 'fixed'

# read in priority target list 
with open('/home/ptamburo/tierras/tierras_analyze/analysis_priority_fields.txt', 'r') as f:
    priority_targets = f.readlines()
priority_targets = [i.strip() for i in priority_targets][::-1]

print('Doing photometry...')
if phot_type == 'fixed':
    ap_radii = np.arange(5,21)
elif phot_type == 'variable':
    ap_radii = np.arange(0.5, 1.6, 0.1)


if single_field == '':
    target_list = sorted(os.listdir(f'/data/tierras/flattened/{date}'))
    # move any fields in analysis_priority_fields.txt to the front of the list 
    
    for i in range(len(priority_targets)):
        shift_ind = np.where([j == priority_targets[i] for j in target_list])[0]
        if len(shift_ind) == 0: # if the target is in the priority list but was not observed, skip it
            continue 
        target_list.remove(priority_targets[i])
        target_list.insert(0, priority_targets[i])
else:
    target_list = [single_field]

if start_field != '':
    ind = np.where(np.array(target_list) == start_field)[0][0]
    target_list = target_list[ind:]

for j in range(len(target_list)):
    target = target_list[j]
    if target == 'TARGET' or target == 'TARGET_red':
        continue
    if 'TEST' in target:
        continue
    # TODO: how to automatically set the rp_mag_limit for really faint/bright targets?
    if target == 'POI-2':
        rp_mag_limit = 17.06
    if target == 'HD60779':
        rp_mag_limit = 14
    else:
        rp_mag_limit = 17.00
    args = f'-target {target} -date {date} -ffname {ffname} -rp_mag_limit {rp_mag_limit} -ap_radii {" ".join(map(str,ap_radii))} -phot_type {phot_type} -plot_source_detection False'
    print(args)
    ap_phot_main(args.split())
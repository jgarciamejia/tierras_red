#!/usr/bin/env python

import os 
import glob
import argparse 

# Deal with command line
ap = argparse.ArgumentParser()
ap.add_argument("-date", required=True, help="Date of observation in YYYYMMDD format.")
ap.add_argument("-target", required=False, help="Name of observed target as shown in raw FITS files. The name need not be complete so long as it contains enough characters to differentiate it from other target names within the same night.")
ap.add_argument("-targets", metavar='N', required=False, type=str, nargs='+', help="List containing name of observed targets as shown in the raw FITS files.")
args = ap.parse_args()

date = args.date
target = args.target
target_list = args.targets

def count_files_in_folder(folder_path):
    count = 0
    items = os.listdir(folder_path)
    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            count += 1
    return count


def count_files_with_string(folder_path, search_string):
    count = 0
    files = glob.glob(os.path.join(folder_path, f"*{search_string}*"))
    for file in files:
        if os.path.isfile(file):
            count += 1
    return count

def count_files_with_strings(folder_path, strings_to_count):
    # Initialize a dictionary to store counts for each string
    counts = {string: 0 for string in strings_to_count}

    # Use os.listdir to get a list of all items (files and directories) in the folder
    items = os.listdir(folder_path)

    # Iterate through the items and count files that contain each string
    for item in items:
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            for string in strings_to_count:
                if string in item:
                    counts[string] += 1

    return counts

incomingpath = '/data/tierras/incoming'
datepath = os.path.join(incomingpath,date)

nfiles_per_date = count_files_in_folder(datepath)

if target is None and target_list is None:
    flist = os.listdir(datepath)
    target_list = set([flist[ind].split('.')[2] for ind in range(len(flist))])
    if 'FLAT001' in target_list:
        target_list = [targetname for targetname in target_list if not targetname.startswith('FLAT')]
        target_list.append('FLAT')
    nfiles_per_targets = count_files_with_strings(datepath,target_list)
    print (target_list)
    print (date)
    nfilesum = 0
    for target,nfiles in nfiles_per_targets.items():
        print (target,nfiles)
        nfilesum += nfiles
    print (nfiles_per_date, nfilesum)

    #print (date,nfiles_per_date)

elif target is not None:
    nfiles_per_target = count_files_with_string(datepath,target)
    print (date,target,nfiles_per_target)
    print (nfiles_per_date)

elif target is None and target_list is not None:
    nfiles_per_targets = count_files_with_strings(datepath,target_list)
    print (date)
    nfilesum = 0
    for target,nfiles in nfiles_per_targets.items():
        print (target,nfiles)
        nfilesum += nfiles
    print (nfiles_per_date, nfilesum)

import argparse 
from sort_and_red_crontab import get_yday_date
import os 
from defocused_wcs_solver import defocused_wcs_solver
import subprocess

def main():
    """WCS of thwomp reference fields is frequently inaccurate, so update it automatically using local Gaia q
    
    """

    ap = argparse.ArgumentParser()
    ap.add_argument('-date', default=None, help='Calendar date for which to reduce data. If not passed, will assume yesterday.')
    args = ap.parse_args() 

    if args.date is None:
        # Access observation info
        date = get_yday_date()
    else:
        date = args.date   

    flattened_dirs = os.listdir(f'/data/tierras/flattened/{date}/')
    thwomp_ref_fields = [i for i in flattened_dirs if '_ref' in i]

    if len(thwomp_ref_fields) == 0:
        print(f'No THWOMP reference fields found on {date}, returning!')
        return 

    # loop over any thwomp ref fields and update wcs
    for i in range(len(thwomp_ref_fields)):      
        
        defocused_wcs_solver(date=date, field=thwomp_ref_fields[i], write=True) 

    return

if __name__ == '__main__':
    main()
#!/opt/cfpython/python-3.11.9/bin/python
import os 
from datetime import datetime, timezone, timedelta
from astroplan import Observer 
from astropy.time import Time
import astropy.units as u
import time 
import logging 
import subprocess
import argparse
import shlex
from glob import glob 
import numpy as np 

logfile = '/data/tierras/log/pipeline.log'
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
emails = ['patrick.tamburo@cfa.harvard.edu', 'juliana.garcia-mejia@cfa.harvard.edu']

def notify_failure(subject): 
    # email Pat / Juliana if the pipeline breaks
    recipients = " ".join(emails)
    safe_subject = shlex.quote(subject)  # adds quotes and escapes special chars
    try: 
        cmd = f'echo | mutt -s {safe_subject} {recipients}'
        os.system(cmd)
    except Exception as e:
        logging.error(f'Failed to send error notification with mutt: {e}')

def run_command(command, description):
    logging.info(f'Starting: {description}')
    try: 
        subprocess.run(command, shell=True, check=True)
        logging.info(f'Completed: {description}')
    except subprocess.CalledProcessError as e:
        # if the command fails, log it and email 
        logging.error(f'Error during: {description}\n{e}')
        notify_failure(f'Pipeline step failed: {description}')
        raise # stop the pipeline

def main(run_now=False, skip_transfer=False, skip_flat=False, skip_reduction=False, skip_photometry=False, skip_light_curves=False, skip_db=False):
    try: 
        if not run_now:
            # sleep until civil twilight at FLWO
            obs = Observer.at_site('Whipple')
            current_ut_time = Time(datetime.now(timezone.utc))
            sunrise_time = obs.sun_rise_time(current_ut_time, horizon=-6*u.deg, which='next')
            delta = int((sunrise_time.value - current_ut_time.jd)*86400)
            if delta < 0: 
                raise RuntimeError('Current time is after sunrise.')
            
            logging.info(f'Sleeping for {delta} seconds.')
            time.sleep(delta)
        else:
            logging.info('Skipping sunrise wait time due to -n/--now flag.')
    except Exception as e:
        logging.error(f'Error before pipeline start: {e}')
        notify_failure(f'Pipeline failed before starting: {e}')
        return

    try: 
        cal_date = Time(datetime.now(timezone.utc)-timedelta(days=1)).strftime('%Y%m%d')

        PYTHON = '/opt/cfpython/python-3.11.9/bin/python' # this is needed to work with crontab

        if not skip_transfer:
            run_command('/home/ptamburo/bin/tierrascopy', 'Data transfer from telescope')
            run_command(f'{PYTHON} /home/ptamburo/tierras/tierras_track/mv_autoobservelog.py', 'Transfer autoobserve log')
            run_command(f'{PYTHON} /home/ptamburo/tierras/tierras_track/mv_teldlog.py', 'Transfer teld log')

        if not skip_flat:
            # check for flats 
            # if they were taken, make a flat and a new superflat
            file_list = np.array(sorted(glob(f'/data/tierras/incoming/{cal_date}/*.fit')))
            n_flats = 0
            flats = False
            for i in range(len(file_list)):
                if 'FLAT' in file_list[i]:
                    n_flats += 1

            if n_flats > 2:
                flats = True

            # if we got more than 2 flats...
            if flats:
                # don't need to pass the python path here since its declared at the start of the flatcombine.py program
                # make the flat 
                run_command(f'/home/ptamburo/tierras/tierras_red/flatcombine.py -date {cal_date}', 'Make flat field')

                # make a new super flat including the newest flat, with a window looking back 60 nights
                run_command(f'{PYTHON} /home/ptamburo/tierras/tierras_red/superflat.py -end_date {cal_date} -day_window 60', 'Make superflat')

            # in the ORIGINAL pipeline, we would not flat field the data
            # run_command(f'{PYTHON} /home/ptamburo/tierras/tierras_red/sort_and_red_crontab.py -ffname flat0000', 'Reduce data')

            # for now, run a parallel reduction that does the flat fielding correction and saves to flat0001 directories

            # get the list of available super flats
            super_flats = np.array(sorted(glob('/data/tierras/flats/SUPERFLAT*')))

            # choose the one whose end date is closest to the current date? 
            super_flat_end_dates = [datetime.strptime(i.split('/')[-1].split('_')[-1].split('.')[0], '%Y%m%d') for i in super_flats]

            deltas = np.array([abs((datetime.strptime(cal_date, '%Y%m%d')- i).days) for i in super_flat_end_dates])

            super_flat = super_flats[np.argmin(deltas)]

        if not skip_reduction:
            # from September 2025 onward, we flat field the data
            run_command(f'{PYTHON} /home/ptamburo/tierras/tierras_red/sort_and_red_crontab.py -ffname flat0000 -f {super_flat}', 'Reduce data')

        if not skip_photometry:
            run_command(f'{PYTHON} /home/ptamburo/tierras/tierras_red/run_photometry.py', 'Run photometry') 

        if not skip_light_curves:

            # Update light curves for ALL fields observed in the past week only on Sunday
            if datetime.now().strftime('%A') == 'Saturday':
                logging.info('Updating light curves for all sources in all fields observed in the past week.')
                run_command(f'{PYTHON} /home/ptamburo/tierras/tierras_analyze/make_light_curves.py -all_fields_last_week True', 'Make light curves')
            else:
                # Otherwise, just update the high priority fields 
                logging.info('Updating light curves for all sources in analysis_priority_fields observed last night.')
                run_command(f'{PYTHON} /home/ptamburo/tierras/tierras_analyze/make_light_curves.py -high_priority_only True', 'Make light curves')


        if not skip_db:
            run_command(f'{PYTHON} /home/ptamburo/tierras/tierras_analyze/update_database.py', 'Update database')

    except Exception: 
        logging.error('Pipeline terminated early due to previous error.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tierras morning pipeline")
    parser.add_argument('-n', '--now', action='store_true',
                        help='Run immediately without waiting for sunrise')
    parser.add_argument('-skip_transfer', action='store_true', help='Skip data transfer from telescope')
    parser.add_argument('-skip_flat', action='store_true', help='Skip creation of flat/superflat')
    parser.add_argument('-skip_reduction', action='store_true', help='Skip reduction of data')
    parser.add_argument('-skip_photometry', action='store_true', help='Skip photometry')
    parser.add_argument('-skip_light_curves', action='store_true', help='Skip creation of light curves')
    parser.add_argument('-skip_db', action='store_true', help='Skip database update')
    args = parser.parse_args()
    main(run_now=args.now, skip_transfer=args.skip_transfer, skip_flat=args.skip_flat, skip_reduction=args.skip_reduction, skip_photometry=args.skip_photometry, skip_light_curves=args.skip_light_curves, skip_db=args.skip_db)

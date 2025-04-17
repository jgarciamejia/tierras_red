#!/opt/cfpython/python-3.11.9/bin/python
import os 
from datetime import datetime, timezone
from astroplan import Observer 
from astropy.time import Time
import astropy.units as u
import time 
import logging 
import subprocess
import argparse
import shlex

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

def main(run_now=False):
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
        run_command('/home/ptamburo/bin/tierrascopy', 'Data transfer from telescope')
        run_command('python /home/ptamburo/tierras/tierras_red/sort_and_red_crontab.py', 'Reduce data')
        run_command('python /home/ptamburo/tierras/tierras_track/mv_autoobservelog.py', 'Transfer autoobserve log')
        run_command('python /home/ptamburo/tierras/tierras_track/mv_teldlog.py', 'Transfer teld log')
        run_command('python /home/ptamburo/tierras/tierras_analyze/process_data.py', 'Run photometry and make light curves')
    except Exception: 
        logging.error('Pipeline terminated early due to previous error.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tierras morning pipeline")
    parser.add_argument('-n', '--now', action='store_true',
                        help='Run immediately without waiting for sunrise')
    args = parser.parse_args()
    main(run_now=args.now)

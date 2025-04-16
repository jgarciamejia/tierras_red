#!/opt/cfpython/python-3.11.9/bin/python
import os 
from datetime import datetime, timezone
from astroplan import Observer 
from astropy.time import Time
import astropy.units as u
import time 

def main():

    # sleep until civil twilight at FLWO
    obs = Observer.at_site('Whipple')
    current_ut_time = Time(datetime.now(timezone.utc))
    sunrise_time = obs.sun_rise_time(current_ut_time, horizon=-6*u.deg, which='next')
    delta = int((sunrise_time.value - current_ut_time.jd)*86400)
    if delta < 0: 
        raise RuntimeError('Current time is after sunrise.')
    
    print(f'Sleeping for {delta} seconds.')
    time.sleep(delta)

    # run data transfer scripts
    os.system('/home/ptamburo/bin/tierrascopy')
    os.system('python /home/ptamburo/tierras/tierras_red/sort_and_red_crontab.py')
    os.system('python /home/ptamburo/tierras/tierras_track/mv_autoobservelog.py')
    os.system('python /home/ptamburo/tierras/tierras_track/mv_teldlog.py')

    # run photometry and make light curves 
    os.system('python /home/ptamburo/tierras/tierras_analyze/process_data.py')

if __name__ == '__main__':
    main()
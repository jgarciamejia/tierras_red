import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
from glob import glob 
from astropy.io import fits
import argparse 
from datetime import datetime, timedelta
from ap_phot import set_tierras_permissions

def main(raw_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('-start_date', required=False, default=None, help='Start date (YYYYMMDD) on which to begin loading flats for combination. If None, starts from the earliest avaiable flat.')
    ap.add_argument('-end_date', required=False, default=None, help='End date (YYYYMMDD) on which to stop loading flats for combination. If None, ends on the latest avaiable flat.')
    ap.add_argument('-day_window', required=False, default=60, type=int, help='If only the end date is passed, the code will look for other flats within day_window dayss of the end date to generate the super flat. This is mainly used in run_transfer_and_analyze.py')
    args = ap.parse_args(raw_args)
    start_date = args.start_date 
    end_date = args.end_date
    day_window = args.day_window

    flats_dir = '/data/tierras/flats/'
    flats = np.array(sorted(glob(flats_dir+'*_FLAT.fit')))
    flat_dates = []
    keep_inds = []
    for i in range(len(flats)):
        if 'SUPER' in flats[i]:
            continue
        flat_dates.append(datetime.strptime(flats[i].split('/')[-1].split('_')[0], '%Y%m%d'))
        keep_inds.append(i)
    flat_dates = np.array(flat_dates)
    flats = flats[keep_inds]

    if end_date is not None:
        end_date = datetime.strptime(end_date, '%Y%m%d')
    else:
        end_date = flat_dates[-1]

    if start_date is not None: 
        start_date = datetime.strptime(start_date, '%Y%m%d')
    else:
        start_date = end_date - timedelta(days=day_window)

    inds = np.array([i >= start_date and i <= end_date for i in flat_dates])

    flat_dates = np.array(flat_dates)[inds]
    flats = np.array(flats)[inds]

    print(f'Median combining {len(flats)} flats between {start_date.strftime("%Y%m%d")} and {end_date.strftime("%Y%m%d")}.')

    flat_data = np.zeros((len(flats), 2048, 4096), dtype='float32')
    for i in range(len(flats)):
        flat_data[i] = fits.open(flats[i])[0].data
    superflat = np.median(flat_data, axis=0)

    hdr = fits.Header()
    hdr['COMMENT'] = f'N_flats = {len(flats)}'
    hdr['COMMENT'] = f'start_date = {start_date.strftime("%Y%m%d")}'
    hdr['COMMENT'] = f'end_date = {end_date.strftime("%Y%m%d")}'
    hdr['COMMENT'] = 'Combined the following flats:'
    for i in range(len(flats)):
        hdr['COMMENT'] = flats[i]

    output_hdul = fits.HDUList([fits.PrimaryHDU(data=superflat, header=hdr)])

    output_path = flats_dir+f'SUPERFLAT_{start_date.strftime("%Y%m%d")}_to_{end_date.strftime("%Y%m%d")}.fit'
    output_hdul.writeto(output_path, overwrite=True)
    set_tierras_permissions(output_path)
    print(f'Wrote superflat to {output_path}!')

    return 

if __name__ == '__main__':
    main()
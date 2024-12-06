import numpy as np 
import matplotlib.pyplot as plt 
plt.ion()
from glob import glob 
from astropy.io import fits
import argparse 
from datetime import datetime 
from ap_phot import set_tierras_permissions

def main(raw_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('-start_date', required=False, default=None, help='Start date (YYYYMMDD) on which to begin loading flats for combination. If None, starts from the earliest avaiable flat.')
    ap.add_argument('-end_date', required=False, default=None, help='End date (YYYYMMDD) on which to stop loading flats for combination. If None, ends on the latest avaiable flat.')
    args = ap.parse_args(raw_args)
    start_date = args.start_date 
    end_date = args.end_date

    flats_dir = '/data/tierras/flats/'
    flats = sorted(glob(flats_dir+'*.fit'))
    flat_dates = np.array([datetime.strptime(i.split('/')[-1].split('_')[0], '%Y%m%d') for i in flats])

    if start_date is not None: 
        start_date = datetime.strptime(start_date, '%Y%m%d')
    else:
        start_date = flat_dates[0]

    if end_date is not None:
        end_date = datetime.strptime(end_date, '%Y%m%d')
    else:
        end_date = flat_dates[-1]

    inds = np.array([i >= start_date and i <= end_date for i in flat_dates])

    flat_dates = np.array(flat_dates)[inds]
    flats = np.array(flats)[inds]

    print(f'Median combining {len(flats)} flats between {start_date.strftime("%Y%m%d")} and {end_date.strftime("%Y%m%d")}.')

    flat_data = np.zeros((len(flats), 2048, 4096))
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
    output_hdul.writeto(output_path)
    set_tierras_permissions(output_path)
    print(f'Wrote superflat to {output_path}!')

    return 

if __name__ == '__main__':
    main()
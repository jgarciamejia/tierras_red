import argparse 
import numpy as np 
import os 
from glob import glob
from astropy.io import fits

# sometimes we make a mistake entering a target's name into Tierras and only catch it later on 
# this will loop through data directories for the current name and rename them with the target name 

def main(raw_args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('-current_name', required=True, help='The name you want to replace')
    ap.add_argument('-new_name', required=True, help='The new name.')
    args = ap.parse_args(raw_args)
    current_name = args.current_name
    new_name = args.new_name 

    # /data/tierras/incoming 
    # need to change file name and update header 
    incoming_path = '/data/tierras/incoming/'
    incoming_dirs = os.listdir(incoming_path)
    for i in range(len(incoming_dirs)):
        #try:
        incoming_files = np.array(glob(incoming_path + incoming_dirs[i]+'/*.fit'))
        incoming_targs = [path.split('.')[-2] for path in incoming_files]
        if current_name in incoming_targs:
            print(incoming_dirs[i])
            file_inds = np.where(np.array(incoming_targs) == current_name)[0]
            for im_num in range(len(file_inds)):
                # replace hdul[0]['OBJECT'] 
                filepath = incoming_files[file_inds[im_num]]
                with fits.open(filepath, mode='update') as filehandle:
                    filehandle[0].header['OBJECT'] = new_name
                # rename file 
                os.rename(filepath, filepath.replace(current_name, new_name))


    # I recommend manually deleting the misnamed flattened/photometry/lightcurve/fields directories, and then just re-running the pipeline on the offending dates

if __name__ == '__main__':
    main()
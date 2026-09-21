from astropy.io import fits
from astropy.wcs import WCS

'''
    Copy the WCS from a "reference" file to a "target" file
'''

flattened_path = '/data/tierras/flattened/'
date           = '20260905'
target         = 'HIP107350_ref'
ref_filenum    = '0200' # the reference file
target_filenum = '0195' # the file to be updated

# 1. Load the reference file and extract its WCS
ref_filename = f'{flattened_path}/{date}/{target}/flat0000/{date}.{ref_filenum}.{target}_red.fit'

with fits.open(ref_filename) as ref_hdul:
    # Usually, the WCS is in the primary header (index 0) or the image extension (index 1)
    ref_wcs = WCS(ref_hdul[0].header) 

# 2. Convert the WCS object back into clean FITS header keywords
wcs_header_cards = ref_wcs.to_header()

# 3. Open the target file in update mode and overwrite its WCS keywords
target_filename = f'{flattened_path}/{date}/{target}/flat0000/{date}.{target_filenum}.{target}_red.fit'
with fits.open(target_filename, mode='update') as target_hdul:
    target_header = target_hdul[0].header  # Or target_hdul['SCI'].header depending on your structure
    
    # Remove existing WCS keywords from the target to avoid conflicting metadata
    # (Optional but highly recommended)
    for card in wcs_header_cards:
        if card in target_header:
            del target_header[card]
            
    # Append the new WCS keywords to the target header
    target_header.extend(wcs_header_cards, update=True)
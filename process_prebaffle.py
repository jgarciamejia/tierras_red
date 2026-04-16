#!/usr/bin/env python
"""
process_prebaffle.py

Process pre-baffle Tierras nights through the full pipeline:
  1. sort_and_red_crontab.py  (image reduction, if flattened files don't exist)
  2. ap_phot.py               (aperture photometry, if parquets don't exist)
  3. Extract per-exposure sky background timeseries from photometry outputs

Pre-baffle dates: 20211013 through 20220214
  (data in /data/tierras/incoming, no flat field applied)

Usage:
    python process_prebaffle.py                        # run full pipeline
    python process_prebaffle.py --dry-run              # preview without executing
    python process_prebaffle.py --extract-only          # skip reduction/photometry, just extract sky data
    python process_prebaffle.py --start 20211101        # start from a specific date
    python process_prebaffle.py --skip-reduction        # skip sort_and_red, assume flattened files exist
    python process_prebaffle.py --skip-photometry       # skip ap_phot, assume parquets exist
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from glob import glob
from collections import Counter
from datetime import datetime
from pathlib import Path

# ---- configuration ----
INCOMING_ROOT = '/data/tierras/incoming'
FLATTENED_ROOT = '/data/tierras/flattened'
PHOTOMETRY_ROOT = '/data/tierras/photometry'
FFNAME = 'flat0000'  # no flat was available for pre-baffle data

# Pre-baffle date range (inclusive)
PREBAFFLE_START = '20211013'
PREBAFFLE_END = '20220214'

# Aperture photometry settings (matching run_photometry.py)
AP_RADII = np.arange(5, 21)  # 5 through 20 pixels, 1-pixel increments
PHOT_TYPE = 'fixed'
RP_MAG_LIMIT = 17.0


def setup_logging(logfile=None):
    fmt = '%(asctime)s  %(levelname)-8s  %(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def get_dates_in_range(start_str, end_str, incoming_root):
    """Return sorted list of YYYYMMDD date strings that exist as directories
    in incoming_root and fall within [start_str, end_str]."""
    all_dirs = sorted(glob(os.path.join(incoming_root, '????????')))
    dates = []
    for d in all_dirs:
        name = os.path.basename(d)
        try:
            datetime.strptime(name, '%Y%m%d')
        except ValueError:
            continue
        if start_str <= name <= end_str:
            dates.append(name)
    return dates


def detect_fields(date, incoming_root):
    """Auto-detect all science field names from raw FITS filenames.

    Raw files follow: YYYYMMDD.NNNN.TARGET.fit
    Returns a list of unique field names, excluding calibration frames.
    """
    pattern = os.path.join(incoming_root, date, f'{date}.*.fit')
    files = glob(pattern)
    if not files:
        pattern = os.path.join(incoming_root, date, f'{date}.*.fits')
        files = glob(pattern)
    if not files:
        return []

    calibration_keywords = {'flat', 'dark', 'bias', 'zero', 'test', 'focus',
                            'pointing', 'flat001', 'target', 'target_red'}
    fields = set()
    for f in files:
        basename = os.path.basename(f)
        parts = basename.split('.')
        if len(parts) < 4:
            continue
        target = parts[2]
        if target.lower() not in calibration_keywords and not target.upper().startswith('FLAT') and 'TEST' not in target.upper():
            fields.add(target)

    return sorted(fields)


def flattened_files_exist(date, field, ffname):
    """Check whether reduced (_red.fit) files exist."""
    flat_path = os.path.join(FLATTENED_ROOT, date, field, ffname)
    if not os.path.isdir(flat_path):
        return False
    red_files = glob(os.path.join(flat_path, '*_red.fit'))
    return len(red_files) > 0


def photometry_exists(date, field, ffname):
    """Check whether ap_phot output parquets exist."""
    phot_path = os.path.join(PHOTOMETRY_ROOT, date, field, ffname)
    if not os.path.isdir(phot_path):
        return False
    parquets = glob(os.path.join(phot_path, '*phot*.parquet'))
    return len(parquets) > 0


def run_sort_and_red(date, dry_run=False):
    """Run sort_and_red_crontab to reduce raw data (no flat).

    sort_and_red_crontab auto-detects all targets for a given date,
    so one call handles every field observed that night.
    We do NOT pass -f since there's no flat for pre-baffle data.
    """
    import subprocess
    cmd = [sys.executable, '/home/jmejia/tierras/photometry/tierras_red/sort_and_red_crontab.py',
           '-date', date, '-ffname', FFNAME]
    logging.info(f'  sort_and_red cmd: {" ".join(cmd)}')

    if dry_run:
        logging.info('  [DRY RUN] skipping reduction')
        return True

    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        logging.error(f'  sort_and_red FAILED: {e}')
        return False


def run_ap_phot(date, field, dry_run=False):
    """Run ap_phot for a given date/field."""
    ap_radii_str = ' '.join(map(str, AP_RADII))
    args_list = ['-target', field,
                 '-date', date,
                 '-ffname', FFNAME,
                 '-rp_mag_limit', str(RP_MAG_LIMIT),
                 '-ap_radii'] + [str(r) for r in AP_RADII] + [
                 '-phot_type', PHOT_TYPE,
                 '-plot_source_detection', 'False']
    logging.info(f'  ap_phot args: {" ".join(args_list)}')

    if dry_run:
        logging.info('  [DRY RUN] skipping photometry')
        return True

    try:
        from ap_phot import main as ap_phot_main
        ap_phot_main(args_list)
        return True
    except Exception as e:
        logging.error(f'  ap_phot FAILED: {e}')
        return False


def extract_sky_data(date, field, ffname):
    """Extract per-exposure sky background timeseries from photometry output.

    Returns a DataFrame with columns:
        date, field, BJD_TDB, Airmass, Exposure_Time, HA, Dome_Humid,
        FWHM_X, FWHM_Y, Median_Sky_ADU, Median_Sky_ADU_per_s
    or None if data cannot be read.
    """
    import pyarrow.parquet as pq

    phot_path = os.path.join(PHOTOMETRY_ROOT, date, field, ffname)

    # read ancillary data (times, airmass, etc.)
    ancillary_files = glob(os.path.join(phot_path, '*ancillary*.parquet'))
    if not ancillary_files:
        logging.warning(f'    No ancillary parquet found for {date}/{field}')
        return None

    try:
        ancillary_tab = pq.read_table(ancillary_files[0])
        ancillary_df = ancillary_tab.to_pandas()
    except Exception as e:
        logging.warning(f'    Could not read ancillary data for {date}/{field}: {e}')
        return None

    # read one photometry file to get per-source sky values
    phot_files = sorted(glob(os.path.join(phot_path, '*phot*.parquet')))
    if not phot_files:
        logging.warning(f'    No photometry parquets found for {date}/{field}')
        return None

    try:
        # use the first (smallest aperture) file — sky values are the same across apertures
        phot_tab = pq.read_table(phot_files[0])
        sky_cols = [c for c in phot_tab.column_names if c.endswith(' Sky')]
        if sky_cols:
            sky_arr = np.column_stack([phot_tab[c].to_numpy() for c in sky_cols]) # shape: (N exposures, M sources)
            median_sky = np.nanmedian(sky_arr, axis=1)  # median across all sources
        else:
            logging.warning(f'    No sky columns found in photometry for {date}/{field}')
            median_sky = np.full(len(ancillary_df), np.nan)
    except Exception as e:
        logging.warning(f'    Could not read photometry for {date}/{field}: {e}')
        return None

    n_exp = len(ancillary_df)

    # build output dataframe
    result = pd.DataFrame({
        'date': [date] * n_exp,
        'field': [field] * n_exp,
    })

    # map ancillary columns
    col_map = {
        'BJD TDB': 'BJD_TDB',
        'Airmass': 'Airmass',
        'Exposure Time': 'Exposure_Time',
        'HA': 'HA',
        'Dome Humid': 'Dome_Humid',
    }
    for orig, new in col_map.items():
        if orig in ancillary_df.columns:
            result[new] = ancillary_df[orig].values
        else:
            result[new] = np.nan

    # FWHM may or may not be present (appended by measure_fwhm_grid)
    for col in ['FWHM X', 'FWHM Y']:
        new_name = col.replace(' ', '_')
        if col in ancillary_df.columns:
            result[new_name] = ancillary_df[col].values
        else:
            result[new_name] = np.nan

    result['Median_Sky_ADU'] = median_sky
    if 'Exposure_Time' in result.columns:
        exp_times = result['Exposure_Time'].values.astype(float)
        exp_times[exp_times == 0] = np.nan  # avoid division by zero
        result['Median_Sky_ADU_per_s'] = median_sky / exp_times
    else:
        result['Median_Sky_ADU_per_s'] = np.nan

    return result


def main():
    ap = argparse.ArgumentParser(
        description='Process pre-baffle Tierras nights: reduce, photometer, and extract sky data.')
    ap.add_argument('--start', default=PREBAFFLE_START,
                    help=f'Start date YYYYMMDD (default: {PREBAFFLE_START})')
    ap.add_argument('--end', default=PREBAFFLE_END,
                    help=f'End date YYYYMMDD (default: {PREBAFFLE_END})')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print what would be done without executing')
    ap.add_argument('--skip-reduction', action='store_true',
                    help='Skip sort_and_red even if flattened files are missing')
    ap.add_argument('--skip-photometry', action='store_true',
                    help='Skip ap_phot even if photometry is missing')
    ap.add_argument('--extract-only', action='store_true',
                    help='Skip reduction and photometry; only extract sky data from existing outputs')
    ap.add_argument('--output', default='/data/tierras/prebaffle_sky_data.csv',
                    help='Path to save the combined sky background CSV')
    ap.add_argument('--logfile', default=None,
                    help='Optional path to a log file')
    args = ap.parse_args()

    if args.extract_only:
        args.skip_reduction = True
        args.skip_photometry = True

    setup_logging(args.logfile)
    logging.info('=' * 60)
    logging.info('Pre-baffle Tierras data processing')
    logging.info(f'Date range: {args.start} to {args.end}')
    logging.info(f'ffname: {FFNAME}')
    logging.info(f'skip_reduction: {args.skip_reduction}')
    logging.info(f'skip_photometry: {args.skip_photometry}')
    logging.info(f'Output: {args.output}')
    logging.info('=' * 60)

    dates = get_dates_in_range(args.start, args.end, INCOMING_ROOT)
    logging.info(f'Found {len(dates)} date directories in range.')

    results = {
        'reduced': [], 'reduction_skipped': [], 'reduction_failed': [],
        'photometered': [], 'photometry_skipped': [], 'photometry_failed': [],
        'extracted': [], 'extraction_failed': [],
        'no_fields': [],
    }

    all_sky_data = []

    for i, date in enumerate(dates):
        logging.info(f'[{i + 1}/{len(dates)}] {date}')

        # auto-detect field(s)
        fields = detect_fields(date, INCOMING_ROOT)
        if not fields:
            logging.warning(f'  No science fields detected. Skipping.')
            results['no_fields'].append(date)
            continue
        logging.info(f'  Fields: {fields}')

        # --- STEP 1: Reduction ---
        # sort_and_red processes ALL fields for a given date in one call,
        # so we only need to check if ANY field is missing flattened files.
        if not args.skip_reduction:
            needs_reduction = any(not flattened_files_exist(date, f, FFNAME) for f in fields)
            if needs_reduction:
                logging.info(f'  Running sort_and_red for {date}...')
                success = run_sort_and_red(date, dry_run=args.dry_run)
                if success:
                    results['reduced'].append(date)
                else:
                    results['reduction_failed'].append(date)
                    continue  # skip photometry/extraction if reduction failed
            else:
                logging.info(f'  Flattened files already exist for all fields.')
                results['reduction_skipped'].append(date)
        else:
            results['reduction_skipped'].append(date)

        # --- STEP 2 & 3: Photometry and extraction, per-field ---
        for field in fields:
            logging.info(f'  Field: {field}')

            # check flattened files exist before attempting photometry
            if not flattened_files_exist(date, field, FFNAME):
                logging.warning(f'    No flattened files for {field}. Skipping.')
                continue

            # photometry
            if not args.skip_photometry:
                if not photometry_exists(date, field, FFNAME):
                    logging.info(f'    Running ap_phot...')
                    success = run_ap_phot(date, field, dry_run=args.dry_run)
                    if success:
                        results['photometered'].append(f'{date}/{field}')
                    else:
                        results['photometry_failed'].append(f'{date}/{field}')
                        continue
                else:
                    logging.info(f'    Photometry already exists.')
                    results['photometry_skipped'].append(f'{date}/{field}')
            else:
                results['photometry_skipped'].append(f'{date}/{field}')

            # extraction
            if not args.dry_run and photometry_exists(date, field, FFNAME):
                logging.info(f'    Extracting sky data...')
                sky_df = extract_sky_data(date, field, FFNAME)
                if sky_df is not None and len(sky_df) > 0:
                    all_sky_data.append(sky_df)
                    results['extracted'].append(f'{date}/{field}')
                    logging.info(f'    Extracted {len(sky_df)} exposures.')
                else:
                    results['extraction_failed'].append(f'{date}/{field}')

    # --- Save combined sky data ---
    if all_sky_data and not args.dry_run:
        combined = pd.concat(all_sky_data, ignore_index=True)
        combined.to_csv(args.output, index=False)
        logging.info(f'Saved combined sky data: {args.output} ({len(combined)} rows)')
        try:
            from ap_phot import set_tierras_permissions
            set_tierras_permissions(args.output)
        except Exception:
            pass

    # --- Summary ---
    logging.info('')
    logging.info('=' * 60)
    logging.info('SUMMARY')
    logging.info(f'  Dates in range:            {len(dates)}')
    logging.info(f'  No fields detected:        {len(results["no_fields"])}')
    logging.info(f'  Reduced:                   {len(results["reduced"])}')
    logging.info(f'  Reduction skipped:         {len(results["reduction_skipped"])}')
    logging.info(f'  Reduction failed:          {len(results["reduction_failed"])}')
    logging.info(f'  Photometry run:            {len(results["photometered"])}')
    logging.info(f'  Photometry skipped:        {len(results["photometry_skipped"])}')
    logging.info(f'  Photometry failed:         {len(results["photometry_failed"])}')
    logging.info(f'  Sky data extracted:        {len(results["extracted"])}')
    logging.info(f'  Extraction failed:         {len(results["extraction_failed"])}')
    if all_sky_data:
        logging.info(f'  Total exposures saved:     {sum(len(df) for df in all_sky_data)}')
    logging.info('=' * 60)

    if results['reduction_failed']:
        logging.info('\nDates where reduction failed:')
        for d in results['reduction_failed']:
            logging.info(f'  {d}')

    if results['photometry_failed']:
        logging.info('\nDate/fields where photometry failed:')
        for d in results['photometry_failed']:
            logging.info(f'  {d}')


if __name__ == '__main__':
    main()

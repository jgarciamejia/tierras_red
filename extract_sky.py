#!/usr/bin/env python
"""
extract_sky.py

Measure per-exposure sigma-clipped sky background from reduced Tierras frames.
Walks *_red.fit files under:
    /data/tierras/flattened/<date>/<field>/<ffname>/            (primary)
    /data/tierras/flattened/<date>/<field>/<ffname>/excluded/   (astrom-failed but
                                                                 photometrically fine)

Each row is one exposure. Output is raw values only:
  - sigma-clipped image statistics (median, std, N pixels used)
  - selected header keywords passed through verbatim

Moon illumination / separation / baffle era are NOT computed here. Feed the
output CSV to compute_moon_per_exposure.py later for those columns.

Usage:
    # contiguous date range
    python extract_sky.py --start 20211013 --end 20220214 --output sky_prebaffle.csv

    # hand-picked non-contiguous dates (one YYYYMMDD per line, '#' comments ok)
    python extract_sky.py --dates tel_baffle_dates.txt --output sky_tel.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

FLATTENED_ROOT = '/data/tierras/flattened'
DEFAULT_FFNAME = 'flat0000'

# Header keywords passed through to the CSV verbatim (missing -> NaN/empty).
HEADER_KEYS = [
    'DATE-OBS', 'MJD-OBS', 'EXPTIME', 'AIRMASS', 'HA',
    'CAT-RA', 'CAT-DEC', 'RA', 'DEC',
    'OBSTYPE', 'OBJECT',
]

# Field dirs whose name starts with any of these are skipped by default.
CAL_PREFIXES = ('BIAS', 'DARK', 'FLAT', 'POINT', 'TEST', 'FOCUS', 'WARM')

# Number of random finite pixels used to estimate the sigma-clipped sky per
# frame. 500k is statistically indistinguishable from the full ~7M finite pixels
# for a robust median, ~10x faster, and avoids the temp-array OOM on cafecol.
_SKY_SAMPLE_SIZE = 500_000
_SKY_RNG = np.random.default_rng(42)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        stream=sys.stdout,
    )


def resolve_dates(args):
    """Return sorted YYYYMMDD list from either --dates file or --start/--end range.

    For --start/--end: intersect the requested window with dates that actually
    exist under FLATTENED_ROOT so we don't waste time on empty nights.
    """
    if args.dates:
        with open(args.dates) as f:
            lines = [line.strip() for line in f]
        dates = [line for line in lines if line and not line.startswith('#')]
        for d in dates:
            datetime.strptime(d, '%Y%m%d')
        return sorted(set(dates))

    all_dirs = sorted(glob(os.path.join(FLATTENED_ROOT, '????????')))
    dates = []
    for d in all_dirs:
        name = os.path.basename(d)
        try:
            datetime.strptime(name, '%Y%m%d')
        except ValueError:
            continue
        if args.start <= name <= args.end:
            dates.append(name)
    return dates


def parse_frame_number(filename):
    """Pull NNNN out of YYYYMMDD.NNNN.TARGET_red.fit."""
    parts = os.path.basename(filename).split('.')
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def extract_one_frame(red_path, date, field, in_excluded):
    """Return one dict of sky stats + header values for a _red.fit, or None."""
    try:
        with fits.open(red_path, memmap=False) as hdul:
            data = hdul[0].data
            hdr = hdul[0].header
    except Exception as e:
        logging.warning(f'    could not open {os.path.basename(red_path)}: {e}')
        return None

    try:
        # Subsample finite pixels before sigma-clipping. Peak memory per frame
        # drops from ~200 MB to ~30 MB (kills the OOM on cafecol), runtime
        # drops from ~1.2 s to ~0.3 s, and the median is statistically
        # indistinguishable from using all 8M pixels.
        finite = data[np.isfinite(data)]
        n_used = int(finite.size)
        if finite.size > _SKY_SAMPLE_SIZE:
            idx = _SKY_RNG.integers(0, finite.size, size=_SKY_SAMPLE_SIZE)
            sample = finite[idx]
        else:
            sample = finite
        _, median, std = sigma_clipped_stats(sample, sigma=3.0)
    except Exception as e:
        logging.warning(f'    sigma_clipped_stats failed on {os.path.basename(red_path)}: {e}')
        return None

    row = {
        'date': date,
        'field': field,
        'frame_number': parse_frame_number(red_path),
        'filename': os.path.basename(red_path),
        'in_excluded': in_excluded,
        'Median_Sky_ADU': float(median),
        'Sky_Std_ADU': float(std),
        'N_pixels_used': n_used,
    }
    for key in HEADER_KEYS:
        row[key] = hdr.get(key)
    return row


def iter_fields(date, ffname, include_calibration_fields):
    """Yield (field_name, flat_dir, excluded_dir) for each field dir on a date."""
    date_root = os.path.join(FLATTENED_ROOT, date)
    if not os.path.isdir(date_root):
        return
    for field in sorted(os.listdir(date_root)):
        if not include_calibration_fields and field.upper().startswith(CAL_PREFIXES):
            continue
        flat_dir = os.path.join(date_root, field, ffname)
        if not os.path.isdir(flat_dir):
            continue
        excluded_dir = os.path.join(flat_dir, 'excluded')
        yield field, flat_dir, excluded_dir


def main():
    ap = argparse.ArgumentParser(
        description='Extract sigma-clipped sky + header values from Tierras _red.fit frames.')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--dates', help='Path to a text file with one YYYYMMDD per line')
    group.add_argument('--start', help='Start date YYYYMMDD (inclusive)')
    ap.add_argument('--end', help='End date YYYYMMDD (inclusive; required with --start)')
    ap.add_argument('--ffname', default=DEFAULT_FFNAME,
                    help=f'Flattened subdir name (default: {DEFAULT_FFNAME})')
    ap.add_argument('--output', required=True, help='Output CSV path')
    ap.add_argument('--include-calibration-fields', action='store_true',
                    help='Include BIAS/DARK/FLAT/POINT/TEST/FOCUS/WARM field dirs '
                         '(skipped by default)')
    args = ap.parse_args()

    if args.start and not args.end:
        ap.error('--end is required with --start')

    setup_logging()

    dates = resolve_dates(args)
    logging.info(f'extract_sky: {len(dates)} dates, ffname={args.ffname}, '
                 f'output={args.output}')

    rows = []
    for i, date in enumerate(dates, 1):
        logging.info(f'[{i}/{len(dates)}] {date}')
        date_rows_before = len(rows)

        for field, flat_dir, excluded_dir in iter_fields(
                date, args.ffname, args.include_calibration_fields):
            flat_files = sorted(glob(os.path.join(flat_dir, '*_red.fit')))
            excluded_files = sorted(glob(os.path.join(excluded_dir, '*_red.fit')))
            if not flat_files and not excluded_files:
                continue
            logging.info(f'  {field}: {len(flat_files)} in flat0000 / '
                         f'{len(excluded_files)} in excluded')

            for rf in flat_files:
                row = extract_one_frame(rf, date, field, in_excluded=False)
                if row is not None:
                    rows.append(row)
            for rf in excluded_files:
                row = extract_one_frame(rf, date, field, in_excluded=True)
                if row is not None:
                    rows.append(row)

        logging.info(f'  → {len(rows) - date_rows_before} exposures from {date}')

    if not rows:
        logging.warning('No rows extracted. Nothing written.')
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    logging.info(f'Wrote {len(df)} rows to {args.output}')


if __name__ == '__main__':
    main()

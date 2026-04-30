#!/usr/bin/env python
"""
extract_sky_from_phot.py

Per-frame sky CSV produced by reading ap_phot's existing parquets, taking
the median of "<source N> Sky" columns across sources for each frame.

Companion to extract_sky.py — output schema is identical, so
compute_moon_per_exposure.py works on this CSV without modification.

Why use this instead of extract_sky.py
--------------------------------------
- Matches the methodology used in the Garcia-Mejia thesis test-photometry
  plots (per-source annulus median).
- Much faster: ap_phot parquets already exist on disk, no FITS reads, no
  sigma_clipped_stats. Hundreds of dates per minute on cafecol.

When to use extract_sky.py instead
----------------------------------
- For nights that don't have ap_phot output (typically: pre-baffle nights
  where astrom failed, so reduction succeeded but photometry was never run).

Column semantics (differences from extract_sky.py)
--------------------------------------------------
- Median_Sky_ADU : nanmedian of "<source> Sky" columns per frame
- Sky_Std_ADU   : nanstd of the same
- N_pixels_used : NUMBER OF SOURCES contributing (not pixel count!) — useful
                  as a quality flag (very small values = sparse field).
- in_excluded   : always False (phot parquets only contain photometered frames).
- MJD-OBS       : BJD_TDB from ancillary parquet minus 2400000.5. The
                  barycentric offset is ≤ ~8 min, negligible for moon calcs.
- DATE-OBS, CAT-RA, CAT-DEC, RA, DEC, OBSTYPE, OBJECT : pulled from the first
                  reduced-FITS header for the (date, field). Constant per
                  field, so read once. NaN if the FITS file isn't on disk.
- frame_number  : 1-based row index in the parquet (frames in time order).
                  filename column is left None since parquets don't store the
                  source FITS filename.

Usage
-----
    python extract_sky_from_phot.py --start 20211025 --end 20211205 \\
        --output sky_prebaffle_phot.csv

    python extract_sky_from_phot.py --dates tel_baffle_dates.txt \\
        --output sky_telbaffle_phot.csv
"""

import argparse
import gc
import logging
import os
import sys
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits

PHOTOMETRY_ROOT = '/data/tierras/photometry'
FLATTENED_ROOT = '/data/tierras/flattened'
DEFAULT_FFNAME = 'flat0000'

# Constant-per-field header keywords pulled from the first reduced FITS
FITS_HEADER_KEYS = ['DATE-OBS', 'CAT-RA', 'CAT-DEC', 'RA', 'DEC', 'OBSTYPE', 'OBJECT']

# ancillary column name → extract_sky-style output column
ANCILLARY_COLUMN_MAP = {
    'Exposure Time': 'EXPTIME',
    'Airmass':       'AIRMASS',
    'HA':            'HA',
}

CAL_PREFIXES = ('BIAS', 'DARK', 'FLAT', 'POINT', 'TEST', 'FOCUS', 'WARM')


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        stream=sys.stdout,
    )


def resolve_dates(args):
    if args.dates:
        with open(args.dates) as f:
            lines = [line.strip() for line in f]
        dates = [line for line in lines if line and not line.startswith('#')]
        for d in dates:
            datetime.strptime(d, '%Y%m%d')
        return sorted(set(dates))

    start = datetime.strptime(args.start, '%Y%m%d')
    end = datetime.strptime(args.end, '%Y%m%d')
    if end < start:
        raise ValueError('--end must be >= --start')
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime('%Y%m%d'))
        cur += timedelta(days=1)
    return dates


def already_processed_dates(output_path):
    if not os.path.exists(output_path):
        return set()
    try:
        existing = pd.read_csv(output_path, usecols=['date'])
        return set(existing['date'].astype(str).unique())
    except Exception as e:
        logging.warning(f'could not read existing output {output_path}: {e}')
        return set()


def get_field_header_metadata(date, field, ffname, flattened_root):
    """Pull constant-per-field header keys from the first available _red.fit.
    Falls back to flat0000/excluded/ if flat0000/ is empty."""
    flat_dir = os.path.join(flattened_root, date, field, ffname)
    files = sorted(glob(os.path.join(flat_dir, '*_red.fit')))
    if not files:
        files = sorted(glob(os.path.join(flat_dir, 'excluded', '*_red.fit')))
    if not files:
        return {k: None for k in FITS_HEADER_KEYS}
    try:
        hdr = fits.getheader(files[0], 0)
        return {k: hdr.get(k) for k in FITS_HEADER_KEYS}
    except Exception as e:
        logging.warning(f'  could not read FITS header for {date}/{field}: {e}')
        return {k: None for k in FITS_HEADER_KEYS}


def extract_one_field(date, field, ffname, photometry_root, flattened_root):
    """Process one (date, field). Returns list of dict rows (one per frame), or []."""
    phot_dir = os.path.join(photometry_root, date, field, ffname)
    if not os.path.isdir(phot_dir):
        return []

    ancillary_files = sorted(glob(os.path.join(phot_dir, '*ancillary*.parquet')))
    phot_files = sorted(glob(os.path.join(phot_dir, '*phot*.parquet')))
    if not ancillary_files or not phot_files:
        logging.warning(f'  {date}/{field}: missing ancillary or phot parquets')
        return []

    try:
        ancillary = pq.read_table(ancillary_files[0]).to_pandas()
    except Exception as e:
        logging.warning(f'  {date}/{field}: could not read ancillary: {e}')
        return []

    try:
        # any phot parquet works — Sky columns are independent of aperture radius
        phot = pq.read_table(phot_files[0]).to_pandas()
    except Exception as e:
        logging.warning(f'  {date}/{field}: could not read phot parquet: {e}')
        return []

    n_exp = len(ancillary)
    if len(phot) != n_exp:
        logging.warning(f'  {date}/{field}: row count mismatch '
                        f'(ancillary={n_exp}, phot={len(phot)}) — skipping')
        return []

    # Per-frame Sky stats across sources
    sky_cols = [c for c in phot.columns if c.endswith(' Sky')]
    if not sky_cols:
        logging.warning(f'  {date}/{field}: no Sky columns in phot parquet')
        return []
    sky_arr = np.column_stack([phot[c].values for c in sky_cols])
    median_sky = np.nanmedian(sky_arr, axis=1)
    std_sky    = np.nanstd(sky_arr, axis=1)
    n_sources  = np.sum(np.isfinite(sky_arr), axis=1)

    # Time: BJD_TDB → MJD-OBS-equivalent
    if 'BJD TDB' in ancillary.columns:
        bjd = pd.to_numeric(ancillary['BJD TDB'], errors='coerce')
        mjd_approx = bjd - 2400000.5
    else:
        mjd_approx = pd.Series([np.nan] * n_exp)

    # Constant header metadata for this (date, field) — read once
    header_meta = get_field_header_metadata(date, field, ffname, flattened_root)

    rows = []
    for i in range(n_exp):
        row = {
            'date':           date,
            'field':          field,
            'frame_number':   i + 1,
            'filename':       None,
            'in_excluded':    False,
            'Median_Sky_ADU': float(median_sky[i]) if np.isfinite(median_sky[i]) else np.nan,
            'Sky_Std_ADU':    float(std_sky[i])    if np.isfinite(std_sky[i])    else np.nan,
            'N_pixels_used':  int(n_sources[i]),
            'DATE-OBS':       header_meta.get('DATE-OBS'),
            'MJD-OBS':        float(mjd_approx.iloc[i]) if pd.notna(mjd_approx.iloc[i]) else np.nan,
        }
        for ancil_key, out_key in ANCILLARY_COLUMN_MAP.items():
            row[out_key] = ancillary.iloc[i].get(ancil_key) if ancil_key in ancillary.columns else None
        for k in ['CAT-RA', 'CAT-DEC', 'RA', 'DEC', 'OBSTYPE', 'OBJECT']:
            row[k] = header_meta.get(k)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description='Extract per-frame median Sky from ap_phot parquets.')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--dates', help='Path to file with one YYYYMMDD per line')
    group.add_argument('--start', help='Start date YYYYMMDD (inclusive)')
    ap.add_argument('--end', help='End date YYYYMMDD (inclusive; required with --start)')
    ap.add_argument('--ffname', default=DEFAULT_FFNAME)
    ap.add_argument('--photometry-root', default=PHOTOMETRY_ROOT,
                    help='Root path containing <date>/<field>/<ffname>/ parquets')
    ap.add_argument('--flattened-root', default=FLATTENED_ROOT,
                    help='Root path containing reduced FITS files (used to read '
                         'constant per-field header metadata)')
    ap.add_argument('--output', required=True, help='Output CSV')
    ap.add_argument('--include-calibration-fields', action='store_true',
                    help='Include BIAS/DARK/FLAT/POINT/TEST/FOCUS/WARM field dirs '
                         '(skipped by default)')
    args = ap.parse_args()

    if args.start and not args.end:
        ap.error('--end is required with --start')

    setup_logging()

    dates = resolve_dates(args)
    logging.info(f'extract_sky_from_phot: {len(dates)} dates -> {args.output}')

    done_dates = already_processed_dates(args.output)
    if done_dates:
        logging.info(f'Resume: {len(done_dates)} dates already in output, will skip them')

    total_written = 0
    for i, date in enumerate(dates, 1):
        logging.info(f'[{i}/{len(dates)}] {date}')
        if date in done_dates:
            logging.info('  already in output, skipping')
            continue

        date_dir = os.path.join(args.photometry_root, date)
        if not os.path.isdir(date_dir):
            logging.info('  no photometry directory; skipping')
            continue

        date_rows = []
        for field in sorted(os.listdir(date_dir)):
            if not args.include_calibration_fields and field.upper().startswith(CAL_PREFIXES):
                continue
            field_rows = extract_one_field(
                date, field, args.ffname, args.photometry_root, args.flattened_root)
            if field_rows:
                logging.info(f'  {field}: {len(field_rows)} frames')
                date_rows.extend(field_rows)

        logging.info(f'  → {len(date_rows)} frames from {date}')

        if date_rows:
            df = pd.DataFrame(date_rows)
            file_exists = os.path.exists(args.output)
            df.to_csv(args.output, mode='a', header=not file_exists, index=False)
            total_written += len(date_rows)
            del df, date_rows
            gc.collect()

    logging.info(f'Done. Wrote {total_written} new rows to {args.output}')


if __name__ == '__main__':
    main()

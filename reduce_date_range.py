#!/usr/bin/env python
"""
reduce_date_range.py

Drive sort_and_red_crontab.py over an inclusive YYYYMMDD date range.
No flat fielding is applied (intentional for the baffle sky-vs-moon
analysis: pre-baffle has no matching flats, and we want apples-to-apples
across eras).

Skips any date where every detected science field already has _red.fit
output in /data/tierras/flattened/<date>/<field>/<ffname>/.

Usage:
    python reduce_date_range.py --start 20211013 --end 20211031
    python reduce_date_range.py --start 20211013 --end 20211031 --dry-run
"""

import os
import sys
import argparse
import logging
import subprocess
from collections import Counter
from datetime import datetime
from glob import glob

INCOMING_ROOT = '/data/tierras/incoming'
FLATTENED_ROOT = '/data/tierras/flattened'
DEFAULT_FFNAME = 'flat0000'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SORT_AND_RED = os.path.join(SCRIPT_DIR, 'sort_and_red_crontab.py')


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        stream=sys.stdout,
    )


def get_dates_in_range(start_str, end_str, incoming_root):
    dates = []
    for d in sorted(glob(os.path.join(incoming_root, '????????'))):
        name = os.path.basename(d)
        try:
            datetime.strptime(name, '%Y%m%d')
        except ValueError:
            continue
        if start_str <= name <= end_str:
            dates.append(name)
    return dates


def detect_fields(date, incoming_root):
    """Sorted list of science fields for a date from raw FITS filenames.

    Raw filename pattern: YYYYMMDD.NNNN.TARGET.fit. Excludes calibration
    frames and deduplicates field names case-insensitively (keeping the
    casing that appears on the most files, e.g. Gl905 over GL905).
    """
    files = glob(os.path.join(incoming_root, date, f'{date}.*.fit'))
    if not files:
        return []

    calib_keywords = {'flat', 'dark', 'bias', 'zero', 'test', 'focus',
                      'pointing', 'flat001', 'target', 'target_red', 'warm'}
    calib_prefixes = ('FLAT', 'POINT', 'TEST')

    counts = Counter()
    for f in files:
        parts = os.path.basename(f).split('.')
        if len(parts) < 4:
            continue
        tgt = parts[2]
        if tgt.lower() in calib_keywords:
            continue
        if tgt.upper().startswith(calib_prefixes):
            continue
        counts[tgt] += 1

    seen = {}
    for tgt, n in counts.items():
        key = tgt.lower()
        if key not in seen or n > seen[key][1]:
            seen[key] = (tgt, n)
    return sorted(v[0] for v in seen.values())


def is_flattened(date, field, ffname):
    path = os.path.join(FLATTENED_ROOT, date, field, ffname)
    if not os.path.isdir(path):
        return False
    return bool(glob(os.path.join(path, '*_red.fit')))


def run_sort_and_red(date, ffname, dry_run=False):
    cmd = [sys.executable, SORT_AND_RED, '-date', date, '-ffname', ffname]
    logging.info(f'  cmd: {" ".join(cmd)}')
    if dry_run:
        logging.info('  [DRY RUN] not executed')
        return True
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f'  sort_and_red FAILED: {e}')
        return False


def main():
    ap = argparse.ArgumentParser(
        description='Run sort_and_red_crontab.py over an inclusive date range.')
    ap.add_argument('--start', required=True, help='Start date YYYYMMDD')
    ap.add_argument('--end', required=True, help='End date YYYYMMDD')
    ap.add_argument('--ffname', default=DEFAULT_FFNAME,
                    help=f'Flattened subdir name (default: {DEFAULT_FFNAME})')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print plan without executing sort_and_red')
    args = ap.parse_args()

    setup_logging()
    logging.info(f'reduce_date_range: [{args.start}, {args.end}] ffname={args.ffname} dry_run={args.dry_run}')

    dates = get_dates_in_range(args.start, args.end, INCOMING_ROOT)
    logging.info(f'Found {len(dates)} incoming date dirs in range.')

    reduced, skipped, no_fields, failed = [], [], [], []

    for i, date in enumerate(dates, 1):
        logging.info(f'[{i}/{len(dates)}] {date}')
        fields = detect_fields(date, INCOMING_ROOT)
        if not fields:
            logging.info('  no science fields; skipping')
            no_fields.append(date)
            continue
        logging.info(f'  fields: {fields}')

        if all(is_flattened(date, f, args.ffname) for f in fields):
            logging.info('  already fully reduced; skipping')
            skipped.append(date)
            continue

        ok = run_sort_and_red(date, args.ffname, dry_run=args.dry_run)
        (reduced if ok else failed).append(date)

    logging.info('=' * 50)
    logging.info(f'Dates in range:    {len(dates)}')
    logging.info(f'Reduced:           {len(reduced)}')
    logging.info(f'Already-reduced:   {len(skipped)}')
    logging.info(f'No fields:         {len(no_fields)}')
    logging.info(f'Failed:            {len(failed)}')
    if failed:
        logging.info('Failed dates:')
        for d in failed:
            logging.info(f'  {d}')


if __name__ == '__main__':
    main()

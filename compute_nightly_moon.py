#!/usr/bin/env python
"""
compute_nightly_moon.py

Survey moon coverage across a date range — one row per night. Used to pick
post-baffle sample nights for the baffle sky-vs-moon analysis. Cheap (no
FITS reads); runs fine on a laptop.

Output columns:
    date                                  YYYYMMDD. "night of <date>" = UT night
                                          beginning evening of <date> local time
                                          at FLWO.
    moon_illumination                     fraction 0-1 at local midnight
    moon_altitude_deg_at_midnight         moon altitude at FLWO at local midnight
    moon_peak_altitude_deg_during_night   max altitude during the astronomical-
                                          night window (sun < -18°)
    moon_hours_above_horizon              hours moon was above horizon during
                                          astronomical night
    n_incoming_frames                     optional. Count of *.fit in
                                          <incoming-root>/<date>/. Only filled
                                          when --incoming-root is given (i.e.
                                          you're on a machine where the
                                          /data/tierras tree is mounted).

Usage:
    # full era survey (run on local Mac is fine):
    python compute_nightly_moon.py --start 20220215 --end 20250828 \\
        --output moon_telbaffle.csv

    # add incoming-frame counts (run on cafecol):
    python compute_nightly_moon.py --start 20220215 --end 20250828 \\
        --output moon_telbaffle.csv \\
        --incoming-root /data/tierras/incoming

    # specific dates only:
    python compute_nightly_moon.py --dates dates.txt --output moon_subset.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astroplan import Observer
from astroplan.moon import moon_illumination
from astropy.coordinates import AltAz, get_body
from astropy.time import Time
import astropy.units as u

OBSERVER = Observer.at_site('Whipple')
FLWO = OBSERVER.location

# Number of grid points across each astronomical-night window for the peak-
# altitude and hours-above-horizon calculations. 120 points over a ~10-hour
# night ≈ 5-minute sampling. Moon moves ≲ 0.1° in 5 minutes — plenty fine.
N_GRID = 120


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        stream=sys.stdout,
    )


def resolve_dates(args):
    """Return YYYYMMDD strings from --dates file or a contiguous --start/--end range."""
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


def night_metrics(date_str, incoming_root=None):
    """Compute moon metrics for the night that begins evening of date_str.

    Reference time is noon UT on date_str (= ~5 AM local at FLWO, well before
    sunset). astroplan's midnight()/twilight() with which='next' then locate
    the upcoming evening's events.
    """
    ref_dt = datetime.strptime(date_str, '%Y%m%d') + timedelta(hours=12)
    ref = Time(ref_dt)

    midnight = OBSERVER.midnight(ref, which='next')

    try:
        eve = OBSERVER.twilight_evening_astronomical(ref, which='next')
        morn = OBSERVER.twilight_morning_astronomical(eve, which='next')
    except Exception as e:
        logging.warning(f'{date_str}: twilight calculation failed: {e}')
        eve = morn = None

    # Moon at local midnight
    midnight_altaz = AltAz(obstime=midnight, location=FLWO)
    moon_at_midnight = get_body('moon', midnight, FLWO).transform_to(midnight_altaz)
    moon_alt_midnight = float(moon_at_midnight.alt.deg)
    illum = float(moon_illumination(midnight))

    # Peak altitude + hours above horizon, sampled across astronomical night
    if eve is not None and morn is not None and morn > eve:
        grid = eve + (morn - eve) * np.linspace(0, 1, N_GRID)
        grid_altaz = AltAz(obstime=grid, location=FLWO)
        moon_alts = get_body('moon', grid, FLWO).transform_to(grid_altaz).alt.deg
        peak_alt = float(np.max(moon_alts))
        frac_up = float((moon_alts > 0).sum()) / len(grid)
        night_hours = (morn - eve).to(u.hour).value
        hours_up = frac_up * night_hours
    else:
        peak_alt = np.nan
        hours_up = np.nan

    row = {
        'date': date_str,
        'moon_illumination': illum,
        'moon_altitude_deg_at_midnight': moon_alt_midnight,
        'moon_peak_altitude_deg_during_night': peak_alt,
        'moon_hours_above_horizon': hours_up,
    }

    if incoming_root is not None:
        date_dir = os.path.join(incoming_root, date_str)
        if os.path.isdir(date_dir):
            row['n_incoming_frames'] = sum(1 for f in os.listdir(date_dir) if f.endswith('.fit'))
        else:
            row['n_incoming_frames'] = np.nan

    return row


def main():
    ap = argparse.ArgumentParser(
        description='Per-night moon survey for post-baffle night selection.')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--dates', help='Path to text file with one YYYYMMDD per line')
    group.add_argument('--start', help='Start date YYYYMMDD (--end required)')
    ap.add_argument('--end', help='End date YYYYMMDD (required with --start)')
    ap.add_argument('--output', required=True, help='Output CSV path')
    ap.add_argument('--incoming-root', default=None,
                    help='If provided, count *.fit under <incoming-root>/<date>/. '
                         'Pass on cafecol; omit on local Mac.')
    args = ap.parse_args()

    if args.start and not args.end:
        ap.error('--end is required with --start')

    setup_logging()

    dates = resolve_dates(args)
    logging.info(f'compute_nightly_moon: {len(dates)} dates -> {args.output}')
    if args.incoming_root:
        logging.info(f'Counting frames under {args.incoming_root}/<date>/')

    rows = []
    for i, d in enumerate(dates, 1):
        if i == 1 or i % 100 == 0 or i == len(dates):
            logging.info(f'[{i}/{len(dates)}] {d}')
        try:
            rows.append(night_metrics(d, incoming_root=args.incoming_root))
        except Exception as e:
            logging.warning(f'{d}: failed: {e}')

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    logging.info(f'Wrote {len(df)} nights to {args.output}')

    if len(df):
        logging.info(f'illumination:      min={df.moon_illumination.min():.3f} '
                     f'median={df.moon_illumination.median():.3f} '
                     f'max={df.moon_illumination.max():.3f}')
        if df.moon_peak_altitude_deg_during_night.notna().any():
            logging.info(f'peak alt (deg):    min={df.moon_peak_altitude_deg_during_night.min():.1f} '
                         f'max={df.moon_peak_altitude_deg_during_night.max():.1f}')
        if 'n_incoming_frames' in df.columns and df.n_incoming_frames.notna().any():
            logging.info(f'incoming frames:   '
                         f'nights with data: {(df.n_incoming_frames > 0).sum()} of {len(df)}')


if __name__ == '__main__':
    main()

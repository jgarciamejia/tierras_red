#!/usr/bin/env python
"""
compute_moon_per_exposure.py

Read a CSV produced by extract_sky.py and append four per-exposure columns:

  - moon_illumination     : fraction 0-1 (0 = new, 1 = full), from astroplan
  - moon_altitude_deg     : altitude at FLWO at exposure time; < 0 means below horizon
  - moon_separation_deg   : angular separation from target pointing (CAT-RA/CAT-DEC)
  - baffle_era            : 'no_baffles' / 'telescope_baffles_only' /
                            'telescope_and_edge_baffles'

Time source: MJD-OBS (preferred) with DATE-OBS fallback. Target pointing:
CAT-RA / CAT-DEC (commanded, per target). Observer: FLWO (via astroplan's
'Whipple' site).

Usage:
    python compute_moon_per_exposure.py --input sky_20211025.csv \
        --output sky_20211025_with_moon.csv

    # write in place (default omits the --output flag: new file with _with_moon.csv suffix)
    python compute_moon_per_exposure.py --input sky_20211025.csv
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, AltAz, get_body
from astropy.time import Time
import astropy.units as u
from astroplan import Observer
from astroplan.moon import moon_illumination

OBSERVER = Observer.at_site('Whipple')
FLWO = OBSERVER.location

# Baffle era cutoffs as YYYYMMDD strings (lower bound inclusive).
BAFFLE_CUTOFFS = [
    ('no_baffles',                 '00000000'),
    ('telescope_baffles_only',     '20220215'),
    ('telescope_and_edge_baffles', '20250829'),
]


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        stream=sys.stdout,
    )


def classify_baffle_era(date_str):
    """Return the baffle-era name for a YYYYMMDD date string."""
    era = 'no_baffles'
    for name, cutoff in BAFFLE_CUTOFFS:
        if date_str >= cutoff:
            era = name
    return era


def build_time_array(df):
    """Return an astropy Time array, preferring MJD-OBS and falling back to DATE-OBS."""
    mjd = pd.to_numeric(df.get('MJD-OBS'), errors='coerce') if 'MJD-OBS' in df else pd.Series([np.nan] * len(df))
    missing = mjd.isna()
    if missing.any() and 'DATE-OBS' in df:
        fallback = df.loc[missing, 'DATE-OBS'].astype(str).values
        try:
            mjd.loc[missing] = Time(fallback, format='isot', scale='utc').mjd
        except Exception as e:
            logging.warning(f'Could not parse {missing.sum()} DATE-OBS fallbacks: {e}')
    if mjd.isna().any():
        raise ValueError(f'{mjd.isna().sum()} rows have neither a valid MJD-OBS nor DATE-OBS')
    return Time(mjd.values, format='mjd', scale='utc')


def build_target_coords(df):
    """Return a SkyCoord array from CAT-RA / CAT-DEC (sexagesimal strings).

    Rows with unparseable coords become NaN separations downstream. Parses
    row-by-row so one bad row doesn't kill the whole array.
    """
    ra_arr = np.full(len(df), np.nan)
    dec_arr = np.full(len(df), np.nan)
    ra_series = df.get('CAT-RA')
    dec_series = df.get('CAT-DEC')
    if ra_series is None or dec_series is None:
        logging.warning('CAT-RA or CAT-DEC column missing; moon_separation_deg will be NaN everywhere')
        return SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg)

    n_bad = 0
    for i, (ra_s, dec_s) in enumerate(zip(ra_series, dec_series)):
        if pd.isna(ra_s) or pd.isna(dec_s):
            n_bad += 1
            continue
        try:
            c = SkyCoord(str(ra_s), str(dec_s), unit=(u.hourangle, u.deg))
            ra_arr[i] = c.ra.deg
            dec_arr[i] = c.dec.deg
        except Exception:
            n_bad += 1
    if n_bad:
        logging.warning(f'{n_bad} / {len(df)} rows had unparseable CAT-RA/CAT-DEC; separation will be NaN for those')
    return SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg)


def main():
    ap = argparse.ArgumentParser(
        description='Annotate an extract_sky.py CSV with moon + baffle-era columns.')
    ap.add_argument('--input', required=True, help='Input CSV from extract_sky.py')
    ap.add_argument('--output', help='Output CSV. Defaults to <input>_with_moon.csv.')
    args = ap.parse_args()

    setup_logging()

    df = pd.read_csv(args.input)
    logging.info(f'Loaded {len(df)} rows from {args.input}')

    # Time & target coords
    times = build_time_array(df)
    targets = build_target_coords(df)

    # Moon illumination (Earth-centered; topocentric difference is <0.1%)
    logging.info('Computing moon illumination...')
    df['moon_illumination'] = moon_illumination(times)

    # Moon position & altitude at FLWO at each exposure time
    logging.info('Computing moon position and altitude at FLWO...')
    moon = get_body('moon', times, FLWO)
    altaz_frame = AltAz(obstime=times, location=FLWO)
    moon_altaz = moon.transform_to(altaz_frame)
    df['moon_altitude_deg'] = moon_altaz.alt.deg

    # Sun altitude — for filtering twilight-contaminated frames at analysis time.
    # Astronomical night = sun altitude < -18°.
    logging.info('Computing sun altitude at FLWO...')
    sun = get_body('sun', times, FLWO)
    df['sun_altitude_deg'] = sun.transform_to(altaz_frame).alt.deg

    # Moon–target angular separation, computed in the observer's AltAz frame
    # (the angle you'd actually measure from FLWO). Avoids astropy's
    # NonRotationTransformationWarning about GCRS↔ICRS direction ambiguity.
    logging.info('Computing moon–target separation...')
    targets_altaz = targets.transform_to(altaz_frame)
    df['moon_separation_deg'] = targets_altaz.separation(moon_altaz).deg

    # Baffle era from the 'date' column (YYYYMMDD strings)
    df['baffle_era'] = df['date'].astype(str).apply(classify_baffle_era)

    output_path = args.output
    if output_path is None:
        root, ext = os.path.splitext(args.input)
        output_path = f'{root}_with_moon{ext}'

    df.to_csv(output_path, index=False)
    logging.info(f'Wrote {len(df)} rows with moon/era columns to {output_path}')

    # Brief summary for sanity
    logging.info(
        f'moon_illumination: min={df.moon_illumination.min():.3f} '
        f'median={df.moon_illumination.median():.3f} max={df.moon_illumination.max():.3f}'
    )
    logging.info(
        f'moon_altitude_deg: min={df.moon_altitude_deg.min():.1f} '
        f'median={df.moon_altitude_deg.median():.1f} max={df.moon_altitude_deg.max():.1f}'
    )
    logging.info(
        f'sun_altitude_deg:  min={df.sun_altitude_deg.min():.1f} '
        f'median={df.sun_altitude_deg.median():.1f} max={df.sun_altitude_deg.max():.1f}'
    )
    n_twilight = (df.sun_altitude_deg > -18).sum()
    logging.info(f'  ({n_twilight} of {len(df)} frames have sun_altitude > -18° = twilight-contaminated)')
    logging.info(f'baffle_era counts: {df.baffle_era.value_counts().to_dict()}')


if __name__ == '__main__':
    main()

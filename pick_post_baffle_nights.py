#!/usr/bin/env python
"""
pick_post_baffle_nights.py

Augment a compute_nightly_moon.py CSV with incoming-frame counts (only on
machines where /data/tierras is mounted, e.g. cafecol), then choose a
stratified sample of nights matched to the pre-baffle moon-illumination
range plus a few high-illumination context nights. Output is a date list
ready to feed to `extract_sky.py --dates`.

Selection strategy
------------------
1. Drop nights with fewer than --min-frames *.fit files in incoming/.
2. Within illumination [--illum-min, --illum-max], divide into --n-bins
   bins. From each bin, pick --per-bin nights spread across
   moon_peak_altitude_deg_during_night for altitude diversity. Spread is
   deterministic (linear-spaced indices on the bin sorted by peak alt) —
   no random seed, fully reproducible.
3. Add --high-illum-n nights from illum >= --high-illum-min, also spread
   across peak altitude.

If you don't like the auto pick, edit the resulting <output> dates.txt by
hand (one YYYYMMDD per line, '#' for comments) — that's all extract_sky.py
needs.

Usage (on cafecol)
------------------
    python pick_post_baffle_nights.py \\
        --input moon_telbaffle.csv \\
        --incoming-root /data/tierras/incoming \\
        --output tel_baffle_dates.txt \\
        --plot-output moon_telbaffle_selection.png

    python pick_post_baffle_nights.py \\
        --input moon_bothbaffles.csv \\
        --incoming-root /data/tierras/incoming \\
        --output both_baffles_dates.txt \\
        --plot-output moon_bothbaffles_selection.png
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        stream=sys.stdout,
    )


def count_incoming_frames(incoming_root, date_str):
    """Return (n_frames, dir_exists). dir absent (closed for monsoon, etc.) → (0, False)."""
    date_dir = os.path.join(incoming_root, str(date_str))
    if not os.path.isdir(date_dir):
        return 0, False
    return sum(1 for f in os.listdir(date_dir) if f.endswith('.fit')), True


def select_within_bin(group, n, sort_by='moon_peak_altitude_deg_during_night'):
    """Pick n rows from a DataFrame group, evenly spread across `sort_by`."""
    if len(group) <= n:
        return group.copy()
    sorted_grp = group.sort_values(sort_by).reset_index(drop=True)
    idx = np.linspace(0, len(sorted_grp) - 1, n).astype(int)
    return sorted_grp.iloc[idx].copy()


def make_plot(df, selected, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    available = df[df.n_incoming_frames > 0]

    ax.scatter(
        available.moon_illumination,
        available.moon_peak_altitude_deg_during_night,
        s=np.clip(available.n_incoming_frames / 10, 5, 100),
        c='lightgray', alpha=0.5, edgecolors='none',
        label=f'available nights with frames (n={len(available)})')

    ax.scatter(
        selected.moon_illumination,
        selected.moon_peak_altitude_deg_during_night,
        s=80, c='crimson', edgecolors='black', linewidths=0.5, alpha=0.85,
        label=f'selected (n={len(selected)})')

    ax.axhline(0, color='gray', ls=':', alpha=0.5, zorder=0)
    ax.set_xlabel('Moon illumination at midnight')
    ax.set_ylabel('Moon peak altitude during astronomical night (deg)')
    ax.set_title('Night selection — point size ∝ # incoming frames')
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f'Wrote plot: {path}')


def main():
    ap = argparse.ArgumentParser(
        description='Augment moon CSV with frame counts; pick stratified nights.')
    ap.add_argument('--input', required=True,
                    help='CSV from compute_nightly_moon.py')
    ap.add_argument('--incoming-root', default=None,
                    help='Path to /data/tierras/incoming. If given, '
                         'n_incoming_frames is (re-)computed and the input '
                         'CSV is updated in place.')
    ap.add_argument('--output', required=True, help='Output dates.txt')
    ap.add_argument('--plot-output', default=None, help='Optional plot PNG')
    ap.add_argument('--min-frames', type=int, default=30,
                    help='Drop nights with fewer than this many incoming frames')
    ap.add_argument('--illum-min', type=float, default=0.05,
                    help='Lower bound for stratified illumination range')
    ap.add_argument('--illum-max', type=float, default=0.75,
                    help='Upper bound for stratified illumination range')
    ap.add_argument('--n-bins', type=int, default=5,
                    help='Illumination bins across [illum-min, illum-max]')
    ap.add_argument('--per-bin', type=int, default=3,
                    help='Nights to pick per illumination bin')
    ap.add_argument('--high-illum-min', type=float, default=0.85,
                    help='Threshold for high-illum context nights')
    ap.add_argument('--high-illum-n', type=int, default=3,
                    help='Number of high-illum context nights to add')
    args = ap.parse_args()

    setup_logging()

    df = pd.read_csv(args.input)
    df['date'] = df['date'].astype(str)
    logging.info(f'Loaded {len(df)} rows from {args.input}')

    if args.incoming_root:
        logging.info(f'Counting incoming frames under {args.incoming_root}/<date>/ ...')
        results = df['date'].map(lambda d: count_incoming_frames(args.incoming_root, d))
        df['n_incoming_frames'] = results.map(lambda t: t[0])
        df['incoming_dir_exists'] = results.map(lambda t: t[1])
        df.to_csv(args.input, index=False)
        logging.info(f'Updated {args.input} with n_incoming_frames + incoming_dir_exists')

    if 'n_incoming_frames' not in df.columns or df.n_incoming_frames.isna().all():
        logging.error('CSV has no n_incoming_frames. Re-run with --incoming-root.')
        sys.exit(1)

    n_total = len(df)
    if 'incoming_dir_exists' in df.columns:
        n_no_dir = int((~df.incoming_dir_exists.astype(bool)).sum())
    else:
        # CSV from older run without the dir-existence column; can't tell empty-dir vs no-dir
        n_no_dir = int((df.n_incoming_frames == 0).sum())
    n_with_dir_no_data = int(((df.n_incoming_frames == 0) &
                              (df.get('incoming_dir_exists', True) == True)).sum()) \
        if 'incoming_dir_exists' in df.columns else 0
    n_with_data = int((df.n_incoming_frames > 0).sum())
    n_usable = int((df.n_incoming_frames >= args.min_frames).sum())
    logging.info(f'{n_total} nights total:')
    logging.info(f'  {n_no_dir} closed (no incoming/<date>/ — monsoon, weather, etc.)')
    if n_with_dir_no_data:
        logging.info(f'  {n_with_dir_no_data} with directory but 0 *.fit files (aborted/error nights)')
    logging.info(f'  {n_with_data} with any frames')
    logging.info(f'  {n_usable} with >= {args.min_frames} frames (usable for selection)')

    usable = df[df.n_incoming_frames >= args.min_frames].copy()

    # Stratified selection within the illumination range
    in_range = usable[(usable.moon_illumination >= args.illum_min) &
                      (usable.moon_illumination <= args.illum_max)].copy()
    in_range['illum_bin'] = pd.cut(in_range.moon_illumination, bins=args.n_bins)

    logging.info(f'\nIn-range stratification ({args.illum_min} <= illum <= {args.illum_max}):')
    in_range_picks = []
    for bin_label, group in in_range.groupby('illum_bin', observed=True):
        if len(group) == 0:
            logging.warning(f'  bin {bin_label}: 0 candidates — none picked')
            continue
        picked = select_within_bin(group, args.per_bin)
        in_range_picks.append(picked)
        logging.info(f'  bin {bin_label}: {len(group)} candidates, picked {len(picked)}')
    in_range_selected = (pd.concat(in_range_picks) if in_range_picks
                         else pd.DataFrame(columns=usable.columns))

    # High-illum context
    high_illum = usable[usable.moon_illumination >= args.high_illum_min].copy()
    if len(high_illum) > 0:
        high_picked = select_within_bin(high_illum, args.high_illum_n)
        logging.info(f'\nHigh-illum (illum >= {args.high_illum_min}): '
                     f'{len(high_illum)} candidates, picked {len(high_picked)}')
    else:
        high_picked = pd.DataFrame(columns=usable.columns)
        logging.info(f'\nHigh-illum (illum >= {args.high_illum_min}): no candidates')

    selected = (
        pd.concat([in_range_selected, high_picked])
          .drop_duplicates('date')
          .sort_values('date')
    )
    logging.info(f'\nTotal selected: {len(selected)}')

    selected['date'].to_csv(args.output, index=False, header=False)
    logging.info(f'Wrote {len(selected)} dates to {args.output}')

    if args.plot_output:
        make_plot(df, selected, args.plot_output)

    print('\nSelected nights:')
    cols = ['date', 'moon_illumination', 'moon_altitude_deg_at_midnight',
            'moon_peak_altitude_deg_during_night', 'n_incoming_frames']
    print(selected[cols].to_string(index=False))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Merge a duplicate target across the entire Tierras filesystem.

Handles:
  - /data/tierras/incoming       : rename .fit files + update OBJECT header
  - /data/tierras/flattened      : rename dirs/files + update OBJECT header
  - /data/tierras/photometry     : rename dirs/files + update log content
  - /data/tierras/lightcurves    : rename dirs/files
  - /data/tierras/fields         : merge directories, rename files, update CSVs

Usage:
  python merge_targets.py -old_name LSPMJ1048+0111 -new_name 2MASSJ1048+0111
  python merge_targets.py -old_name LSPMJ1048+0111 -new_name 2MASSJ1048+0111 --dry-run
"""

import argparse
import logging
import os
import shutil
from glob import glob

import pandas as pd
from astropy.io import fits

TIERRAS_BASE = '/data/tierras'

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_rename(src: str, dst: str, dry: bool) -> None:
    if src == dst:
        return
    if os.path.exists(dst):
        log.warning('  SKIP rename (destination exists): %s -> %s', src, dst)
        return
    log.info('  rename: %s  ->  %s', src, dst)
    if not dry:
        try:
            os.rename(src, dst)
        except:
            log.info(f'COULD NOT RENAME {src} to {dst}')


def update_fits_object(path: str, old: str, new: str, dry: bool) -> None:
    try:
        # In dry-run mode use readonly so we don't need write permission
        mode = 'readonly' if dry else 'update'
        with fits.open(path, mode=mode) as hdul:
            hdr = hdul[0].header
            if hdr.get('OBJECT', '') == old:
                log.info('    FITS OBJECT: %s -> %s  in  %s', old, new, path)
                if not dry:
                    hdr['OBJECT'] = new
    except Exception as exc:
        log.warning('    Could not update FITS header in %s: %s', path, exc)


def rename_in_string(s: str, old: str, new: str) -> str:
    return s.replace(old, new)


def rename_file_in_dir(dirpath: str, old: str, new: str, dry: bool,
                       update_fits: bool = False) -> None:
    """Rename every file inside dirpath whose name contains old."""
    for fname in sorted(os.listdir(dirpath)):
        if old not in fname:
            continue
        fpath = os.path.join(dirpath, fname)
        if not os.path.isfile(fpath):
            continue
        new_fname = fname.replace(old, new)
        new_fpath = os.path.join(dirpath, new_fname)
        if update_fits and fname.endswith('.fit'):
            update_fits_object(fpath, old, new, dry)
        safe_rename(fpath, new_fpath, dry)


def _read_tierras_csv(path: str):
    """Read a Tierras CSV that may start with a '#' comment line.

    Returns (dataframe, comment_line_or_None).
    """
    comment = None
    with open(path) as fh:
        first = fh.readline()
        if first.startswith('#'):
            comment = first.rstrip('\n')
    df = pd.read_csv(path, comment='#')
    return df, comment


# ---------------------------------------------------------------------------
# incoming
# ---------------------------------------------------------------------------

def process_incoming(old: str, new: str, dry: bool) -> None:
    base = os.path.join(TIERRAS_BASE, 'incoming')
    if not os.path.isdir(base):
        log.warning('incoming directory not found: %s', base)
        return

    log.info('=== incoming ===')
    for date_dir in sorted(os.listdir(base)):
        date_path = os.path.join(base, date_dir)
        if not os.path.isdir(date_path):
            continue
        fits_files = glob(os.path.join(date_path, f'*.{old}.fit'))
        if not fits_files:
            continue
        log.info('  date: %s  (%d files)', date_dir, len(fits_files))
        for fpath in sorted(fits_files):
            update_fits_object(fpath, old, new, dry)
            new_fpath = fpath.replace(old, new)
            safe_rename(fpath, new_fpath, dry)


# ---------------------------------------------------------------------------
# Date-based directories (flattened, photometry, lightcurves)
# ---------------------------------------------------------------------------

def merge_flat0000_dirs(old_flat: str, new_flat: str,
                        old: str, new: str, dry: bool,
                        update_fits: bool = False) -> None:
    """Move (and rename) all files from old_flat into new_flat."""
    if not os.path.isdir(new_flat) and not dry:
        os.makedirs(new_flat, exist_ok=True)

    for fname in sorted(os.listdir(old_flat)):
        fpath = os.path.join(old_flat, fname)
        if not os.path.isfile(fpath):
            continue
        new_fname = fname.replace(old, new)
        new_fpath = os.path.join(new_flat, new_fname)
        if update_fits and fname.endswith('.fit'):
            update_fits_object(fpath, old, new, dry)
        if os.path.exists(new_fpath):
            log.warning('    SKIP move (dest exists): %s', new_fpath)
            continue
        log.info('    move+rename: %s -> %s', fpath, new_fpath)
        if not dry:
            shutil.move(fpath, new_fpath)

    if not dry and os.path.isdir(old_flat) and not os.listdir(old_flat):
        os.rmdir(old_flat)


def process_date_based(dirname: str, old: str, new: str, dry: bool,
                       update_fits: bool = False) -> list:
    """Returns list of conflict dates (both names existed under this directory)."""
    base = os.path.join(TIERRAS_BASE, dirname)
    if not os.path.isdir(base):
        log.warning('%s directory not found: %s', dirname, base)
        return []

    conflict_dates = []
    log.info('=== %s ===', dirname)
    for date_dir in sorted(os.listdir(base)):
        date_path = os.path.join(base, date_dir)
        if not os.path.isdir(date_path):
            continue

        old_targ = os.path.join(date_path, old)
        new_targ = os.path.join(date_path, new)
        if not os.path.isdir(old_targ):
            continue

        log.info('  date: %s', date_dir)

        if os.path.isdir(new_targ):
            conflict_dates.append(date_dir)
            # Both names exist on this date — need to merge contents
            log.info('    CONFLICT: both names exist, merging %s into %s', old, new)

            # Rename and merge files directly in old_targ (above flat0000)
            for fname in sorted(os.listdir(old_targ)):
                fpath = os.path.join(old_targ, fname)
                if os.path.isfile(fpath) and old in fname:
                    new_fname = fname.replace(old, new)
                    new_fpath = os.path.join(new_targ, new_fname)
                    if update_fits and fname.endswith('.fit'):
                        update_fits_object(fpath, old, new, dry)
                    if os.path.exists(new_fpath):
                        log.warning('    SKIP move (dest exists): %s', new_fpath)
                    else:
                        log.info('    move+rename: %s -> %s', fpath, new_fpath)
                        if not dry:
                            shutil.move(fpath, new_fpath)
                elif os.path.isdir(fpath):
                    new_subdir = os.path.join(new_targ, fname)
                    if os.path.isdir(new_subdir):
                        merge_flat0000_dirs(fpath, new_subdir, old, new, dry,
                                           update_fits=update_fits)
                    else:
                        # Subdir (e.g. flat0000) only in old — just rename contents and move
                        if not dry:
                            os.makedirs(new_subdir, exist_ok=True)
                        rename_file_in_dir(fpath, old, new, dry,
                                          update_fits=update_fits)
                        for f in sorted(os.listdir(fpath)) if not dry else []:
                            src = os.path.join(fpath, f)
                            dst = os.path.join(new_subdir, f)
                            shutil.move(src, dst)
                        if not dry and not os.listdir(fpath):
                            os.rmdir(fpath)

            if not dry and os.path.isdir(old_targ) and not os.listdir(old_targ):
                os.rmdir(old_targ)
                log.info('    removed empty dir: %s', old_targ)

        else:
            # Only old name exists — rename sub-files then the directory itself
            for subname in sorted(os.listdir(old_targ)):
                subpath = os.path.join(old_targ, subname)
                if os.path.isdir(subpath):
                    rename_file_in_dir(subpath, old, new, dry,
                                      update_fits=update_fits)
                elif os.path.isfile(subpath) and old in subname:
                    if update_fits and subname.endswith('.fit'):
                        update_fits_object(subpath, old, new, dry)
                    new_subpath = subpath.replace(old, new)
                    safe_rename(subpath, new_subpath, dry)

            safe_rename(old_targ, new_targ, dry)

    return conflict_dates


# ---------------------------------------------------------------------------
# fields (target-based, not date-based)
# ---------------------------------------------------------------------------

def update_ancillary_csv(path: str, old: str, new: str, dry: bool) -> None:
    """Update Filename column in global_ancillary_data.csv."""
    try:
        df, comment = _read_tierras_csv(path)
        if 'Filename' not in df.columns:
            return
        mask = df['Filename'].str.contains(old, na=False)
        if mask.any():
            log.info('    update Filename column in %s (%d rows)', path, mask.sum())
            if not dry:
                df.loc[mask, 'Filename'] = df.loc[mask, 'Filename'].str.replace(
                    old, new, regex=False)
                with open(path, 'w') as fh:
                    if comment:
                        fh.write(comment + '\n')
                    df.to_csv(fh, index=False)
    except Exception as exc:
        log.warning('    Could not update CSV %s: %s', path, exc)


def merge_ancillary_csvs(old_csv: str, new_csv: str, old: str, new: str,
                         dry: bool) -> None:
    """Combine two global_ancillary_data.csv files into one (at new_csv)."""
    try:
        df_old, _ = _read_tierras_csv(old_csv)
        if 'Filename' in df_old.columns:
            df_old['Filename'] = df_old['Filename'].str.replace(
                old, new, regex=False)
        df_new, comment = _read_tierras_csv(new_csv)
        combined = pd.concat([df_new, df_old], ignore_index=True)
        if 'BJD TDB' in combined.columns:
            combined.sort_values('BJD TDB', inplace=True)
        combined.drop_duplicates(inplace=True)
        log.info('    merge ancillary: %d + %d = %d rows -> %s',
                 len(df_new), len(df_old), len(combined), new_csv)
        if not dry:
            with open(new_csv, 'w') as fh:
                if comment:
                    fh.write(comment + '\n')
                combined.to_csv(fh, index=False)
    except Exception as exc:
        log.warning('    Could not merge ancillary CSVs: %s', exc)


def merge_csv_files(old_path: str, new_path: str, dry: bool) -> None:
    """Concatenate two Tierras CSV files, sort by BJD TDB if present, drop duplicates."""
    try:
        df_old, comment_old = _read_tierras_csv(old_path)
        df_new, comment_new = _read_tierras_csv(new_path)
        combined = pd.concat([df_new, df_old], ignore_index=True)
        if 'BJD TDB' in combined.columns:
            combined.sort_values('BJD TDB', inplace=True)
        combined.drop_duplicates(inplace=True)
        comment = comment_new or comment_old
        log.info('  merge CSV: %d + %d = %d rows -> %s',
                 len(df_new), len(df_old), len(combined), new_path)
        if not dry:
            with open(new_path, 'w') as fh:
                if comment:
                    fh.write(comment + '\n')
                combined.to_csv(fh, index=False)
            os.remove(old_path)
    except Exception as exc:
        log.warning('  Could not merge CSV %s into %s: %s', old_path, new_path, exc)


def merge_fields_subdirs(old_dir: str, new_dir: str, old: str, new: str,
                         dry: bool) -> None:
    """Recursively merge old_dir into new_dir, renaming files with old in name."""
    for item in sorted(os.listdir(old_dir)):
        old_path = os.path.join(old_dir, item)
        new_item = item.replace(old, new)
        new_path = os.path.join(new_dir, new_item)

        if os.path.isdir(old_path):
            if not os.path.isdir(new_path):
                log.info('  mkdir: %s', new_path)
                if not dry:
                    os.makedirs(new_path, exist_ok=True)
            merge_fields_subdirs(old_path, new_path, old, new, dry)
            if not dry and os.path.isdir(old_path) and not os.listdir(old_path):
                os.rmdir(old_path)
        elif os.path.isfile(old_path):
            if os.path.exists(new_path):
                # For CSV files that exist in both dirs, merge them
                if old_path.endswith('.csv'):
                    merge_csv_files(old_path, new_path, dry)
                else:
                    log.warning('  SKIP (dest exists): %s', new_path)
            else:
                log.info('  move+rename: %s -> %s', old_path, new_path)
                if not dry:
                    shutil.move(old_path, new_path)


def process_fields(old: str, new: str, dry: bool) -> None:
    base = os.path.join(TIERRAS_BASE, 'fields')
    if not os.path.isdir(base):
        log.warning('fields directory not found: %s', base)
        return

    log.info('=== fields ===')
    old_dir = os.path.join(base, old)
    new_dir = os.path.join(base, new)

    if not os.path.isdir(old_dir):
        log.info('  no fields directory for %s, skipping', old)
        return

    if os.path.isdir(new_dir):
        log.info('  both field dirs exist — merging %s into %s', old, new)

        # global_ancillary_data.csv: merge
        old_anc = os.path.join(old_dir, 'global_ancillary_data.csv')
        new_anc = os.path.join(new_dir, 'global_ancillary_data.csv')
        if os.path.isfile(old_anc) and os.path.isfile(new_anc):
            merge_ancillary_csvs(old_anc, new_anc, old, new, dry)
            # Remove the now-merged old copy so it isn't picked up again below
            if not dry and os.path.isfile(old_anc):
                os.remove(old_anc)
        elif os.path.isfile(old_anc):
            update_ancillary_csv(old_anc, old, new, dry)
            log.info('  move ancillary: %s -> %s', old_anc, new_anc)
            if not dry:
                shutil.move(old_anc, new_anc)

        # sources/ subdir — merge recursively
        old_sources = os.path.join(old_dir, 'sources')
        new_sources = os.path.join(new_dir, 'sources')
        if os.path.isdir(old_sources):
            if not os.path.isdir(new_sources) and not dry:
                os.makedirs(new_sources, exist_ok=True)
            merge_fields_subdirs(old_sources, new_sources, old, new, dry)
            if not dry and os.path.isdir(old_sources) and not os.listdir(old_sources):
                os.rmdir(old_sources)

        # Any other files at the top of old_dir (skip already-handled ones)
        handled = {'global_ancillary_data.csv', 'sources'}
        if os.path.isdir(old_dir):
            for item in sorted(os.listdir(old_dir)):
                if item in handled:
                    continue
                old_path = os.path.join(old_dir, item)
                new_item = item.replace(old, new)
                new_path = os.path.join(new_dir, new_item)
                if os.path.isfile(old_path):
                    if os.path.exists(new_path):
                        log.warning('  SKIP (dest exists): %s', new_path)
                    else:
                        log.info('  move+rename: %s -> %s', old_path, new_path)
                        if not dry:
                            shutil.move(old_path, new_path)

        if not dry and os.path.isdir(old_dir) and not os.listdir(old_dir):
            os.rmdir(old_dir)
            log.info('  removed empty dir: %s', old_dir)
        elif not dry and os.path.isdir(old_dir):
            log.warning('  old fields dir not empty after merge: %s', old_dir)

    else:
        log.info('  only %s exists — renaming to %s', old, new)

        # Update global_ancillary_data.csv in place
        anc = os.path.join(old_dir, 'global_ancillary_data.csv')
        if os.path.isfile(anc):
            update_ancillary_csv(anc, old, new, dry)

        # Rename files recursively before renaming the top-level dir
        for root, dirs, files in os.walk(old_dir):
            for fname in sorted(files):
                if old in fname:
                    fpath = os.path.join(root, fname)
                    new_fpath = os.path.join(root, fname.replace(old, new))
                    safe_rename(fpath, new_fpath, dry)

        safe_rename(old_dir, new_dir, dry)


# ---------------------------------------------------------------------------
# ap_phot re-run
# ---------------------------------------------------------------------------

def rerun_phot(dates: list, target: str, ffname: str, dry: bool) -> None:
    from ap_phot import main as ap_phot_main
    import numpy as np

    ap_radii = list(map(str, range(5, 21)))  # ap_phot auto-overrides for THWOMP targets

    log.info('=== re-running ap_phot on %d conflict date(s) ===', len(dates))
    for date in sorted(dates):
        log.info('  ap_phot: date=%s  target=%s', date, target)
        if not dry:
            ap_phot_main([
                '-date', date,
                '-target', target,
                '-ffname', ffname,
                '-ap_radii', *ap_radii,
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(raw_args=None):
    ap = argparse.ArgumentParser(
        description='Merge a Tierras target from old_name to new_name '
                    'across all data directories.')
    ap.add_argument('-old_name', required=True,
                    help='Current (wrong) target name to replace')
    ap.add_argument('-new_name', required=True,
                    help='Correct target name to use')
    ap.add_argument('-ffname', required=False, default='flat0000',
                    help='Flat-field folder name (default: flat0000)')
    ap.add_argument('--dry-run', action='store_true',
                    help='Print what would happen without making any changes')
    ap.add_argument('--rerun-phot', action='store_true',
                    help='After merging, re-run ap_phot on all conflict dates')
    args = ap.parse_args(raw_args)

    old = args.old_name
    new = args.new_name

    if old == new:
        log.error('old_name and new_name are identical — nothing to do.')
        return

    if args.dry_run:
        log.info('*** DRY RUN — no files will be changed ***')

    log.info('Merging: %s  ->  %s', old, new)

    process_incoming(old, new, args.dry_run)
    # Conflict dates are tracked from flattened (what ap_phot reads)
    conflict_dates = process_date_based('flattened', old, new, args.dry_run, update_fits=True)
    process_date_based('photometry', old, new, args.dry_run, update_fits=False)
    process_date_based('lightcurves', old, new, args.dry_run, update_fits=False)
    process_fields(old, new, args.dry_run)

    if conflict_dates:
        log.info('%d date(s) had observations under both names and need photometry re-run:',
                 len(conflict_dates))
        for d in sorted(conflict_dates):
            log.info('  %s', d)

    if args.rerun_phot:
        if not conflict_dates:
            log.info('No conflict dates — nothing to re-run.')
        else:
            rerun_phot(conflict_dates, new, args.ffname, args.dry_run)

    log.info('Done.')


if __name__ == '__main__':
    main()

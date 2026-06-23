#!/usr/bin/env python
"""
wcs_solver.py  —  local Gaia DR3 WCS solver for Tierras images.

Recovers a correct WCS for images where the imastrom pipeline failed,
by querying the on-disk Gaia DR3 catalog and fitting a new solution.

Usage (standalone):
    python wcs_solver.py <flat_red.fit> [<flat_red.fit> ...]

The script detects sources with SEP, cross-matches them against the
local Gaia DR3 catalog, fits a new CD matrix + CRVAL (keeping CRPIX
and PV2 distortion terms fixed), and updates the FITS header in-place.
"""

import argparse
import math
import os
import re
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy_healpix import HEALPix
import sep

GAIA_DIR = '/data/tierras/gaia_dr3/gaia_source'
GAIA_EPOCH = 2016.0          # catalog reference epoch
HEALPIX_LEVEL = 8            # nside = 256, as used in filenames
PLATE_SCALE = 0.432          # arcsec/pix (nominal)
DEFAULT_MAG_LIMIT = 17.5     # Gaia G mag; bright enough to detect reliably
DEFAULT_SEARCH_RAD = 0.40    # degrees (~24 arcmin) — covers FOV + buffer
DEFAULT_MATCH_TOL  = 10.0    # arcsec; generous to cope with ~5′ CRVAL offset
MIN_MATCHES        = 10      # minimum matched pairs to accept a solution


# ---------------------------------------------------------------------------
# Gaia catalog helpers
# ---------------------------------------------------------------------------

def _build_file_index():
    """Return list of (hp8_lo, hp8_hi, filename) for every Gaia file."""
    pat = re.compile(r'GaiaSource_(\d+)-(\d+)_sub\.fits')
    index = []
    for fname in os.listdir(GAIA_DIR):
        m = pat.match(fname)
        if m:
            index.append((int(m.group(1)), int(m.group(2)),
                          os.path.join(GAIA_DIR, fname)))
    return index


_GAIA_INDEX = None   # cached after first call


def _gaia_index():
    global _GAIA_INDEX
    if _GAIA_INDEX is None:
        _GAIA_INDEX = _build_file_index()
    return _GAIA_INDEX


def query_gaia_local(ra_deg, dec_deg, radius_deg,
                     mag_limit=DEFAULT_MAG_LIMIT,
                     obs_epoch=2026.0):
    """
    Return Gaia DR3 stars within *radius_deg* of (ra_deg, dec_deg).

    Proper motions are applied to bring positions from GAIA_EPOCH to
    *obs_epoch*.  Returns a dict with numpy arrays:
        ra, dec  — degrees, at obs_epoch
        gmag     — Gaia G magnitude
    """
    hp = HEALPix(nside=2**HEALPIX_LEVEL, order='nested', frame='icrs')
    center = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame='icrs')
    target_pixels = set(hp.cone_search_skycoord(center,
                                                radius=radius_deg * u.deg))

    ra_all, dec_all, gmag_all = [], [], []

    for hp8_lo, hp8_hi, fpath in _gaia_index():
        # quick range check — skip files with no overlap
        if hp8_hi < min(target_pixels) or hp8_lo > max(target_pixels):
            continue
        if not os.path.exists(fpath):
            continue

        with fits.open(fpath, memmap=True) as hdul:
            d = hdul[1].data

        # magnitude filter first (cheap)
        gmag = d['phot_g_mean_mag']
        keep = gmag <= mag_limit
        if not np.any(keep):
            continue

        ra   = np.radians(d['ra'][keep])
        dec  = np.radians(d['dec'][keep])
        gmag = gmag[keep]
        pmra = d['pmra'][keep]   # mas/yr in RA*cos(dec)
        pmdec= d['pmdec'][keep]  # mas/yr in Dec

        # apply proper motion (linear approximation)
        dep  = obs_epoch - GAIA_EPOCH
        pmra_safe  = np.where(np.isfinite(pmra),  pmra,  0.0)
        pmdec_safe = np.where(np.isfinite(pmdec), pmdec, 0.0)
        mas2rad = math.pi / (180.0 * 3600.0 * 1000.0)
        ra  = ra  + dep * pmra_safe  * mas2rad / np.cos(dec)
        dec = dec + dep * pmdec_safe * mas2rad

        # cone filter in angular distance
        cos_sep = (np.sin(dec) * math.sin(math.radians(dec_deg)) +
                   np.cos(dec) * math.cos(math.radians(dec_deg)) *
                   np.cos(ra - math.radians(ra_deg)))
        cos_limit = math.cos(math.radians(radius_deg))
        inside = cos_sep >= cos_limit
        if not np.any(inside):
            continue

        ra_all.append(np.degrees(ra[inside]))
        dec_all.append(np.degrees(dec[inside]))
        gmag_all.append(gmag[inside])

    if not ra_all:
        return None

    return dict(ra=np.concatenate(ra_all),
                dec=np.concatenate(dec_all),
                gmag=np.concatenate(gmag_all))


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def detect_sources(img, detection_sigma=5.0, min_area=5):
    """Run SEP on *img*, return (x, y, flux) sorted brightest-first."""
    data = img.astype(np.float64)
    mask = ~np.isfinite(data)
    data[mask] = 0.0
    bkg  = sep.Background(data, mask=mask)
    data -= bkg
    sources = sep.extract(data, detection_sigma, err=bkg.globalrms,
                          minarea=min_area, mask=mask)
    order = np.argsort(sources['flux'])[::-1]
    return sources['x'][order], sources['y'][order], sources['flux'][order]


# ---------------------------------------------------------------------------
# WCS fitting
# ---------------------------------------------------------------------------

def _header_ra_dec(hdr):
    """
    Parse the RA / DEC header keywords (sexagesimal or decimal degrees).
    Returns (ra_deg, dec_deg) or raises KeyError.
    """
    ra_str  = hdr['RA']
    dec_str = hdr['DEC']

    def _sexa(s):
        s = s.strip()
        sign = -1 if s.startswith('-') else 1
        s = s.lstrip('+-')
        parts = s.split(':')
        val = float(parts[0])
        if len(parts) > 1:
            val += float(parts[1]) / 60.0
        if len(parts) > 2:
            val += float(parts[2]) / 3600.0
        return sign * val

    ra_h  = _sexa(ra_str)    # hours
    dec_d = _sexa(dec_str)   # degrees
    return ra_h * 15.0, dec_d


def _obs_epoch_from_header(hdr):
    mjd = float(hdr.get('MJD-OBS', 51544.5))
    return 2000.0 + (mjd - 51544.5) / 365.25


def fit_wcs(hdr, src_x, src_y,
            ra_center, dec_center,
            gaia,
            match_tol_arcsec=DEFAULT_MATCH_TOL,
            min_matches=MIN_MATCHES):
    """
    Fit a new CRVAL + CD matrix given detected source pixel positions and
    a Gaia catalog.

    Strategy
    --------
    1. Use the header WCS (CRVAL + CRPIX + CD) to project Gaia stars to pixels.
       When CRVAL is wrong, this introduces a constant pixel shift — it does NOT
       change the relative layout of Gaia stars on the detector.
    2. Find that shift by voting: for every (source, Gaia) candidate pair compute
       the implied (dx, dy) and bin into a 2D histogram. The peak gives the
       global offset between the two sets, even when it is hundreds of pixels.
    3. Apply the shift, re-match with a tight tolerance, fit CRVAL + CD via
       linear least squares in gnomonic coordinates.
    """
    crpix1 = float(hdr.get('CRPIX1', 2051.0))
    crpix2 = float(hdr.get('CRPIX2', 1032.9))

    # Use the header WCS as-is (ZPN with distortion terms) for the initial
    # Gaia projection. A wrong CRVAL in a ZPN WCS still produces an
    # approximately constant pixel shift near field centre, which the vote
    # recovers. Using TAN here instead would ignore the ZPN distortion
    # (up to 22" at the detector corners), causing edge sources to miss
    # the match tolerance entirely.
    if 'CRVAL1' in hdr and 'CD1_1' in hdr:
        w = WCS(hdr)
    else:
        w = WCS(naxis=2)
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.crval = [ra_center, dec_center]
        w.wcs.crpix = [crpix1, crpix2]
        ps = PLATE_SCALE / 3600.0
        w.wcs.cd = np.array([[0.0, ps], [-ps, 0.0]])
        w.wcs.set()

    # Project Gaia to pixel coords; keep a generous margin around the detector
    ny, nx = int(hdr.get('NAXIS2', 2048)), int(hdr.get('NAXIS1', 4096))
    vote_radius_pix = 1200  # ~8.6 arcmin — covers worst observed CRVAL errors
    gx_all, gy_all = w.all_world2pix(gaia['ra'], gaia['dec'], 1)
    on_det = ((gx_all > -vote_radius_pix) & (gx_all < nx + vote_radius_pix) &
              (gy_all > -vote_radius_pix) & (gy_all < ny + vote_radius_pix))
    on_det_idx = np.where(on_det)[0]   # maps sub-indices → full gaia indices
    gx = gx_all[on_det]
    gy = gy_all[on_det]
    gaia_sub = {k: v[on_det] for k, v in gaia.items()}

    if len(gx) == 0:
        return None, 0, None, [], []

    # ---- Vote for the global pixel shift ----------------------------------
    # Limit to brightest sources / Gaia stars to keep the N^2 cost manageable
    n_vote_src  = min(len(src_x), 200)
    n_vote_gaia = min(len(gx), 500)
    vsx = src_x[:n_vote_src]
    vsy = src_y[:n_vote_src]
    vgx = gx[:n_vote_gaia]
    vgy = gy[:n_vote_gaia]

    dxs = (vsx[:, None] - vgx[None, :]).ravel()
    dys = (vsy[:, None] - vgy[None, :]).ravel()

    bin_size = 5.0
    bins = np.arange(-vote_radius_pix, vote_radius_pix + bin_size, bin_size)
    hist, _, _ = np.histogram2d(dxs, dys, bins=[bins, bins])
    peak = np.unravel_index(np.argmax(hist), hist.shape)
    shift_x = (bins[peak[0]] + bins[peak[0] + 1]) / 2.0
    shift_y = (bins[peak[1]] + bins[peak[1] + 1]) / 2.0

    # ---- Tight match with the found shift ---------------------------------
    tol_pix = match_tol_arcsec / PLATE_SCALE
    gx_s = gx + shift_x
    gy_s = gy + shift_y

    matched_src_idx  = []
    matched_gaia_idx = []
    used_gaia = set()

    for i in range(len(src_x)):
        dx = gx_s - src_x[i]
        dy = gy_s - src_y[i]
        d2 = dx*dx + dy*dy
        j  = int(np.argmin(d2))
        if d2[j] < tol_pix**2 and j not in used_gaia:
            matched_src_idx.append(i)
            matched_gaia_idx.append(j)
            used_gaia.add(j)

    n = len(matched_src_idx)
    if n < min_matches:
        return None, n, None, [], []

    # ---- Fit CRVAL + CD in ZPN intermediate coordinates -------------------
    # Simply fitting in gnomonic (TAN) coords and then applying a ZPN WCS
    # double-counts the polynomial distortion (systematic residuals up to 22"
    # at detector corners).  Instead we work directly in ZPN intermediate
    # coordinates.
    #
    # A temporary ZPN WCS with CD=identity and CRPIX=(0,0) lets wcslib compute
    # the ZPN intermediate coords for each Gaia position.  Those coords are the
    # exact targets for the CD matrix fit, so no distortion is double-counted.
    #
    # We iterate twice: first a CRVAL-only step (the CD may be badly wrong for
    # a failed imastrom solve), then a full CRVAL+CD re-fit once matching is
    # clean enough to constrain the CD accurately.

    gaia_ra_m  = gaia_sub['ra'][matched_gaia_idx]
    gaia_dec_m = gaia_sub['dec'][matched_gaia_idx]

    def _delta_crval(wcs_cur, keep_mask):
        """Sigma-clipped median sky offset between source pixels and Gaia."""
        pred_ra, pred_dec = wcs_cur.all_pix2world(
            src_x[matched_src_idx], src_y[matched_src_idx], 1)
        dra  = (gaia_ra_m  - pred_ra)  * np.cos(np.radians(gaia_dec_m))
        ddec = gaia_dec_m - pred_dec
        for _ in range(3):
            sig = np.std(np.hypot(dra[keep_mask], ddec[keep_mask]))
            new_mask = np.hypot(dra, ddec) < 3.0 * sig
            if new_mask.sum() < min_matches or np.array_equal(new_mask, keep_mask):
                break
            keep_mask = new_mask
        return np.median(dra[keep_mask]), np.median(ddec[keep_mask]), keep_mask

    def _fit_cd(crval_ra, crval_dec, keep_mask):
        """Fit CD matrix using ZPN intermediate coords from a temp WCS.

        We include a constant intercept column in the design matrix so that
        any small residual CRVAL offset is absorbed rather than rotated into
        the CD diagonal terms.  The intercept is applied back as a further
        CRVAL correction after the fit.
        """
        tmp = w.deepcopy()
        tmp.wcs.crval = [crval_ra, crval_dec]
        tmp.wcs.crpix = [0.0, 0.0]
        tmp.wcs.cd    = np.eye(2)
        tmp.wcs.set()
        xi_t, eta_t = tmp.all_world2pix(gaia_ra_m[keep_mask],
                                         gaia_dec_m[keep_mask], 1)
        px = src_x[matched_src_idx][keep_mask] - crpix1
        py = src_y[matched_src_idx][keep_mask] - crpix2
        # [dx, dy, 1]: intercept absorbs any residual CRVAL shift
        A  = np.column_stack([px, py, np.ones(keep_mask.sum())])
        p1, _, _, _ = np.linalg.lstsq(A, xi_t,  rcond=None)
        p2, _, _, _ = np.linalg.lstsq(A, eta_t, rcond=None)
        cd_fit = np.array([p1[:2], p2[:2]])
        # Return CD and the intercept offsets (in degrees) for CRVAL refinement
        return cd_fit, p1[2], p2[2]

    # Step 1: CRVAL-only correction using the header CD (which may be wrong
    # but is good enough to locate the field centre).
    keep = np.ones(n, dtype=bool)
    dra_med, ddec_med, keep = _delta_crval(w, keep)
    old_crval = w.wcs.crval
    crval_ra  = old_crval[0] + dra_med / np.cos(np.radians(old_crval[1]))
    crval_dec = old_crval[1] + ddec_med

    # Step 2: fit CD with the corrected CRVAL; intercept refines CRVAL further.
    cd_fit, xi_off, eta_off = _fit_cd(crval_ra, crval_dec, keep)
    crval_ra  += xi_off  / np.cos(np.radians(crval_dec))
    crval_dec += eta_off

    # Step 3: rebuild WCS, refine CRVAL once more.
    wout = w.deepcopy()
    wout.wcs.crval = [crval_ra, crval_dec]
    wout.wcs.cd    = cd_fit
    wout.wcs.set()
    dra_med2, ddec_med2, keep = _delta_crval(wout, keep)
    crval_ra  += dra_med2 / np.cos(np.radians(crval_dec))
    crval_dec += ddec_med2
    wout.wcs.crval = [crval_ra, crval_dec]
    wout.wcs.set()

    # ---- Second-pass re-match using the refined WCS -----------------------
    # The initial vote can miss edge sources when the header CRVAL is very
    # wrong.  Now that we have a good WCS, re-project all Gaia stars and
    # do a fresh tight match to recover those sources.
    gx2, gy2 = wout.all_world2pix(gaia_sub['ra'], gaia_sub['dec'], 1)
    # Use 2× the first-pass RMS as tolerance, clamped to [2", match_tol]
    pred_ra_p1, pred_dec_p1 = wout.all_pix2world(src_x[matched_src_idx],
                                                   src_y[matched_src_idx], 1)
    sep_p1 = SkyCoord(pred_ra_p1 * u.deg, pred_dec_p1 * u.deg).separation(
             SkyCoord(gaia_sub['ra'][matched_gaia_idx] * u.deg,
                      gaia_sub['dec'][matched_gaia_idx] * u.deg)).arcsec
    rms_p1 = float(np.sqrt(np.mean(sep_p1**2)))
    tol2_arcsec = float(np.clip(2.0 * rms_p1, 2.0, match_tol_arcsec))
    tol_pix2 = tol2_arcsec / PLATE_SCALE
    matched_src_idx2  = []
    matched_gaia_idx2 = []
    used_gaia2 = set()

    for i in range(len(src_x)):
        dx = gx2 - src_x[i]
        dy = gy2 - src_y[i]
        d2 = dx*dx + dy*dy
        j  = int(np.argmin(d2))
        if d2[j] < tol_pix2**2 and j not in used_gaia2:
            matched_src_idx2.append(i)
            matched_gaia_idx2.append(j)
            used_gaia2.add(j)

    if len(matched_src_idx2) >= n:
        matched_src_idx  = matched_src_idx2
        matched_gaia_idx = matched_gaia_idx2
        n = len(matched_src_idx2)
        # Update the Gaia arrays the closures reference
        gaia_ra_m  = gaia_sub['ra'][np.array(matched_gaia_idx)]
        gaia_dec_m = gaia_sub['dec'][np.array(matched_gaia_idx)]
        # Re-fit CD and CRVAL with the larger, cleaner match set
        keep2 = np.ones(n, dtype=bool)
        dra_m3, ddec_m3, keep2 = _delta_crval(wout, keep2)
        crval_ra  += dra_m3 / np.cos(np.radians(crval_dec))
        crval_dec += ddec_m3
        cd_fit2, xi_off2, eta_off2 = _fit_cd(crval_ra, crval_dec, keep2)
        crval_ra  += xi_off2  / np.cos(np.radians(crval_dec))
        crval_dec += eta_off2
        wout.wcs.crval = [crval_ra, crval_dec]
        wout.wcs.cd    = cd_fit2
        wout.wcs.set()
        dra_m4, ddec_m4, keep2 = _delta_crval(wout, keep2)
        crval_ra  += dra_m4 / np.cos(np.radians(crval_dec))
        crval_dec += ddec_m4
        wout.wcs.crval = [crval_ra, crval_dec]
        wout.wcs.set()

    # Residuals
    pred_ra, pred_dec = wout.all_pix2world(src_x[matched_src_idx],
                                           src_y[matched_src_idx], 1)
    sep_arcsec = SkyCoord(pred_ra * u.deg, pred_dec * u.deg).separation(
                 SkyCoord(gaia_sub['ra'][matched_gaia_idx] * u.deg,
                          gaia_sub['dec'][matched_gaia_idx] * u.deg)
                 ).arcsec
    rms = float(np.sqrt(np.mean(sep_arcsec**2)))

    # Map matched_gaia_idx (indices into gaia_sub) → indices into full gaia
    matched_gaia_idx_full = on_det_idx[matched_gaia_idx]

    return wout, n, rms, matched_src_idx, matched_gaia_idx_full


# ---------------------------------------------------------------------------
# Top-level solver
# ---------------------------------------------------------------------------

def solve_file(fitsfile,
               search_radius_deg=DEFAULT_SEARCH_RAD,
               mag_limit=DEFAULT_MAG_LIMIT,
               match_tol_arcsec=DEFAULT_MATCH_TOL,
               min_matches=MIN_MATCHES,
               write=False,
               verbose=True):
    """
    Solve (and optionally update) the WCS for *fitsfile*.

    Returns a dict with keys:
        wcs            — astropy.wcs.WCS (fitted solution), or None on failure
        nmatches       — int
        rms_arcsec     — float or None
        gaia           — dict(ra, dec, gmag) for all queried Gaia stars
        src_x, src_y   — detected source pixel coords (brightest-first)
        matched_src_idx, matched_gaia_idx — index arrays into src_*/gaia
    """
    def log(msg):
        if verbose:
            print(msg)

    with fits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        img = hdul[0].data.astype(np.float32)

    try:
        ra_center, dec_center = _header_ra_dec(hdr)
    except KeyError:
        log('ERROR: no RA/DEC keywords in header')
        return dict(wcs=None, nmatches=0, rms_arcsec=None,
                    gaia=None, src_x=None, src_y=None,
                    matched_src_idx=None, matched_gaia_idx=None)

    obs_epoch = _obs_epoch_from_header(hdr)
    log(f'Field centre: RA={ra_center:.4f}  Dec={dec_center:.4f}  epoch={obs_epoch:.2f}')

    log(f'Querying local Gaia DR3 within {search_radius_deg:.2f} deg ...')
    gaia = query_gaia_local(ra_center, dec_center, search_radius_deg,
                            mag_limit=mag_limit, obs_epoch=obs_epoch)
    if gaia is None or len(gaia['ra']) == 0:
        log('ERROR: no Gaia stars found')
        return dict(wcs=None, nmatches=0, rms_arcsec=None,
                    gaia=None, src_x=None, src_y=None,
                    matched_src_idx=None, matched_gaia_idx=None)
    log(f'  Found {len(gaia["ra"])} Gaia stars (G <= {mag_limit})')

    log('Detecting sources ...')
    src_x, src_y, src_flux = detect_sources(img)
    log(f'  Detected {len(src_x)} sources')

    log(f'Fitting WCS (tol={match_tol_arcsec:.1f}")...')
    result = fit_wcs(hdr, src_x, src_y,
                     ra_center, dec_center,
                     gaia,
                     match_tol_arcsec=match_tol_arcsec,
                     min_matches=min_matches)
    wcs_fit, nmatches, rms, matched_src_idx, matched_gaia_idx = result

    if wcs_fit is None:
        log(f'FAILED: only {nmatches} matches (need {min_matches})')
        return dict(wcs=None, nmatches=nmatches, rms_arcsec=None,
                    gaia=gaia, src_x=src_x, src_y=src_y,
                    matched_src_idx=None, matched_gaia_idx=None)

    old_crval1 = hdr.get('CRVAL1', 'N/A')
    old_crval2 = hdr.get('CRVAL2', 'N/A')
    log(f'Solution: {nmatches} matches, RMS={rms:.3f}"')
    log(f'  Old CRVAL: ({old_crval1}, {old_crval2})')
    log(f'  New CRVAL: ({wcs_fit.wcs.crval[0]:.6f}, {wcs_fit.wcs.crval[1]:.6f})')

    if write:
        _update_header(fitsfile, wcs_fit, nmatches, rms)
        log(f'Updated header in {fitsfile}')

    return dict(wcs=wcs_fit, nmatches=nmatches, rms_arcsec=rms,
                gaia=gaia, src_x=src_x, src_y=src_y,
                matched_src_idx=np.array(matched_src_idx),
                matched_gaia_idx=np.array(matched_gaia_idx))


def plot_solution(fitsfile, wcs, gaia, src_x, src_y,
                  matched_src_idx, matched_gaia_idx,
                  rms_arcsec, outfile=None):
    """
    Display the image with the fitted WCS, detected sources, and matched
    Gaia stars overlaid.

    Parameters
    ----------
    fitsfile : str
    wcs : astropy.wcs.WCS  — the fitted solution
    gaia : dict with 'ra', 'dec', 'gmag' arrays (full catalog, epoch-corrected)
    src_x, src_y : detected source pixel coords (brightest-first)
    matched_src_idx, matched_gaia_idx : indices into the above arrays
    rms_arcsec : float
    outfile : str or None  — if given, save to file instead of displaying
    """
    import matplotlib.pyplot as plt
    plt.ion()
    from matplotlib.patches import Circle
    from astropy.visualization import ZScaleInterval

    with fits.open(fitsfile) as hdul:
        hdr = hdul[0].header
        img = hdul[0].data.astype(np.float32)

    img[~np.isfinite(img)] = np.nan
    ny, nx = img.shape

    fig = plt.figure(figsize=(14, 7))

    # ---- Left panel: image with native WCS axes and overlays -------------
    ax = fig.add_subplot(1, 2, 1, projection=wcs)

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(img[np.isfinite(img)])
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
              interpolation='nearest', aspect='equal')

    # All detected sources — small grey crosses
    ax.plot(src_x, src_y, '+', color='grey', ms=4, lw=0.5,
            transform=ax.get_transform('pixel'),
            label=f'Detected ({len(src_x)})', zorder=2)

    # Matched Gaia stars — circles sized by magnitude
    gaia_ra_m  = gaia['ra'][matched_gaia_idx]
    gaia_dec_m = gaia['dec'][matched_gaia_idx]
    gaia_g_m   = gaia['gmag'][matched_gaia_idx]
    gx_m, gy_m = wcs.all_world2pix(gaia_ra_m, gaia_dec_m, 1)
    r_pix = 3.0 + (DEFAULT_MAG_LIMIT - gaia_g_m).clip(0) * 1.2
    for xc, yc, rr in zip(gx_m, gy_m, r_pix):
        circ = Circle((xc, yc), rr, linewidth=0.8, edgecolor='lime',
                      facecolor='none',
                      transform=ax.get_transform('pixel'), zorder=3)
        ax.add_patch(circ)
    ax.plot([], [], 'o', ms=8, mec='lime', mfc='none',
            label=f'Gaia matched ({len(gaia_ra_m)}, RMS={rms_arcsec:.2f}")')

    # Unmatched Gaia stars on detector — tiny cyan dots
    gx_all, gy_all = wcs.all_world2pix(gaia['ra'], gaia['dec'], 1)
    on_det = (gx_all > 0) & (gx_all < nx) & (gy_all > 0) & (gy_all < ny)
    unmatched_mask = np.ones(len(gaia['ra']), dtype=bool)
    unmatched_mask[matched_gaia_idx] = False
    show = on_det & unmatched_mask
    ax.plot(gx_all[show], gy_all[show], '.', color='cyan', ms=2, lw=0,
            transform=ax.get_transform('pixel'),
            label=f'Gaia unmatched ({show.sum()})', zorder=2)

    ax.coords[0].set_axislabel('RA')
    ax.coords[1].set_axislabel('Dec')
    ax.coords.grid(color='white', alpha=0.2, linestyle='--')
    ax.legend(loc='upper right', fontsize=7, framealpha=0.6)
    ax.set_title(os.path.basename(fitsfile), fontsize=9)

    # ---- Right panel: residuals scatter plot ----------------------------
    ax2 = fig.add_subplot(1, 2, 2)

    pred_ra, pred_dec = wcs.all_pix2world(src_x[matched_src_idx],
                                          src_y[matched_src_idx], 1)
    dra  = (pred_ra  - gaia_ra_m)  * np.cos(np.radians(gaia_dec_m)) * 3600.0
    ddec = (pred_dec - gaia_dec_m) * 3600.0

    sc2 = ax2.scatter(dra, ddec, c=gaia_g_m, cmap='viridis_r',
                      s=20, vmin=gaia_g_m.min(), vmax=DEFAULT_MAG_LIMIT,
                      zorder=3)
    plt.colorbar(sc2, ax=ax2, label='Gaia G mag')

    lim = max(abs(dra).max(), abs(ddec).max()) * 1.1
    ax2.axhline(0, color='k', lw=0.5)
    ax2.axvline(0, color='k', lw=0.5)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel('Delta RA (arcsec)')
    ax2.set_ylabel('Delta Dec (arcsec)')
    ax2.set_title(f'Residuals  RMS={rms_arcsec:.3f}"  N={len(dra)}', fontsize=9)
    ax2.set_aspect('equal')

    plt.tight_layout()

    breakpoint()

    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved plot to {outfile}')
    else:
        plt.show()

    plt.close(fig)


def _update_header(fitsfile, wcs, nmatches, rms_arcsec):
    """Write fitted WCS keywords back into the primary header of *fitsfile*."""
    with fits.open(fitsfile, mode='update') as hdul:
        hdr = hdul[0].header
        hdr['CRVAL1'] = wcs.wcs.crval[0]
        hdr['CRVAL2'] = wcs.wcs.crval[1]
        hdr['CD1_1']  = wcs.wcs.cd[0, 0]
        hdr['CD1_2']  = wcs.wcs.cd[0, 1]
        hdr['CD2_1']  = wcs.wcs.cd[1, 0]
        hdr['CD2_2']  = wcs.wcs.cd[1, 1]
        hdr['STDCRMS'] = rms_arcsec
        hdr['NUMBRMS'] = nmatches
        hdr.add_history('WCS refit by wcs_solver.py (local Gaia DR3)')
        hdul.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+', help='Reduced FITS files (_red.fit)')
    ap.add_argument('-r', '--radius', type=float, default=DEFAULT_SEARCH_RAD,
                    metavar='DEG', help='Gaia search radius in degrees (default %(default)s)')
    ap.add_argument('-m', '--maglim', type=float, default=DEFAULT_MAG_LIMIT,
                    metavar='MAG', help='Gaia G magnitude limit (default %(default)s)')
    ap.add_argument('-t', '--tol', type=float, default=DEFAULT_MATCH_TOL,
                    metavar='ARCSEC', help='Match tolerance in arcsec (default %(default)s)')
    ap.add_argument('-n', '--min-matches', type=int, default=MIN_MATCHES,
                    help='Minimum matches to accept solution (default %(default)s)')
    ap.add_argument('-w', '--write', action='store_true',
                    help='Write corrected WCS back to file')
    ap.add_argument('-p', '--plot', action='store_true',
                    help='Show diagnostic plot (image + Gaia overlay + residuals)')
    ap.add_argument('-o', '--outdir', type=str, default=None,
                    metavar='DIR',
                    help='Save plots to this directory instead of displaying interactively')
    args = ap.parse_args()

    for f in args.files:
        print(f'\n=== {f} ===')
        result = solve_file(f,
                            search_radius_deg=args.radius,
                            mag_limit=args.maglim,
                            match_tol_arcsec=args.tol,
                            min_matches=args.min_matches,
                            write=args.write)

        if args.plot and result['wcs'] is not None:
            outfile = None
            if args.outdir:
                stem = os.path.splitext(os.path.basename(f))[0]
                outfile = os.path.join(args.outdir, stem + '_wcs.png')
            plot_solution(f,
                          result['wcs'],
                          result['gaia'],
                          result['src_x'],
                          result['src_y'],
                          result['matched_src_idx'],
                          result['matched_gaia_idx'],
                          result['rms_arcsec'],
                          outfile=outfile)

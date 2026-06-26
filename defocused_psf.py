import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from astropy.time import Time
from astropy.visualization import simple_norm 

from photutils.background import Background2D
from photutils.psf import (
    EPSFBuilder,
    extract_stars,
)

from ap_phot import plot_image, source_selection

def build_epsf(image_sub, stars_tbl, r_outer,
               oversampling=2, max_stars=40, min_separation=None):
    """
    Build an empirical PSF from isolated stars.
    Adapted for defocused (large) PSFs.

    Parameters
    ----------
    image_sub      : 2D background-subtracted image
    stars_tbl      : table with 'x', 'y' columns
    r_outer        : outer PSF radius estimate [pixels]
    oversampling   : ePSF oversampling factor (1–2 is enough for large PSFs)
    max_stars      : cap number of stars used
    min_separation : minimum star separation [pixels]
    """
    if min_separation is None:
        min_separation = 3 * r_outer

    # --- Filter: remove stars too close to edges or each other ---
    ny, nx = image_sub.shape
    margin = int(r_outer * 2.5) + 5
    box_half = int(r_outer * 2) + 5
    box_size = 2 * box_half + 1

    mask_edge = (
        (stars_tbl["x"] > margin) & (stars_tbl["x"] < nx - margin) &
        (stars_tbl["y"] > margin) & (stars_tbl["y"] < ny - margin)
    )
    clean = stars_tbl[mask_edge]

    # Remove crowded stars (simple nearest-neighbour check)
    from scipy.spatial import cKDTree
    coords = np.column_stack([clean["x"], clean["y"]])
    tree   = cKDTree(coords)
    pairs  = tree.query_pairs(r=min_separation)
    bad    = set(j for pair in pairs for j in pair)
    good   = [i for i in range(len(clean)) if i not in bad]
    clean  = clean[good]

    # Sort by brightness, take brightest max_stars
    clean.sort("phot_rp_mean_mag")
    clean = clean[:max_stars]
    print(f"Using {len(clean)} stars to build ePSF (box={box_size}px, "
          f"oversampling={oversampling})")

    # --- Extract star cutouts ---
    stars_tbl_epsf = Table()
    stars_tbl_epsf["x"] = clean["x"]
    stars_tbl_epsf["y"] = clean["y"]

    nddata = NDData(data=image_sub)
    stars  = extract_stars(nddata, stars_tbl_epsf, size=box_size)
    print(f"Extracted {len(stars)} star cutouts")

    # --- Build ePSF ---
    epsf_builder = EPSFBuilder(
        oversampling     = oversampling,
        maxiters         = 10,
        progress_bar     = True,
        smoothing_kernel = "quadratic",  # or 'quartic', numpy array
        # recentering_maxiters=3,        # reduce if donut centroiding drifts
        # center_accuracy=0.5,
    )
    epsf, fitted_stars = epsf_builder(stars)
    return epsf, fitted_stars

def save_epsf_fits(epsf):
    """
    Save a photutils EPSFModel / FittableImageModel / ImagePSF to FITS.

    Stores the PSF array as the primary image and encodes all
    reconstruction parameters (oversampling, origin) in the header.

    Parameters
    ----------
    epsf      : EPSFModel  — the PSF to save
    filepath  : str/Path   — output file path (e.g. 'psf.fits')
    overwrite : bool — overwrite existing file
    """
    home_dir = str(Path.home())

    hdu = fits.PrimaryHDU(data=epsf.data.astype(np.float64))
    hdr = hdu.header

    # ---- Oversampling (scalar or (x, y) tuple) ----
    os_arr = np.atleast_1d(np.asarray(epsf.oversampling, dtype=int))
    hdr['OVERSMPX'] = (int(os_arr[0]),   'PSF oversampling factor along x')
    hdr['OVERSMPY'] = (int(os_arr[-1]),  'PSF oversampling factor along y')

    # ---- Origin: center of PSF in data-array pixel coordinates ----
    origin = np.atleast_1d(np.asarray(epsf.origin, dtype=float))
    hdr['ORIG_X']   = (float(origin[0]),   'PSF origin x [data array px]')
    hdr['ORIG_Y']   = (float(origin[-1]),  'PSF origin y [data array px]')

    # ---- Provenance ----
    hdr['DATE']    = (Time.now().isot,   'File creation date (UTC)')
    hdr['CREATOR'] = ('photutils',       'PSF builder software')

    fits.HDUList([hdu]).writeto(f'{home_dir}/tierras/tierras_red/psfs/defocused_psf.fits', overwrite=True)
    return 

def load_epsf_fits(filepath):
    """
    Load an EPSFModel from a FITS file saved by save_epsf_fits().

    Returns an EPSFModel instance ready for PSFPhotometry /
    IterativePSFPhotometry, regardless of photutils version.
    """
    # Version-aware import (photutils API changed across versions)
    try:
        from photutils.psf import EPSFModel             # photutils < 2.0
    except ImportError:
        try:
            from photutils.psf import FittableImageModel as EPSFModel
        except ImportError:
            from photutils.psf import ImagePSF as EPSFModel  # photutils >= 1.9

    with fits.open(filepath) as hdul:
        data = hdul[0].data.astype(np.float64)
        hdr  = hdul[0].header

        os_x   = int(hdr.get('OVERSMPX', 1))
        os_y   = int(hdr.get('OVERSMPY', os_x))
        orig_x = float(hdr.get('ORIG_X',  (data.shape[1] - 1) / 2.0))
        orig_y = float(hdr.get('ORIG_Y',  (data.shape[0] - 1) / 2.0))

    oversampling = os_x if (os_x == os_y) else (os_x, os_y)
    origin       = (orig_x, orig_y)

    epsf = EPSFModel(data=data, oversampling=oversampling, origin=origin)
    print(f"Loaded ePSF  shape={data.shape}  "
          f"oversampling={oversampling}  ← {filepath}")
    return epsf

def gaia_rp_to_counts(g_rp, exptime, coeff=9.7e8, gain=5.9):
    """
    Convert Gaia RP magnitudes to expected total counts in an exposure.

    counts = t_exp × 10^( (ZP - G_RP) / 2.5 )

    Parameters
    ----------
    g_rp    : array-like — Gaia RP magnitudes
    exptime : float      — exposure time [s]
    coeff   : float      — the calibrating flux value for G_RP = 0 from instrument patper [e- / s]
    gain    : floag      — the gain of the image (e- / count)
    Returns
    -------
    counts : ndarray — expected total counts per star
    """
    return exptime * coeff * 10.0**((- np.asarray(g_rp, dtype=float)) / 2.5) / gain

def render_psf_model(
    epsf,
    x_positions,
    y_positions,
    fluxes,
    image_shape = (2048, 4096),
    psf_half_size = 150,   # [px] half-width of rendering box; auto-detected if None
):
    """
    Render each star as a scaled ePSF stamp and accumulate into a model image.

    For defocused images the PSF is large; psf_half_size must exceed r_outer.
    Edge stars are rendered partially and flagged.

    Parameters
    ----------
    epsf          : EPSFModel — normalised photutils ePSF  (sum ≈ 1 per unit flux)
    x_positions   : array     — star x pixel centres  (0-indexed)
    y_positions   : array     — star y pixel centres
    fluxes        : array     — total counts per star  [same units as image]
    image_shape   : tuple     — (nrows, ncols)
    psf_half_size : int|None  — half-size of stamp used for evaluation [px]

    Returns
    -------
    model     : 2D ndarray  — model image
    info_table: Table       — per-star: flux_input, pixel_sum, flux_fraction, status
    """
    nrows, ncols = image_shape
    model = np.zeros(image_shape, dtype=np.float64)

    # ── Auto-detect minimum safe stamp size ─────────────────────────────────
    if psf_half_size is None:
        os          = np.atleast_1d(np.asarray(epsf.oversampling, dtype=float))
        native_size = np.asarray(epsf.data.shape) / os
        psf_half_size = int(np.ceil(max(native_size) / 2)) + 5

    x_arr = np.asarray(x_positions, dtype=float)
    y_arr = np.asarray(y_positions, dtype=float)
    f_arr = np.asarray(fluxes,      dtype=float)

    statuses, psums, ffracs = [], [], []

    for xc, yc, flux in zip(x_arr, y_arr, f_arr):
        # ── Skip bad / zero-flux entries ────────────────────────────────────
        if flux <= 0 or not np.isfinite(flux):
            statuses.append('skipped'); psums.append(0.0); ffracs.append(0.0)
            continue

        # ── Pixel-aligned bounding box clipped to image ──────────────────────
        xi = int(round(xc));  yi = int(round(yc))
        x0 = max(0,     xi - psf_half_size)
        x1 = min(ncols, xi + psf_half_size + 1)
        y0 = max(0,     yi - psf_half_size)
        y1 = min(nrows, yi + psf_half_size + 1)

        if x0 >= x1 or y0 >= y1:
            statuses.append('outside'); psums.append(0.0); ffracs.append(0.0)
            continue

        # ── Evaluate ePSF on pixel grid ──────────────────────────────────────
        yy, xx = np.mgrid[y0-y0:y1-y0, x0-x0:x1-x0].astype(float)
        stamp  = epsf.evaluate(x=xx, y=yy, flux=flux, x_0=psf_half_size/2 + epsf.shape[0]/2 - 8 , y_0=psf_half_size/2 + epsf.shape[1]/2 - 9)
        model[y0:y1, x0:x1] += stamp

        psum   = float(stamp.sum())
        ffrac  = psum / flux
        status = 'edge' if (
            xi < psf_half_size or xi > ncols - psf_half_size or
            yi < psf_half_size or yi > nrows - psf_half_size
        ) else 'ok'

        statuses.append(status)
        psums.append(psum)
        ffracs.append(ffrac)

        # breakpoint()

    info_table = Table({
        'x'            : x_arr,
        'y'            : y_arr,
        'flux_input'   : f_arr,
        'pixel_sum'    : psums,
        'flux_fraction': ffracs,   # ≈ 1.0 for interior stars; < 1.0 at edges
        'status'       : statuses,
    })

    counts = {s: statuses.count(s) for s in ['ok', 'edge', 'outside', 'skipped']}
    print(f"Rendered: {counts['ok']} ok | {counts['edge']} edge | "
          f"{counts['outside']} outside | {counts['skipped']} skipped")

    return model, info_table

def extract_cutout_grid(image, xc, yc, half_size):
    """
    Extract a square cutout centred on (xc, yc) and return its
    absolute pixel coordinate arrays — needed so PSF evaluation
    uses the same coordinate system as the model.

    Returns
    -------
    cutout : 2D array
    xx     : 2D array of absolute x pixel centres
    yy     : 2D array of absolute y pixel centres
    slices : (y_slice, x_slice) used to index back into the image
    """
    nrows, ncols = image.shape
    xi, yi = int(round(xc)), int(round(yc))

    x0 = max(0,     xi - half_size);  x1 = min(ncols, xi + half_size + 1)
    y0 = max(0,     yi - half_size);  y1 = min(nrows, yi + half_size + 1)

    cutout = image[y0:y1, x0:x1].copy()
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(float)
    #         ↑ absolute coords         ↑ absolute coords
    return cutout, xx, yy, (slice(y0, y1), slice(x0, x1))

def compute_psf_moments(cutout, xx, yy, xc, yc, sky=0.0):
    """
    Compute quadrupole moments, size, and ellipticity of a PSF stamp.
    These should be ~constant if the PSF is truly uniform.

    R² = Mxx + Myy  — isotropic size [px²]
    e1 = (Mxx - Myy) / R²  — elongation along x vs y  (ideal: 0)
    e2 = 2 Mxy / R²        — elongation at 45°         (ideal: 0)
    """
    d = np.maximum(cutout - sky, 0.0)
    total = d.sum()
    if total <= 0:
        nan = np.nan
        return dict(R2=nan, e1=nan, e2=nan, Mxx=nan, Myy=nan, Mxy=nan,
                    x_centroid=nan, y_centroid=nan, concentration=nan)

    d_n  = d / total
    dx   = xx - xc
    dy   = yy - yc

    # First moments (centroid residual — should be ~0 if position is right)
    x_cen = float(np.sum(d_n * dx))
    y_cen = float(np.sum(d_n * dy))

    # Second moments
    Mxx = float(np.sum(d_n * dx**2))
    Myy = float(np.sum(d_n * dy**2))
    Mxy = float(np.sum(d_n * dx * dy))
    R2  = Mxx + Myy
    e1  = (Mxx - Myy) / R2 if R2 > 0 else np.nan
    e2  = 2.0 * Mxy  / R2 if R2 > 0 else np.nan

    # Concentration: flux fraction inside inner vs outer ring
    r         = np.sqrt(dx**2 + dy**2)
    r_scale   = np.sqrt(R2) if R2 > 0 else 1.0
    flux_core = d[r <= 0.5 * r_scale].sum()
    flux_wing = d[(r > 0.5 * r_scale) & (r <= 1.5 * r_scale)].sum()
    conc      = flux_core / (flux_wing + 1e-30)

    return dict(R2=R2, e1=e1, e2=e2,
                Mxx=Mxx, Myy=Myy, Mxy=Mxy,
                x_centroid=x_cen, y_centroid=y_cen,
                concentration=conc)

def diagnose_psf_variation(image, stars_tbl, half_size,
                            saturation=None, sky=0.0,
                            mag_col='phot_rp_mean_mag'):
    """
    Measure PSF shape metrics for every star and return a diagnostic table.
    Use plot_psf_variation() to visualise the results.
    """
    rows = []
    for row in stars_tbl:
        xc = float(row['x_pix'] if 'x_pix' in row.colnames else row['x'])
        yc = float(row['y_pix'] if 'y_pix' in row.colnames else row['y'])
        cutout, xx, yy, _ = extract_cutout_grid(image, xc, yc, half_size)

        is_sat = (saturation is not None) and bool(np.any(cutout >= saturation))
        m = compute_psf_moments(cutout, xx, yy, xc, yc, sky=sky)

        entry = {'x': xc, 'y': yc,
                 'peak': float(cutout.max()),
                 'total_flux': float(cutout.sum()),
                 'saturated': is_sat, **m}
        if mag_col in stars_tbl.colnames:
            entry['mag'] = float(row[mag_col])
        rows.append(entry)

    return Table(rows)

def plot_psf_variation(diag_tbl, image_shape=None, figsize=(16, 10)):
    """
    Six-panel diagnostic plot:
      Top row    — shape metrics vs magnitude (brightness dependence)
      Bottom row — shape metric maps across the detector (position dependence)
    """
    ok  = ~np.asarray(diag_tbl['saturated'], dtype=bool)
    has_mag = 'mag' in diag_tbl.colnames

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

    x_data  = diag_tbl['mag'][ok]    if has_mag else diag_tbl['total_flux'][ok]
    x_label = 'G_RP [mag]'           if has_mag else 'Total flux [counts]'

    metrics = [
        ('R2',          r'$R^2 = M_{xx} + M_{yy}$ [px²]', 'Isotropic size'),
        ('e1',          r'Ellipticity $e_1$',               'x vs y elongation'),
        ('concentration', 'Concentration index',            'Core-to-ring ratio'),
    ]

    # ── Top row: metric vs magnitude ─────────────────────────────────────────
    for col, (key, ylabel, title) in enumerate(metrics):
        ax  = fig.add_subplot(gs[0, col])
        val = np.asarray(diag_tbl[key][ok], dtype=float)
        med = np.nanmedian(val)
        std = np.nanstd(val)

        ax.scatter(x_data, val, s=12, alpha=0.6, c='steelblue', zorder=3)
        ax.axhline(med, color='crimson', lw=1.2, ls='--',
                   label=f'median = {med:.3f}')
        ax.fill_between([x_data.min(), x_data.max()],
                        med - std, med + std,
                        color='crimson', alpha=0.1, label=f'±1σ = {std:.3f}')

        # Mark saturated stars
        if np.any(~ok):
            ax.scatter(
                diag_tbl['mag'][~ok] if has_mag else diag_tbl['total_flux'][~ok],
                np.full((~ok).sum(), med),
                marker='x', c='orange', s=30, zorder=4, label='saturated'
            )
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(ylabel,  fontsize=9)
        ax.set_title(f'{title}\nvs magnitude', fontsize=9)
        ax.legend(fontsize=7)

    # ── Bottom row: 2D spatial maps ───────────────────────────────────────────
    for col, (key, ylabel, title) in enumerate(metrics):
        ax  = fig.add_subplot(gs[1, col])
        val = np.asarray(diag_tbl[key][ok], dtype=float)
        vmed = np.nanmedian(val)
        vrange = max(np.nanpercentile(np.abs(val - vmed), 95), 1e-6)

        sc = ax.scatter(
            diag_tbl['x'][ok], diag_tbl['y'][ok],
            c=val, s=35,
            cmap='RdBu_r',
            norm=Normalize(vmin=vmed - vrange, vmax=vmed + vrange),
            zorder=3,
        )
        plt.colorbar(sc, ax=ax, label=ylabel, fraction=0.04, pad=0.03)

        if image_shape is not None:
            ax.set_xlim(0, image_shape[1])
            ax.set_ylim(0, image_shape[0])
        ax.set_xlabel('x [px]', fontsize=9)
        ax.set_ylabel('y [px]', fontsize=9)
        ax.set_title(f'{title}\nacross detector', fontsize=9)

    fig.suptitle('PSF Shape Variation Diagnostics', y=1.01, fontsize=12)
    plt.show()


def plot_radial_profile_comparison(image, stars_tbl, epsf, half_size,
                                    n_bins=30, mag_col='phot_rp_mean_mag',
                                    n_mag_bins=3):
    """
    Compare azimuthally-averaged radial profiles in bins of magnitude.
    Each bin should overlay cleanly if the PSF shape is brightness-independent.
    """
    has_mag = mag_col in stars_tbl.colnames
    fig, ax = plt.subplots(figsize=(8, 5))

    # Overplot ePSF reference profile
    half_epsf = epsf.data.shape[0] // 2
    yy_e, xx_e = np.mgrid[-half_epsf:half_epsf+1,
                           -half_epsf:half_epsf+1].astype(float)
    epsf_stamp = epsf.evaluate(xx_e, yy_e, 1.0, 0.0, 0.0)
    epsf_stamp /= epsf_stamp.max()
    r_e = np.sqrt(xx_e**2 + yy_e**2).ravel()
    bin_edges = np.linspace(0, r_e.max(), n_bins + 1)
    bin_cen   = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    epsf_prof = np.array([
        np.mean(epsf_stamp.ravel()[(r_e >= bin_edges[i]) & (r_e < bin_edges[i+1])])
        if np.any((r_e >= bin_edges[i]) & (r_e < bin_edges[i+1])) else np.nan
        for i in range(n_bins)
    ])
    ax.plot(bin_cen, epsf_prof, 'k-', lw=2, label='ePSF reference')

    if has_mag:
        mags    = np.asarray(stars_tbl[mag_col], dtype=float)
        mag_bins = np.percentile(mags[np.isfinite(mags)], np.linspace(0, 100, n_mag_bins + 1))
        colors  = plt.cm.plasma(np.linspace(0.1, 0.9, n_mag_bins))

        for k in range(n_mag_bins):
            mask = (mags >= mag_bins[k]) & (mags < mag_bins[k + 1])
            profiles = []
            for row in stars_tbl[mask]:
                xc = float(row['x_pix'] if 'x_pix' in row.colnames else row['x'])
                yc = float(row['y_pix'] if 'y_pix' in row.colnames else row['y'])
                cutout, xx, yy, _ = extract_cutout_grid(image, xc, yc, half_size)
                cutout_norm = cutout / max(cutout.max(), 1e-10)
                r = np.sqrt((xx - xc)**2 + (yy - yc)**2).ravel()
                prof = np.array([
                    np.mean(cutout_norm.ravel()[(r >= bin_edges[i]) & (r < bin_edges[i+1])])
                    if np.any((r >= bin_edges[i]) & (r < bin_edges[i+1])) else np.nan
                    for i in range(n_bins)
                ])
                profiles.append(prof)
            if profiles:
                med_prof = np.nanmedian(profiles, axis=0)
                lbl = f'G_RP {mag_bins[k]:.1f}–{mag_bins[k+1]:.1f}  (N={mask.sum()})'
                ax.plot(bin_cen, med_prof, color=colors[k], lw=1.5, label=lbl)

    ax.set_xlabel('Radius [px]')
    ax.set_ylabel('Normalised profile')
    ax.set_title('Radial profile by magnitude bin\n'
                 '(curves should overlap if PSF is brightness-independent)')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    restore = True # if False, generate using the image defined below

    home_dir = str(Path.home())

    # if the user does not already have the psf, generate
    if not os.path.exists(f'{home_dir}/tierras/tierras_red/psfs/defocused_psf.fits'):
        print('Defocused PSF does not exist! Generating.')
        restore = False
        # if the user does not have a psfs folder in tierras_red, generate it
        if not os.path.exists(f'{home_dir}/tierras/tierras_red/psfs/'):
            os.mkdir(f'{home_dir}/tierras/tierras_red/psfs/')

    date    = '20260621'
    target  = 'HIP107350'
    filenum = '0369'

    file_list = [f'/data/tierras/flattened/{date}/{target}/flat0000/{date}.{filenum}.{target}_red.fit']
    hdul = fits.open(file_list[0])

    image   = hdul[0].data
    header  = hdul[0].header
    exptime = header['EXPTIME']

    stars_tbl = Table.from_pandas(source_selection(file_list, rp_mag_limit=14))
    stars_tbl.rename_column('X pix', 'x')
    stars_tbl.rename_column('Y pix', 'y')

    bkg_2d = Background2D(image, 32, filter_size=31).background

    image_sub = image - bkg_2d

    if not restore:

        epsf, fitted_stars = build_epsf(image_sub, stars_tbl, r_outer=40, oversampling=2, max_stars=40)

        print(f"ePSF array shape : {epsf.data.shape}")
        print(f"ePSF oversampling: {epsf.oversampling}")

        save_epsf_fits(epsf)
    else:
        epsf = load_epsf_fits(f'{home_dir}/tierras/tierras_red/psfs/defocused_psf.fits')

    plt.figure(figsize=(6, 5))
    plt.imshow(epsf.data, origin="lower", cmap="inferno")
    plt.plot(epsf.shape[0]/2, epsf.shape[1]/2, 'rx')
    plt.colorbar(label="Normalized flux")
    plt.title(f"Empirical PSF  (oversampling={epsf.oversampling[0]}×)")
    plt.tight_layout()
    plt.show()

    breakpoint()

    # now generate a model image using the stars used for the epsf

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)

    norm = simple_norm(image_sub, min_percent=1, max_percent=99)
    ax[0].imshow(image_sub, origin='lower', norm=norm)
    ax[0].plot(stars_tbl['x'], stars_tbl['y'], 'rx')

    fluxes = gaia_rp_to_counts(stars_tbl['phot_rp_mean_mag'], exptime, coeff=9.7e8/250)
    model, info_tbl = render_psf_model(epsf, stars_tbl['x'], stars_tbl['y'], fluxes, )

    ax[1].imshow(model, origin='lower', norm=norm)
    ax[1].plot(stars_tbl['x'], stars_tbl['y'], 'rx')


    res_img = image_sub - model 
    ax[2].imshow(res_img, origin='lower', norm=norm)
    ax[2].plot(stars_tbl['x'], stars_tbl['y'], 'rx')


    HALF_SIZE = 150
    SATURATION = 55000.
    diag = diagnose_psf_variation(image_sub, stars_tbl, HALF_SIZE, saturation=SATURATION)

    plot_psf_variation(diag, image_shape=image_sub.shape)
    plot_radial_profile_comparison(image_sub, stars_tbl, epsf, HALF_SIZE)
    breakpoint()
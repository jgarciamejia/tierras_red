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
from astropy.modeling import fitting
from astropy.nddata import NDData, Cutout2D

from photutils.background import Background2D
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.psf import (
    EPSFBuilder,
    extract_stars,
    EPSFFitter,
    PSFPhotometry
)

from scipy.optimize import least_squares, curve_fit
from scipy.stats import sigmaclip

from tierras_red_utils import source_selection, load_epsf_fits, generate_defocused_psf, epsf_interp

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

    fits.HDUList([hdu]).writeto(f'/data/tierras/psfs/defocused_psf.fits', overwrite=True)
    return 

def make_cutout(image, xi, yi, half):
        """
        Cut out a (2*half x 2*half) region centred on (xi, yi),
        clipping at image edges.

        Returns
        -------
        cutout  : 2D array  — the clipped pixel data
        x_c     : float     — star x position inside the cutout
        y_c     : float     — star y position inside the cutout
        """
        ny, nx = image.shape

        # Desired bounds
        x0 = int(xi) - half;  x1 = int(xi) + half
        y0 = int(yi) - half;  y1 = int(yi) + half

        # Clip to image
        x0c = max(x0, 0);  x1c = min(x1, nx)
        y0c = max(y0, 0);  y1c = min(y1, ny)

        cutout = image[y0c:y1c, x0c:x1c]

        # Star centre in cutout coordinates
        x_c = float(xi) - x0c
        y_c = float(yi) - y0c

        return cutout, x_c, y_c

def flux_model(x, A):
    """
        fittable model of flux (e-/s) as a function of magnitude
    """
    return A*10**(-x/2.5)

def fit_epsf(data, epsf_interp):
    """
        fit epsf to Tierras data 
    """
    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]].astype(float)

    def residuals(p):
        xc, yc, flux = p
        model = flux * epsf_interp((yy - yc).ravel(),
                            (xx - xc).ravel(), grid=False).reshape(cutout.shape)
        return (cutout - model).ravel()
    
    res = least_squares(residuals, x0=[x_c0, y_c0, float(data.sum())], method='lm')

    return res


if __name__ == '__main__':

    restore = True # if False, generate using the image defined below

 
    # if the user does not already have the psf, generate
    if not os.path.exists('/data/tierras/psfs/defocused_psf.fits'):
        print('Defocused PSF does not exist! Generating.')
        restore = False

    date    = '20260621'
    target  = 'HIP107350'
    filenum = '0369'

    file_list = [f'/data/tierras/flattened/{date}/{target}/flat0000/{date}.{filenum}.{target}_red.fit']
    hdul = fits.open(file_list[0])

    image   = hdul[0].data
    header  = hdul[0].header
    exptime = header['EXPTIME']

    stars_tbl = Table.from_pandas(source_selection(file_list, rp_mag_limit=15, overwrite=True))
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
        epsf = load_epsf_fits('/data/tierras/psfs/defocused_psf.fits')

    plt.figure(figsize=(6, 5))
    plt.imshow(epsf.data, origin="lower", cmap="inferno")
    plt.plot(epsf.shape[0]/2, epsf.shape[1]/2, 'rx')
    plt.colorbar(label="Normalized flux")
    plt.title(f"Empirical PSF  (oversampling={epsf.oversampling[0]}×)")
    plt.tight_layout()
    plt.show()

    breakpoint()

    i    = 0
    plot = False
    GAIN = 5.9
    half = 100

    epsf_interp = epsf_interp(epsf)
    
    psf_flux_per_s = np.zeros(len(stars_tbl))
    ap_flux_per_s  = np.zeros_like(psf_flux_per_s)
    for i in range(len(stars_tbl)):
        print(f'Fitting star {i+1} of {len(stars_tbl)}')
        xi, yi = float(stars_tbl['x'][i]), float(stars_tbl['y'][i])
        cutout, x_c0, y_c0 = make_cutout(image_sub, xi, yi, half)

        yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]].astype(float)

        r = fit_epsf(cutout, epsf_interp)

        x_fit, y_fit, flux_fit = r.x
        print(f"x_fit={x_fit:.2f}  y_fit={y_fit:.2f}  flux={flux_fit:.1f}")

        # ── Render fitted model ───────────────────────────────────────────────────────
        model_image = flux_fit * epsf_interp((yy - y_fit).ravel(),
                                        (xx - x_fit).ravel(),
                                        grid=False).reshape(cutout.shape)

        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
            norm = simple_norm(cutout, min_percent=1, max_percent=99)

            axes[0].imshow(cutout,               origin='lower', norm=norm)
            axes[0].plot(x_fit, y_fit, 'rx', ms=10)
            axes[0].set_title('Data')

            axes[1].imshow(model_image,          origin='lower', norm=norm)
            axes[1].plot(x_fit, y_fit, 'rx', ms=10)
            axes[1].set_title(f'Model  ({x_fit:.1f}, {y_fit:.1f})')

            axes[2].imshow(cutout - model_image, origin='lower',
                        norm=simple_norm(cutout - model_image, min_percent=1, max_percent=99))
            axes[2].set_title('Residual')

            plt.tight_layout()

        psf_flux_per_s[i] = flux_fit / exptime * GAIN

        # compare with aperture photometry 
        ap = CircularAperture((half, half), r=50)
        
        if plot:
            ap.plot(ax=axes[0], color='r')
            breakpoint()

        phot_tbl = aperture_photometry(cutout, ap)
        ap_flux_per_s[i] = phot_tbl['aperture_sum'][0]/exptime * GAIN


    # do an initial baselining of the fluxes so that we can sigma clip outliers 
    model_flux_init = 9.7e8*10**(-stars_tbl['phot_rp_mean_mag']/2.5) # coefficient from fit in instrument paper
    baselined_fluxes_init = psf_flux_per_s / model_flux_init
    v, lo, hi, = sigmaclip(baselined_fluxes_init, 3.5, 3.5)
    keep_inds = np.where((baselined_fluxes_init > lo) & (baselined_fluxes_init < hi))[0]

    # now fit to fluxes without outliers 
    x = stars_tbl['phot_rp_mean_mag'][keep_inds]
    y = psf_flux_per_s[keep_inds]

    plt.figure()
    plt.plot(x, y, '.')
    plt.yscale('log')

    popt, pcov = curve_fit(flux_model, x, y, p0=[9.7e8])

    model_mags   = np.arange(4, 17, 0.1)
    model_fluxes = flux_model(model_mags, popt[0])

    plt.plot(model_mags, model_fluxes, label=f'Calibration from Defocused PSF fluxes: A={popt[0]:.1e} e-/s')
    plt.plot(model_mags, flux_model(model_mags, 9.7e8), label=f'Calibration from instrument paper: A=9.7e+08 e-/s')
    plt.legend()
    plt.ylabel('e- / s', fontsize=14)
    plt.xlabel('$G_\\text{RP}$ (mag)', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    breakpoint()
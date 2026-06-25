import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from astropy.io import fits
from astropy.nddata import NDData
from astropy.table import Table
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter, TRFLSQFitter

from photutils.background import Background2D, MedianBackground, MMMBackground
from photutils.detection import DAOStarFinder, find_peaks
from photutils.psf import (
    EPSFBuilder,
    extract_stars,
    FittableImageModel,   # photutils < 1.9
    # ImagePSF,           # photutils >= 1.9 (replacement for FittableImageModel)
    PSFPhotometry,
    IterativePSFPhotometry,
)
from photutils.aperture import CircularAperture

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
    clean.sort("peak_value", reverse=True)
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



if __name__ == '__main__':


    file_list = ['/data/tierras/flattened/20260622/HIP70497/flat0000/20260622.0077.HIP70497_red.fit']
    hdul = fits.open(file_list[0])

    image = hdul[0].data

    stars_tbl = source_selection(file_list, rp_mag_limit=11)
    breakpoint()

    epsf, fitted_stars = build_epsf(image_sub, stars_tbl, r_outer=r_outer_guess, oversampling=2, max_stars=40)

    plt.figure(figsize=(6, 5))
    plt.imshow(epsf.data, origin="lower", cmap="inferno")
    plt.colorbar(label="Normalized flux")
    plt.title(f"Empirical PSF  (oversampling={epsf.oversampling[0]}×)")
    plt.tight_layout()
    plt.show()

    print(f"ePSF array shape : {epsf.data.shape}")
    print(f"ePSF oversampling: {epsf.oversampling}")


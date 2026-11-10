### ==============================
# SPHEREx datacube builder — v4


# Core Python & Scientific Stack
# ==============================
import os
import sys
import time
import logging
import socket
import concurrent.futures

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.cluster.hierarchy import linkage, fcluster



# ==============================
# Astropy
# ==============================
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.stats import SigmaClip, sigma_clipped_stats
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

# ==============================
# Photometry & VO Access
# ==============================
from photutils.aperture import (
    CircularAperture,
    SkyCircularAperture,
    SkyCircularAnnulus,
    aperture_photometry,
    ApertureStats,
)
import pyvo
import urllib.error
import urllib.request

# ==============================
# GUI / File Dialogs
# ==============================
try:                                    # file dialogs (interactive menu only)
    import tkinter as tk
    from tkinter import filedialog
except ImportError:                     # headless / batch use
    tk = filedialog = None

matplotlib.use("Agg")   # Safe for headless execution / multiprocessing

# -------------------------------------------------
# Suppress astropy warnings
# -------------------------------------------------
logging.getLogger("astropy").setLevel(logging.ERROR)
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

# -------------------------------------------------
# SPHEREx Level-2 FLAGS bits (Explanatory Supplement, Table 16)
# -------------------------------------------------
SPHEREX_FLAG_BITS = {
    "TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2, "NONFUNC": 6, "DICHROIC": 7,
    "MISSING_DATA": 9, "HOT": 10, "COLD": 11, "FULLSAMPLE": 12, "PHANMISS": 14,
    "NONLINEAR": 15, "PERSIST": 17, "OUTLIER": 19, "CROSSTALK": 20, "SOURCE": 21,
    "GHOST": 22, "GHOST_FPA": 23, "GHOST_EXT": 24, "STREAK": 25, "BLOOM": 26,
    "SNOWBALL": 27, "HALO": 28, "SATELLITE_HALO": 29,
}

# Pixels with any of these bits are set to NaN before resampling.
#  * TRANSIENT is the pipeline's on-board cosmic-ray flag (replaces the old 3x3-median
#    cosmic-ray filter, which clipped the cores of real, undersampled stars).
#  * OVERFLOW / NONLINEAR / BLOOM are NOT masked by default: they mark bright-star
#    cores, and masking them would punch a hole in every bright star.
#  * SOURCE is informational only and is never masked.
DEFAULT_BAD_FLAGS = (
    "TRANSIENT", "SUR_ERROR", "NONFUNC", "DICHROIC", "MISSING_DATA",
    "HOT", "COLD", "PHANMISS", "PERSIST", "STREAK", "SNOWBALL", "SATELLITE_HALO",
)


def flag_bitmask(names=DEFAULT_BAD_FLAGS):
    mask = 0
    for name in names:
        mask |= 1 << SPHEREX_FLAG_BITS[name]
    return mask

# -------------------------------------------------
# FITS open with retry
# -------------------------------------------------
def open_fits_with_retry(url, retries=10, base_delay=1, cache=False, timeout=120):
    """Open a (remote) FITS file, retrying on network errors.

    Remote files are read straight into memory.  (astropy's fits.open(url) writes every
    download to a temporary file, astropy-download-<pid>-*, that is only deleted when Python
    exits, so a long run over many targets filled the disk.)  Local paths open normally;
    cache=True keeps the old behaviour (astropy download cache)."""
    import io
    is_remote = str(url).startswith(("http://", "https://", "ftp://"))
    for attempt in range(1, retries + 1):
        try:
            if not is_remote or cache:
                return fits.open(url, cache=cache)
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            return fits.open(io.BytesIO(data))
        except (urllib.error.HTTPError,
                urllib.error.URLError,
                socket.gaierror,
                socket.timeout,
                OSError) as e:
            if attempt == retries:
                raise
            wait = base_delay * 2 ** (attempt - 1)
            print(f"Retry {attempt}/{retries} ({wait}s): {e}")
            time.sleep(wait)

# -------------------------------------------------
# TAP query
# -------------------------------------------------
def get_data(ra, dec, size):
    bandpasses = [f"SPHEREx-D{i}" for i in range(1, 7)]
    bandpass_sql = ",".join(f"'{b}'" for b in bandpasses)

    service = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")

    query = f"""
    SELECT
        'https://irsa.ipac.caltech.edu/' || a.uri ||
        '?center={ra.value},{dec.value}d&size={size.to(u.deg).value}' AS uri,
        p.time_bounds_lower,
        p.energy_bandpassname
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(
        POINT('ICRS', {ra.value}, {dec.value}),
        p.poly
    )
    AND p.energy_bandpassname IN ({bandpass_sql})
    ORDER BY p.time_bounds_lower
    """
    t0 = time.time()
    results = service.search(query)
    print(f"TAP query time: {time.time() - t0:.2f}s")
    print(f"Found {len(results)} images")
    return results

# -------------------------------------------------
# Cutout processing (ZODI subtraction, FLAGS masking, astrometry quality)
# -------------------------------------------------
def process_cutout_return(row, ra, dec, subtract_zodi=True, mask_flags=True,
                          bad_flag_mask=None):
    if bad_flag_mask is None:
        bad_flag_mask = flag_bitmask()
    try:
        with open_fits_with_retry(row["uri"]) as hdul:
            img_hdu = hdul["IMAGE"]
            wcs = WCS(img_hdu.header, relax=True)

            img_data = img_hdu.data.astype(np.float64)

            if subtract_zodi and "ZODI" in hdul and hdul["ZODI"].data is not None:
                zodi_data = hdul["ZODI"].data.astype(np.float64)
                img_data -= zodi_data

            if mask_flags and "FLAGS" in hdul and hdul["FLAGS"].data is not None:
                flags_data = hdul["FLAGS"].data.astype(np.int64)
                img_data[(flags_data & bad_flag_mask) != 0] = np.nan

            bunit = img_hdu.header.get("BUNIT", "MJy/sr").strip()

            # Fine-astrometry quality: 0 = solution rms < 1" (NaN if keyword absent)
            finast = img_hdu.header.get("FINAST", None)
            finast = np.nan if finast is None else float(finast)

            coord = SkyCoord(ra=ra, dec=dec)
            x, y = wcs.world_to_pixel(coord)

            spec_wcs = WCS(img_hdu.header, fobj=hdul, key="W", relax=True)
            spec_wcs.sip = None
            lam, _ = spec_wcs.pixel_to_world(x, y)
            lam = lam.to(u.micron).value

            ny, nx = img_data.shape
            y_idx, x_idx = np.indices((ny, nx))
            native_wave_map, _ = spec_wcs.pixel_to_world(x_idx, y_idx)
            native_wave_map = native_wave_map.to(u.micron).value

            # R7 spectral calibration: per-pixel band centre / width from the parameter maps
            # (verified detector coordinates), else WCS (+6.4 nm in bands 5-6)
            band_i = band_from_name(row.get("energy_bandpassname", ""), lam)
            native_wave_map, bw_map, lam_cal, wave_src = calibrated_wavelengths(
                img_hdu.header, native_wave_map, band_i, x, y)
            if lam_cal is not None:
                lam = lam_cal
            elif wave_src == "WCS+MWIR":
                lam = lam + MWIR_BC_SHIFT_UM

            hdu = fits.ImageHDU(data=img_data, header=img_hdu.header.copy())
            hdu.header["EXTNAME"] = f"IMAGE{row['cutout_index']}"
            hdu.header["BUNIT"] = bunit

            wave_hdu = fits.ImageHDU(data=native_wave_map, header=img_hdu.header)
            bw_hdu = (fits.ImageHDU(data=np.asarray(bw_map, np.float32), header=img_hdu.header)
                      if bw_map is not None else None)

            return {
                "cutout_index": row["cutout_index"],
                "band": band_i,
                "central_wavelength": lam,
                "wave_src": wave_src,
                "bw_hdu": bw_hdu,
                "time_bounds_lower": row.get("time_bounds_lower", np.nan),
                "bunit": bunit,
                "uri": row["uri"],
                "finast": finast,
                "hdu": hdu,
                "wave_hdu": wave_hdu
            }

    except Exception as e:
        print(f" Failed cutout {row['cutout_index']}: {e}")
        return {
            "cutout_index": row["cutout_index"],
            "band": -1,
            "central_wavelength": np.nan,
            "time_bounds_lower": row.get("time_bounds_lower", np.nan),
            "bunit": "MJy/sr",
            "uri": row["uri"],
            "finast": np.nan,
            "hdu": None,
            "wave_hdu": None,
            "wave_src": "",
            "bw_hdu": None,
        }

# -------------------------------------------------
# ONE master spatial grid for every epoch:
# north-up TAN projection centred on the target, user-chosen pixel scale.
# Every exposure of every epoch is resampled onto exactly this grid.
# -------------------------------------------------
def build_master_spatial_wcs(center_coord, size_arcsec, pixel_scale_arcsec=6.15,
                             bunit="MJy/sr"):
    """
    Parameters
    ----------
    center_coord       : SkyCoord of the target (becomes the grid centre).
    size_arcsec        : side length of the square output grid (arcsec).
    pixel_scale_arcsec : output pixel size; 6.15" is the native SPHEREx pixel.

    Returns
    -------
    master_header : 2-D FITS header (north-up TAN, NAXIS1/2 set)
    ny, nx        : output image dimensions in pixels (odd, target on the central pixel)
    """
    n = int(np.ceil(size_arcsec / pixel_scale_arcsec))
    if n % 2 == 0:
        n += 1

    w = WCS(naxis=2)
    w.wcs.ctype  = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval  = [center_coord.ra.deg, center_coord.dec.deg]
    w.wcs.crpix  = [(n + 1) / 2.0, (n + 1) / 2.0]
    w.wcs.cdelt  = [-pixel_scale_arcsec / 3600.0, pixel_scale_arcsec / 3600.0]
    w.wcs.cunit  = ["deg", "deg"]
    w.wcs.radesys = "ICRS"

    master_header = w.to_header()
    master_header["NAXIS"]  = 2
    master_header["NAXIS1"] = n
    master_header["NAXIS2"] = n
    master_header["BUNIT"]  = bunit

    print(f"[INFO] Master grid: north-up TAN, {pixel_scale_arcsec:.3f}\"/pix, "
          f"{n}×{n} pixels centred on RA={center_coord.ra.deg:.6f}, "
          f"Dec={center_coord.dec.deg:.6f}")
    return master_header, n, n


# -------------------------------------------------
# Build a common wavelength grid from all epoch grids
# -------------------------------------------------
def build_common_wave_grid(epoch_rows_list, oversample=1.0):
    """
    epoch_rows_list : list of astropy Tables, one per epoch,
                      each already sorted by central_wavelength.
    oversample      : grid spacing = median_dlambda / oversample.

    Returns
    -------
    common_waves : 1-D ndarray of wavelengths in µm,
                   uniformly spaced on [global_min, global_max].
    """
    all_waves = []
    for rows in epoch_rows_list:
        w = np.array(rows["central_wavelength"], dtype=float)
        all_waves.append(w[~np.isnan(w)])

    all_waves_concat = np.concatenate(all_waves)
    global_min = all_waves_concat.min()
    global_max = all_waves_concat.max()

    median_dlambda_list = []
    for w in all_waves:
        w_sorted = np.sort(w)
        if len(w_sorted) > 1:
            median_dlambda_list.append(np.median(np.diff(w_sorted)))
    if not median_dlambda_list:
        raise ValueError("Cannot determine wavelength spacing — no valid epoch data.")

    dlambda = np.median(median_dlambda_list) / oversample
    n_grid = max(int(np.round((global_max - global_min) / dlambda)) + 1, 2)
    common_waves = np.linspace(global_min, global_max, n_grid)
    print(f"[INFO] Common wavelength grid: {n_grid} points, "
          f"{global_min:.4f}–{global_max:.4f} µm, "
          f"Δλ={dlambda*1e3:.2f} nm")
    return common_waves


# -------------------------------------------------
# Spatial resampling
# -------------------------------------------------
# Images are drizzled with STScI's `drizzle` package (the drizzle engine behind
# AstroDrizzle / drizzlepac; Fruchter & Hook 2002).  Each input pixel is shrunk to a
# drop of side `pixfrac` and its surface brightness is shared among the output pixels
# the drop overlaps, by exact overlap area: no ringing, flux conserved.
# Bilinear interpolation is used only internally, for the smooth per-pixel wavelength maps.
RESAMPLE_KERNELS = {
    "bilinear": (np.array([0, 1]), lambda t: np.clip(1.0 - np.abs(t), 0.0, None)),
}
RESAMPLE_METHODS = ("drizzle",)

# ---- Drizzle engine: STScI drizzle package (required: pip install drizzle) ----
try:
    import drizzle as _drizzle_pkg
    from drizzle.resample import Drizzle as _STScIDrizzle
    from drizzle.utils import calc_pixmap as _calc_pixmap
except ImportError as _e:
    raise ImportError("Cube_builder_sphereX_v2 needs STScI's drizzle package: "
                      "pip install drizzle") from _e
DRIZZLE_ENGINE = f"stsci drizzle {getattr(_drizzle_pkg, '__version__', '')}".strip()


def drizzle_to_grid(data, wcs_in, wcs_out, shape_out, pixfrac=0.6, weight=None):
    """
    Drizzle one image onto the output grid with STScI's drizzle (square kernel).

    Surface brightness in, surface brightness out.  calc_pixmap maps every input pixel
    through both WCSs (SIP distortion included); the C code computes the exact overlap of
    each drop with each output pixel.  weight: optional per-pixel weight map (e.g.
    1/variance); default 1 for finite pixels.  NaN input pixels get zero weight; output
    pixels that no drop reaches are NaN.
    """
    data = np.asarray(data, dtype=np.float32)
    good = np.isfinite(data)
    if weight is None:
        wht = good.astype(np.float32)
    else:
        wht = np.where(good & np.isfinite(weight), weight, 0.0).astype(np.float32)
    pixmap = _calc_pixmap(wcs_in, wcs_out, shape=data.shape)
    drz = _STScIDrizzle(kernel="square", out_shape=tuple(shape_out), fillval=np.nan,
                        disable_ctx=True)
    # scale=1: input and output are both surface brightness (MJy/sr), so no pixel-area scaling
    drz.add_image(np.where(good, data, 0.0).astype(np.float32), exptime=1.0, pixmap=pixmap,
                  weight_map=wht, pixfrac=pixfrac, scale=1.0, in_units="cps")
    out = np.asarray(drz.out_img, dtype=np.float64)
    out[~(np.asarray(drz.out_wht) > 0)] = np.nan
    return out


def resample_to_grid(data, xin, yin, method="bilinear", min_weight=0.5, chunk=50000):
    """
    Sample `data` at fractional input-pixel positions (xin, yin), 0-based.

    NaN / masked input pixels get zero weight and the kernel is renormalised.
    The output is NaN where the nearest input pixel is NaN, where less than
    `min_weight` of the kernel weight is available, or outside the input image.
    """
    data = np.asarray(data, dtype=np.float64)
    ny_in, nx_in = data.shape
    xin = np.asarray(xin, dtype=np.float64).ravel()
    yin = np.asarray(yin, dtype=np.float64).ravel()
    out = np.full(xin.shape, np.nan)

    inside = (np.isfinite(xin) & np.isfinite(yin)
              & (xin > -0.5) & (xin < nx_in - 0.5)
              & (yin > -0.5) & (yin < ny_in - 0.5))
    if not inside.any():
        return out

    if method not in RESAMPLE_KERNELS:
        raise ValueError("resample_to_grid is only used (bilinear) for wavelength maps; "
                         "images are drizzled with drizzle_to_grid.")
    taps, kern = RESAMPLE_KERNELS[method]

    finite = np.isfinite(data)
    filled = np.where(finite, data, 0.0)

    idx = np.nonzero(inside)[0]
    for start in range(0, idx.size, chunk):
        sel = idx[start:start + chunk]
        x, y = xin[sel], yin[sel]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)

        ix = x0[:, None] + taps[None, :]
        iy = y0[:, None] + taps[None, :]
        wx = kern((x - x0)[:, None] - taps[None, :])
        wy = kern((y - y0)[:, None] - taps[None, :])

        okx = (ix >= 0) & (ix < nx_in)
        oky = (iy >= 0) & (iy < ny_in)
        ixc = np.clip(ix, 0, nx_in - 1)[:, None, :]
        iyc = np.clip(iy, 0, ny_in - 1)[:, :, None]

        w2   = wy[:, :, None] * wx[:, None, :]
        good = finite[iyc, ixc] & oky[:, :, None] & okx[:, None, :]
        wg   = np.where(good, w2, 0.0)

        num = np.einsum("nij,nij->n", wg, filled[iyc, ixc])
        den = wg.sum(axis=(1, 2))
        tot = w2.sum(axis=(1, 2))

        near_ok = finite[np.clip(np.rint(y).astype(int), 0, ny_in - 1),
                         np.clip(np.rint(x).astype(int), 0, nx_in - 1)]
        ok = near_ok & (den > min_weight * tot)

        res = np.full(sel.size, np.nan)
        res[ok] = num[ok] / den[ok]
        out[sel] = res
    return out


# -------------------------------------------------
# Create Rectified Cube with PCHIP Interpolation.
# All epochs must pass the SAME master_header / (ny, nx)
# so that their output cubes are pixel-aligned.
# -------------------------------------------------
def create_rectified_cube(
    valid_rows, target_ra, target_dec,
    common_waves=None,
    master_header=None,
    master_ny=None,
    master_nx=None,
    resample_method="drizzle",
    pixfrac=0.6,
):
    """
    Parameters
    ----------
    valid_rows      : astropy Table rows for this epoch, sorted by wavelength.
    target_ra/dec   : SkyCoord quantities (kept for API compatibility).
    common_waves    : 1-D array of output wavelength nodes (µm).
    master_header   : 2-D FITS header defining the common celestial grid
                      (from build_master_spatial_wcs; same for every epoch).
    master_ny/nx    : output image dimensions (pixels).
    resample_method : spatial method, one of RESAMPLE_METHODS.
    pixfrac         : drop size for resample_method="drizzle" (input pixels).
    """
    if master_header is None:
        raise ValueError("master_header must be provided to create_rectified_cube; "
                         "call build_master_spatial_wcs first.")

    waves = np.array([row["central_wavelength"] for row in valid_rows], dtype=float)
    mjds  = np.array([row["time_bounds_lower"]  for row in valid_rows], dtype=float)
    bunit = valid_rows[0]["bunit"] if valid_rows[0]["bunit"] else "MJy/sr"

    n_y, n_x = int(master_ny), int(master_nx)

    if common_waves is None:
        common_waves = np.linspace(np.nanmin(waves), np.nanmax(waves), len(valid_rows))
    num_slices = len(common_waves)

    # Interpolate MJD array onto common wavelength nodes
    sort_w = np.argsort(waves)
    valid_mjd_mask = ~np.isnan(mjds[sort_w]) & ~np.isnan(waves[sort_w])
    if np.sum(valid_mjd_mask) > 1:
        common_mjds = np.interp(
            common_waves,
            waves[sort_w][valid_mjd_mask],
            mjds[sort_w][valid_mjd_mask]
        )
    else:
        common_mjds = np.full_like(common_waves, np.nan)

    # Sky position of every output pixel — computed once, reused for every slice
    master_wcs = WCS(master_header).celestial
    yy, xx = np.mgrid[0:n_y, 0:n_x]
    sky_out = master_wcs.pixel_to_world(xx.ravel(), yy.ravel())

    n_in = len(valid_rows)
    all_reprojected = np.full((n_in, n_y, n_x), np.nan)
    all_waves_rep   = np.full((n_in, n_y, n_x), np.nan)

    how = f"{DRIZZLE_ENGINE}, pixfrac={pixfrac}"
    print(f" Resampling {n_in} slices onto master grid ({n_y}×{n_x}) with {how} …")
    for s, row in enumerate(valid_rows):
        hdu      = row["hdu"]
        wave_hdu = row["wave_hdu"]

        # Input pixel coordinates of every output pixel (includes SIP distortion)
        wcs_in = WCS(hdu.header, relax=True).celestial
        xin, yin = wcs_in.world_to_pixel(sky_out)

        all_reprojected[s] = drizzle_to_grid(
            hdu.data, wcs_in, master_wcs, (n_y, n_x), pixfrac=pixfrac)
        # The wavelength map is smooth, so plain bilinear is exact enough
        all_waves_rep[s] = resample_to_grid(
            wave_hdu.data, xin, yin, method="bilinear").reshape(n_y, n_x)

    print(f" PCHIP interpolation onto {num_slices}-node common wavelength grid …")
    cube_data = np.full((num_slices, n_y, n_x), np.nan, dtype=np.float32)

    for i in range(n_y):
        for j in range(n_x):
            p_vals  = all_reprojected[:, i, j]
            p_waves = all_waves_rep[:, i, j]

            valid_mask = ~np.isnan(p_vals) & ~np.isnan(p_waves)
            if valid_mask.sum() <= 2:
                continue

            p_waves_c = p_waves[valid_mask]
            p_vals_c  = p_vals[valid_mask]

            sort_idx = np.argsort(p_waves_c)
            x_s = p_waves_c[sort_idx]
            y_s = p_vals_c[sort_idx]

            x_uniq, uniq_idx = np.unique(x_s, return_index=True)
            y_uniq = y_s[uniq_idx]

            if len(x_uniq) > 2:
                in_range = (common_waves >= x_uniq[0]) & (common_waves <= x_uniq[-1])
                pchip = PchipInterpolator(x_uniq, y_uniq, extrapolate=False)
                cube_data[in_range, i, j] = pchip(common_waves[in_range])

    # Build the 3-D output header (master celestial WCS + spectral axis)
    out_header = master_header.copy()
    out_header["BUNIT"]   = bunit
    out_header["WCSAXES"] = 3
    out_header["NAXIS"]   = 3
    out_header["NAXIS3"]  = num_slices
    out_header["CTYPE3"]  = "WAVE"
    out_header["CRPIX3"]  = 1.0
    out_header["CRVAL3"]  = float(common_waves[0])
    out_header["CDELT3"]  = float(common_waves[1] - common_waves[0])
    out_header["CUNIT3"]  = "um"
    out_header["RESAMPLE"] = (resample_method, "spatial resampling method")
    if resample_method == "drizzle":
        out_header["PIXFRAC"] = (float(pixfrac), "drizzle drop size (input pixels)")
        out_header["DRZENGIN"] = (DRIZZLE_ENGINE, "spatial resampling engine")

    if "PC1_1" in out_header:
        for tag in ["3_3", "1_3", "2_3", "3_1", "3_2"]:
            out_header[f"PC{tag}"] = 1.0 if tag == "3_3" else 0.0
    elif "CD1_1" in out_header:
        out_header["CD3_3"] = out_header["CDELT3"]
        for tag in ["1_3", "2_3", "3_1", "3_2"]:
            out_header[f"CD{tag}"] = 0.0

    return cube_data, out_header, common_waves, common_mjds


# -------------------------------------------------
# v2: spectral kernel binning onto the common grid
# -------------------------------------------------
# band -> (lambda_min, lambda_max, R)
SPHEREX_BANDS = {
    1: (0.75, 1.11, 41.0), 2: (1.11, 1.64, 41.0), 3: (1.64, 2.42, 41.0),
    4: (2.42, 3.82, 35.0), 5: (3.82, 4.42, 110.0), 6: (4.42, 5.00, 130.0),
}

# u = (lambda_sample - lambda_node) / kernel_FWHM
SPECTRAL_KERNELS = {
    "tent":  (lambda u: np.clip(1.0 - np.abs(u), 0.0, None), 1.0),       # triangle, half-base = FWHM
}

# ---- Instrument spectral response ("response" kernel) ----
# Hui et al. 2026, ApJS 284, 10 and the SSDC R7 spectral-calibration delivery
# ("SPHEREx spectral fiducial functions", Hui, July 2026):
#   band centre  lambda_c = 50% point of the cumulative response;  dlambda = integral of the
#   peak-normalised response;  R = lambda_c / dlambda.
#   The fiducial responses are FLAT-TOPPED with steep edges (not Gaussian).  The middle-pixel
#   (Cent) curve keeps the true per-pixel bandwidth, and a pixel away from the channel centre is
#   well described by shifting that curve in wavelength (delivery note, Sect. 3).
# With kernel="response" each sample s is weighted by ITS OWN bandpass evaluated at the node,
# T_s(lambda_k): the voxel is the response-weighted mean of every measurement that has
# sensitivity at lambda_k.
#   * response_file given (the SSDC R7 delivery folder, its MiddlePixel/SubChannel17 folder, or
#     a single table): the measured Cent curve of the nearest channel, shifted to lambda_s.
#   * no file: a flat-topped super-Gaussian, T = exp(-ln2 |2 du / FWHM|^4), whose equivalent
#     width is dlambda = kernel_fwhm * lambda_s / R(lambda_s), with R(lambda) interpolated
#     from the per-channel table below (delivery note Table 2: median of the BC/BW maps).
SUPERGAUSS_P = 4.0
_SG_EQW = 0.9934        # equivalent width / FWHM for p = 4: Gamma(1+1/p) / ln2**(1/p)

# Table 2 of the delivery note: per band, 17 channel centres [um] and median R
SPHEREX_CHANNELS = {
    1: ([0.753, 0.771, 0.790, 0.809, 0.828, 0.848, 0.868, 0.889, 0.910, 0.932, 0.955, 0.978,
         1.001, 1.025, 1.050, 1.075, 1.102],
        [42, 42, 41, 40, 40, 40, 40, 39, 39, 39, 39, 38, 38, 38, 38, 37, 37]),
    2: ([1.113, 1.139, 1.167, 1.195, 1.224, 1.254, 1.284, 1.315, 1.346, 1.378, 1.412, 1.446,
         1.481, 1.517, 1.554, 1.592, 1.631],
        [43, 42, 42, 41, 41, 41, 42, 42, 42, 42, 41, 41, 41, 41, 41, 41, 41]),
    3: ([1.656, 1.695, 1.735, 1.775, 1.818, 1.861, 1.905, 1.949, 1.995, 2.041, 2.089, 2.139,
         2.189, 2.240, 2.292, 2.345, 2.399],
        [42, 42, 41, 41, 41, 41, 41, 41, 41, 40, 40, 40, 40, 40, 40, 40, 41]),
    4: ([2.445, 2.506, 2.575, 2.646, 2.719, 2.794, 2.871, 2.951, 3.033, 3.118, 3.204, 3.293,
         3.383, 3.476, 3.570, 3.667, 3.768],
        [42, 34, 35, 35, 35, 35, 34, 34, 34, 34, 35, 35, 35, 36, 36, 36, 36]),
    5: ([3.833, 3.867, 3.901, 3.935, 3.970, 4.005, 4.040, 4.076, 4.112, 4.148, 4.184, 4.220,
         4.257, 4.294, 4.331, 4.368, 4.405],
        [109, 109, 109, 108, 109, 110, 110, 111, 112, 113, 113, 115, 114, 115, 116, 116, 117]),
    6: ([4.435, 4.467, 4.500, 4.533, 4.566, 4.600, 4.635, 4.669, 4.704, 4.739, 4.773, 4.808,
         4.843, 4.879, 4.915, 4.951, 4.988],
        [128, 125, 127, 126, 127, 125, 127, 126, 128, 130, 131, 132, 131, 130, 130, 132, 135]),
}


def spherex_R(lam, band):
    """Resolving power at lam (um) in detector band, interpolated over the 17 channels."""
    lc, R = SPHEREX_CHANNELS[int(band)]
    return np.interp(lam, lc, np.asarray(R, float))


# ---- SSDC R7 response profiles generated from Table 2 of the R7 delivery note ----
# For every one of the 102 channels (SPHEREX_CHANNELS above = Table 2: median band centre
# lambda_c and median R = lambda_c / dlambda) a peak-normalised, flat-topped profile is built,
# T(lambda) = exp(-ln2 |2 (lambda - lambda_c) / FWHM|^4), with FWHM chosen so that its
# equivalent width (integral of T, the R7 definition of dlambda) equals lambda_c / R.
# The flat top / steep edges follow the fiducial curves shown in the note (Figs. 3-5).
# In use, a pixel's profile is the one of its nearest channel shifted to the pixel's own
# wavelength (the note's prescription for off-centre pixels).

def r7_table_profiles(nsamp_per_dlambda=40, extent=1.5):
    """Response profiles of the 102 SPHEREx channels from Table 2, in load_response_table format."""
    out = {}
    for b, (lc, R) in SPHEREX_CHANNELS.items():
        lc = np.asarray(lc, float); dl = lc / np.asarray(R, float)
        fwhm = dl / _SG_EQW
        step = dl.min() / nsamp_per_dlambda
        span = extent * dl.max()
        offs = np.arange(-span, span + step, step)
        prof = np.exp(-np.log(2) * np.abs(2.0 * offs[None, :] / fwhm[:, None]) ** SUPERGAUSS_P)
        prof[prof < 1e-4] = 0.0
        out[b] = (lc, offs, prof)
    out["_path"] = "SSDC R7 Table 2 (generated profiles)"
    return out


def resolve_response(response_file):
    """"auto" -> R7 CSV delivery if found, else profiles generated from R7 Table 2;
    "table2" -> the Table-2 profiles; "model" -> None (same shape, R interpolated between
    channels); a path -> that CSV file/folder; a dict -> itself."""
    if isinstance(response_file, dict):
        return response_file
    if response_file in (None, "", "model"):
        return None
    if response_file in ("table2", "builtin"):
        return r7_table_profiles()
    if response_file == "auto":
        rf = find_ssdc_r7()
        if rf:
            return load_response_table(rf)
        print("[INFO] Using SSDC R7 response profiles generated from Table 2 of the R7 note "
              "(lambda_c and R of all 102 channels, flat-top shape).")
        return r7_table_profiles()
    return load_response_table(response_file)


# Where the SSDC R7 spectral-calibration delivery is looked for when response_file="auto"
# (the default).  Put the unpacked delivery (the folder containing MiddlePixel/) next to this
# script or in the working directory, or set SPHEREX_SSDC_R7=/path/to/SSDC_R7.
SSDC_R7_NAMES = ("SSDC_R7", "SSDC R7", "SSDC_R7_spectral", "SPHERExSpectralFiles", "ssdc_r7")


def _calib_search_roots():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    home = os.path.expanduser("~")
    bases = [os.getcwd(), here, os.path.dirname(here), os.path.dirname(os.getcwd()),
             os.path.join(home, "Desktop"), os.path.join(home, "Downloads"), home]
    return list(dict.fromkeys(b for b in bases if b and os.path.isdir(b)))


# members of the delivery that the cube builder needs (fiducial curves + parameter maps)
SSDC_TAR_MEMBERS = ("MiddlePixel/SubChannel17", "FidSpecSummary/SubChannel17", "ParameterFiles")


def _extract_ssdc_tar(tar_path, dest_root):
    """Unpack the parts of SPHERExSpectralFiles_*.tar the builder uses into dest_root/<tar stem>."""
    import tarfile
    out = os.path.join(dest_root, os.path.splitext(os.path.basename(tar_path))[0])
    if os.path.isdir(os.path.join(out, "MiddlePixel")):
        return out
    print(f"[INFO] Unpacking the SPHEREx spectral calibration {tar_path} -> {out} (once) ...")
    os.makedirs(out, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers()
                   if any(m.name.lstrip("./").startswith(p) for p in SSDC_TAR_MEMBERS)
                   and not m.issym() and not m.islnk() and ".." not in m.name]
        tf.extractall(out, members=members)
    return out


def find_ssdc_r7():
    """Return the SSDC R7 spectral-calibration folder (SPHERExSpectralFiles_202607 or similar),
    or None (then the built-in R7 Table-2 model is used).  Looked for in $SPHEREX_SSDC_R7, the
    working directory, this script's folder, their parents, ~/Desktop, ~/Downloads and ~; a
    SPHERExSpectralFiles_*.tar found there is unpacked (needed members only) next to this script."""
    import glob
    env = os.environ.get("SPHEREX_SSDC_R7")
    cands = [env] if env else []
    tars = []
    for base in _calib_search_roots():
        cands += [os.path.join(base, n) for n in SSDC_R7_NAMES]
        cands += sorted(glob.glob(os.path.join(base, "SPHERExSpectralFiles*")))
        cands += sorted(glob.glob(os.path.join(base, "*", "SPHERExSpectralFiles*")))
        cands += glob.glob(os.path.join(base, "*", "MiddlePixel", ".."))
    for r in cands:
        if r and r.endswith(".tar") and os.path.isfile(r):
            tars.append(r)
            continue
        if r and os.path.isdir(r) and glob.glob(os.path.join(r, "**", "FiducialSpec*.csv"),
                                                recursive=True):
            return os.path.normpath(r)
    if env and env.endswith(".tar") and os.path.isfile(env):
        tars.insert(0, env)
    if tars:
        here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
        try:
            return _extract_ssdc_tar(sorted(tars)[-1], here)
        except Exception as e:
            print(f"[WARNING] Could not unpack {tars[-1]}: {e}")
    return None


# ---- per-pixel calibration: band-centre (BC) and band-width (BW) maps ----
USE_BC_MAPS = True            # per-pixel wavelengths / bandwidths from the R7 parameter maps
APPLY_MWIR_SHIFT = True       # if the maps cannot be used: R7 band-centre correction of the
MWIR_BC_SHIFT_UM = 0.0064     # WCS wavelengths in bands 5-6 (+6.4 nm, R7 delivery note Sect. 4)
BC_MATCH_TOL_UM = 0.02        # map vs WCS: |median difference| allowed when validating the
BC_MATCH_SCATTER_UM = 0.004   # detector coordinates, and the allowed robust scatter
_PARAM_MAPS = {}


def load_param_map(kind, band, root=None):
    """R7 parameter map ('BC' band centre or 'BW' band width, um) of detector `band`, 2040x2040,
    or None when the delivery / file is not available.  Cached."""
    import glob
    key = (kind, int(band))
    if key in _PARAM_MAPS:
        return _PARAM_MAPS[key]
    root = root or find_ssdc_r7()
    arr = None
    if root:
        f = sorted(glob.glob(os.path.join(root, "**", f"*SSDC_{kind}_Band{int(band)}.fits"),
                             recursive=True))
        if not f:   # maps may sit in another copy of the delivery (e.g. only one has ParameterFiles)
            for base in _calib_search_roots():
                f = sorted(glob.glob(os.path.join(base, "*SPHERExSpectralFiles*", "**",
                                                  f"*SSDC_{kind}_Band{int(band)}.fits"), recursive=True)
                           + glob.glob(os.path.join(base, "*", "*SPHERExSpectralFiles*", "**",
                                                    f"*SSDC_{kind}_Band{int(band)}.fits"), recursive=True))
                if f:
                    break
        if f:
            with fits.open(f[0]) as h:
                d = next(x.data for x in h if x.data is not None and np.ndim(x.data) == 2)
            arr = np.asarray(d, dtype=np.float32)
            if np.nanmedian(arr) > 100:                     # nm -> um
                arr = arr / 1000.0
    _PARAM_MAPS[key] = arr
    return arr


def detector_coords(header, shape, wave_map, band):
    """Detector pixel coordinates (X, Y), 0-based, of every pixel of a cutout, VERIFIED against
    the image's spectral WCS: the R7 band-centre map sampled there must agree with the WCS
    wavelengths (median offset < BC_MATCH_TOL_UM, robust scatter < BC_MATCH_SCATTER_UM).
    Candidates: the alternate WCS 'A' (parent-image pixel coordinates of IRSA cutouts, 0- or
    1-based), IRAF LTV1/LTV2 offsets, and the image itself if it is a full 2040x2040 frame.
    Returns (X, Y, median offset BC - WCS) or None."""
    bc = load_param_map("BC", band)
    if bc is None or wave_map is None:
        return None
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    cands = []
    try:
        if "CTYPE1A" in header:
            wa = WCS(header, key="A", relax=True)
            if not wa.has_celestial and wa.naxis >= 2:
                X, Y = wa.sub(2).wcs_pix2world(xx, yy, 0)
                cands += [("WCS-A", X, Y), ("WCS-A-1", X - 1.0, Y - 1.0)]
    except Exception:
        pass
    if "LTV1" in header and "LTV2" in header:
        cands.append(("LTV", xx - float(header["LTV1"]), yy - float(header["LTV2"])))
    if (ny, nx) == bc.shape:
        cands.append(("full-frame", xx, yy))
    best = None
    sub = (slice(None, None, max(1, ny // 12)), slice(None, None, max(1, nx // 12)))
    for name, X, Y in list(cands) + [(n + "-T", Yc, Xc) for n, Xc, Yc in cands]:
        Xi, Yi = np.rint(X).astype(int), np.rint(Y).astype(int)
        if Xi.min() < 0 or Yi.min() < 0 or Xi.max() >= bc.shape[1] or Yi.max() >= bc.shape[0]:
            continue
        d = (bc[Yi[sub], Xi[sub]] - wave_map[sub]).ravel()
        d = d[np.isfinite(d)]
        if d.size < 10:
            continue
        med = float(np.median(d))
        sc = float(1.4826 * np.median(np.abs(d - med)))
        if abs(med) < BC_MATCH_TOL_UM and sc < BC_MATCH_SCATTER_UM and (best is None or sc < best[3]):
            best = (name, X, Y, sc, med)
    if best is None:
        return None
    return best[1], best[2], best[4]


def _bilinear(img, X, Y):
    x0 = np.clip(np.floor(X).astype(int), 0, img.shape[1] - 2)
    y0 = np.clip(np.floor(Y).astype(int), 0, img.shape[0] - 2)
    fx, fy = X - x0, Y - y0
    return ((1 - fx) * (1 - fy) * img[y0, x0] + fx * (1 - fy) * img[y0, x0 + 1]
            + (1 - fx) * fy * img[y0 + 1, x0] + fx * fy * img[y0 + 1, x0 + 1])


def calibrated_wavelengths(header, wave_map, band, x_src=None, y_src=None):
    """Per-pixel central wavelength and bandwidth maps of one exposure with the R7 calibration.
    Returns (wave_map, bw_map or None, lam_at_source or None, how) where how is
    'BCmap' (parameter maps at verified detector coordinates), 'WCS+MWIR' (WCS wavelengths with
    the +6.4 nm band 5-6 correction) or 'WCS'."""
    if USE_BC_MAPS and band in range(1, 7):
        dc = detector_coords(header, wave_map.shape, wave_map, band)
        if dc is not None:
            X, Y, _ = dc
            bc, bw = load_param_map("BC", band), load_param_map("BW", band)
            wmap = _bilinear(bc, X, Y)
            bmap = _bilinear(bw, X, Y) if bw is not None else None
            lam_src = None
            if x_src is not None:
                lam_src = float(_bilinear(wmap, np.atleast_1d(float(x_src)), np.atleast_1d(float(y_src)))[0])
            return wmap, bmap, lam_src, "BCmap"
    if APPLY_MWIR_SHIFT and band in (5, 6):
        return wave_map + MWIR_BC_SHIFT_UM, None, None, "WCS+MWIR"
    return wave_map, None, None, "WCS"


def check_calibration(l2_path_or_url, x=None, y=None):
    """Diagnostic for one SPHEREx L2 image or cutout: which wavelength source the builder would
    use (BC maps or WCS), the map-vs-WCS offset, and the wavelength at (x, y)."""
    with fits.open(l2_path_or_url) as hdul:
        h = hdul["IMAGE"].header
        det = str(h.get("DETECTOR", h.get("BAND", "")))
        band = int(det[-1]) if det[-1:].isdigit() else None
        ny, nx = hdul["IMAGE"].data.shape
        sw = WCS(h, fobj=hdul, key="W", relax=True)
        sw.sip = None
        yy, xx = np.indices((ny, nx))
        wl = sw.pixel_to_world(xx, yy)[0].to(u.micron).value
    if band in (None, -1):
        band = int(band_of_wavelength(np.nanmedian(wl)))
    dc = detector_coords(h, (ny, nx), wl, band)
    print(f"band {band}: WCS wavelengths {np.nanmin(wl):.4f}-{np.nanmax(wl):.4f} um")
    print(f"  R7 calibration folder: {find_ssdc_r7()}")
    if dc is None:
        print("  detector coordinates NOT verified -> WCS wavelengths"
              + (f" + {MWIR_BC_SHIFT_UM * 1e3:.1f} nm (bands 5-6)" if band in (5, 6) and APPLY_MWIR_SHIFT else ""))
    else:
        print(f"  BC maps used; median BC - WCS = {dc[2] * 1e3:+.2f} nm")
    return dc


def load_response_table(path):
    """
    Read fiducial channel responses into {band: (centres[nch], offsets[noff], prof[nch, noff])}.

    Accepted layouts (FITS table or CSV/ECSV/ASCII readable by astropy Table):
      * a wavelength column (name starting 'wav' or 'lam', um) plus one response column per
        channel (any other numeric column), or
      * a 2-D FITS image, response[channel, wavelength], with the wavelength vector in a
        second HDU / a 'WAVELENGTH' column / the header axis 1 WCS.
    Each channel is peak-normalised; its centre is the 50 % cumulative point (paper Eq. 3) and its
    band is taken from that centre.  Responses below 1e-3 of the peak, and anything further than
    4 x lambda_c/R from the centre (out-of-band leaks in bands 1-2), are set to zero.
    """
    from astropy.table import Table
    import glob
    if os.path.isdir(path):
        return _load_response_dir(path)
    wave, resp = None, None
    try:
        with fits.open(path) as hl:
            for h in hl:
                if isinstance(h, fits.ImageHDU) or (isinstance(h, fits.PrimaryHDU) and h.data is not None):
                    if h.data is not None and h.data.ndim == 2:
                        resp = np.asarray(h.data, float)
                        hdr = h.header
                elif isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                    names = [c.lower() for c in h.columns.names]
                    for c in h.columns.names:
                        if c.lower().startswith(("wav", "lam")):
                            wave = np.asarray(h.data[c], float).ravel()
                    if resp is None and wave is not None and len(names) > 1:
                        cols = [c for c in h.columns.names if not c.lower().startswith(("wav", "lam"))]
                        resp = np.array([np.asarray(h.data[c], float) for c in cols])
            if resp is not None and wave is None:
                n = resp.shape[1]
                wave = hdr.get("CRVAL1", 0) + (np.arange(n) + 1 - hdr.get("CRPIX1", 1)) * hdr.get("CDELT1", 1)
    except OSError:
        t = Table.read(path)
        wcol = [c for c in t.colnames if c.lower().startswith(("wav", "lam"))][0]
        wave = np.asarray(t[wcol], float)
        resp = np.array([np.asarray(t[c], float) for c in t.colnames if c != wcol])
    if resp is None or wave is None:
        raise ValueError(f"Could not find wavelength + response arrays in {path}")
    if resp.shape[0] == wave.size and resp.shape[1] != wave.size:
        resp = resp.T
    if np.nanmedian(wave) > 100:          # nm -> um
        wave = wave / 1000.0

    per_band = {}
    for r in resp:
        r = np.nan_to_num(r, nan=0.0)
        if r.max() <= 0:
            continue
        r = r / r.max()
        cum = np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(wave))
        lc = np.interp(0.5 * cum[-1], cum, 0.5 * (wave[1:] + wave[:-1]))
        b = int(band_of_wavelength(lc))
        R = float(spherex_R(lc, b))
        off = wave - lc
        r = np.where((np.abs(off) <= 4 * lc / R) & (r >= 1e-3), r, 0.0)
        per_band.setdefault(b, []).append((lc, off, r))
    out = {}
    for b, chans in per_band.items():
        chans.sort(key=lambda c: c[0])
        span = max(np.max(np.abs(c[1][c[2] > 0])) for c in chans)
        step = min(np.median(np.diff(c[1])) for c in chans)
        offs = np.arange(-span, span + step, step)
        prof = np.array([np.interp(offs, c[1], c[2], left=0.0, right=0.0) for c in chans])
        out[b] = (np.array([c[0] for c in chans]), offs, prof)
    print(f" Loaded spectral response for bands {sorted(out)} "
          f"({', '.join(f'B{b}: {len(v[0])} ch' for b, v in sorted(out.items()) if b != '_path')}) "
          f"from {path}")
    out["_path"] = str(path)
    return out


def _load_response_dir(path, nsub=17, case="Cent"):
    """
    SSDC R7 spectral-calibration delivery: CSV files
    <root>/MiddlePixel/SubChannel<N>/Band<b>/FiducialSpec_Band<b>_SubCh<XX>_<N>_Cent.csv
    (columns: Wavelength [um], NormTransmission).  `path` may be the delivery root, the
    MiddlePixel folder, a SubChannel<N> folder or a Band<b> folder.  Middle-pixel (Cent) curves
    are used, as recommended for per-pixel work; off-centre pixels are handled by the shift
    in _response_weights.  The 17 nominal sub-channels are used unless only another N is present.
    """
    import glob
    import re
    files = sorted(glob.glob(os.path.join(path, "**", f"*FiducialSpec*{case}*.csv"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(path, "**", "*FiducialSpec*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No FiducialSpec*.csv files under {path}")

    def nsub_of(f):
        m = re.search(r"SubCh\D?(\d+)\D(\d+)", os.path.basename(f))
        return int(m.group(2)) if m else None
    avail = {nsub_of(f) for f in files}
    use_n = nsub if nsub in avail else sorted(a for a in avail if a is not None)[0] if avail - {None} else None
    if use_n is not None:
        files = [f for f in files if nsub_of(f) == use_n]

    per_band = {}
    for f in files:
        try:
            arr = np.genfromtxt(f, delimiter=",", comments="#", invalid_raise=False)
            arr = arr[np.all(np.isfinite(arr[:, :2]), axis=1)]
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[0] < 5:
            continue
        wave, r = arr[:, 0], arr[:, 1]
        o = np.argsort(wave)
        wave, r = wave[o], np.clip(r[o], 0.0, None)
        if np.nanmedian(wave) > 100:
            wave = wave / 1000.0
        if r.max() <= 0:
            continue
        r = r / r.max()
        cum = np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(wave))
        lc = np.interp(0.5 * cum[-1], cum, 0.5 * (wave[1:] + wave[:-1]))
        mb = re.search(r"Band\D?(\d)", os.path.basename(f))
        b = int(mb.group(1)) if mb else int(band_of_wavelength(lc))
        R = float(spherex_R(lc, b))
        off = wave - lc
        r = np.where((np.abs(off) <= 3 * lc / R) & (r >= 2e-3), r, 0.0)   # drop leaks / noise floor
        per_band.setdefault(b, []).append((lc, off, r))

    out = {}
    for b, chans in per_band.items():
        chans.sort(key=lambda c: c[0])
        span = max(np.max(np.abs(c[1][c[2] > 0])) for c in chans)
        step = min(np.median(np.diff(c[1])) for c in chans)
        offs = np.arange(-span, span + step, step)
        prof = np.array([np.interp(offs, c[1], c[2], left=0.0, right=0.0) for c in chans])
        out[b] = (np.array([c[0] for c in chans]), offs, prof)
    if not out:
        raise ValueError(f"Could not read any fiducial response from {path}")
    print(f" Loaded SPHEREx fiducial responses ({case}, {use_n} sub-channels) for bands "
          f"{sorted(out)} from {path}")
    out["_path"] = str(path)
    return out


def _channel_eqw(table, band):
    """Equivalent width (integral of the peak-normalised curve) of each channel's response."""
    key = ("_eqw", band)
    if key not in table:
        centres, offs, prof = table[band]
        trap = getattr(np, "trapezoid", None) or np.trapz
        table[key] = trap(prof, offs, axis=1)
    return table[key]


def _response_weights(lam_s, lam_k, band, kernel_fwhm=1.0, table=None, bw_s=None):
    """T_s(lam_k) for sample wavelengths lam_s (any shape) of one band.  bw_s (same shape, um):
    the samples' own bandwidths from the R7 BW map; the channel curve is then stretched in
    wavelength so its equivalent width equals bw_s (shape from the fiducial curve)."""
    if table is not None and band in table:
        centres, offs, prof = table[band]
        flat = lam_s.ravel()
        W = np.zeros(flat.shape)
        ok = np.isfinite(flat)
        ch = np.clip(np.searchsorted(centres, flat[ok]), 1, len(centres) - 1)
        ch = np.where(np.abs(flat[ok] - centres[ch - 1]) < np.abs(flat[ok] - centres[ch]), ch - 1, ch)
        d = lam_k - flat[ok]
        if bw_s is not None:
            bw = np.asarray(bw_s, float).ravel()[ok]
            eqw = _channel_eqw(table, band)[ch]
            scale = np.where(np.isfinite(bw) & (bw > 0), eqw / bw, 1.0)
            d = d * scale
        w = np.zeros(d.shape)
        for c in np.unique(ch):
            m = ch == c
            w[m] = np.interp(d[m], offs, prof[c], left=0.0, right=0.0)
        W[ok] = w
        return W.reshape(lam_s.shape)
    fw = kernel_fwhm * lam_s / spherex_R(lam_s, band) / _SG_EQW        # FWHM of the flat-top profile
    with np.errstate(invalid="ignore"):
        u = 2.0 * np.abs(lam_k - lam_s) / fw
        return np.where(u <= 2.0, np.exp(-np.log(2) * u ** SUPERGAUSS_P), 0.0)


def band_from_name(name, lam=None):
    """'SPHEREx-D3' -> 3; otherwise the band containing lam."""
    s = str(name).strip()
    if s[-1:].isdigit():
        return int(s[-1])
    return band_of_wavelength(lam)


def band_of_wavelength(lam):
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    out = np.full(lam.shape, -1, dtype=int)
    for b, (lo, hi, _) in SPHEREX_BANDS.items():
        out[(lam >= lo) & (lam < hi)] = b
    out[lam < SPHEREX_BANDS[1][0]] = 1
    out[lam >= SPHEREX_BANDS[6][1]] = 6
    return out if out.size > 1 else int(out[0])


def create_rectified_cube_kernel(
    valid_rows, target_ra, target_dec,
    common_waves=None,
    master_header=None,
    master_ny=None,
    master_nx=None,
    resample_method="drizzle",
    pixfrac=0.6,
    kernel="response",
    kernel_fwhm=1.0,
    min_samples=1,
    response_file="auto",
    fill_gaps=True,
):
    """
    One epoch -> cube on the common wavelength grid by spectral kernel binning.

    Each exposure is resampled onto the master RA/Dec grid (same as v1); its per-pixel
    wavelength map is resampled too.  For node lambda_k in band b, every sample of band b
    gets the weight K((lambda_sample - lambda_k) / (kernel_fwhm * lambda_k / R_b)) and the
    voxel is the weighted mean.  kernel_fwhm is in units of the resolution element.
    kernel="response" instead weights each sample by its own SPHEREx bandpass at lambda_k
    (the SSDC R7 fiducial middle-pixel curves when response_file is given, else a flat-topped
    model with equivalent width kernel_fwhm * lambda_s / R(lambda_s); see load_response_table).
    Voxels with fewer than min_samples samples are NaN.
    Same return values as create_rectified_cube.
    """
    if master_header is None:
        raise ValueError("master_header must be provided; call build_master_spatial_wcs first.")
    resp_table = None
    if kernel == "response":
        resp_table = resolve_response(response_file)
        kfun, ksupp = None, 2.2
    else:
        kfun, ksupp = SPECTRAL_KERNELS[kernel]

    waves = np.array([row["central_wavelength"] for row in valid_rows], dtype=float)
    mjds  = np.array([row["time_bounds_lower"]  for row in valid_rows], dtype=float)
    bands = np.array([row["band"] if "band" in valid_rows.colnames else band_of_wavelength(w)
                      for row, w in zip(valid_rows, waves)], dtype=int)
    bunit = valid_rows[0]["bunit"] if valid_rows[0]["bunit"] else "MJy/sr"
    n_y, n_x = int(master_ny), int(master_nx)
    num_slices = len(common_waves)

    master_wcs = WCS(master_header).celestial
    yy, xx = np.mgrid[0:n_y, 0:n_x]
    sky_out = master_wcs.pixel_to_world(xx.ravel(), yy.ravel())

    n_in = len(valid_rows)
    vals = np.full((n_in, n_y, n_x), np.nan, dtype=np.float32)
    lams = np.full((n_in, n_y, n_x), np.nan, dtype=np.float32)
    have_bw = "bw_hdu" in valid_rows.colnames and any(r["bw_hdu"] is not None for r in valid_rows)
    bws = np.full((n_in, n_y, n_x), np.nan, dtype=np.float32) if have_bw else None
    how = f"{DRIZZLE_ENGINE}, pixfrac={pixfrac}"
    print(f" Resampling {n_in} exposures onto master grid ({n_y}×{n_x}) with {how} …")
    for s, row in enumerate(valid_rows):
        wcs_in = WCS(row["hdu"].header, relax=True).celestial
        xin, yin = wcs_in.world_to_pixel(sky_out)
        vals[s] = drizzle_to_grid(row["hdu"].data, wcs_in, master_wcs, (n_y, n_x), pixfrac=pixfrac)
        lams[s] = resample_to_grid(row["wave_hdu"].data, xin, yin,
                                   method="bilinear").reshape(n_y, n_x)
        if have_bw and row["bw_hdu"] is not None:
            bws[s] = resample_to_grid(row["bw_hdu"].data, xin, yin,
                                      method="bilinear").reshape(n_y, n_x)
    lam_lo = np.nanmin(lams.reshape(n_in, -1), axis=1)
    lam_hi = np.nanmax(lams.reshape(n_in, -1), axis=1)

    if kernel == "response":
        desc = (f"SSDC R7 responses: {resp_table.get('_path')}" if resp_table is not None
                else f"flat-top bandpass, width = {kernel_fwhm} × λ/R(λ) (SSDC R7 Table 2)")
        print(f" Response-weighted binning ({desc}) onto "
              f"{num_slices}-node common wavelength grid …")
    else:
        print(f" Kernel binning ({kernel}, FWHM = {kernel_fwhm} × λ/R) onto "
              f"{num_slices}-node common wavelength grid …")
    cube_data = np.full((num_slices, n_y, n_x), np.nan, dtype=np.float32)
    nsamp = np.zeros((num_slices, n_y, n_x), dtype=np.int16)
    common_mjds = np.full(num_slices, np.nan)
    node_band = band_of_wavelength(common_waves)
    for k, lam_k in enumerate(common_waves):
        b = int(node_band[k])
        if kernel == "response":
            if resp_table is not None and b in resp_table:
                half = 1.15 * float(np.max(np.abs(resp_table[b][1][resp_table[b][2].max(axis=0) > 0])))
            else:
                half = 1.1 * kernel_fwhm * lam_k / float(np.min(SPHEREX_CHANNELS[b][1])) / _SG_EQW
        else:
            fw = kernel_fwhm * lam_k / SPHEREX_BANDS[b][2]      # kernel FWHM in um
            half = ksupp * fw
        cand = np.nonzero((bands == b) & (lam_lo - half <= lam_k) & (lam_hi + half >= lam_k))[0]
        if cand.size == 0:
            continue
        D = vals[cand].astype(np.float64)
        if kernel == "response":
            W = _response_weights(lams[cand].astype(np.float64), lam_k, b,
                                  kernel_fwhm=kernel_fwhm, table=resp_table,
                                  bw_s=(bws[cand].astype(np.float64) if have_bw else None))
        else:
            with np.errstate(invalid="ignore"):
                W = kfun((lams[cand].astype(np.float64) - lam_k) / fw)
        ok = np.isfinite(D) & np.isfinite(W)
        W = np.where(ok, W, 0.0)
        S0 = W.sum(axis=0)
        cnt = (W > 0).sum(axis=0)
        good = (S0 > 0) & (cnt >= min_samples)
        with np.errstate(invalid="ignore", divide="ignore"):
            f = (W * np.where(ok, D, 0.0)).sum(axis=0) / S0
        cube_data[k][good] = f[good]
        nsamp[k] = cnt
        wsum = W.reshape(cand.size, -1).sum(axis=1)
        if wsum.sum() > 0:
            common_mjds[k] = np.sum(wsum * mjds[cand]) / wsum.sum()

    n_filled = 0
    if fill_gaps:
        # As in v1: each pixel's spectrum is PCHIP-interpolated across wavelengths where it has
        # no samples (saturated star cores, flagged pixels).  Filled voxels keep NSAMP = 0, so
        # (finite value & NSAMP == 0) marks interpolated, not measured, voxels.
        finite = np.isfinite(cube_data)
        need = (~finite).any(axis=0) & (finite.sum(axis=0) >= 3)
        for yy, xx in zip(*np.nonzero(need)):
            ok = finite[:, yy, xx]
            f = PchipInterpolator(common_waves[ok], cube_data[ok, yy, xx].astype(float),
                                  extrapolate=False)
            v = f(common_waves[~ok])
            cube_data[~ok, yy, xx] = v
            n_filled += int(np.isfinite(v).sum())
        print(f" Spectral gap filling (PCHIP, as v1): {n_filled} voxels "
              f"({100 * n_filled / cube_data.size:.2f}%) interpolated; NSAMP = 0 marks them.")

    out_header = master_header.copy()
    out_header["BUNIT"]   = bunit
    out_header["WCSAXES"] = 3
    out_header["NAXIS"]   = 3
    out_header["NAXIS3"]  = num_slices
    out_header["CTYPE3"]  = "WAVE"
    out_header["CRPIX3"]  = 1.0
    out_header["CRVAL3"]  = float(common_waves[0])
    out_header["CDELT3"]  = float(common_waves[1] - common_waves[0])
    out_header["CUNIT3"]  = "um"
    out_header["RESAMPLE"] = (resample_method, "spatial resampling method")
    if resample_method == "drizzle":
        out_header["PIXFRAC"] = (float(pixfrac), "drizzle drop size (input pixels)")
        out_header["DRZENGIN"] = (DRIZZLE_ENGINE, "spatial resampling engine")
    out_header["SPECMETH"] = ("kernel", "spectral binning (v2)")
    out_header["KERNEL"]   = (kernel, "spectral kernel")
    out_header["KFWHM"]    = (float(kernel_fwhm), "kernel FWHM in units of lambda/R")
    out_header["GAPFILL"]  = (bool(fill_gaps), "spectral PCHIP fill of empty voxels (v1-like)")
    out_header["NFILLED"]  = (int(n_filled), "voxels filled by interpolation (NSAMP=0)")
    srcs = sorted({str(r["wave_src"]) for r in valid_rows}) if "wave_src" in valid_rows.colnames else ["WCS"]
    out_header["WAVESRC"] = ("+".join(srcs)[:60], "pixel wavelengths: BCmap (R7 maps) / WCS(+MWIR)")
    out_header["MWIRSHFT"] = (float(MWIR_BC_SHIFT_UM if "WCS+MWIR" in srcs else 0.0),
                              "um added to WCS wavelengths of bands 5-6")
    out_header["BWMAP"] = (bool(have_bw), "per-pixel bandwidth from the R7 BW map")
    if kernel == "response":
        out_header["RESPONSE"] = (("r7_table2" if "Table 2" in str(resp_table.get("_path")) else "ssdc_r7_cent")
                                  if resp_table is not None else "supergauss",
                                  "bandpass model (SSDC R7 / Hui+2026)")
        rpath = (resp_table.get("_path") if isinstance(resp_table, dict) else None)
        if rpath:
            out_header["RESPFILE"] = (os.path.basename(str(rpath))[:60], "response table")
    return cube_data, out_header, common_waves, common_mjds, nsamp


# -------------------------------------------------
# Pure median stack.  Because all epoch cubes are
# built on the same master grid + wavelength grid they
# have identical shapes (NaN-padding kept as a safety net).
# -------------------------------------------------
def median_stack_epochs(epoch_cubes, min_epochs=2):
    """
    Parameters
    ----------
    epoch_cubes : list of 3-D float32 arrays, each (n_wave, ny, nx).
    min_epochs  : voxels with fewer finite epochs → NaN in output.

    Returns
    -------
    median_cube : (n_wave, ny, nx) float32
    mad_cube    : (n_wave, ny, nx) float32  — MAD-based uncertainty
    n_good_cube : (n_wave, ny, nx) int16
    """
    if len(epoch_cubes) == 0:
        raise ValueError("No epoch cubes provided for stacking.")

    n_ep   = len(epoch_cubes)
    shapes = [c.shape for c in epoch_cubes]
    n_wave = shapes[0][0]
    ny_max = max(s[1] for s in shapes)
    nx_max = max(s[2] for s in shapes)

    if any(s[0] != n_wave for s in shapes):
        raise ValueError(
            f"Epoch cubes have inconsistent wavelength axes: {[s[0] for s in shapes]}"
        )

    if len(set(shapes)) > 1:
        # Should not happen when the master grid is used, but handle gracefully
        print(f" [WARN] Unexpected shape mismatch {[s[1:] for s in shapes]}; "
              f"NaN-padding to ({ny_max}, {nx_max}).")

    stack = np.full((n_ep, n_wave, ny_max, nx_max), np.nan, dtype=np.float64)
    for ep_idx, cube in enumerate(epoch_cubes):
        _, ny, nx = cube.shape
        stack[ep_idx, :, :ny, :nx] = cube.astype(np.float64)

    print(f" Median-stacking {n_ep} aligned epoch cubes → "
          f"({n_wave}, {ny_max}, {nx_max}), min {min_epochs} valid epochs per voxel …")

    median_cube = np.full((n_wave, ny_max, nx_max), np.nan, dtype=np.float32)
    mad_cube    = np.full((n_wave, ny_max, nx_max), np.nan, dtype=np.float32)
    n_good_cube = np.zeros((n_wave, ny_max, nx_max),          dtype=np.int16)

    MAD_TO_SIGMA = 1.4826

    for k in range(n_wave):
        plane  = stack[:, k, :, :]
        n_good = np.sum(np.isfinite(plane), axis=0)
        valid  = n_good >= min_epochs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN voxels
            med = np.nanmedian(plane, axis=0)
            mad = np.nanmedian(np.abs(plane - med[np.newaxis, :, :]), axis=0)
        unc = np.where(
            n_good > 0,
            1.253 * MAD_TO_SIGMA * mad / np.sqrt(np.maximum(n_good, 1)),
            np.nan
        )

        median_cube[k][valid] = med[valid].astype(np.float32)
        mad_cube[k][valid]    = unc[valid].astype(np.float32)
        n_good_cube[k]        = n_good.astype(np.int16)

    n_total = n_wave * ny_max * nx_max
    n_nan   = np.sum(~np.isfinite(median_cube))
    print(f" Stack done.  Valid voxels: {n_total - n_nan}/{n_total} "
          f"({100*(n_total - n_nan)/n_total:.1f}%)")

    return median_cube, mad_cube, n_good_cube


# -------------------------------------------------
# Cluster Observations into Epochs via Hierarchical/Density Clustering
# -------------------------------------------------
def assign_epochs_clustered(table, max_gap_days=14.0, method="hierarchical"):
    mjds = np.array(table["time_bounds_lower"], dtype=float)
    valid_mask = ~np.isnan(mjds)
    epochs = np.full(len(table), -1, dtype=int)

    if np.sum(valid_mask) > 0:
        valid_mjds = mjds[valid_mask].reshape(-1, 1)

        if len(valid_mjds) == 1:
            cluster_labels = np.array([1])
        elif method == "dbscan" and HAS_SKLEARN:
            db = DBSCAN(eps=max_gap_days, min_samples=2).fit(valid_mjds)
            cluster_labels = db.labels_
            if np.any(cluster_labels >= 0):
                cluster_labels[cluster_labels >= 0] += 1
        else:
            Z = linkage(valid_mjds, method="single")
            cluster_labels = fcluster(Z, t=max_gap_days, criterion="distance")

        epochs[valid_mask] = cluster_labels

    table["epoch"] = epochs
    return table

# -------------------------------------------------
# Diagnostic Plot: Epoch & MJD Distribution
# -------------------------------------------------
def plot_epoch_distribution(table, target_name):
    os.makedirs("./spectra", exist_ok=True)
    valid = table[~np.isnan(table["time_bounds_lower"])].copy()
    if len(valid) == 0:
        print("[WARNING] No valid timestamps available for epoch distribution plot.")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    epochs = valid["epoch"]
    unique_epochs = np.unique(epochs)
    cmap = matplotlib.colormaps.get_cmap("tab10")
    n_colors = max(len(unique_epochs), 1)

    for i, ep in enumerate(unique_epochs):
        ep_mask = epochs == ep
        sub = valid[ep_mask]
        label = f"Epoch {ep}" if ep != -1 else "Unclustered / Noise"
        color = "gray" if ep == -1 else cmap(i % n_colors)
        ax1.scatter(
            sub["time_bounds_lower"], sub["central_wavelength"],
            c=[color], label=f"{label} (N={len(sub)})",
            alpha=0.85, edgecolors="k", s=45
        )

    ax1.set_ylabel("Central Wavelength (µm)", fontsize=10)
    ax1.set_title(f"Observation Epoch Clusters — {target_name}",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
               borderaxespad=0., fontsize=9)

    mjds = valid["time_bounds_lower"]
    ax2.hist(mjds, bins=max(15, len(unique_epochs) * 5),
             color="lightskyblue", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Time (MJD)", fontsize=10)
    ax2.set_ylabel("Cutout Count", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_plot = os.path.join("spectra", f"{target_name}_epoch_distribution.png")
    plt.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved Epoch Distribution Plot: {out_plot}")


# -------------------------------------------------
# Astrometry-quality filter (FINAST header keyword)
# -------------------------------------------------
def drop_poor_astrometry(rows, label=""):
    """Keep exposures with FINAST == 0 (fine-astrometry rms < 1").
    Exposures whose header has no FINAST keyword are kept."""
    fin = np.array(rows["finast"], dtype=float)
    if not np.any(np.isfinite(fin)):
        print(f"[INFO] {label}: no FINAST keyword in headers; astrometry cut skipped.")
        return rows
    bad = np.isfinite(fin) & (fin != 0)
    if bad.any():
        print(f"[INFO] {label}: dropping {bad.sum()}/{len(rows)} exposures with "
              f"FINAST != 0 (astrometric rms > 1\").")
    return rows[~bad]


# -------------------------------------------------
# Main Datacube Builder
# -------------------------------------------------
def make_datacubes(
    ra, dec, size, target,
    color_by_wavelength=False,
    subtract_zodi=True,
    mask_flags=True,
    gap_days=14.0,
    min_epochs_for_stack=2,
    wave_oversample=1.0,
    min_cutouts_per_epoch=100,
    pixel_scale_arcsec=6.15,
    resample_method="drizzle",
    require_good_astrometry=True,
    bad_flags=DEFAULT_BAD_FLAGS,
    pixfrac=0.6,
    spectral_method="kernel",
    kernel="response",
    kernel_fwhm=1.0,
    min_samples=1,
    response_file="auto",
    fill_gaps=True,
):
    """spectral_method: "kernel" (v2: per-epoch kernel binning on the common grid) or
    "pchip" (v1: per-pixel PCHIP interpolation).  kernel: "response" (default; SSDC R7 SPHEREx
    bandpass weighting, response_file="auto" finds the R7 delivery or falls back to the built-in
    R7 model) or "tent" (triangle of FWHM kernel_fwhm * lambda/R);
    kernel_fwhm in units of the resolution element lambda/R.  The common grid is uniform
    with step = median exposure spacing / wave_oversample (wave_oversample > 1 subsamples)."""
    ra  = ra  * u.deg if not isinstance(ra,  u.Quantity) else ra
    dec = dec * u.deg if not isinstance(dec, u.Quantity) else dec

    if resample_method not in RESAMPLE_METHODS:
        raise ValueError(f"resample_method must be one of {RESAMPLE_METHODS}")
    if resample_method == "drizzle" and pixfrac + pixel_scale_arcsec / 6.15 < 1.2:
        print(f"[WARNING] pixfrac={pixfrac} with {pixel_scale_arcsec}\" pixels can leave "
              f"empty pixels in single exposures; use pixfrac >= "
              f"{1.2 - pixel_scale_arcsec / 6.15:.2f}.")
    bad_flag_mask = flag_bitmask(bad_flags)
    resp_obj = None
    if spectral_method == "kernel" and kernel == "response":
        resp_obj = resolve_response(response_file)

    results      = get_data(ra, dec, size)
    center_coord = SkyCoord(ra=ra, dec=dec)
    table        = results.to_table()

    table["cutout_index"]       = np.arange(1, len(table) + 1)
    table["central_wavelength"] = np.full(len(table), np.nan)
    table["finast"]             = np.full(len(table), np.nan)
    table["band"]               = np.full(len(table), -1, dtype=int)
    table["bunit"]              = np.full(len(table), "MJy/sr", dtype=object)
    table["hdu"]                = np.full(len(table), None, dtype=object)
    table["wave_hdu"]           = np.full(len(table), None, dtype=object)
    table["bw_hdu"]             = np.full(len(table), None, dtype=object)
    table["wave_src"]           = np.full(len(table), "", dtype=object)

    print(" Downloading cutouts …")
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                process_cutout_return, row, ra, dec,
                subtract_zodi=subtract_zodi, mask_flags=mask_flags,
                bad_flag_mask=bad_flag_mask,
            ): i
            for i, row in enumerate(table)
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            r   = future.result()
            idx = r["cutout_index"] - 1
            table["central_wavelength"][idx] = r["central_wavelength"]
            table["finast"][idx]             = r["finast"]
            table["band"][idx]               = r["band"]
            table["bunit"][idx]              = r.get("bunit", "MJy/sr")
            table["hdu"][idx]                = r["hdu"]
            table["wave_hdu"][idx]           = r["wave_hdu"]
            table["bw_hdu"][idx]             = r.get("bw_hdu")
            table["wave_src"][idx]           = r.get("wave_src", "")
            table["time_bounds_lower"][idx]  = r["time_bounds_lower"]
            table["uri"][idx]                = r["uri"]
            if i % 10 == 0 or i == len(table):
                print(f"   → {i}/{len(table)} downloaded "
                      f"({(time.time() - t_start)/60:.2f} min)")

    valid = table[~np.isnan(table["central_wavelength"])]
    if len(valid) == 0:
        raise RuntimeError("No valid cutouts retrieved.")
    srcs, nsrc = np.unique(np.asarray(valid["wave_src"], dtype=str), return_counts=True)
    print("[INFO] Wavelength calibration of the exposures: "
          + ", ".join(f"{s or '?'} x{n}" for s, n in zip(srcs, nsrc))
          + "  (BCmap = R7 band-centre/width maps; WCS+MWIR = WCS + 6.4 nm in bands 5-6)")

    valid = assign_epochs_clustered(valid, max_gap_days=gap_days)
    plot_epoch_distribution(valid, target)

    unique_epochs = [e for e in np.unique(valid["epoch"]) if e != -1]
    print(f"[INFO] Detected {len(unique_epochs)} epoch cluster(s): {list(unique_epochs)}")

    # ── Step 1: keep only epochs with >min_cutouts_per_epoch cutouts ──────
    epoch_row_lists = []
    for ep in unique_epochs:
        ep_rows = valid[valid["epoch"] == ep].copy()
        ep_rows.sort("central_wavelength")
        n_ep = len(ep_rows)
        if n_ep <= min_cutouts_per_epoch:
            print(f"[INFO] Skipping Epoch {ep}: only {n_ep} cutouts "
                  f"(need >{min_cutouts_per_epoch}).")
            continue
        if n_ep < 3:
            print(f"[INFO] Skipping Epoch {ep}: fewer than 3 valid slices.")
            continue
        epoch_row_lists.append(ep_rows)

    if not epoch_row_lists:
        raise RuntimeError(
            f"No epochs survived the >{min_cutouts_per_epoch}-cutout filter; "
            "cannot proceed."
        )
    print(f"[INFO] {len(epoch_row_lists)} epoch(s) pass the "
          f">{min_cutouts_per_epoch}-cutout threshold.")

    # ── Step 1b: drop exposures with poor astrometry (they smear the stack) ─
    if require_good_astrometry:
        kept = []
        for ep_rows in epoch_row_lists:
            ep = ep_rows["epoch"][0]
            ep_rows = drop_poor_astrometry(ep_rows, label=f"Epoch {ep}")
            if len(ep_rows) < 3:
                print(f"[INFO] Skipping Epoch {ep}: fewer than 3 exposures left "
                      "after the astrometry cut.")
                continue
            kept.append(ep_rows)
        epoch_row_lists = kept
        if not epoch_row_lists:
            raise RuntimeError("No epochs left after the astrometry cut.")

    # ── Step 2: ONE common wavelength grid across all qualifying epochs ────
    common_waves = build_common_wave_grid(epoch_row_lists,
                                          oversample=wave_oversample)

    # ── Step 3: ONE master RA/Dec grid (north-up, centred on the target) ──
    first_bunit = epoch_row_lists[0][0]["bunit"] or "MJy/sr"
    master_header, master_ny, master_nx = build_master_spatial_wcs(
        center_coord, size.to(u.arcsec).value,
        pixel_scale_arcsec=pixel_scale_arcsec, bunit=first_bunit,
    )

    # ── Step 4: build per-epoch cubes, all on master grid + common_waves ──
    epoch_cube_data    = []
    epoch_mjds_list    = []
    combined_3d_header = None

    created_cubes = []

    for ep_rows in epoch_row_lists:
        ep = ep_rows["epoch"][0]
        print(f"\n--- Epoch {ep}: building cube on master grid "
              f"({len(ep_rows)} cutouts) ---")

        e_nsamp = None
        if spectral_method == "kernel":
            e_data, e_head, _, e_mjds, e_nsamp = create_rectified_cube_kernel(
                ep_rows, center_coord.ra, center_coord.dec, common_waves=common_waves,
                master_header=master_header, master_ny=master_ny, master_nx=master_nx,
                resample_method=resample_method, pixfrac=pixfrac,
                kernel=kernel, kernel_fwhm=kernel_fwhm, min_samples=min_samples,
                response_file=resp_obj, fill_gaps=fill_gaps)
        else:
            e_data, e_head, _, e_mjds = create_rectified_cube(
                ep_rows,
                center_coord.ra,
                center_coord.dec,
                common_waves=common_waves,
                master_header=master_header,
                master_ny=master_ny,
                master_nx=master_nx,
                resample_method=resample_method,
                pixfrac=pixfrac,
            )

        ep_cube = f"spherex_cube_{target}_epoch{ep}.fits"
        hdus = [fits.PrimaryHDU(data=e_data, header=e_head),
                fits.ImageHDU(data=e_mjds, name="MJD")]
        if e_nsamp is not None:
            hdus.append(fits.ImageHDU(data=e_nsamp, name="NSAMP"))
        fits.HDUList(hdus).writeto(ep_cube, overwrite=True)
        created_cubes.append((ep_cube, f"Epoch {ep}"))
        print(f" Epoch {ep} cube written: {ep_cube}  shape={e_data.shape}")

        epoch_cube_data.append(e_data)
        epoch_mjds_list.append(e_mjds)
        if combined_3d_header is None:
            combined_3d_header = e_head   # same for every epoch

    # ── Step 5: median stack ───────────────────────────────────────────────
    if len(epoch_cube_data) >= min_epochs_for_stack:
        print(f"\n--- Median-stacking {len(epoch_cube_data)} epoch cubes ---")
        median_cube, mad_cube, n_good_cube = median_stack_epochs(
            epoch_cube_data, min_epochs=min_epochs_for_stack
        )

        # Use mean MJD per wavelength plane across epochs
        stacked_mjds = np.nanmean(np.array(epoch_mjds_list), axis=0)

        # Header dimensions already match (all epochs share the master grid)
        combined_3d_header["NAXIS2"] = median_cube.shape[1]
        combined_3d_header["NAXIS1"] = median_cube.shape[2]
        combined_3d_header["NEPOCHS"] = (len(epoch_cube_data), "epochs median-stacked")

        combined_cube = f"spherex_cube_{target}_combined.fits"
        fits.HDUList([
            fits.PrimaryHDU(data=median_cube,    header=combined_3d_header),
            fits.ImageHDU(data=mad_cube,          name="UNCERTAINTY"),
            fits.ImageHDU(data=n_good_cube,       name="NGOOD"),
            fits.ImageHDU(data=stacked_mjds,      name="MJD"),
        ]).writeto(combined_cube, overwrite=True)
        created_cubes.insert(0, (combined_cube, "Combined"))
        print(f" Combined (median-stacked) Datacube written: {combined_cube}")
        print(f"   Extensions: PRIMARY=flux, UNCERTAINTY=σ_MAD, NGOOD=N_epochs, MJD=time")

    else:
        print(f"\n[WARNING] Only {len(epoch_cube_data)} epoch(s) available — "
              f"skipping median stack (need ≥{min_epochs_for_stack}).")
        print(" Falling back to a single cube from all exposures …")
        valid_comb = valid.copy()
        valid_comb.sort("central_wavelength")
        if require_good_astrometry:
            valid_comb = drop_poor_astrometry(valid_comb, label="All cutouts")
        if spectral_method == "kernel":
            cube_data, cube_header, _, comb_mjds, _ = create_rectified_cube_kernel(
                valid_comb, center_coord.ra, center_coord.dec, common_waves=common_waves,
                master_header=master_header, master_ny=master_ny, master_nx=master_nx,
                resample_method=resample_method, pixfrac=pixfrac,
                kernel=kernel, kernel_fwhm=kernel_fwhm, min_samples=min_samples,
                response_file=resp_obj, fill_gaps=fill_gaps)
        else:
            cube_data, cube_header, _, comb_mjds = create_rectified_cube(
                valid_comb,
                center_coord.ra,
                center_coord.dec,
                common_waves=common_waves,
                master_header=master_header,
                master_ny=master_ny,
                master_nx=master_nx,
                resample_method=resample_method,
                pixfrac=pixfrac,
            )
        combined_cube = f"spherex_cube_{target}_combined.fits"
        fits.HDUList([
            fits.PrimaryHDU(data=cube_data, header=cube_header),
            fits.ImageHDU(data=comb_mjds,   name="MJD"),
        ]).writeto(combined_cube, overwrite=True)
        created_cubes.insert(0, (combined_cube, "Combined"))
        print(f" Combined Datacube written: {combined_cube}")

    return created_cubes


# -------------------------------------------------
# Spectral extraction helpers (v2.1)
# -------------------------------------------------
# 1. Recentring: the aperture is centred on the source as SPHEREx sees it (flux-weighted
#    centroid on the wavelength-median image), not on the input coordinates, which can be
#    several arcsec off (proper motion, catalogue epoch, blended pairs).
# 2. Aperture correction (OFF by default; aperture_correct=True to use): the fraction of a point source's light inside the aperture is
#    measured at every wavelength from isolated field stars in the cube (curve of growth out
#    to APCORR_REF_PIX); the aperture flux is divided by it.  Small apertures then give
#    total fluxes with less background noise.
# 3. Background: the annulus is sigma-clipped and other sources in it are masked.
APCORR_REF_PIX = 3.0          # reference radius (pixels, 18.5") treated as the total for point sources
APCORR_SMOOTH = 9             # running median over wavelength nodes for the correction


def _median_image(data_cube):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.asarray(data_cube, float), axis=0)


def _local_peaks(img, edge=8, min_sep=4):
    """Bright, isolated local maxima of an image (x, y, peak S/N), brightest first."""
    from scipy import ndimage
    f = np.nan_to_num(img, nan=np.nanmedian(img))
    smooth = ndimage.median_filter(f, 9)
    resid = f - smooth
    noise = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    peak = (f == ndimage.maximum_filter(f, 2 * min_sep + 1)) & (resid > 10 * noise)
    ny, nx = f.shape
    yy, xx = np.nonzero(peak)
    keep = (xx >= edge) & (xx < nx - edge) & (yy >= edge) & (yy < ny - edge)
    xx, yy = xx[keep], yy[keep]
    snr = resid[yy, xx] / noise
    order = np.argsort(-snr)
    return xx[order], yy[order], snr[order]


def find_source_centroid(data_cube, wcs, skycoord, search_arcsec=12.0, min_snr=5.0):
    """Centroid of the source nearest the input position on the wavelength-median image.
    Returns (SkyCoord, offset_arcsec); the input position if no clear source is found."""
    img = _median_image(data_cube)
    pix = float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
    x0, y0 = [float(v) for v in wcs.world_to_pixel(skycoord)]
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    rr = np.hypot(xx - x0, yy - y0)
    rs = search_arcsec / pix
    ring = (rr > rs + 2) & (rr < rs + 5)
    if not np.isfinite(img[rr <= rs]).any() or not ring.any():
        return skycoord, 0.0
    bg = np.nanmedian(img[ring])
    noise = 1.4826 * np.nanmedian(np.abs(img[ring] - bg))
    sub = np.where(rr <= rs, np.nan_to_num(img - bg), -np.inf)
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    if not (noise > 0 and sub[iy, ix] > min_snr * noise):
        return skycoord, 0.0
    w = np.clip(np.nan_to_num(img - bg), 0, None) * (np.hypot(xx - ix, yy - iy) <= 1.5)
    cx, cy = (w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()
    new = wcs.pixel_to_world(cx, cy)
    return new, float(skycoord.separation(new).arcsec)


def source_mask(img, exclude_xy=None, exclude_r=0.0, nsig=5.0, grow=2):
    """True where other sources sit (to be left out of the background annulus)."""
    from scipy import ndimage
    f = np.nan_to_num(img, nan=np.nanmedian(img))
    resid = f - ndimage.median_filter(f, 9)
    noise = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    m = ndimage.binary_dilation(resid > nsig * noise, iterations=grow)
    if exclude_xy is not None and exclude_r > 0:
        yy, xx = np.mgrid[:f.shape[0], :f.shape[1]]
        m &= np.hypot(xx - exclude_xy[0], yy - exclude_xy[1]) > exclude_r
    return m


def aperture_correction(data_cube, r_ap_pix, avoid_xy=None, ref_pix=APCORR_REF_PIX,
                        max_stars=30, smooth=APCORR_SMOOTH, self_snr_min=30.0):
    """
    Per-wavelength point-source aperture correction (>= 1) from field stars in the cube:
    1 / median over stars of F(r_ap) / F(ref_pix), local background in a masked annulus
    (ref_pix+1 .. ref_pix+3 pixels), smoothed along wavelength.  Returns (corr, source),
    source = number of field stars used, or "target" when fewer than 3 field stars exist
    (small cutouts) and the target itself is bright (S/N > self_snr_min): its own curve of
    growth is then used, which is valid if it is an isolated point source.
    """
    from scipy.ndimage import median_filter
    cube = np.asarray(data_cube, float)
    nw, ny, nx = cube.shape
    if r_ap_pix >= ref_pix:
        return np.ones(nw), 0
    img = _median_image(cube)
    xs, ys, snr = _local_peaks(img, edge=int(ref_pix + 4), min_sep=int(ref_pix))
    if avoid_xy is not None:
        far = np.hypot(xs - avoid_xy[0], ys - avoid_xy[1]) > ref_pix + 3
        xs, ys, snr = xs[far], ys[far], snr[far]
    # drop stars with blank pixels in their core (saturated / masked) in many slices
    ok = [np.isnan(cube[:, y - 1:y + 2, x - 1:x + 2]).mean() < 0.05 for x, y in zip(xs, ys)]
    xs, ys = xs[np.array(ok, bool)][:max_stars], ys[np.array(ok, bool)][:max_stars]
    source = len(xs)
    if len(xs) < 3:
        if avoid_xy is None:
            print(f"[WARNING] Aperture correction: only {len(xs)} usable field stars, not applied.")
            return np.ones(nw), 0
        tx, ty = int(round(avoid_xy[0])), int(round(avoid_xy[1]))
        f = np.nan_to_num(img, nan=np.nanmedian(img))
        yy0, xx0 = np.mgrid[:ny, :nx]
        ring = (np.hypot(xx0 - tx, yy0 - ty) > ref_pix + 1) & (np.hypot(xx0 - tx, yy0 - ty) < ref_pix + 4)
        noise = 1.4826 * np.nanmedian(np.abs(f[ring] - np.nanmedian(f[ring])))
        tsnr = (f[ty, tx] - np.nanmedian(f[ring])) / noise if noise > 0 else 0
        inside = (ref_pix + 3 <= tx < nx - ref_pix - 3) and (ref_pix + 3 <= ty < ny - ref_pix - 3)
        print(f"[INFO] Target peak S/N on the median image: {tsnr:.0f}")
        if tsnr < self_snr_min or not inside:
            print(f"[WARNING] Aperture correction: only {len(xs)} usable field stars and the target "
                  f"is too faint or too near the edge for its own curve of growth; not applied.")
            return np.ones(nw), 0
        print(f"[INFO] Aperture correction from the target's own curve of growth "
              f"(only {len(xs)} field stars in this cutout; assumes an isolated point source).")
        xs, ys, source = np.array([tx]), np.array([ty]), "target"
    srcmask = source_mask(img)
    yy, xx = np.mgrid[:ny, :nx]
    fr = np.full((len(xs), nw), np.nan)
    bgimg = np.nanmedian(img)
    for j, (x, y) in enumerate(zip(xs, ys)):
        # sub-pixel centroid of the star (the aperture fraction is very sensitive to centring)
        wgt = np.clip(np.nan_to_num(img - bgimg), 0, None) * (np.hypot(xx - x, yy - y) <= 1.5)
        xc, yc = (wgt * xx).sum() / wgt.sum(), (wgt * yy).sum() / wgt.sum()
        rr = np.hypot(xx - x, yy - y)
        cy0, cy1, cx0, cx1 = y - int(ref_pix) - 4, y + int(ref_pix) + 5, x - int(ref_pix) - 4, x + int(ref_pix) + 5
        stamp = cube[:, cy0:cy1, cx0:cx1]
        r = rr[cy0:cy1, cx0:cx1]
        ann = (r >= ref_pix + 1) & (r <= ref_pix + 3) & ~srcmask[cy0:cy1, cx0:cx1]
        cx_, cy_ = xc - cx0, yc - cy0
        ap_s = CircularAperture((cx_, cy_), r=r_ap_pix)
        ap_r = CircularAperture((cx_, cy_), r=ref_pix)
        for k in range(nw):
            sl = stamp[k]
            if not np.isfinite(sl[r <= ref_pix]).all():
                continue
            bg = np.nanmedian(sl[ann])
            fs = aperture_photometry(sl - bg, ap_s)["aperture_sum"][0]
            fref = aperture_photometry(sl - bg, ap_r)["aperture_sum"][0]
            if fref > 0:
                fr[j, k] = fs / fref
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        frac = np.nanmedian(fr, axis=0)
    good = np.isfinite(frac)
    if good.sum() < 3:
        return np.ones(nw), 0
    frac = np.interp(np.arange(nw), np.nonzero(good)[0], frac[good])
    frac = median_filter(frac, size=min(smooth, nw), mode="nearest")
    frac = np.clip(frac, 0.2, 1.0)
    return 1.0 / frac, source


def plot_spectrum_from_cubes(cube_list, ra_deg, dec_deg,
                              aperture_arcsec=14.0, wl_plot=2.0,
                              pixel_scale_arcsec=None, recenter=True,
                              aperture_correct=False, mask_sources=True):
    """recenter: centre the aperture on the source centroid in the cube (within 12").
    aperture_correct (default False = off): divide by the point-source fraction inside the aperture, measured at
    each wavelength from field stars (curve of growth to 3 px = 18.5").  Only then does the
    CSV get an `apcorr` column with the factor applied to the annulus-subtracted fluxes.
    mask_sources: leave other sources out of the background annulus.

    pixel_scale_arcsec=None → read the pixel scale from each cube's WCS
    (needed for the MJy/sr → mJy conversion when the output pixel scale changes).

    When two or more epoch cubes are given, a 'median_of_epochs' spectrum is also
    written: the aperture spectrum of each epoch, median-combined per wavelength.
    This is the recommended spectrum — a voxel-by-voxel median cube mixes epochs
    that sampled the (undersampled) PSF at different sub-pixel positions, which
    adds a few % of wavelength-to-wavelength scatter to aperture photometry."""
    os.makedirs("./spectra", exist_ok=True)

    first_cube_path, _ = cube_list[0]
    base   = os.path.basename(first_cube_path)
    target = (base.replace("spherex_cube_", "")
                  .replace("_combined.fits", "")
                  .replace(".fits", ""))

    fig = plt.figure(figsize=(9, 10))
    ax1 = fig.add_subplot(211)

    primary_wcs    = None
    primary_slice  = None
    primary_wl_sel = None
    skycoord = SkyCoord(ra_deg, dec_deg, unit="deg")

    # ── v2.1: recentre, aperture correction and annulus mask, from the first cube ──
    apcorr, src_mask_img = None, None
    with fits.open(cube_list[0][0]) as h0:
        cube0, wcs0 = h0[0].data, WCS(h0[0].header).celestial
        pix0 = float(np.mean(proj_plane_pixel_scales(wcs0)) * 3600.0)
        if recenter and aperture_arcsec > 0:
            new, off = find_source_centroid(cube0, wcs0, skycoord)
            if off > 0:
                print(f"[INFO] Aperture recentred on the SPHEREx centroid: {off:.1f}\" from the "
                      f"input position (RA {new.ra.deg:.6f}, Dec {new.dec.deg:.6f}).")
                skycoord = new
        xy0 = [float(v) for v in wcs0.world_to_pixel(skycoord)]
        if aperture_correct and aperture_arcsec > 0:
            apcorr, n_st = aperture_correction(cube0, aperture_arcsec / pix0, avoid_xy=xy0)
            if n_st == "target" or n_st >= 3:
                src = "the target" if n_st == "target" else f"{n_st} field stars"
                print(f"[INFO] Aperture correction from {src}: "
                      f"{apcorr.min():.3f}-{apcorr.max():.3f} (r = {aperture_arcsec}\" vs "
                      f"{APCORR_REF_PIX * pix0:.1f}\").")
        if mask_sources and aperture_arcsec > 0:
            src_mask_img = source_mask(_median_image(cube0), exclude_xy=xy0,
                                       exclude_r=aperture_arcsec / pix0 + 1.0)

    colors   = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    sigclip  = SigmaClip(sigma=3.0, maxiters=5)
    generated_csvs = []
    epoch_spectra  = []          # (wavelengths, out_data) for every "Epoch N" cube
    all_spectra    = []          # (label, wavelengths, out_data) for every cube

    for idx, (cube_path, label) in enumerate(cube_list):
        hdul       = fits.open(cube_path)
        data_cube  = hdul[0].data
        header     = hdul[0].header

        has_unc  = "UNCERTAINTY" in hdul
        unc_cube = hdul["UNCERTAINTY"].data if has_unc else None

        mjds = (hdul["MJD"].data if "MJD" in hdul
                else np.full(header["NAXIS3"], np.nan))

        n_waves = header["NAXIS3"]
        crval3  = header["CRVAL3"]
        cdelt3  = header["CDELT3"]
        crpix3  = header["CRPIX3"]
        wavelengths = crval3 + (np.arange(1, n_waves + 1) - crpix3) * cdelt3

        wcs = WCS(header).celestial

        # MJy/sr per pixel → mJy, using this cube's actual pixel solid angle
        pix_arcsec = (pixel_scale_arcsec if pixel_scale_arcsec is not None
                      else float(np.mean(proj_plane_pixel_scales(wcs)) * 3600.0))
        omega_pixel       = ((pix_arcsec * u.arcsec).to(u.rad).value) ** 2
        mjy_to_mjy_factor = omega_pixel * 1e9

        fluxes_raw, fluxes_med_sub, fluxes_ann_sub = [], [], []
        bg_medians, bg_annuli, flux_errors          = [], [], []

        if aperture_arcsec <= 0:
            px, py = wcs.world_to_pixel(skycoord)
            px, py = int(np.round(px)), int(np.round(py))
            fluxes_raw     = data_cube[:, py, px] * mjy_to_mjy_factor
            fluxes_med_sub = fluxes_raw
            fluxes_ann_sub = fluxes_raw
            bg_medians     = np.zeros(n_waves)
            bg_annuli      = np.zeros(n_waves)
            flux_errors    = (unc_cube[:, py, px] * mjy_to_mjy_factor
                              if has_unc else np.zeros(n_waves))
            flux_errors    = np.where(np.isfinite(flux_errors), flux_errors, 0.0)
        else:
            sky_ap  = SkyCircularAperture(skycoord, r=aperture_arcsec * u.arcsec)
            sky_ann = SkyCircularAnnulus(
                skycoord,
                r_in=1.5  * aperture_arcsec * u.arcsec,
                r_out=2.5 * aperture_arcsec * u.arcsec,
            )
            ap_pix  = sky_ap.to_pixel(wcs)
            ann_pix = sky_ann.to_pixel(wcs)
            ap_area = ap_pix.area

            for i in range(n_waves):
                sl = np.asarray(data_cube[i, :, :], float)

                raw = aperture_photometry(np.nan_to_num(sl, nan=0.0),
                                          ap_pix)["aperture_sum"][0]
                fluxes_raw.append(raw * mjy_to_mjy_factor)

                bg_med_val = np.nanmedian(sl)
                fluxes_med_sub.append((raw - bg_med_val * ap_area) * mjy_to_mjy_factor)
                bg_medians.append(bg_med_val * ap_area * mjy_to_mjy_factor)

                ann_stats  = ApertureStats(sl, ann_pix, sigma_clip=sigclip,
                                           mask=src_mask_img)
                bg_ann_val = ann_stats.median if np.isfinite(ann_stats.median) else 0.0
                bg_ann_std = ann_stats.std    if np.isfinite(ann_stats.std)    else 0.0
                bg_ann_tot = bg_ann_val * ap_area

                fluxes_ann_sub.append((raw - bg_ann_tot) * mjy_to_mjy_factor)
                bg_annuli.append(bg_ann_tot * mjy_to_mjy_factor)

                if has_unc:
                    unc_sl  = np.asarray(unc_cube[i, :, :], float)
                    unc_sum = aperture_photometry(
                        np.nan_to_num(unc_sl**2, nan=0.0), ap_pix
                    )["aperture_sum"][0]
                    flux_errors.append(np.sqrt(max(unc_sum, 0)) * mjy_to_mjy_factor)
                else:
                    n_ann = float(getattr(ann_stats, "n_pixels",
                                          getattr(ann_stats, "npixels", 1.0)))
                    n_ann = n_ann if n_ann > 0 else 1.0
                    var_bg = ap_area * bg_ann_std**2 * (1.0 + ap_area / n_ann)
                    flux_errors.append(np.sqrt(var_bg) * mjy_to_mjy_factor)

        hdul.close()

        fluxes_raw     = np.array(fluxes_raw,     dtype=float)
        fluxes_med_sub = np.array(fluxes_med_sub, dtype=float)
        fluxes_ann_sub = np.array(fluxes_ann_sub, dtype=float)
        bg_medians     = np.array(bg_medians,     dtype=float)
        bg_annuli      = np.array(bg_annuli,      dtype=float)
        flux_errors    = np.array(flux_errors,    dtype=float)
        corr = (apcorr if (apcorr is not None and apcorr.size == n_waves)
                else np.ones(n_waves))
        fluxes_ann_sub = fluxes_ann_sub * corr
        flux_errors    = flux_errors * corr

        out_data = np.c_[
            wavelengths, mjds, fluxes_raw, fluxes_med_sub,
            fluxes_ann_sub, bg_medians, bg_annuli, flux_errors,
        ]
        all_spectra.append((label, wavelengths, out_data))
        if label.lower().startswith("epoch"):
            epoch_spectra.append((wavelengths, out_data))

        color = colors[idx % len(colors)]
        ax1.errorbar(
            wavelengths, fluxes_ann_sub, yerr=flux_errors,
            fmt="o-", lw=1.2, label=f"{label} (Annulus Sub)",
            color=color, capsize=2,
        )

        if idx == 0:
            i_img = (np.argmin(np.abs(wavelengths - wl_plot))
                     if wl_plot else np.argmax(fluxes_ann_sub))
            primary_slice  = data_cube[i_img, :, :]
            primary_wcs    = wcs
            primary_wl_sel = wavelengths[i_img]

    # ── ONE CSV: combined cube (standard columns), every epoch, median of epoch spectra ──
    names8 = ["MJD", "flux_raw_mJy", "flux_median_sub_mJy", "flux_annulus_sub_mJy",
              "bg_median_total_mJy", "bg_annulus_total_mJy", "flux_err_mJy"]
    ref_label, ref_w, ref_d = all_spectra[0]
    table_out = {"wavelength_um": ref_w}
    for j, nm in enumerate(names8, start=1):
        table_out[nm] = ref_d[:, j]
    if apcorr is not None and apcorr.size == len(ref_w) and np.any(apcorr != 1):
        table_out["apcorr"] = apcorr        # only when aperture_correct=True
    same = lambda w: len(w) == len(ref_w) and np.allclose(w, ref_w)
    for label, w, d in all_spectra[1:]:
        if not same(w):
            print(f"[WARNING] {label}: different wavelength grid, left out of the CSV.")
            continue
        tag = label.lower().replace(" ", "")
        table_out[f"{tag}_MJD"] = d[:, 1]
        table_out[f"{tag}_flux_annulus_sub_mJy"] = d[:, 4]
        table_out[f"{tag}_flux_err_mJy"] = d[:, 7]
    if len(epoch_spectra) >= 2 and all(same(w) for w, _ in epoch_spectra):
        stack = np.array([d for _, d in epoch_spectra])            # (n_epoch, n_wave, 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med  = np.nanmedian(stack[:, :, 4], axis=0)
            n_ok = np.sum(np.isfinite(stack[:, :, 4]), axis=0)
            mad  = np.nanmedian(np.abs(stack[:, :, 4] - med[None]), axis=0)
            sig  = np.fmax(np.nanmedian(stack[:, :, 7], axis=0), 1.4826 * mad)
        table_out["median_of_epochs_flux_mJy"] = med
        table_out["median_of_epochs_err_mJy"] = 1.253 * sig / np.sqrt(np.maximum(n_ok, 1))
        ax1.errorbar(ref_w, med, yerr=table_out["median_of_epochs_err_mJy"],
                     fmt="-", lw=2.0, color="black", capsize=0, zorder=5,
                     label=f"Median of {len(epoch_spectra)} epoch spectra")
    csv_name = os.path.join("spectra", f"{target}_spectra_ap_{aperture_arcsec}.csv")
    cols = list(table_out)
    np.savetxt(csv_name, np.column_stack([table_out[c] for c in cols]), delimiter=",",
               comments="# ", header=(
                   f"SPHEREx Extracted Spectra — Target: {target} "
                   f"(first columns: {ref_label} cube; then each epoch; then median of epoch spectra)\n"
                   f"Units: mJy; annulus-subtracted fluxes "
                   f"{'are aperture-corrected (apcorr)' if (apcorr is not None and np.any(apcorr != 1)) else 'are NOT aperture-corrected'}, "
                   f"aperture centre RA {skycoord.ra.deg:.6f} Dec {skycoord.dec.deg:.6f}\n"
                   + ",".join(cols)))
    generated_csvs.append(csv_name)
    print(f"Spectra CSV written: {csv_name}")

    ax1.set_xlabel("Wavelength (µm)")
    ax1.set_ylabel("Flux Density (mJy)")
    ax1.set_title(f"{target} — Multi-Epoch Extracted Spectra")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    if primary_wcs is not None:
        ax2 = fig.add_subplot(212, projection=primary_wcs)
        masked = np.ma.masked_invalid(primary_slice)
        norm   = (simple_norm(masked.compressed(), "log", percent=99)
                  if masked.count() > 0 else None)
        ax2.imshow(masked, origin="lower", cmap="inferno", norm=norm,
                   interpolation="nearest")

        if aperture_arcsec > 0:
            sky_ap  = SkyCircularAperture(skycoord, r=aperture_arcsec * u.arcsec)
            sky_ann = SkyCircularAnnulus(
                skycoord,
                r_in=1.5  * aperture_arcsec * u.arcsec,
                r_out=2.5 * aperture_arcsec * u.arcsec,
            )
            sky_ap.to_pixel(primary_wcs).plot(
                ax=ax2, color="cyan",   lw=1.5, label="Aperture")
            sky_ann.to_pixel(primary_wcs).plot(
                ax=ax2, color="yellow", lw=1.0, ls="--", label="Annulus")

        ax2.set_title(f"{target} Combined @ {primary_wl_sel:.4g} µm")
        ax2.set_xlabel("RA")
        ax2.set_ylabel("Dec")

    plt.tight_layout()
    png_name = os.path.join("spectra", f"{target}_multiepoch_cutout.png")
    plt.savefig(png_name, dpi=200)
    plt.close(fig)
    print(f"Spectra saved to CSVs (mJy) and plot written: {png_name}")
    return generated_csvs


def validate_spectrum_against_catalogs(csv_path, ra_deg, dec_deg,
                                        radius_arcsec=5.0):
    print(f"\n[INFO] Validating {csv_path} …")

    df_sp = pd.read_csv(csv_path, skiprows=2)
    df_sp.columns = (df_sp.columns
                     .str.replace(r"^#\s*", "", regex=True)
                     .str.strip())

    if "flux_annulus_sub_mJy" in df_sp.columns:
        flux_col, err_col = "flux_annulus_sub_mJy", "flux_err_mJy"
    elif "flux_annulus_sub" in df_sp.columns:
        flux_col, err_col = "flux_annulus_sub", "flux_err"
    else:
        flux_col, err_col = df_sp.columns[4], df_sp.columns[7]

    coord     = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    v         = Vizier(columns=["*", "e_*"])
    res_2mass = v.query_region(coord, radius=radius_arcsec * u.arcsec,
                               catalog="II/246/out")
    res_wise  = v.query_region(coord, radius=radius_arcsec * u.arcsec,
                               catalog="II/328/allwise")

    zeropoints = {
        "2MASS:J":  {"zp": 1594.0, "wl": 1.235, "m": "Jmag",  "em": "e_Jmag",  "res": res_2mass},
        "2MASS:H":  {"zp": 1024.0, "wl": 1.662, "m": "Hmag",  "em": "e_Hmag",  "res": res_2mass},
        "2MASS:Ks": {"zp":  666.7, "wl": 2.159, "m": "Kmag",  "em": "e_Kmag",  "res": res_2mass},
        "WISE:W1":  {"zp":  309.54,"wl": 3.352, "m": "W1mag", "em": "e_W1mag", "res": res_wise},
        "WISE:W2":  {"zp":  171.78,"wl": 4.603, "m": "W2mag", "em": "e_W2mag", "res": res_wise},
    }

    cat_wl, cat_flux, cat_err, cat_labels = [], [], [], []
    for name, cfg in zeropoints.items():
        if len(cfg["res"]) > 0 and len(cfg["res"][0]) > 0:
            row = cfg["res"][0][0]
            if cfg["m"] in row.colnames:
                m  = row[cfg["m"]]
                em = row[cfg["em"]] if cfg["em"] in row.colnames else np.nan
                if not np.isnan(m):
                    f_jy  = cfg["zp"] * 10 ** (-0.4 * m)
                    ef_jy = 0.921 * f_jy * em if not np.isnan(em) else 0.0
                    cat_wl.append(cfg["wl"])
                    cat_flux.append(f_jy  * 1e3)
                    cat_err.append(ef_jy  * 1e3)
                    cat_labels.append(name)

    plt.figure(figsize=(9, 5))
    plt.errorbar(df_sp["wavelength_um"], df_sp[flux_col], yerr=df_sp[err_col],
                 fmt=".-", color="black", alpha=0.75, label="SPHEREx (mJy)")
    if cat_wl:
        plt.errorbar(cat_wl, cat_flux, yerr=cat_err,
                     fmt="ro", capsize=4, label="2MASS / AllWISE Catalog")
        for x, y, lbl in zip(cat_wl, cat_flux, cat_labels):
            plt.annotate(lbl, (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    else:
        print("[WARNING] No 2MASS/WISE matches found within search radius.")

    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Flux Density (mJy)")
    plt.title(f"Photometric Validation — {os.path.basename(csv_path)}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_plot = csv_path.replace(".csv", "_validated.png")
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"Validation plot saved: {out_plot}")


# =================================================
# HELPER FUNCTIONS & MAIN LOOP
# =================================================
def parse_coordinates(ra_val, dec_val):
    try:
        c = SkyCoord(ra=float(ra_val)*u.degree,
                     dec=float(dec_val)*u.degree, frame="icrs")
    except (ValueError, TypeError):
        c = SkyCoord(ra=str(ra_val), dec=str(dec_val),
                     unit=(u.hourangle, u.deg), frame="icrs")
    return c.ra.deg, c.dec.deg

def resolve_simbad(name):
    result = Simbad.query_object(name)
    if result is None or len(result) == 0:
        raise ValueError(f"SIMBAD could not resolve '{name}'")
    return parse_coordinates(result["ra"][0], result["dec"][0])

def process_target(
    ra, dec, name, size, aperture_arcsec,
    color_by_wavelength=False, subtract_zodi=True, mask_flags=True,
    validate_photometry=False, gap_days=14.0, wave_oversample=1.0,
    min_cutouts_per_epoch=100, pixel_scale_arcsec=6.15,
    resample_method="drizzle", require_good_astrometry=True,
    pixfrac=0.6, spectral_method="kernel", kernel="response", kernel_fwhm=1.0,
    response_file="auto", fill_gaps=True,
):
    print(f"\n[INFO] Processing {name} (RA={ra:.6f}, Dec={dec:.6f})")
    try:
        created_cubes = make_datacubes(
            ra, dec, size, name,
            color_by_wavelength=color_by_wavelength,
            subtract_zodi=subtract_zodi,
            mask_flags=mask_flags,
            gap_days=gap_days,
            wave_oversample=wave_oversample,
            min_cutouts_per_epoch=min_cutouts_per_epoch,
            pixel_scale_arcsec=pixel_scale_arcsec,
            resample_method=resample_method,
            require_good_astrometry=require_good_astrometry,
            pixfrac=pixfrac,
            spectral_method=spectral_method,
            kernel=kernel,
            kernel_fwhm=kernel_fwhm,
            response_file=response_file,
            fill_gaps=fill_gaps,
        )
        csv_files = plot_spectrum_from_cubes(
            created_cubes, ra, dec, aperture_arcsec=aperture_arcsec
        )
        if validate_photometry and csv_files:
            for csv_path in csv_files:
                validate_spectrum_against_catalogs(csv_path, ra, dec)
    except Exception as e:
        import traceback
        print(f"[ERROR] {name}: {e}")
        traceback.print_exc()

def validate_radec(ra, dec):
    if not (0.0 <= ra < 360.0):
        raise ValueError(f"Invalid RA: {ra}. Must be 0–360°")
    if not (-90.0 <= dec <= 90.0):
        raise ValueError(f"Invalid Dec: {dec}. Must be −90 – +90°")

if __name__ == "__main__":
    while True:
        print("\n======================================")
        print("SPHEREx 3D Datacube Builder v2 (kernel binning per epoch + median stack)")
        print("======================================")
        print("  1 → CSV catalog")
        print("  2 → Manual RA/Dec [DEFAULT]")
        print("  3 → SIMBAD name resolution")
        print("  4 → Open existing Datacube + extract spectrum")
        print("  5 → Standalone: Validate existing Spectrum CSV vs 2MASS/WISE")
        print("  0 → Exit")

        mode_input = input("Enter mode (0–5) [default=2]: ").strip()
        mode = mode_input or "2"

        if mode not in {"0","1","2","3","4","5"}:
            print(f"[WARNING] Invalid option '{mode}'.")
            continue
        if mode == "0":
            print("Exiting.")
            sys.exit(0)

        try:
            if mode == "5":
                root = tk.Tk(); root.withdraw()
                csv_file = filedialog.askopenfilename(
                    title="Select SPHEREx Spectrum CSV",
                    filetypes=[("CSV files","*.csv"),("All files","*.*")])
                if not csv_file:
                    raise RuntimeError("No file selected")
                ra_in  = input("Target RA (deg or hh:mm:ss): ").strip()
                dec_in = input("Target Dec (deg or dd:mm:ss): ").strip()
                ra, dec = parse_coordinates(ra_in, dec_in)
                validate_radec(ra, dec)
                validate_spectrum_against_catalogs(csv_file, ra, dec)
                continue

            cutout_arcmin = float(input("Cutout size (arcmin) [default=15]: ").strip() or 15)
            aperture_arcsec = float(input("Aperture radius (arcsec) [default=14]: ").strip() or 14)
            gap_days = float(input("Max epoch gap (days) [default=14]: ").strip() or 14)
            wave_oversample = float(input("Wavelength grid oversample (>1 = finer uniform grid) "
                                          "[default=1.0]: ").strip() or 1.0)
            min_cutouts_per_epoch = int(input("Keep epochs with more than N cutouts, N [default=100]: ").strip() or 100)
            pixel_scale_arcsec = float(input("Output pixel scale (arcsec) [default=6.15 = native]: ").strip() or 6.15)
            resample_method = "drizzle"                  # STScI drizzle
            pixfrac = float(input("Drizzle pixfrac (smaller = sharper, noisier) "
                                  "[default=0.6]: ").strip() or 0.6)
            cutout_size = cutout_arcmin * u.arcminute
            spectral_method = (input("Spectral method: kernel (v2) / pchip (v1) [default=kernel]: ")
                               .strip().lower() or "kernel")
            kernel, kernel_fwhm, response_file = "response", 1.0, "auto"
            if spectral_method == "kernel":
                kernel = (input("Spectral kernel: response (SSDC R7 filter response) / tent "
                                "[default=response]: ").strip().lower() or "response")
                if kernel not in ("response",) + tuple(SPECTRAL_KERNELS):
                    print(f"[WARNING] Unknown kernel '{kernel}', using response.")
                    kernel = "response"
                if kernel == "response":
                    response_file = (input("SSDC R7 responses [default=auto: R7 CSV folder if present, "
                                           "else profiles from R7 Table 2 built into this script]: ")
                                     .strip() or "auto")
                kernel_fwhm = float(input("Kernel FWHM in resolution elements lambda/R "
                                          "[default=1.0]: ").strip() or 1.0)

            subtract_zodi = input("Subtract Zodiacal Light? (y/n) [default=y]: ").strip().lower() in ("y","yes","")
            mask_flags    = input("Mask bad-pixel FLAGS (cosmic rays, hot/cold, persistence; never SOURCE)? "
                                  "(y/n) [default=y]: ").strip().lower() in ("y","yes","")
            color_by_wavelength = input("Color by wavelength? (y/n) [default=n]: ").strip().lower() in ("y","yes")
            validate_photometry = input("Validate vs 2MASS/WISE? (y/n) [default=n]: ").strip().lower() in ("y","yes")

            run_kwargs = dict(
                color_by_wavelength=color_by_wavelength,
                subtract_zodi=subtract_zodi, mask_flags=mask_flags,
                validate_photometry=validate_photometry,
                gap_days=gap_days, wave_oversample=wave_oversample,
                min_cutouts_per_epoch=min_cutouts_per_epoch,
                pixel_scale_arcsec=pixel_scale_arcsec,
                resample_method=resample_method,
                pixfrac=pixfrac,
                spectral_method=spectral_method,
                kernel=kernel,
                kernel_fwhm=kernel_fwhm,
                response_file=response_file,
            )

            if mode == "1":
                root = tk.Tk(); root.withdraw()
                csv_file = filedialog.askopenfilename(
                    title="Select Target CSV Catalog",
                    filetypes=[("CSV files","*.csv"),("All files","*.*")])
                if not csv_file:
                    raise RuntimeError("No catalog selected")
                df = pd.read_csv(csv_file)
                df.columns = df.columns.str.strip()
                ra_col   = input("RA column [default=RA]: ").strip()   or "RA"
                dec_col  = input("Dec column [default=Dec]: ").strip() or "Dec"
                name_col = input("Name column [default=Name]: ").strip() or "Name"
                for i, row in df.iterrows():
                    try:
                        ra, dec = parse_coordinates(row[ra_col], row[dec_col])
                        validate_radec(ra, dec)
                        tname = str(row[name_col]).strip() if name_col in df.columns else f"Target_{i+1}"
                        process_target(ra, dec, tname, cutout_size, aperture_arcsec, **run_kwargs)
                    except Exception as row_err:
                        print(f"[WARNING] Skipping row {i}: {row_err}")

            elif mode == "2":
                ra_in  = input("RA (deg) [default=277.479646]: ").strip() or "277.479646"
                dec_in = input("Dec (deg) [default=1.239842]: ").strip()  or "1.239842"
                ra, dec = parse_coordinates(ra_in, dec_in)
                validate_radec(ra, dec)
                target_name = input("Target name [default=Serp_test]: ").strip() or "Serp_test"
                process_target(ra, dec, target_name, cutout_size, aperture_arcsec, **run_kwargs)

            elif mode == "3":
                target_name = input("SIMBAD name: ").strip()
                if not target_name:
                    raise ValueError("Empty target name.")
                ra, dec = resolve_simbad(target_name)
                validate_radec(ra, dec)
                print(f"[INFO] Resolved → RA={ra:.6f}, Dec={dec:.6f}")
                process_target(ra, dec, target_name, cutout_size, aperture_arcsec, **run_kwargs)

            elif mode == "4":
                root = tk.Tk(); root.withdraw()
                cube_file = filedialog.askopenfilename(
                    title="Select SPHEREx Datacube FITS",
                    filetypes=[("FITS files","*.fits"),("All files","*.*")])
                if not cube_file:
                    raise RuntimeError("No file selected")
                ra_in  = input("Target RA: ").strip()
                dec_in = input("Target Dec: ").strip()
                ra, dec = parse_coordinates(ra_in, dec_in)
                validate_radec(ra, dec)
                wl_plot = float(input("Image wavelength (µm) [default=2.0]: ").strip() or 2.0)
                csv_files = plot_spectrum_from_cubes(
                    [(cube_file, "Existing Datacube")], ra, dec,
                    aperture_arcsec=aperture_arcsec, wl_plot=wl_plot)
                if validate_photometry and csv_files:
                    for csv_path in csv_files:
                        validate_spectrum_against_catalogs(csv_path, ra, dec)

        except Exception as e:
            import traceback
            print(f"\n[ERROR] {e}")
            traceback.print_exc()
            print("Returning to main menu …\n")
            continue

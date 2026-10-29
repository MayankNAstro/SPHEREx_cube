### ==============================
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
    SkyCircularAperture,
    SkyCircularAnnulus,
    aperture_photometry,
    ApertureStats,
)
import pyvo
import urllib.error

# ==============================
# GUI / File Dialogs
# ==============================
import tkinter as tk
from tkinter import filedialog

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
def open_fits_with_retry(url, retries=10, base_delay=1, cache=False):
    for attempt in range(1, retries + 1):
        try:
            return fits.open(url, cache=cache)
        except (urllib.error.HTTPError,
                urllib.error.URLError,
                socket.gaierror,
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

            hdu = fits.ImageHDU(data=img_data, header=img_hdu.header.copy())
            hdu.header["EXTNAME"] = f"IMAGE{row['cutout_index']}"
            hdu.header["BUNIT"] = bunit

            wave_hdu = fits.ImageHDU(data=native_wave_map, header=img_hdu.header)

            return {
                "cutout_index": row["cutout_index"],
                "central_wavelength": lam,
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
            "central_wavelength": np.nan,
            "time_bounds_lower": row.get("time_bounds_lower", np.nan),
            "bunit": "MJy/sr",
            "uri": row["uri"],
            "finast": np.nan,
            "hdu": None,
            "wave_hdu": None
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
# Resolution-preserving, NaN-aware image resampling
# -------------------------------------------------
# SPHEREx's PSF (FWHM ~6") is undersampled by its 6.15" pixels.
#  * "drizzle" (default): each input pixel is shrunk to a drop of side `pixfrac`
#    and its flux is shared among the output pixels the drop overlaps.  All
#    weights are positive, so it cannot ring, and it conserves flux, while
#    being sharper than bilinear (simulation: FWHM 6.4" vs 7.5" bilinear).
#  * "lanczos3/lanczos2/cubic": sharpest, but their negative lobes ring around
#    bright undersampled stars (dark moat ~2 px out, 2-8% of the peak).
#  * "bilinear": smooth, no ringing, blurs the PSF by about one pixel.
def _lanczos(t, a):
    t = np.asarray(t, dtype=float)
    return np.where(np.abs(t) < a, np.sinc(t) * np.sinc(t / a), 0.0)


def _keys_cubic(t, a=-0.5):
    t = np.abs(np.asarray(t, dtype=float))
    return np.where(
        t <= 1.0, (a + 2.0) * t**3 - (a + 3.0) * t**2 + 1.0,
        np.where(t < 2.0, a * t**3 - 5.0 * a * t**2 + 8.0 * a * t - 4.0 * a, 0.0))


# name -> (tap offsets relative to floor(x), 1-D kernel)
RESAMPLE_KERNELS = {
    "lanczos3": (np.arange(-2, 4), lambda t: _lanczos(t, 3)),
    "lanczos2": (np.arange(-1, 3), lambda t: _lanczos(t, 2)),
    "cubic":    (np.arange(-1, 3), _keys_cubic),
    "bilinear": (np.array([0, 1]), lambda t: np.clip(1.0 - np.abs(t), 0.0, None)),
}
RESAMPLE_METHODS = ("drizzle",) + tuple(RESAMPLE_KERNELS) + ("nearest",)


def drizzle_to_grid(data, wcs_in, wcs_out, shape_out, pixfrac=0.6, n_sub=4):
    """
    Flux-conserving 'drop' resampling of one image onto the output grid.

    Every finite input pixel is shrunk about its centre to a square drop of
    side `pixfrac` (input pixels), split into n_sub x n_sub sub-points that are
    mapped through both WCSs (SIP included) and binned into output pixels.
    Output = overlap-weighted mean surface brightness; NaN where no drop lands.
    """
    data = np.asarray(data, dtype=np.float64)
    ny, nx = data.shape
    n_out_y, n_out_x = shape_out

    # Output-pixel coordinates of every input pixel CORNER (one WCS call each way)
    yc, xc = np.mgrid[0:ny + 1, 0:nx + 1] - 0.5
    xo, yo = wcs_out.world_to_pixel(wcs_in.pixel_to_world(xc.ravel(), yc.ravel()))
    xo = np.asarray(xo, dtype=np.float64).reshape(ny + 1, nx + 1)
    yo = np.asarray(yo, dtype=np.float64).reshape(ny + 1, nx + 1)

    good = np.isfinite(data)
    vals = data[good]
    corners_x = [c[good] for c in (xo[:-1, :-1], xo[:-1, 1:], xo[1:, :-1], xo[1:, 1:])]
    corners_y = [c[good] for c in (yo[:-1, :-1], yo[:-1, 1:], yo[1:, :-1], yo[1:, 1:])]

    frac = 0.5 + ((np.arange(n_sub) + 0.5) / n_sub - 0.5) * pixfrac   # sub-point offsets in (0, 1)
    acc  = np.zeros(n_out_y * n_out_x)
    wsum = np.zeros(n_out_y * n_out_x)
    for a in frac:            # along x
        for b in frac:        # along y
            w00, w10, w01, w11 = (1 - a) * (1 - b), a * (1 - b), (1 - a) * b, a * b
            X = w00 * corners_x[0] + w10 * corners_x[1] + w01 * corners_x[2] + w11 * corners_x[3]
            Y = w00 * corners_y[0] + w10 * corners_y[1] + w01 * corners_y[2] + w11 * corners_y[3]
            ix, iy = np.rint(X).astype(int), np.rint(Y).astype(int)
            ok = (ix >= 0) & (ix < n_out_x) & (iy >= 0) & (iy < n_out_y)
            lin = iy[ok] * n_out_x + ix[ok]
            acc  += np.bincount(lin, weights=vals[ok], minlength=acc.size)
            wsum += np.bincount(lin, minlength=wsum.size)

    out = np.full(acc.size, np.nan)
    hit = wsum > 0
    out[hit] = acc[hit] / wsum[hit]
    return out.reshape(shape_out)


def resample_to_grid(data, xin, yin, method="lanczos3", min_weight=0.5, chunk=50000):
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

    if method == "nearest":
        ix = np.clip(np.rint(xin[inside]).astype(int), 0, nx_in - 1)
        iy = np.clip(np.rint(yin[inside]).astype(int), 0, ny_in - 1)
        out[inside] = data[iy, ix]
        return out

    if method == "drizzle":
        raise ValueError("resample_to_grid interpolates; for 'drizzle' call drizzle_to_grid "
                         "(it needs the input and output WCS).")
    if method not in RESAMPLE_KERNELS:
        raise ValueError(f"Unknown resample method '{method}'. "
                         f"Choose from {RESAMPLE_METHODS}.")
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

    how = (f"drizzle, pixfrac={pixfrac}" if resample_method == "drizzle"
           else f"'{resample_method}' kernel")
    print(f" Resampling {n_in} slices onto master grid ({n_y}×{n_x}) with {how} …")
    for s, row in enumerate(valid_rows):
        hdu      = row["hdu"]
        wave_hdu = row["wave_hdu"]

        # Input pixel coordinates of every output pixel (includes SIP distortion)
        wcs_in = WCS(hdu.header, relax=True).celestial
        xin, yin = wcs_in.world_to_pixel(sky_out)

        if resample_method == "drizzle":
            all_reprojected[s] = drizzle_to_grid(
                hdu.data, wcs_in, master_wcs, (n_y, n_x), pixfrac=pixfrac)
        else:
            all_reprojected[s] = resample_to_grid(
                hdu.data, xin, yin, method=resample_method).reshape(n_y, n_x)
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

    if "PC1_1" in out_header:
        for tag in ["3_3", "1_3", "2_3", "3_1", "3_2"]:
            out_header[f"PC{tag}"] = 1.0 if tag == "3_3" else 0.0
    elif "CD1_1" in out_header:
        out_header["CD3_3"] = out_header["CDELT3"]
        for tag in ["1_3", "2_3", "3_1", "3_2"]:
            out_header[f"CD{tag}"] = 0.0

    return cube_data, out_header, common_waves, common_mjds


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
):
    ra  = ra  * u.deg if not isinstance(ra,  u.Quantity) else ra
    dec = dec * u.deg if not isinstance(dec, u.Quantity) else dec

    if resample_method not in RESAMPLE_METHODS:
        raise ValueError(f"resample_method must be one of {RESAMPLE_METHODS}")
    if resample_method == "drizzle" and pixfrac + pixel_scale_arcsec / 6.15 < 1.2:
        print(f"[WARNING] pixfrac={pixfrac} with {pixel_scale_arcsec}\" pixels can leave "
              f"empty pixels in single exposures; use pixfrac >= "
              f"{1.2 - pixel_scale_arcsec / 6.15:.2f}.")
    bad_flag_mask = flag_bitmask(bad_flags)

    results      = get_data(ra, dec, size)
    center_coord = SkyCoord(ra=ra, dec=dec)
    table        = results.to_table()

    table["cutout_index"]       = np.arange(1, len(table) + 1)
    table["central_wavelength"] = np.full(len(table), np.nan)
    table["finast"]             = np.full(len(table), np.nan)
    table["bunit"]              = np.full(len(table), "MJy/sr", dtype=object)
    table["hdu"]                = np.full(len(table), None, dtype=object)
    table["wave_hdu"]           = np.full(len(table), None, dtype=object)

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
            table["bunit"][idx]              = r.get("bunit", "MJy/sr")
            table["hdu"][idx]                = r["hdu"]
            table["wave_hdu"][idx]           = r["wave_hdu"]
            table["time_bounds_lower"][idx]  = r["time_bounds_lower"]
            table["uri"][idx]                = r["uri"]
            if i % 10 == 0 or i == len(table):
                print(f"   → {i}/{len(table)} downloaded "
                      f"({(time.time() - t_start)/60:.2f} min)")

    valid = table[~np.isnan(table["central_wavelength"])]
    if len(valid) == 0:
        raise RuntimeError("No valid cutouts retrieved.")

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
        fits.HDUList([
            fits.PrimaryHDU(data=e_data,  header=e_head),
            fits.ImageHDU(data=e_mjds,    name="MJD"),
        ]).writeto(ep_cube, overwrite=True)
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
        print(" Falling back to single-epoch PCHIP cube …")
        valid_comb = valid.copy()
        valid_comb.sort("central_wavelength")
        if require_good_astrometry:
            valid_comb = drop_poor_astrometry(valid_comb, label="All cutouts")
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


def plot_spectrum_from_cubes(cube_list, ra_deg, dec_deg,
                              aperture_arcsec=14.0, wl_plot=2.0,
                              pixel_scale_arcsec=None):
    """pixel_scale_arcsec=None → read the pixel scale from each cube's WCS
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

    colors   = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    sigclip  = SigmaClip(sigma=3.0, maxiters=5)
    generated_csvs = []
    epoch_spectra  = []          # (wavelengths, out_data) for every "Epoch N" cube

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

                ann_stats  = ApertureStats(sl, ann_pix, sigma_clip=sigclip)
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

        clean_label = label.lower().replace(" ", "_")
        csv_name = os.path.join("spectra",
                                f"{target}_{clean_label}_ap_{aperture_arcsec}.csv")
        out_data = np.c_[
            wavelengths, mjds, fluxes_raw, fluxes_med_sub,
            fluxes_ann_sub, bg_medians, bg_annuli, flux_errors,
        ]
        hdr_txt = (
            f"SPHEREx Extracted Spectrum — Target: {target} ({label})\n"
            f"Units: mJy\n"
            "wavelength_um,MJD,flux_raw_mJy,flux_median_sub_mJy,"
            "flux_annulus_sub_mJy,bg_median_total_mJy,"
            "bg_annulus_total_mJy,flux_err_mJy"
        )
        np.savetxt(csv_name, out_data, delimiter=",",
                   header=hdr_txt, comments="# ")
        generated_csvs.append(csv_name)
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

    # ── Median of the per-epoch spectra (recommended combined spectrum) ──────
    same_grid = (len(epoch_spectra) >= 2 and all(
        len(w) == len(epoch_spectra[0][0]) and np.allclose(w, epoch_spectra[0][0])
        for w, _ in epoch_spectra))
    if same_grid:
        stack = np.array([d for _, d in epoch_spectra])          # (n_epoch, n_wave, 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med_data = np.nanmedian(stack, axis=0)
            med_data[:, 1] = np.nanmean(stack[:, :, 1], axis=0)   # MJD: mean epoch time
            n_ok = np.sum(np.isfinite(stack[:, :, 4]), axis=0)
            mad  = np.nanmedian(np.abs(stack[:, :, 4] - med_data[None, :, 4]), axis=0)
            sig  = np.fmax(np.nanmedian(stack[:, :, 7], axis=0), 1.4826 * mad)
            med_data[:, 7] = 1.253 * sig / np.sqrt(np.maximum(n_ok, 1))
        med_csv = os.path.join("spectra",
                               f"{target}_median_of_epochs_ap_{aperture_arcsec}.csv")
        np.savetxt(med_csv, med_data, delimiter=",", comments="# ", header=(
            f"SPHEREx Extracted Spectrum — Target: {target} "
            f"(median of {len(epoch_spectra)} epoch spectra)\n"
            f"Units: mJy\n"
            "wavelength_um,MJD,flux_raw_mJy,flux_median_sub_mJy,"
            "flux_annulus_sub_mJy,bg_median_total_mJy,"
            "bg_annulus_total_mJy,flux_err_mJy"))
        generated_csvs.insert(0, med_csv)
        ax1.errorbar(med_data[:, 0], med_data[:, 4], yerr=med_data[:, 7],
                     fmt="-", lw=2.0, color="black", capsize=0, zorder=5,
                     label=f"Median of {len(epoch_spectra)} epoch spectra (recommended)")
        print(f"Median-of-epochs spectrum written: {med_csv}")

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
    pixfrac=0.6,
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
        print("SPHEREx 3D Datacube Builder (Common Grid + Median Stack)")
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
            wave_oversample = float(input("Wavelength grid oversample [default=1.0]: ").strip() or 1.0)
            min_cutouts_per_epoch = int(input("Keep epochs with more than N cutouts, N [default=100]: ").strip() or 100)
            pixel_scale_arcsec = float(input("Output pixel scale (arcsec) [default=6.15 = native]: ").strip() or 6.15)
            resample_method = (input("Resampling (drizzle/lanczos3/lanczos2/cubic/bilinear/nearest) "
                                     "[default=drizzle]: ").strip().lower() or "drizzle")
            if resample_method not in RESAMPLE_METHODS:
                print(f"[WARNING] Unknown method '{resample_method}', using drizzle.")
                resample_method = "drizzle"
            pixfrac = 0.6
            if resample_method == "drizzle":
                pixfrac = float(input("Drizzle pixfrac (smaller = sharper, noisier) "
                                      "[default=0.6]: ").strip() or 0.6)
            cutout_size = cutout_arcmin * u.arcminute

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

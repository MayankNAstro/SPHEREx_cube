### ==============================
# Core Python
# ==============================
import os
import sys
import time
import logging
import socket
import concurrent.futures

# ==============================
# Scientific stack
# ==============================
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import median_filter

# ==============================
# Astropy & Reproject
# ==============================
import astropy.units as u
from astropy.time import Time
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.stats import SigmaClip
from astroquery.simbad import Simbad
from reproject import reproject_interp

# ==============================
# Photometry
# ==============================
from photutils.aperture import (
    SkyCircularAperture,
    SkyCircularAnnulus,
    aperture_photometry,
    ApertureStats,
)

# ==============================
# VO access
# ==============================
import pyvo
import urllib.error

# ==============================
# GUI / File Dialogs
# ==============================
import tkinter as tk
from tkinter import filedialog

matplotlib.use("Agg")   # Safe for clusters / multiprocessing

# -------------------------------------------------
# Suppress astropy warnings
# -------------------------------------------------
logging.getLogger("astropy").setLevel(logging.ERROR)
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

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
# Cutout processing (With Optional ZODI Subtraction & DQ Masking)
# -------------------------------------------------
def process_cutout_return(row, ra, dec, subtract_zodi=True, mask_dq=True):
    try:
        with open_fits_with_retry(row["uri"]) as hdul:
            img_hdu = hdul["IMAGE"]
            wcs = WCS(img_hdu.header, relax=True)

            img_data = img_hdu.data.astype(np.float64)

            # 1. ZODI Background Subtraction (Optional, default True)
            if subtract_zodi and "ZODI" in hdul and hdul["ZODI"].data is not None:
                zodi_data = hdul["ZODI"].data.astype(np.float64)
                img_data -= zodi_data

            # 2. DQ Flag Masking (Optional, default True)
            if mask_dq and "FLAGS" in hdul and hdul["FLAGS"].data is not None:
                flags_data = hdul["FLAGS"].data
                img_data[flags_data != 0] = np.nan

            bunit = img_hdu.header.get("BUNIT", "MJy/sr").strip()

            # Get central wavelength for sorting
            coord = SkyCoord(ra=ra, dec=dec)
            x, y = wcs.world_to_pixel(coord)

            spec_wcs = WCS(img_hdu.header, fobj=hdul, key="W", relax=True)
            spec_wcs.sip = None
            lam, _ = spec_wcs.pixel_to_world(x, y)
            lam = lam.to(u.micron).value

            # Calculate full 2D wavelength map natively
            ny, nx = img_data.shape
            y_idx, x_idx = np.indices((ny, nx))
            native_wave_map, _ = spec_wcs.pixel_to_world(x_idx, y_idx)
            native_wave_map = native_wave_map.to(u.micron).value

            # Create updated HDUs
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
            "hdu": None,
            "wave_hdu": None
        }

# -------------------------------------------------
# Create Rectified Cube with PCHIP Interpolation
# -------------------------------------------------
def create_rectified_cube(valid_rows, target_ra, target_dec, size_arcmin=2.0, cr_threshold=50.0):
    waves = np.array([row["central_wavelength"] for row in valid_rows])
    mjds = np.array([row["time_bounds_lower"] for row in valid_rows], dtype=float)
    bunit = valid_rows[0]["bunit"] if "bunit" in valid_rows.colnames and valid_rows[0]["bunit"] else "MJy/sr"

    ref_idx = np.argmin(np.abs(waves - 3.0))
    ref_hdu = valid_rows[ref_idx]["hdu"]

    print(f" Using slice at {waves[ref_idx]:.3f} µm as master spatial reference...")

    ref_wcs = WCS(ref_hdu.header, relax=True).celestial
    master_header = ref_wcs.to_header()

    n_x = ref_hdu.header['NAXIS1']
    n_y = ref_hdu.header['NAXIS2']

    master_header['NAXIS'] = 2
    master_header['NAXIS1'] = n_x
    master_header['NAXIS2'] = n_y
    master_header['BUNIT'] = bunit

    min_wave = np.min(waves)
    max_wave = np.max(waves)
    num_slices = len(valid_rows)
    common_waves = np.linspace(min_wave, max_wave, num_slices)

    sort_w = np.argsort(waves)
    valid_mjd_mask = ~np.isnan(mjds[sort_w]) & ~np.isnan(waves[sort_w])
    if np.sum(valid_mjd_mask) > 1:
        common_mjds = np.interp(common_waves, waves[sort_w][valid_mjd_mask], mjds[sort_w][valid_mjd_mask])
    else:
        common_mjds = np.full_like(common_waves, np.nan)

    cube_data = np.zeros((num_slices, n_y, n_x))
    all_reprojected = []
    all_waves = []

    print(" Cleaning cosmic rays and reprojecting slices...")
    total_crs_removed = 0

    for row in valid_rows:
        hdu = row["hdu"]
        wave_hdu = row["wave_hdu"]

        # Cosmic Ray Filter
        if cr_threshold is not None and cr_threshold > 0:
            data = hdu.data
            surrounding_flux = median_filter(np.nan_to_num(data, nan=0.0), size=3)
            cr_mask = (data > (cr_threshold * surrounding_flux)) & (surrounding_flux > 0)

            if np.sum(cr_mask) > 0:
                data[cr_mask] = surrounding_flux[cr_mask]
                hdu.data = data
                total_crs_removed += np.sum(cr_mask)

        rep_im, _ = reproject_interp(hdu, master_header)
        rep_wave, _ = reproject_interp(wave_hdu, master_header)

        all_reprojected.append(rep_im)
        all_waves.append(rep_wave)

    all_reprojected = np.array(all_reprojected)
    all_waves = np.array(all_waves)

    print(" Interpolating spectra per pixel using PCHIP...")
    for i in range(n_y):
        for j in range(n_x):
            p_vals = all_reprojected[:, i, j]
            p_waves = all_waves[:, i, j]

            valid_mask = ~np.isnan(p_vals) & ~np.isnan(p_waves)

            if np.sum(valid_mask) > 2:
                p_vals_clean = p_vals[valid_mask]
                p_waves_clean = p_waves[valid_mask]

                sort_idx = np.argsort(p_waves_clean)
                x_s = p_waves_clean[sort_idx]
                y_s = p_vals_clean[sort_idx]

                # Deduplicate wavelengths to maintain strict monotonicity
                x_uniq, uniq_idx = np.unique(x_s, return_index=True)
                y_uniq = y_s[uniq_idx]

                if len(x_uniq) > 2:
                    pchip = PchipInterpolator(x_uniq, y_uniq, extrapolate=False)
                    cube_data[:, i, j] = pchip(common_waves)
                else:
                    cube_data[:, i, j] = np.nan
            else:
                cube_data[:, i, j] = np.nan

    # Update Header for 3D
    master_header['WCSAXES'] = 3
    master_header['NAXIS'] = 3
    master_header['NAXIS3'] = num_slices
    master_header['CTYPE3'] = 'WAVE'
    master_header['CRPIX3'] = 1.0
    master_header['CRVAL3'] = common_waves[0]
    master_header['CDELT3'] = common_waves[1] - common_waves[0]
    master_header['CUNIT3'] = 'um'

    if 'PC1_1' in master_header:
        for idx in ['3_3', '1_3', '2_3', '3_1', '3_2']:
            master_header[f'PC{idx}'] = 1.0 if idx == '3_3' else 0.0
    elif 'CD1_1' in master_header:
        master_header['CD3_3'] = master_header['CDELT3']
        for idx in ['1_3', '2_3', '3_1', '3_2']:
            master_header[f'CD{idx}'] = 0.0

    return cube_data, master_header, common_waves, common_mjds

# -------------------------------------------------
# Helper: Cluster Cutouts into Epochs by MJD Gaps
# -------------------------------------------------
def assign_epochs(table, gap_days=14.0):
    mjds = np.array(table["time_bounds_lower"], dtype=float)
    sort_idx = np.argsort(mjds)

    epochs = np.ones(len(table), dtype=int)
    current_epoch = 1

    for k in range(1, len(sort_idx)):
        prev_i = sort_idx[k - 1]
        curr_i = sort_idx[k]

        mjd_prev = mjds[prev_i]
        mjd_curr = mjds[curr_i]

        if not np.isnan(mjd_prev) and not np.isnan(mjd_curr):
            if (mjd_curr - mjd_prev) > gap_days:
                current_epoch += 1
        epochs[curr_i] = current_epoch

    table["epoch"] = epochs
    return table

# -------------------------------------------------
# Main Datacube Builder
# -------------------------------------------------
def make_datacubes(ra, dec, size, target, color_by_wavelength=False, subtract_zodi=True, mask_dq=True):
    ra = ra * u.deg if not isinstance(ra, u.Quantity) else ra
    dec = dec * u.deg if not isinstance(dec, u.Quantity) else dec

    results = get_data(ra, dec, size)
    center_coord = SkyCoord(ra=ra, dec=dec)
    table = results.to_table()

    table["cutout_index"] = np.arange(1, len(table) + 1)
    table["central_wavelength"] = np.full(len(table), np.nan)
    table["bunit"] = np.full(len(table), "MJy/sr", dtype=object)
    table["hdu"] = np.full(len(table), None, dtype=object)
    table["wave_hdu"] = np.full(len(table), None, dtype=object)

    print(" Downloading cutouts...")
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                process_cutout_return, row, ra, dec,
                subtract_zodi=subtract_zodi, mask_dq=mask_dq
            ): i
            for i, row in enumerate(table)
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            r = future.result()
            idx = r["cutout_index"] - 1
            table["central_wavelength"][idx] = r["central_wavelength"]
            table["bunit"][idx] = r.get("bunit", "MJy/sr")
            table["hdu"][idx] = r["hdu"]
            table["wave_hdu"][idx] = r["wave_hdu"]
            table["time_bounds_lower"][idx] = r["time_bounds_lower"]
            table["uri"][idx] = r["uri"]

            if i % 10 == 0 or i == len(table):
                print(f"  → {i}/{len(table)} downloaded ({(time.time() - t_start)/60:.2f} min)")

    valid = table[~np.isnan(table["central_wavelength"])]
    if len(valid) == 0:
        raise RuntimeError("No valid cutouts retrieved.")

    valid = assign_epochs(valid, gap_days=30.0)
    unique_epochs = np.unique(valid["epoch"])
    print(f"[INFO] Detected {len(unique_epochs)} epoch pass(es): {list(unique_epochs)}")

    created_cubes = []

    # 1. Generate Master Combined Datacube
    valid_comb = valid.copy()
    valid_comb.sort("central_wavelength")
    cube_data, cube_header, common_waves, common_mjds = create_rectified_cube(
        valid_comb, center_coord.ra, center_coord.dec, size.to(u.arcmin).value
    )
    combined_cube = f"spherex_cube_{target}_combined.fits"
    fits.HDUList([
        fits.PrimaryHDU(data=cube_data, header=cube_header),
        fits.ImageHDU(data=common_mjds, name="MJD")
    ]).writeto(combined_cube, overwrite=True)
    created_cubes.append((combined_cube, "Combined"))
    print(f" Combined Datacube written: {combined_cube}")

    # 2. Generate Individual Epoch Datacubes
    if len(unique_epochs) > 1:
        for ep in unique_epochs:
            ep_rows = valid[valid["epoch"] == ep].copy()
            ep_rows.sort("central_wavelength")

            if len(ep_rows) < 3:
                print(f"[WARNING] Skipping Epoch {ep}: Insufficient slices ({len(ep_rows)})")
                continue

            e_data, e_head, e_waves, e_mjds = create_rectified_cube(
                ep_rows, center_coord.ra, center_coord.dec, size.to(u.arcmin).value
            )
            ep_cube = f"spherex_cube_{target}_epoch{ep}.fits"
            fits.HDUList([
                fits.PrimaryHDU(data=e_data, header=e_head),
                fits.ImageHDU(data=e_mjds, name="MJD")
            ]).writeto(ep_cube, overwrite=True)
            created_cubes.append((ep_cube, f"Epoch {ep}"))
            print(f" Epoch {ep} Datacube written: {ep_cube}")

    return created_cubes

# -------------------------------------------------
# Extract Multi-spectrum & Backgrounds from Cube(s)
# -------------------------------------------------
def plot_spectrum_from_cubes(cube_list, ra_deg, dec_deg, aperture_arcsec=14.0, wl_plot=2.0):
    os.makedirs("./spectra", exist_ok=True)

    first_cube_path, _ = cube_list[0]
    base = os.path.basename(first_cube_path)
    target = base.replace("spherex_cube_", "").replace("_combined.fits", "").replace(".fits", "")

    fig = plt.figure(figsize=(9, 10))
    ax1 = fig.add_subplot(211)

    primary_wcs = None
    primary_slice = None
    primary_wl_sel = None
    units = "MJy/sr"

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    sigclip = SigmaClip(sigma=3.0, maxiters=5)

    for idx, (cube_path, label) in enumerate(cube_list):
        hdul = fits.open(cube_path)
        data_cube = hdul[0].data
        header = hdul[0].header
        units = header.get("BUNIT", "MJy/sr").strip()

        mjds = hdul["MJD"].data if "MJD" in hdul else np.full(header['NAXIS3'], np.nan)

        n_waves = header['NAXIS3']
        crval3 = header['CRVAL3']
        cdelt3 = header['CDELT3']
        crpix3 = header['CRPIX3']
        wavelengths = crval3 + (np.arange(1, n_waves + 1) - crpix3) * cdelt3

        wcs = WCS(header).celestial
        skycoord = SkyCoord(ra_deg, dec_deg, unit="deg")

        fluxes_raw, fluxes_med_sub, fluxes_ann_sub = [], [], []
        bg_medians, bg_annuli, flux_errors = [], [], []

        if aperture_arcsec <= 0:
            px, py = wcs.world_to_pixel(skycoord)
            px, py = int(np.round(px)), int(np.round(py))
            fluxes_raw = data_cube[:, py, px]
            fluxes_med_sub = fluxes_raw
            fluxes_ann_sub = fluxes_raw
            bg_medians = np.zeros(n_waves)
            bg_annuli = np.zeros(n_waves)
            flux_errors = np.zeros(n_waves)
        else:
            sky_ap = SkyCircularAperture(skycoord, r=aperture_arcsec * u.arcsec)
            sky_ann = SkyCircularAnnulus(
                skycoord,
                r_in=1.5 * aperture_arcsec * u.arcsec,
                r_out=2.5 * aperture_arcsec * u.arcsec
            )
            ap_pix = sky_ap.to_pixel(wcs)
            ann_pix = sky_ann.to_pixel(wcs)
            ap_area = ap_pix.area

            for i in range(n_waves):
                slice_data = np.asarray(data_cube[i, :, :], float)

                phot_raw = aperture_photometry(np.nan_to_num(slice_data, nan=0.0), ap_pix)["aperture_sum"][0]
                fluxes_raw.append(phot_raw)

                bg_med_val = np.nanmedian(slice_data)
                bg_med_total = bg_med_val * ap_area
                fluxes_med_sub.append(phot_raw - bg_med_total)
                bg_medians.append(bg_med_total)

                ann_stats = ApertureStats(slice_data, ann_pix, sigma_clip=sigclip)
                bg_ann_val = ann_stats.median if not np.isnan(ann_stats.median) else 0.0
                bg_ann_std = ann_stats.std if not np.isnan(ann_stats.std) else 0.0
                bg_ann_total = bg_ann_val * ap_area

                fluxes_ann_sub.append(phot_raw - bg_ann_total)
                bg_annuli.append(bg_ann_total)

                n_ann = getattr(ann_stats, 'n_pixels', getattr(ann_stats, 'npixels', 1.0))
                n_ann = float(n_ann) if n_ann > 0 else 1.0

                var_bg = ap_area * (bg_ann_std ** 2) * (1.0 + ap_area / n_ann)
                flux_errors.append(np.sqrt(var_bg))

        hdul.close()

        fluxes_raw = np.array(fluxes_raw)
        fluxes_med_sub = np.array(fluxes_med_sub)
        fluxes_ann_sub = np.array(fluxes_ann_sub)
        bg_medians = np.array(bg_medians)
        bg_annuli = np.array(bg_annuli)
        flux_errors = np.array(flux_errors)

        clean_label = label.lower().replace(" ", "_")
        csv_name = os.path.join("spectra", f"{target}_{clean_label}_ap_{aperture_arcsec}.csv")
        out_data = np.c_[
            wavelengths, mjds, fluxes_raw, fluxes_med_sub,
            fluxes_ann_sub, bg_medians, bg_annuli, flux_errors
        ]
        header_text = (
            f"SPHEREx Extracted Spectrum — Target: {target} ({label})\n"
            f"Units: {units}\n"
            "wavelength_um,MJD,flux_raw,flux_median_sub,flux_annulus_sub,bg_median_total,bg_annulus_total,flux_err"
        )
        np.savetxt(csv_name, out_data, delimiter=",", header=header_text, comments="# ")

        color = colors[idx % len(colors)]
        ax1.errorbar(
            wavelengths, fluxes_ann_sub, yerr=flux_errors,
            fmt="o-", lw=1.2, label=f"{label} (Annulus Sub)", color=color, capsize=2
        )

        if idx == 0:
            i_img = np.argmin(np.abs(wavelengths - wl_plot)) if wl_plot else np.argmax(fluxes_ann_sub)
            primary_slice = data_cube[i_img, :, :]
            primary_wcs = wcs
            primary_wl_sel = wavelengths[i_img]

    ax1.set_xlabel("Wavelength (µm)")
    ax1.set_ylabel(f"Flux / Intensity ({units})")
    ax1.set_title(f"{target} — Multi-Epoch Extracted Spectra")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)

    if primary_wcs is not None:
        ax2 = fig.add_subplot(212, projection=primary_wcs)
        masked_slice = np.ma.masked_invalid(primary_slice)
        norm = simple_norm(masked_slice.compressed(), "log", percent=99) if masked_slice.count() > 0 else None

        ax2.imshow(masked_slice, origin="lower", cmap="inferno", norm=norm)

        if aperture_arcsec > 0:
            sky_ap = SkyCircularAperture(skycoord, r=aperture_arcsec * u.arcsec)
            sky_ann = SkyCircularAnnulus(
                skycoord,
                r_in=1.5 * aperture_arcsec * u.arcsec,
                r_out=2.5 * aperture_arcsec * u.arcsec
            )
            sky_ap.to_pixel(primary_wcs).plot(ax=ax2, color="cyan", lw=1.5, label="Aperture")
            sky_ann.to_pixel(primary_wcs).plot(ax=ax2, color="yellow", lw=1.0, ls="--", label="Annulus")

        ax2.set_title(f"{target} Combined @ {primary_wl_sel:.4g} µm")
        ax2.set_xlabel("RA")
        ax2.set_ylabel("Dec")

    plt.tight_layout()
    png_name = os.path.join("spectra", f"{target}_multiepoch_cutout.png")
    plt.savefig(png_name, dpi=200)
    plt.close(fig)
    print(f" Spectra saved to CSVs and plot written: {png_name}")

# =================================================
# HELPER FUNCTIONS & MAIN LOOP
# =================================================
def parse_coordinates(ra_val, dec_val):
    try:
        ra_float = float(ra_val)
        dec_float = float(dec_val)
        c = SkyCoord(ra=ra_float*u.degree, dec=dec_float*u.degree, frame='icrs')
    except (ValueError, TypeError):
        c = SkyCoord(ra=str(ra_val), dec=str(dec_val), unit=(u.hourangle, u.deg), frame='icrs')
    return c.ra.deg, c.dec.deg

def resolve_simbad(name):
    result = Simbad.query_object(name)
    if result is None or len(result) == 0:
        raise ValueError(f"SIMBAD could not resolve '{name}'")

    ra_str = result['ra'][0]
    dec_str = result['dec'][0]
    return parse_coordinates(ra_str, dec_str)

def process_target(ra, dec, name, size, aperture_arcsec, color_by_wavelength=False, subtract_zodi=True, mask_dq=True):
    print(f"\n[INFO] Processing {name} (RA={ra:.6f}, Dec={dec:.6f})")
    try:
        created_cubes = make_datacubes(
            ra, dec, size, name,
            color_by_wavelength=color_by_wavelength,
            subtract_zodi=subtract_zodi,
            mask_dq=mask_dq
        )
        plot_spectrum_from_cubes(
            created_cubes, ra, dec, aperture_arcsec=aperture_arcsec
        )
    except Exception as e:
        print(f"[ERROR] {name}: {e}")

def validate_radec(ra, dec):
    if not (0.0 <= ra < 360.0):
        raise ValueError(f"Invalid RA: {ra}. Must be between 0 and 360 degrees")
    if not (-90.0 <= dec <= 90.0):
        raise ValueError(f"Invalid Dec: {dec}. Must be between -90 and +90 degrees")

if __name__ == "__main__":
    while True:
        print("\n======================================")
        print("SPHEREx 3D Datacube Builder (Multi-Epoch)")
        print("======================================")
        print("Select input mode:")
        print("  1 → CSV catalog (Supports deg or HMS/DMS)")
        print("  2 → Manual RA/Dec (Supports deg or HMS/DMS) [DEFAULT]")
        print("  3 → SIMBAD name resolution")
        print("  4 → Open existing Datacube + extract spectrum at new RA/Dec")
        print("  0 → Exit")

        mode_input = input("Enter mode (0/1/2/3/4) [default=2]: ").strip()
        mode = mode_input if mode_input else "2"

        if mode not in {"0", "1", "2", "3", "4"}:
            print(f"[WARNING] Invalid option '{mode}'. Please select 0–4.")
            continue

        if mode == "0":
            print("Exiting.")
            sys.exit(0)

        try:
            cutout_input = input("Cutout size (arcmin) [default=15]: ").strip()
            cutout_arcmin = float(cutout_input) if cutout_input else 15.0

            aperture_input = input("Aperture radius (arcsec) [default=14]: ").strip()
            aperture_arcsec = float(aperture_input) if aperture_input else 14.0

            cutout_size = cutout_arcmin * u.arcminute

            zodi_input = input("Subtract Zodiacal Light? (y/n) [default=y]: ").strip().lower()
            subtract_zodi = zodi_input not in ["n", "no"]

            dq_input = input("Mask Data Quality (DQ) flags? (y/n) [default=y]: ").strip().lower()
            mask_dq = dq_input not in ["n", "no"]

            color_by_wavelength = input(
                "Color by wavelength? (y/n) [default=n]: "
            ).strip().lower() == "y"

            if mode == "1":
                root = tk.Tk(); root.withdraw()
                csv_file = filedialog.askopenfilename(
                    title="Select CSV file",
                    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
                )
                if not csv_file:
                    raise RuntimeError("No CSV selected")

                df = pd.read_csv(csv_file)
                df.columns = [c.strip() for c in df.columns]

                ra_col   = input("RA column [default=RA]: ").strip() or "RA"
                dec_col  = input("Dec column [default=Dec]: ").strip() or "Dec"
                name_col = input("Name column [default=Name]: ").strip() or "Name"

                for col in [ra_col, dec_col, name_col]:
                    if col not in df.columns:
                        raise RuntimeError(f"Column '{col}' not found")

                for i, row in df.iterrows():
                    try:
                        ra, dec = parse_coordinates(row[ra_col], row[dec_col])
                        validate_radec(ra, dec)

                        name = str(row[name_col]).strip() or f"RA{ra:.5f}_Dec{dec:.5f}"
                        process_target(
                            ra, dec, name, cutout_size, aperture_arcsec,
                            color_by_wavelength=color_by_wavelength,
                            subtract_zodi=subtract_zodi,
                            mask_dq=mask_dq
                        )

                    except Exception as e:
                        print(f"[WARNING] Skipping row {i}: {e}")
                        continue

            elif mode == "2":
                ra_in = input("Enter RA (deg or hh:mm:ss) [default=277.479646]: ").strip() or "277.479646"
                dec_in = input("Enter Dec (deg or dd:mm:ss) [default=1.239842]: ").strip() or "1.239842"

                ra, dec = parse_coordinates(ra_in, dec_in)
                validate_radec(ra, dec)

                name_in = input("Target name [default=Serp_test]: ").strip()
                name = name_in if name_in else "Serp_test"
                process_target(
                    ra, dec, name, cutout_size, aperture_arcsec,
                    color_by_wavelength=color_by_wavelength,
                    subtract_zodi=subtract_zodi,
                    mask_dq=mask_dq
                )

            elif mode == "3":
                name = input("Enter SIMBAD name: ").strip()
                ra, dec = resolve_simbad(name)
                validate_radec(ra, dec)
                print(f"[INFO] SIMBAD resolved → RA={ra:.6f}, Dec={dec:.6f}")
                process_target(
                    ra, dec, name, cutout_size, aperture_arcsec,
                    color_by_wavelength=color_by_wavelength,
                    subtract_zodi=subtract_zodi,
                    mask_dq=mask_dq
                )

            elif mode == "4":
                root = tk.Tk(); root.withdraw()
                cube_file = filedialog.askopenfilename(
                    title="Select SPHEREx Datacube",
                    filetypes=[("FITS files", "*.fits"), ("All files", "*.*")]
                )
                if not cube_file:
                    raise RuntimeError("No Datacube selected")

                ra_in = input("Enter NEW RA (deg or hh:mm:ss): ").strip()
                dec_in = input("Enter NEW Dec (deg or dd:mm:ss): ").strip()

                ra, dec = parse_coordinates(ra_in, dec_in)
                validate_radec(ra, dec)

                wl_input = input("Image wavelength (µm) [auto]: ").strip()
                wl_plot = float(wl_input) if wl_input else None

                plot_spectrum_from_cubes(
                    [(cube_file, "Single Cube")], ra, dec,
                    aperture_arcsec=aperture_arcsec,
                    wl_plot=wl_plot
                )

        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("Returning to main menu...\n")
            continue

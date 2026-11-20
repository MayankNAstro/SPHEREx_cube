import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import AsinhStretch, ImageNormalize, ManualInterval


def parse_wave_input(raw_str, default_val):
    """Parses either a single float ('2.3') or two space-separated floats ('2.1 2.4')."""
    cleaned = raw_str.strip()
    if not cleaned:
        return default_val
    parts = cleaned.split()
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    else:
        raise ValueError(f"Expected 1 or 2 wavelength values, got {len(parts)}: '{raw_str}'")


def extract_channel_slice(data, wavelengths, target_wave):
    """
    Extracts a 2D image for a channel:
    - Single float: picks the nearest wavelength slice.
    - Tuple/list of two floats (w1, w2): computes the pixel-wise median across all slices in [min, max].
    """
    if isinstance(target_wave, str):
        target_wave = parse_wave_input(target_wave, target_wave)

    if isinstance(target_wave, (tuple, list, np.ndarray)):
        if len(target_wave) != 2:
            raise ValueError(f"Wavelength range must have exactly 2 values, got {target_wave}")
        w_min, w_max = sorted(float(w) for w in target_wave)
        indices = np.where((wavelengths >= w_min) & (wavelengths <= w_max))[0]

        if len(indices) == 0:
            mid_wave = 0.5 * (w_min + w_max)
            idx = np.argmin(np.abs(wavelengths - mid_wave))
            actual_wave = wavelengths[idx]
            label = f"{actual_wave:.2f} µm"
            print(
                f"  → Warning: No slices found in {w_min:.2f}–{w_max:.2f} µm; "
                f"falling back to nearest slice {idx} ({actual_wave:.3f} µm)"
            )
            return data[idx, :, :], label

        master_slice = np.nanmedian(data[indices, :, :], axis=0)
        actual_min = wavelengths[indices].min()
        actual_max = wavelengths[indices].max()

        if len(indices) == 1:
            label = f"{actual_min:.2f} µm"
            print(
                f"  → Channel range {w_min:.2f}–{w_max:.2f} µm matched 1 slice "
                f"(index {indices[0]}: {actual_min:.3f} µm)"
            )
        else:
            label = f"{actual_min:.2f}–{actual_max:.2f} µm"
            print(
                f"  → Channel range {w_min:.2f}–{w_max:.2f} µm median-combined "
                f"{len(indices)} slices ({actual_min:.3f}–{actual_max:.3f} µm)"
            )
        return master_slice, label

    # Single wavelength mode
    target_val = float(target_wave)
    idx = np.argmin(np.abs(wavelengths - target_val))
    actual_wave = wavelengths[idx]
    label = f"{actual_wave:.2f} µm"
    print(f"  → Channel target {target_val:.2f} µm mapped to slice index {idx} ({actual_wave:.3f} µm)")
    return data[idx, :, :], label


def generate_spherex_rgb(
    fits_path,
    target_name="Serp",
    r_wave=4.7,
    g_wave=3.3,
    b_wave=2.3,
    r_cut=(None, None),  # (vmin, vmax) for Red
    g_cut=(None, None),  # (vmin, vmax) for Green
    b_cut=(None, None),  # (vmin, vmax) for Blue
    default_pmin=1.0,
    default_pmax=99.8,
    asinh_a=0.05,
    out_png=None
):
    """Generates an asinh-scaled RGB image supporting single wavelengths or (min, max) median ranges."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

        # Reconstruct spatial WCS and wavelength grid
        wcs_2d = WCS(header).celestial
        n_waves = header['NAXIS3']
        crval3 = header['CRVAL3']
        cdelt3 = header['CDELT3']
        crpix3 = header['CRPIX3']
        wavelengths = crval3 + (np.arange(1, n_waves + 1) - crpix3) * cdelt3

    # Extract single slices or median-combined range slices
    r_img, r_label = extract_channel_slice(data, wavelengths, r_wave)
    g_img, g_label = extract_channel_slice(data, wavelengths, g_wave)
    b_img, b_label = extract_channel_slice(data, wavelengths, b_wave)

    channels = [
        ("Red", r_img, r_label, r_cut),
        ("Green", g_img, g_label, g_cut),
        ("Blue", b_img, b_label, b_cut)
    ]

    normalized_channels = []
    stretch = AsinhStretch(a=asinh_a)

    for name, ch_raw, ch_label, (user_vmin, user_vmax) in channels:
        clean_data = np.nan_to_num(ch_raw, nan=np.nanmedian(ch_raw))

        # Determine channel-specific vmin and vmax
        vmin = user_vmin if user_vmin is not None else np.percentile(clean_data, default_pmin)
        vmax = user_vmax if user_vmax is not None else np.percentile(clean_data, default_pmax)

        print(f"  → {name} ({ch_label}) Scaling: vmin = {vmin:.4f}, vmax = {vmax:.4f}")

        interval = ManualInterval(vmin=vmin, vmax=vmax)
        norm = ImageNormalize(clean_data, interval=interval, stretch=stretch, clip=True)
        normalized_channels.append(norm(clean_data))

    # Stack into an (NY, NX, 3) RGB array
    rgb_composite = np.dstack(normalized_channels)

    # Render RGB composite with WCS projection
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection=wcs_2d)
    ax.imshow(rgb_composite, origin='lower')

    ax.set_xlabel('Right Ascension', fontsize=11)
    ax.set_ylabel('Declination', fontsize=11)
    ax.set_title(
        f"SPHEREx RGB Composite — {target_name}\n"
        f"R: {r_label} | G: {g_label} | B: {b_label}",
        fontsize=12, fontweight='bold'
    )

    if out_png is None:
        base_name = os.path.splitext(os.path.basename(fits_path))[0]
        out_png = f"{base_name}_RGB.png"

    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[INFO] Image successfully saved to: {out_png}")


if __name__ == "__main__":
    fits_input = input("Enter path to SPHEREx 3D datacube FITS file: ").strip()
    if not fits_input:
        fits_input = "spherex_cube_Serp_test_combined.fits"

    target_name_in = input("Enter Target Name (e.g., HOPS 87): ").strip()
    target_name = target_name_in if target_name_in else "Serpens"

    print("\nEnter a single wavelength (e.g., 2.3) or a space-separated range (e.g., 2.1 2.4):")
    b_in = input("Enter Blue wavelength or range (µm) [default=2.3]: ")
    g_in = input("Enter Green wavelength or range (µm) [default=3.3]: ")
    r_in = input("Enter Red wavelength or range (µm) [default=4.7]: ")

    b_wave = parse_wave_input(b_in, 2.3)
    g_wave = parse_wave_input(g_in, 3.3)
    r_wave = parse_wave_input(r_in, 4.7)

    print("\n--- PER-CHANNEL CUTOFF CONFIGURATION ---")
    print("(Press Enter to keep automatic percentile limits for any field)")

    r_min_in = input("Red vmin: ").strip()
    r_max_in = input("Red vmax: ").strip()
    g_min_in = input("Green vmin: ").strip()
    g_max_in = input("Green vmax: ").strip()
    b_min_in = input("Blue vmin: ").strip()
    b_max_in = input("Blue vmax: ").strip()

    r_cut = (float(r_min_in) if r_min_in else None, float(r_max_in) if r_max_in else None)
    g_cut = (float(g_min_in) if g_min_in else None, float(g_max_in) if g_max_in else None)
    b_cut = (float(b_min_in) if b_min_in else None, float(b_max_in) if b_max_in else None)

    generate_spherex_rgb(
        fits_input,
        target_name=target_name,
        r_wave=r_wave,
        g_wave=g_wave,
        b_wave=b_wave,
        r_cut=r_cut,
        g_cut=g_cut,
        b_cut=b_cut
    )

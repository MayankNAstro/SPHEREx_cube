# SPHEREx_cube

SPHEREx 3D Datacube Builder & Photometry Pipeline

A Python tool designed to query IRSA TAP services, download SPHEREx image cutouts across all 6 spectral bandpasses, and construct 3D spatial-spectral datacubes with optional sky subtraction and artifact filtering. It automates multi-epoch observational clustering, PCHIP spectral interpolation, and background-subtracted aperture photometry.

Key Features
Automated Data Retrieval: Queries the IRSA TAP service (spherex.artifact and spherex.plane) to retrieve bandpass cutouts using concurrent multi-threaded downloads with retry logic.

3D Datacube Construction: Reprojects spatial slices onto a uniform reference grid (reproject) and interpolates continuous pixel-level spectra using monotonic Piecewise Cubic Hermite Interpolation (PchipInterpolator).

Multi-Epoch Pass Clustering: Groups observations by MJD timestamps to generate separate 3D datacubes for distinct sky survey passes alongside a master combined cube.

Image Cleaning & Masking:

+++ Zodiacal light subtraction (ZODI HDU).

+++ Quality flag masking (FLAGS HDU).

+++ Cosmic ray rejection via local median filtering.

Aperture Photometry & Extraction: Measures target fluxes using photutils with local annulus and global median background estimation, exporting time-stamped CSV spectra and diagnostic spatial plots.

Flexible Input Modes: Supports batch processing from standard CSV catalogs, manual equatorial coordinates (Decimal Degrees or HMS/DMS), direct SIMBAD name resolution, or local FITS re-analysis.





Output File Structure
Processed outputs are automatically organized into FITS files and a local ./spectra/ directory:


├── spherex_cube_<target>_combined.fits     # Master 3D Datacube (Data + MJD HDUs)

├── spherex_cube_<target>_epoch1.fits       # Epoch-specific 3D Datacube

└── spectra/

    ├── <target>_combined_ap_<radius>.csv   # Extracted 1D spectrum, MJDs & uncertainties

    
    └── <target>_multiepoch_cutout.png      # 1D Spectrum plot & 2D spatial aperture overlay
    

    

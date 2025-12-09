import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from isofit.utils.apply_oe import apply_oe
import xarray as xr
from spectral import envi
import numpy as np

#!/usr/bin/env bash

# TODO: sky view and slope should be created prior to this.

# Number of parallel cores
n_cores=42

# instrument specific elements
SENSOR="emit"

# RDN, LOC, and OBS file paths for hyperspectral data
rdn_file="/store/bawilder/melt-metrics/data/EMIT/inputs/20230216/emit20230216T211835_000"
loc_file="/store/bawilder/melt-metrics/data/EMIT/inputs/20230216/emit20230216T211835_000_LOC"
obs_file="/store/bawilder/melt-metrics/data/EMIT/inputs/20230216/emit20230216T211835_000_OBS"

# pixel size (for bkg term) [meters]
PIXEL_SIZE=60

# Create wavelength file that reads from envi file and then set the path here
wavelength_file = os.path.join(os.path.dirname(obs_file), "wave.txt")
hdr_file = rdn_file + '.hdr'
radiance_dataset = envi.open(hdr_file)
wavelengths = np.array([float(w) for w in radiance_dataset.metadata["wavelength"]])
fwhm = np.array([float(f) for f in radiance_dataset.metadata["fwhm"]])
with open(wavelength_file, 'w') as f:
    for i, (wl, fw) in enumerate(zip(wavelengths, fwhm), start=1):
        f.write(f"{i} {wl} {fw}\n")

# set atmosphere type for RTM (winter for snow , NOTE: this is not used for sRTMnet emulator)
ATMOS="ATM_MIDLAT_WINTER"
#ATMOS="ATM_SUBARC_SUMMER"

# Path to surface config
SURFACE_CONFIG_DIR="/home/bawilder/store/20250626_emit_sierras/surface/multicomponent_surface.json"

# SNOW SURFACE
SURFACE="snow_surface"

# Output directory. Will be created if it doesn't exist.
OUTPUT_DIR="/store/bawilder/melt-metrics/data/EMIT/state/20230216"

# MODTRAN
PREBUILT_LUT="/store/bawilder/20250319_modtran_test/modtran_luts/EMIT_Sierras/emit_20230216T211835.nc"

LUT_CONFIG="/store/bawilder/melt-metrics/isofit/scripts/config-isofit-lut.json"

# LOGGING
LOGGING="INFO"

# SET INVERSION WINDOWS
INVERSION_WINDOWS = [[380.0, 1325.0], [1435, 1770.0], [1965.0, 2500.0]]

apply_oe(rdn_file, loc_file, obs_file, OUTPUT_DIR, SENSOR,
         surface_path= SURFACE_CONFIG_DIR,
         wavelength_path=wavelength_file,
         lut_config_file=LUT_CONFIG,
         prebuilt_lut=PREBUILT_LUT,
         surface_category=SURFACE,
         atmosphere_type=ATMOS,
         logging_level=LOGGING,
         inversion_windows=INVERSION_WINDOWS,
         n_cores=n_cores,
         pixel_size=PIXEL_SIZE)
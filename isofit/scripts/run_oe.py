import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from isofit.utils.apply_oe import apply_oe
from isofit.utils.skyview import skyview
from isofit.utils.shadow import run
from isofit.core.common import find_header

import xarray as xr
from spectral import envi
import numpy as np

#!/usr/bin/env bash


# Number of parallel cores
n_cores=42

# instrument specific elements
SENSOR="emit"

# RDN, LOC, and OBS file paths for hyperspectral data
rdn_file="/store/bawilder/20251217_highmontainasia/emit20230423T064937/emit20230423T064937_rdn.img"
loc_file="/store/bawilder/20251217_highmontainasia/emit20230423T064937/emit20230423T064937_loc.img"
obs_file="/store/bawilder/20251217_highmontainasia/emit20230423T064937/emit20230423T064937_obs.img"

# pixel size (for bkg term) [meters]
PIXEL_SIZE=60

# Create wavelength file that reads from envi file and then set the path here
wavelength_file = os.path.join(os.path.dirname(obs_file), "wave.txt")
hdr_file = find_header(rdn_file)
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
SURFACE_CONFIG_DIR="/store/bawilder/isofit/recipe/bw/multicomponent_surface.json"

# SNOW SURFACE
SURFACE="snow_surface"

# Output directory. Will be created if it doesn't exist.
OUTPUT_DIR="/store/bawilder/20251217_highmontainasia/emit20230423T064937"

# MODTRAN
PREBUILT_LUT="/store/bawilder/20250319_modtran_test/modtran_luts/HighMountainAsiaEMIT/emit_20230423.nc"

LUT_CONFIG="/store/bawilder/isofit/isofit/scripts/config-isofit-lut.json"

# LOGGING
LOGGING="INFO"

# SET INVERSION WINDOWS
INVERSION_WINDOWS = [[380.0, 1325.0], [1435, 1770.0], [1965.0, 2500.0]]


# Run skyview and shadow
skyview(input=loc_file,
        output_directory=os.path.dirname(loc_file),
        resolution=PIXEL_SIZE,
        obs_or_loc="loc",
        method="horizon",
        n_cores=n_cores,
        n_angles=72,
        )

run(input_loc=loc_file,
    input_obs=obs_file,
    pix_size=PIXEL_SIZE,
    n_cpus=n_cores)

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
         pixel_size=PIXEL_SIZE,
         stars=False)
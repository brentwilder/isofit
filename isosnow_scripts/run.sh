#!/usr/bin/env bash
set -euxo pipefail

# Number of parallel cores
n_cores=12

# instrument specific elements
SENSOR="emit"
wavelength_file="/Users/bawilder/Code/isofit-snow/emit/emit-wave.txt"

# RDN, LOC, and OBS file paths for hyperspectral data
rdn_file="/Users/bawilder/Code/sister/output/test2/clip/emit20250327T212148_000"
loc_file="/Users/bawilder/Code/sister/output/test2/clip/emit20250327T212148_000_LOC"
obs_file="/Users/bawilder/Code/sister/output/test2/clip/emit20250327T212148_000_OBS"

# Path to emulator
#EMULATOR_PATH="/Users/bawilder/Documents/sRTMnet/sRTMnet.h5"
EMULATOR_PATH="/Users/bawilder/Documents/sRTMnet/20251206_6c_5layer_-1.6c"

# PREBUILT LUT
LUT="/Users/bawilder/Code/isofit-PRs/local/test/20260828_snowmodel/lut_full/lut.zarr"

# set atmosphere type for RTM
ATMOS="ATM_MIDLAT_WINTER"

# Path to surface config
SURFACE_CONFIG_DIR="/Users/bawilder/Code/isofit-PRs/isosnow_scripts/surfacelut.json"

# Output directory. Will be created if it doesn't exist.
OUTPUT_DIR="/Users/bawilder/Code/isofit-PRs/local/test/20260902_snowmodel"

#LUT_CONFIG="/Users/bawilder/Code/isofit-PRs/local/config-isofit-lut.json"

# LOGGING
LOGGING="INFO"

# SKYVIEW
SKYVIEW="/Users/bawilder/Code/sister/output/test2/clip/sky_view_factor"

# Ancillary data for postprocessing albedo
VEG="/Users/bawilder/Documents/SNOW/EMIT/VEG_TESTING/modis_lakemary"

ALBEDO="/Users/bawilder/Code/snow/LUT/EMIT_L3/EMIT_DISORT_20260828_ALBEDO_2.nc"


# Run iso
isofit apply_oe "${rdn_file}" "${loc_file}" "${obs_file}" "${OUTPUT_DIR}" "${SENSOR}" \
  --surface_path="${SURFACE_CONFIG_DIR}" \
  --wavelength_path="${wavelength_file}" \
  --prebuilt_lut="${LUT}" \
  --emulator_base="${EMULATOR_PATH}" \
  --n_cores=${n_cores} \
  --atmosphere_type="${ATMOS}" \
  --logging_level="${LOGGING}" \
  --surface_category="lut_surface" \
  --skyview_factor="${SKYVIEW}" \
  --veg_fraction_file="${VEG}" \
  --albedo_lut="${ALBEDO}" \
  --use_background_rfl \
  --presolve
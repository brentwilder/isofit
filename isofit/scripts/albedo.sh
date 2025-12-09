#!/usr/bin/env bash

STATES="/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/ang20240815t160909_008/ang20240815t160909_state.hdr"
UNCERT="/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/ang20240815t160909_008/ang20240815t160909_uncert.hdr"
SVF="/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/svf.tif"
SLOPE="/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/slope.tif"
SHADOW="/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/shadow.npy"
LOC="/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/ang20240815t160909_008_L1B_ORT_7b1b5d77_LOC.hdr"
OBS="/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/ang20240815t160909_008_L1B_ORT_7b1b5d77_OBS.hdr"
MODTRAN="/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/ang20240815t160909_008/ang_20240815.nc"
OUTPUT="/Users/bawilder/Documents/ARCSIX/ANG/ang20240815t160909_008_L1B_RDN_cc451d4a_RDN/ang20240815t160909_008"

python ./isofit/isofit/utils/snow_model_outputs.py --states "${STATES}" \
--uncert "${UNCERT}" \
--svf "${SVF}" \
--obs "${OBS}" \
--loc "${LOC}" \
--modtran "${MODTRAN}" \
--canopy "NaN" \
--slope "${SLOPE}" \
--shadow "${SHADOW}" \
--output "${OUTPUT}" 

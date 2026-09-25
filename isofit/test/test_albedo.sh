#!/usr/bin/env bash
set -euxo pipefail


n_cores=12

SENSOR="emit"
wavelength_file="/Users/bawilder/Code/isofit-snow/emit/emit-wave.txt"
EMULATOR_PATH="/Users/bawilder/Documents/sRTMnet/20251206_6c_5layer_-1.6c"
ATMOS="ATM_MIDLAT_WINTER"
SURFACE_CONFIG_DIR="/Users/bawilder/Code/isofit-PRs/isosnow_scripts/surfacelut.json"
LOGGING="INFO"
ALBEDO="/Users/bawilder/Code/snow/LUT/EMIT_L3/EMIT_DISORT_20260828_ALBEDO_2.nc"

ALBEDO_PAIRS_DIR="/Users/bawilder/Code/isofit-PRs/isofit/test/emit_albedo_pairs"







for site_dir in "$ALBEDO_PAIRS_DIR"/*; do
    if [ ! -d "$site_dir" ]; then
        continue
    fi

    for date_dir in "$site_dir"/*; do
        if [ ! -d "$date_dir" ]; then
            continue
        fi

        data_dir="$date_dir/data"
        if [ ! -d "$data_dir" ]; then
            echo "  No 'data' directory found in $date_dir, skipping..."
            continue
        fi

        rdn_file=$(find "$data_dir" -maxdepth 1 -name "emit*T*" ! -name "*_LOC" ! -name "*_OBS" ! -name "*_bgrfl*" ! -name "*_bgtopo*" ! -name "*.hdr" | head -n 1)

        if [ -z "$rdn_file" ]; then
            echo "  No rdn file found in $data_dir, skipping..."
            continue
        fi

        loc_file="${rdn_file}_LOC"
        obs_file="${rdn_file}_OBS"
        skyview_file="${data_dir}/sky_view_factor"

        OUTPUT_DIR="$date_dir"

        isofit apply_oe "${rdn_file}" "${loc_file}" "${obs_file}" "${OUTPUT_DIR}" "${SENSOR}" \
          --surface_path="${SURFACE_CONFIG_DIR}" \
          --wavelength_path="${wavelength_file}" \
          --emulator_base="${EMULATOR_PATH}" \
          --n_cores=${n_cores} \
          --atmosphere_type="${ATMOS}" \
          --logging_level="${LOGGING}" \
          --surface_category="lut_surface" \
          --skyview_factor="${skyview_file}" \
          --albedo_lut="${ALBEDO}" \
          --use_background_rfl
    done
done

echo "Done."
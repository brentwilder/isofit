#!/usr/bin/env bash
set -euxo pipefail

max_jobs=4

SENSOR="emit"
wavelength_file="/Users/bawilder/Code/isofit-snow/emit/emit-wave.txt"
EMULATOR_PATH="/Users/bawilder/Documents/sRTMnet/20251206_6c_5layer_-1.6c"
ATMOS="ATM_MIDLAT_WINTER"
SURFACE_CONFIG_DIR="/Users/bawilder/Code/isofit-PRs/isosnow_scripts/surfacelut.json"
LOGGING="INFO"
ALBEDO="/Users/bawilder/Code/snow/LUT/EMIT_L3/EMIT_DISORT_20260828_ALBEDO_2.nc"

ALBEDO_PAIRS_DIR="/Users/bawilder/Code/isofit-PRs/local/EMIT_SNOW/fig_v2/pairs_flat"




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
            continue
        fi

        rdn_file=$(find "$data_dir" -maxdepth 1 -name "emit*T*" ! -name "*_LOC" ! -name "*_OBS" ! -name "*_bgrfl*" ! -name "*_bgtopo*" ! -name "*.hdr" | head -n 1)

        if [ -z "$rdn_file" ]; then
            continue
        fi
        

        loc_file="${rdn_file}_LOC"
        obs_file="${rdn_file}_OBS"
        skyview_file="${data_dir}/sky_view_factor"
        OUTPUT_DIR="$date_dir"

        rm -rf "${date_dir}/output"

        isofit apply_oe "${rdn_file}" "${loc_file}" "${obs_file}" "${OUTPUT_DIR}" "${SENSOR}" \
          --surface_path="${SURFACE_CONFIG_DIR}" \
          --wavelength_path="${wavelength_file}" \
          --emulator_base="${EMULATOR_PATH}" \
          --n_cores=1 \
          --atmosphere_type="${ATMOS}" \
          --logging_level="${LOGGING}" \
          --surface_category="lut_surface" \
          --skyview_factor="${skyview_file}" \
          --albedo_lut="${ALBEDO}" \
          --use_background_rfl &

        while [ $(jobs -r | wc -l) -ge $max_jobs ]; do
            sleep 1
        done
    done
done

wait

echo "Done."
#!/usr/bin/env bash
set -euo pipefail

artifact_root="${1:-/mnt/sdb/jinming.hu/UDPET/hdf5_cache_v1_full_20260917}"
manifest_root="/mnt/sdb/jinming.hu/UDPET/metrics_step4/input"
repository="/home/jinming.hu/LeqMod_PET_denoising_reproduction"
python="/usr/bin/python3"
builder="${repository}/UDPET/code/build_hdf5_patch_cache.py"

export PYTHONPATH="/mnt/sdb/jinming.hu/UDPET/hdf5_cache_env/pydeps:/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps"
mkdir -p "${artifact_root}/logs"

minimum_free_bytes=$((200 * 1024 * 1024 * 1024))

run_cache() {
    local label="$1"
    local manifest_name="$2"
    local split_name="$3"
    local output_root="${artifact_root}/${label}"
    local available_bytes

    available_bytes="$(df --output=avail -B1 "${artifact_root}" | tail -n 1 | tr -d ' ')"
    if (( available_bytes < minimum_free_bytes )); then
        echo "CACHE_ABORT label=${label} reason=free_space_below_200_GiB available_bytes=${available_bytes}" >&2
        return 1
    fi

    mkdir -p "${output_root}/data"
    echo "CACHE_SPLIT_START label=${label} manifest=${manifest_name} split=${split_name}"
    ionice -c 2 -n 7 nice -n 10 "${python}" "${builder}" \
        --input-csv "${manifest_root}/${manifest_name}" \
        --output-root "${output_root}/data" \
        --output-index "${output_root}/index.json" \
        --output-csv "${output_root}/cache_manifest.csv" \
        --split "${split_name}" \
        --max-studies 100000 \
        --chunk-size 80 80 80 \
        --compression lzf \
        --resume-existing
    echo "CACHE_SPLIT_COMPLETE label=${label}"
}

{
    echo "CACHE_BUILD_START artifact_root=${artifact_root}"
    run_cache train train.csv train
    run_cache val val.csv val
    run_cache test_bern test_bern.csv test_bern
    run_cache test_ruijin test_ruijin.csv test_ruijin
    touch "${artifact_root}/BUILD_COMPLETE"
    echo "CACHE_BUILD_COMPLETE artifact_root=${artifact_root}"
} 2>&1 | tee -a "${artifact_root}/logs/build_all.log"

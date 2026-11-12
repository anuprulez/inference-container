#!/usr/bin/env bash
set -euo pipefail

accelerator="${ACCELERATOR:-cpu}"
image_name="${IMAGE_NAME:-ml-inference:${accelerator}}"
models_dir="${MODELS_DIR:-$(pwd)/models}"
data_dir="${DATA_DIR:-$(pwd)/data}"

mkdir -p "${models_dir}" "${data_dir}"

gpu_args=()
if [[ "${accelerator}" == "cuda" ]]; then
  gpu_args=(--gpus all)
fi

docker run --rm -it \
  "${gpu_args[@]}" \
  -v "${models_dir}:/models" \
  -v "${data_dir}:/data" \
  "${image_name}" "$@"


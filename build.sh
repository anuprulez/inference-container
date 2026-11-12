#!/usr/bin/env bash
set -euo pipefail

accelerator="${ACCELERATOR:-cpu}"
image_name="${IMAGE_NAME:-ml-inference:${accelerator}}"
install_tensorflow="${INSTALL_TENSORFLOW:-1}"

case "${accelerator}" in
  cpu) base_image="python:3.11-slim-bookworm" ;;
  cuda) base_image="nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04" ;;
  *) echo "ACCELERATOR must be cpu or cuda" >&2; exit 2 ;;
esac

docker build \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "ACCELERATOR=${accelerator}" \
  --build-arg "INSTALL_TENSORFLOW=${install_tensorflow}" \
  --tag "${image_name}" \
  "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "Built ${image_name}"


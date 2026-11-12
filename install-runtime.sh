#!/usr/bin/env bash
set -euo pipefail

accelerator="${1:-cpu}"
install_tensorflow="${2:-1}"

python3 -m pip install --upgrade pip setuptools wheel

case "${accelerator}" in
  cpu)
    python3 -m pip install \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r /opt/inference/requirements.txt
    python3 -m pip install onnxruntime
    ;;
  cuda)
    # The cu124 wheel works with the CUDA 12.4 base selected by build.sh.
    python3 -m pip install \
      --index-url https://download.pytorch.org/whl/cu124 \
      --extra-index-url https://pypi.org/simple \
      -r /opt/inference/requirements.txt
    python3 -m pip install onnxruntime-gpu
    ;;
  *)
    echo "ACCELERATOR must be 'cpu' or 'cuda', got: ${accelerator}" >&2
    exit 2
    ;;
esac

# TensorFlow covers SavedModel directories, .pb graphs, and Keras models. It is
# optional because it adds substantially to the image size.
if [[ "${install_tensorflow}" == "1" ]]; then
  python3 -m pip install tensorflow
fi


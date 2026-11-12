# Multi-format ML inference container

This image provides Python runtimes for PyTorch (`.pt`, `.pth`), Hugging Face
PyTorch checkpoints (`.bin`), SafeTensors, ONNX, Hugging Face models, and (by
default) TensorFlow SavedModel, `.pb`, `.h5`, and `.keras` artifacts. It also
supports trusted Python pickle/joblib (`.pkl`, `.pickle`, `.joblib`) models,
including common scikit-learn artifacts. A model file contains weights and
sometimes a graph; it does not always contain preprocessing, architecture, or
inference code.

PyTorch, TorchVision, TorchAudio, and the remaining dependencies are listed in
`requirements.txt`. The installer selects the official PyTorch CPU or CUDA
package index at build time. The PyTorch family is pinned to matching releases
(`torch`/`torchaudio` 2.6.0 and `torchvision` 0.21.0) for reproducible builds.

## Build

CPU image (default):

```bash
./build.sh
```

NVIDIA GPU image (requires a compatible NVIDIA driver and NVIDIA Container Toolkit):

```bash
ACCELERATOR=cuda ./build.sh
```

To omit the large TensorFlow dependency:

```bash
INSTALL_TENSORFLOW=0 ./build.sh
```

Set `IMAGE_NAME=registry/name:tag` to choose another image name.

## Run and verify

`run.sh` mounts `./models` at `/models` and `./data` at `/data`:

```bash
./run.sh /models/model.onnx
./run.sh /models/model.safetensors
./run.sh /models/pytorch_model.bin
./run.sh /models/model.pt
./run.sh /models/checkpoint.pth
./run.sh --allow-unsafe-pickle /models/model.pkl
docker run --rm --entrypoint python3 ml-inference:cpu /opt/inference/healthcheck.py
```

### Multi-file Hugging Face model directories

Place all files belonging to one model directly in the host `models` directory.
The entire directory is mounted at `/models` and treated as one model. This
supports configuration, tokenizer assets, and single-file or sharded
`.safetensors`/`.bin` weights:

```text
models/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
└── model.safetensors.index.json
```

Running without arguments inspects the complete `/models` directory:

```bash
./run.sh
```

Transformers automatically resolves configuration and sharded weight index
files when `/models` is passed to `from_pretrained`:

```python
from transformers import AutoModel, AutoTokenizer

model_dir = "/models"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, device_map="auto")
```

Run your own inference program by overriding the entrypoint:

```bash
docker run --rm -v "$PWD:/workspace" -v "$PWD/models:/models" \
  --entrypoint python3 ml-inference:cpu /workspace/predict.py
```

For CUDA, add `--gpus all` and use `ml-inference:cuda`.

Only load pickle files from a trusted source: Python pickle/joblib loading can
execute arbitrary code. For that reason, the inspector requires the explicit
`--allow-unsafe-pickle` option for `.pkl`, `.pickle`, and `.joblib` files.

## Format note

`.pf` is not a standard interchange format. The inspector treats it like a
PyTorch checkpoint in case it is a locally renamed `.pt`/`.pth` file. If it was
produced by another framework, its corresponding loader must be added. Loading
pickle-based PyTorch checkpoints from untrusted sources can execute malicious
code; the included inspector uses the safer `weights_only=True` mode.

`.bin` is a generic extension. The inspector treats it as a PyTorch checkpoint,
which covers common Hugging Face `pytorch_model.bin` files. Other `.bin` files
(for example GGML/GGUF or vendor-specific binaries) require their matching
runtime and cannot be identified reliably from the extension alone.

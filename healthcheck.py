#!/usr/bin/env python3
import importlib
import json

packages = ["torch", "safetensors", "onnx", "onnxruntime", "transformers", "accelerate", "tensorflow"]
status = {}
for package in packages:
    try:
        module = importlib.import_module(package)
        status[package] = getattr(module, "__version__", "installed")
    except ImportError:
        status[package] = "not installed (may be optional)"

try:
    import torch
    status["cuda_available"] = torch.cuda.is_available()
except ImportError:
    status["cuda_available"] = False

print(json.dumps(status, indent=2))


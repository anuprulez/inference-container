#!/usr/bin/env python3
"""Inspect common ML model files without guessing their architecture."""

import argparse
import json
from pathlib import Path


def inspect(path: Path, allow_unsafe_pickle: bool = False) -> dict:
    suffix = path.suffix.lower()
    result = {"path": str(path), "format": suffix or "directory"}

    if path.is_dir():
        files = sorted(str(p.relative_to(path)) for p in path.rglob("*") if p.is_file())
        result["file_count"] = len(files)
        result["files"] = files[:100]
        config_path = path / "config.json"
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as config_file:
                config = json.load(config_file)
            result["model_type"] = config.get("model_type")
            result["architectures"] = config.get("architectures", [])
        index_files = sorted(path.glob("*.index.json"))
        if index_files:
            result["weight_indexes"] = [item.name for item in index_files]
            result["sharded"] = True
    elif suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(path, framework="pt", device="cpu") as model:
            result["tensors"] = {key: list(model.get_slice(key).get_shape()) for key in model.keys()}
    elif suffix == ".onnx":
        import onnx
        model = onnx.load(str(path), load_external_data=False)
        result["ir_version"] = model.ir_version
        result["inputs"] = [item.name for item in model.graph.input]
        result["outputs"] = [item.name for item in model.graph.output]
    elif suffix in {".bin", ".pt", ".pth", ".pf"}:
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=True)
        result["object_type"] = type(obj).__name__
        if isinstance(obj, dict):
            result["keys"] = list(obj)[:100]
    elif suffix in {".pkl", ".pickle", ".joblib"}:
        if not allow_unsafe_pickle:
            result["warning"] = (
                "Pickle files can execute arbitrary code. Re-run with "
                "--allow-unsafe-pickle only if you trust the file."
            )
            return result
        import joblib
        obj = joblib.load(path)
        result["object_type"] = f"{type(obj).__module__}.{type(obj).__name__}"
        if isinstance(obj, dict):
            result["keys"] = [str(key) for key in list(obj)[:100]]
    elif suffix in {".pb", ".h5", ".keras"}:
        result["loader"] = "TensorFlow/Keras (architecture or signature is required to execute it)"
    else:
        result["warning"] = "Unknown extension; use the library that created the model."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path("/models"),
        help="model file or multi-file model directory (default: /models)",
    )
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="allow loading trusted .pkl/.pickle/.joblib files (can execute code)",
    )
    args = parser.parse_args()
    if not args.model.exists():
        parser.error(f"not found: {args.model}")
    print(json.dumps(inspect(args.model, args.allow_unsafe_pickle), indent=2, default=str))


if __name__ == "__main__":
    main()


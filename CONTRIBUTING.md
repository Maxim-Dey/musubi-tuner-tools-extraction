# Contributing

This extraction supports Krea2 Raw LoRA training and its image caching, training samples, logging and save/resume workflow. Keep changes within that scope and preserve existing training/data/artifact contracts. Follow the repository constitution and applicable Spec Kit feature documents.

## Development environment

Use Python 3.10–3.12, compatible PyTorch/torchvision and the dependencies in `pyproject.toml`. Install development tools from the `dev` dependency group in your chosen environment. See [README](README.md) for model preparation and operational commands.

## Local checks

Run from the repository root with `PYTHONPATH` pointing at `src`. For CPU checks set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `CUDA_VISIBLE_DEVICES` to an empty value:

```text
python -m pytest -q tests
ruff check src tests
ruff format --check src tests
python krea2_cache_latents.py --help
python krea2_cache_text_encoder_outputs.py --help
python krea2_train_network.py --help
```

Use meaningful tests for changed behavior and actual imports/parsers. Record failed, skipped and unavailable checks accurately. Local CPU checks do not demonstrate real model training. This local development stage does not run training, GPU tests, model downloads, package builds or remote verification.

## Attribution

Preserve applicable upstream license headers and attribution when editing or moving code. The [README license section](README.md#license) records retained code and upstream provenance. Include concrete changes and verification results in contribution descriptions.

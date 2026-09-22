# Contributing to this Qwen-Image extraction

Scope is original Qwen-Image adapter training, its two caches, training sample images, logging and save/resume. Read the [README](README.md), [constitution](.specify/memory/constitution.md) and [feature specification](specs/001-scope-qwen-image-lora/spec.md). Preserve existing training algorithms, valid option semantics and source attribution. Changes outside this scope require a separate request.

Use an existing Python >=3.10,<3.13 environment with the declared dependencies. Keep Python indentation at four spaces and follow the existing Ruff configuration. Add focused tests for substantial behavior changes; do not replace real dependencies/readers with fake modules or weaken assertions to accommodate a regression.

Local checks use CPU/offline settings and `python -B -m pytest -p no:cacheprovider -q tests`. Real `--help`/module imports, small cache/adapter tensors and Accelerate serialization are appropriate. Training, GPU runs, weight/tokenizer downloads, package builds and server verification are separate operational work. Record unavailable checks honestly. See [local verification](specs/001-scope-qwen-image-lora/quickstart.md).

Review changes for accidental user-data deletion, modified templates, broken links and excluded import/dispatch paths. Keep notices and copyright headers. This extraction derives from [Musubi Tuner by kohya_ss](https://github.com/kohya-ss/musubi-tuner); Qwen/Diffusers, Hunyuan-derived helpers and LyCORIS attribution remain applicable. The original project is Apache License 2.0 except where its retained third-party notices specify otherwise.

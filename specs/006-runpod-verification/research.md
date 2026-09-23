# Research: RunPod Verification

## Observed environment and unresolved gates

Read-only inventory on 2026-09-23 found the selected Pod shell `root@e25a02a5585f`, `/workspace` with about 203 GiB free, one NVIDIA L40S (46,068 MiB total; 45,458 MiB then free), driver 580.178.04, Python 3.12.3, global torch 2.8.0+cu128, and `uv`/Git. Accelerate and safetensors were absent globally. These are observations to refresh before execution, not evidence of a working full-model load. `/workspace/qwen_image_2512_bf16.safetensors`, `/workspace/qwen_2.5_vl_7b.safetensors`, and `/workspace/qwen_image_vae.safetensors` exist. A raw DiT header has 1,933 BF16 original-like tensors and blocks 0–59; loader compatibility remains unverified. Captioned PNG/TXT files are visible in `/workspace/datasets/{train,val_familiar,val_unfamiliar}`. Matching names alone do not establish the required hash relationship.

Basic SSH interactive PTY works. A noninteractive remote command returned “Your SSH client doesn't support PTY” with exit 0, and `ssh -tt` ignored its supplied command. Therefore first test a small PTY round-trip and durable file retrieval, then transfer a byte-preserving archive in bounded base64 blocks. If either check fails, request exposed-TCP SSH. Preserve ordinary host-key verification and never print or transfer the private key.

## Decisions

| Decision | Reason and verification |
| --- | --- |
| Fetch/checkout `e8af43abc32d967153e3aa3c31d50c77d3f7c47e` from remote branch `001-scope-qwen-image-lora`; overlay the finished local changes. | Remote default HEAD is a different commit. The overlay inventory was 38 modified/untracked runtime and G08 files (~697 KiB raw, no deletions). Rebuild the manifest at transfer time, archive exact bytes despite local `core.autocrlf=true`, and compare archive and every file SHA-256 remotely. |
| Create one `/workspace/musubi-val-loss-tests/<run_id>/` tree and a Python 3.12 venv. | Isolates source, dependencies, fixture copies, caches, outputs, and evidence. Do not modify original models, data, results, or global Python. |
| Install declared project dependencies plus verification-only TensorBoard and pytest; use CUDA 12.8 torch 2.8.0/torchvision 0.23.0. | No `uv.lock` exists. `pyproject.toml` declares dependencies but names an absent `README.md`, so run scripts using `PYTHONPATH=<run>/code/src` without editable installation. Record exact resolved package versions, imports, CUDA, AdamW8bit and available VRAM. [PyTorch's version table](https://pytorch.org/get-started/previous-versions/) lists this CUDA pair. |
| Freeze only existing user-designated image/caption members and their hashes before caching. | Target up to 3 train, 2 familiar, 3 unfamiliar; prove familiar hashes are in train and unfamiliar hashes are disjoint. Copy and rehash; ask one role question if ambiguous. Build both Qwen cache types for canonical sources and G03 renamed/reordered copies with distinct source-bound cache paths/manifests, then freeze both variants before evaluation. |
| Use the finite [G01–G10 matrix](verification-plan.md). | Exact case budgets sum to 38 completed updates on the observed one GPU. G09 adds two only if fresh inventory of the same Pod finds two suitable GPUs. Discrete inputs/state are exact; numeric tolerances are declared before results. No extra training to chase outcomes. |

No new public interface needs a separate contract. The protocol, run manifest, effective configs, and evidence files form the operational contract.

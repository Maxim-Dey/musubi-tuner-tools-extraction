# Implementation Plan: Finite RunPod Verification

**Branch**: `001-scope-qwen-image-lora` (current checkout; feature directory `006-runpod-verification`) | **Date**: 2026-09-23 | **Spec**: [spec.md](spec.md)

## Summary

Verify the finished Stage 1–4 original Qwen-Image LoRA tool on the user's selected Pod through a finite G01–G10 matrix. Preparation proves exact source identity, isolates files and dependencies, freezes user-designated fixtures, and checks short-run budgets before any model update. Verification permits at most 38 real-model optimizer updates on the observed single GPU, or 40 only if the same Pod later has two suitable GPUs. The [verification protocol](verification-plan.md) fixes commands, event counts, tolerances, and evidence in advance.

## Technical Context

**Language/Version**: Python 3.12 and Bash on the Linux Pod.  
**Primary Dependencies**: Project `pyproject.toml` dependencies; isolated PyTorch 2.8.0 + torchvision 0.23.0 CUDA 12.8 pair, Accelerate 1.6.0, bitsandbytes, safetensors; verification-only TensorBoard and pytest. No `uv.lock` is present.  
**Storage**: `/workspace/musubi-val-loss-tests/<run_id>/` for source, `.venv`, fixture copies, caches, test configs, output, scripts, logs, and evidence. Existing models/datasets remain read-only.  
**Testing**: Existing CPU G08 seams plus bounded real Qwen forwards and six short training branches; no 1,600-step run.  
**Target Platform**: User-selected RunPod. Read-only inventory observed one NVIDIA L40S (46,068 MiB, 45,458 MiB free), driver 580.178.04, Python 3.12.3, global torch 2.8.0+cu128, and about 203 GiB free; refresh these observations before execution. They do not establish H200 performance or full model compatibility.  
**Project Type**: Operational verification of existing CLI tools; no product-code change.  
**Performance Goals**: None; memory and duration are observations, not a benchmark.  
**Constraints**: Basic SSH interactive PTY works but arbitrary noninteractive commands do not. No SCP/SFTP, reliable bulk transfer, isolated dependencies, or full model load has yet been verified. Never stop/replace the Pod or user processes, copy the private key, disable host-key checking, overwrite source data, or relax tolerances after results.

## Constitution Check

| Principle | Gate |
| --- | --- |
| I–II Language/task fidelity | PASS: English Spec Kit artifacts; exact G cases and limits preserved. |
| III Minimal compatible changes | PASS: existing trainer/cache entrypoints and Stage 1–4 test seams; only small one-off wrappers and report files. |
| IV Training invariants | PASS: G03–G05 compare fixed inputs, state, and controlled next updates; no changed training math. |
| V Memory/performance | PASS: sequential model processes, no second full base model in memory, no unrequested benchmark. |
| VI Configuration/errors | PASS: effective TOMLs and all budget/role/model paths checked before first update. |
| VII Verification/docs | PASS: evidence/status per case, README reconciliation, failed/unavailable cases visible. |
| Local Stage | PASS: this planning step creates local documents only. Remote preparation/verification are separate later Stage 5 work; no model or GPU is run now. |

**Post-design gate**: PASS. Verified code transfer, fixture-role/hash checks, and isolated environment are hard prerequisites. Their current unverified state is recorded, not treated as completed preparation.

## Project Structure

```text
specs/006-runpod-verification/
  spec.md  plan.md  research.md  data-model.md  quickstart.md
  verification-plan.md
  verification.md             # later measured report
  verification.json           # later compact statuses/identities

/workspace/musubi-val-loss-tests/<run_id>/
  code/                       # exact base plus verified byte-preserving overlay
  .venv/                      # separate dependencies
  fixtures/                   # copied image/caption subsets and caches
  experiments/                # separate train/val TOMLs and output per case
  scripts/                    # bounded launch/probe/inspection helpers only
  outputs/                    # separate experiment tree per training branch
  evidence/                   # manifest, hashes, commands, logs, events, metadata
```

No new public interface is introduced, so a separate `contracts/` directory is unnecessary; [verification-plan.md](verification-plan.md) is the case/evidence contract.

## Phase 0: Research and decisions

See [research.md](research.md). The remote Git default is **not** the tested base: it resolves to `bf478...`; the remote `001-scope-qwen-image-lora` branch resolves to local base `e8af43abc32d967153e3aa3c31d50c77d3f7c47e`. Clone/fetch that exact base and overlay finished local modified/untracked source bytes. A 38-file runtime/G08 overlay (~697 KiB raw at inventory time) must be archived with file SHA-256s; a small PTY transfer canary must round-trip before chunked base64 transfer. Verify archive and per-file hashes remotely. If this cannot be done reliably, stop and request exposed-TCP SSH; never test default HEAD or publish the patch merely to transfer it.

Observed model files are `/workspace/qwen_image_2512_bf16.safetensors` (~39 GiB), `/workspace/qwen_2.5_vl_7b.safetensors` (~16 GiB), and `/workspace/qwen_image_vae.safetensors` (~243 MiB). A header-only DiT read found BF16 original-style keys and 60 blocks; complete loader compatibility is **not yet verified**. `/workspace/datasets/{train,val_familiar,val_unfamiliar}` has captioned PNG/TXT files; matching names alone do not prove role membership. Freeze exact image/caption hashes first.

Create a Python 3.12 venv, install declared runtime requirements plus verification-only TensorBoard and pytest there, and pin the CUDA pair `torch==2.8.0`, `torchvision==0.23.0` from the CUDA 12.8 wheel index. Do not use `pip install -e .`: `pyproject.toml` names an absent `README.md`. Use `PYTHONPATH=<run>/code/src` and existing root scripts. Record resolved versions, CUDA imports, bitsandbytes/AdamW8bit, and free VRAM. No lock file exists; the installed environment is evidence. The pair is listed in [PyTorch's previous-version instructions](https://pytorch.org/get-started/previous-versions/).

## Phase 1: Design and acceptance

### Preparation

1. Recheck selected Pod identity, host-key-pinned interactive SSH, process inventory, free space, GPU and model paths. Keep the user's SSH key local. Generate one collision-free UTC `run_id`, write it once to the run manifest, and never reuse an earlier tree.
2. Clone/fetch exact base in `code/`, verify its commit, transfer byte-preserving overlay only after a round-trip canary, and compare archive plus every source-file SHA-256 with the local manifest. Include later one-off scripts/configs in their own manifest. Stop for exposed-TCP SSH if PTY transport or durable evidence retrieval cannot be verified.
3. Install dependencies only into `.venv`; verify versions, CUDA, model headers/paths and available memory without an optimizer update. If full 1024/BF16/no-FP8/no-block-swap execution fails for environment reasons, report it; do not silently weaken the target.
4. Choose up to 3 train, 2 familiar, 3 unfamiliar existing user-designated images. Verify every familiar image hash exists in train and no unfamiliar hash overlaps; copy images/captions into `fixtures/`, rehash copies, and record model/source hashes. Ask one role question if assignment is ambiguous. Build latent and text caches with existing Qwen commands against canonical train/val TOMLs in an isolated cache experiment. Also prepare G03 renamed/reordered image/caption copies and their own source-bound latent/text caches with a separate manifest. Freeze both fixture/cache variants before any validation; never rename a source while reusing its old cache path.
5. Generate separate short-case TOMLs from Stage 4 example. Enumerate test deviations: budgets/accumulation/intervals/window, warmup 2, `N1=4,N2=2` (G09 `2×1`, separate G03 `10×1`), one fixed prompt/seed with four inference steps, and opt-out comparison settings. G05-off is a legacy training config with neither `experiment_dir` nor `val_dataset_config`, with samples disabled and an isolated output directory. Keep original model, LoRA 16/16, 1024 buckets, BF16, SDPA, AdamW8bit, `5e-5`, gradient checkpointing, FP32 state, and FP8/block swap disabled. Static preflight must read every effective `max_train_steps` and refuse inherited 1600 before the first update.
6. Create only a bounded shell launcher writing command/PID/stdout/stderr/exit files, a small real-model probe for G03–G05, and a read-only evidence inspector. Check original PID and output if SSH drops before any retry. No duplicate launch, infinite watcher, second trainer, or Pod management layer.

### Verification

Run [G01–G09](verification-plan.md) in dependency order: source/cache gates → G01 and G02 → fixed-check/formula probes G03/G04 → sequential paired G05 → isolated current/best G06 → read-only G07 inspection → bounded G08 controlled branches → conditional G09. Each case has predeclared update and validation-call maxima, exact expected events, numeric tolerances, and evidence. Stop a dependent case when its required state/fixture/source is unavailable; record NOT RUN with reason. A functional failure is FAIL, not permission to train longer or change the case. On the observed one-L40S Pod G09 is NOT RUN unless fresh inventory of the **same** Pod finds two suitable GPUs.

### Report

Write `verification.md` and compact `verification.json` under this feature, with G01–G10 PASS/FAIL/NOT RUN, commands, exit codes, source/environment/model/fixture hashes, numeric differences, log/event/state evidence locations, and limits. Keep bulky raw evidence under the unique remote run tree; retrieve only readable summaries and selected evidence via a verified channel, marking anything left remote with its path and retrieval status. Review `README.ru.md` against observed behavior and link the report. Inspect agent-owned test PIDs, stop only remaining agent tests, and leave the Pod and user processes running.

## Complexity Tracking

No exception is planned. Transfer remains a hard prerequisite rather than a workaround that weakens source identity.

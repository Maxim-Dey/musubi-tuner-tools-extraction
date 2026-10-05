# Implementation Plan: Automatic cache preparation

**Branch**: codex/auto-cache-training | **Date**: 2026-10-06 | **Spec**: [spec.md](spec.md)

## Summary

Add opt-in preparation at the existing Qwen preflight boundary. Reuse source/cache validation and native cache subprocesses. Do not introduce a wrapper or another training engine.

## Technical Context

Python3.10–3.12, existing PyTorch/Accelerate1.6, TOML/argparse, safetensors and Pillow. Supported local .venv has Python3.12/CPU torch2.7.1. No dependencies added. Filesystem cache storage and native Qwen original encoder entrypoints remain unchanged. Actual model execution is substituted in local tests; no GPU/remote acceptance.

## Constitution Check

All7 principles apply: English artifacts/Russian dialogue; preserve scope; reuse existing mechanisms; preserve RNG/loss/precision; sequential encoder processes avoid simultaneous models; effective configuration is checked first; use meaningful local regressions and factual results. No conflict. New feature has its own original two-round post-implementation correction allowance; old feature evidence and its explicit continuation are not rewritten. User timestep edits remain untouched and outside prior validation commit.

## Structure and Ownership

- Coordinator: SpecKit artifacts; qwen_image_train_network.py integration/parser; main TOML; README.ru.md/docs/qwen_image.md; existing profile assertion.
- Data agent: training/validation_inputs.py source-only inventory, tests/test_auto_cache_inputs.py.
- Orchestration agent: new training/auto_cache.py, tests/test_auto_cache.py.
- Independent reviewer: interfaces, boundary tests and final convergence; no overlapping file writes.

## Phase 0: Research Decisions

See [research.md](research.md). Inspect from sources before training buckets because training drops uncached items. Existing cache mains load encoders unconditionally, so plan stages first. Existing loops skip by existence and may delete extras: always prevalidate existing files and pass skip_existing/keep_cache. Experiment-only auto mode supplies provenance without inventing a new cache flag or changing legacy paths. Child exit releases memory/global accelerator state. Reject distributed automatic mode before side effects; no distributed coordinator in scope.

## Phase 1: Design

Public inventory contract in validation_inputs: inspect_cache_inputs(args) returns tuple[CacheInventory], with dataset_config:str, validation:bool, missing_latents:tuple[str,...], missing_text:tuple[str,...]. Build ImageDataset from real BlueprintGenerator without preparing cached training buckets. Factor common source-only train/val disjointness/destination checks used by existing validation. Validate all existing source-required files through current cache contracts; require provenance for auto experiment caches. Reject unexpected training latent caches that the existing loader would otherwise consume. Distinguish truly absent paths from directories/broken links. Do not delete or encode. Existing strict resume checks stay authoritative.

Orchestrator prepare_missing_caches(args) returns immediately when auto_cache is false. Validate strict boolean, experiment_mode and rank environment first. Under preserve_rng_state, inspect all inputs, build up to4 stage jobs, prevalidate all required VAE/TE files, then run sequential subprocess.run argv lists with sys.executable -m and check=True. Use native modules, absolute dataset/model paths, model_version original, experiment_mode, skip_existing, keep_cache, batch_size1 and num_workers1; add validation only for val and fp8_vl only to TE when selected. Keep child stderr visible. Do not modify global argv/environment or parent cwd. A contextual error stops training on child failure. Reinspect all required cache files after jobs; incomplete or invalid output fails before training.

Qwen validate_training_inputs calls native validate_training_args then the helper, before seed/session, dataset preparation, Accelerator, sampling models and DiT. Qwen parser exposes mutually exclusive --auto_cache/--no_auto_cache using existing boolean actions; defaultfalse, CLIoverridesTOML. Main user TOML adds auto_cache=true and retains all50 existing fields including user min_timestep/max_timestep/preserve_distribution_shape. Accepted prior profile tests are updated to current user fields rather than editing user settings.

## Verification

Pre-implementation analyze finding I1 (HIGH, FR-006): native text encoding also requires the Qwen/Qwen-Image tokenizer. Before any cache stage, when a text stage is planned, resolve/load the same Qwen2Tokenizer using the native QWEN_IMAGE_ID/subfolder under the RNG guard, then release it. Do not load an encoder for this preflight. Missing-tokenizer failure must precede all subprocesses; complete caches bypass this check. Existing Hugging Face cache/offline behavior is retained. CPU tests substitute tokenizer loading and perform no downloads.

Single-process launcher refinement: pass each child an environment copy without rank/world/master rendezvous variables (including supported MPI equivalents); retain device visibility/PYTHONPATH and other model settings. Never mutate the parent environment. Explicitly test WORLD_SIZE=1/RANK=0 startup and early rejection of multi-process/rank>0 contexts.

Tests precede owned implementation. Real source images/TOMLs/safetensors establish inventory correctness and preservation. Real cache entrypoints are exercised with expensive encoders replaced by local fixtures; native training entrypoint is stopped at model-load boundary. Verify train/val/no-val, none/partial/complete, stale/corrupt/colliding/overlap/orphan, source paths and other cwd, required resource preflight, failure/incomplete output, disabled compatibility, boolean/CLI guards, subprocess argv, parent RNG and resume fingerprint guard. No tests for cosmetic details.

Run focused new+affected regressions, scoped Ruff/format, full retained CPU suite once after integration, CLI help and diff checks. Baseline current profile test has1failure/154passes because the user added3 timestep settings after previous acceptance; preserve them and adjust expected profile. After initial implementation, independent append-only converge adds only confirmed gaps; fixes are separate recorded passes, at most2. Stop when checks pass and no gaps remain.

## Complexity Tracking

One inventory record and one small orchestrator are sufficient. No queue, lock service, model manager, generalized pipeline, new cache format or new encoder implementation.

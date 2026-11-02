# Implementation Plan: Deterministic Krea2 validation
**Branch**: `codex/004-krea2-deterministic-val` | **Date**: 2026-10-27 | **Spec**: [spec.md](spec.md)

## Summary
Implement A→B→C→D→E→F in the existing Krea2 trainer. Separate input/cache and numerical helpers are necessary to test fail-fast behavior without models and reuse existing cache tools. Keep shared trainer changes opt-in and owned by one implementer. Source commands 1–10 remain normative.

## Technical Context
Python >=3.10,<3.13; PyTorch, Accelerate 1.6.0, safetensors, TOML/voluptuous and existing Krea2 loaders. Filesystem manifests/checkpoints and TensorBoard events. pytest CPU fixtures locally; real Linux/CUDA Pod checks in F. Dataset inventoried as 107 train/20 val PNGs, every caption present/nonempty, no content overlap/duplicates or stem collisions. No performance benchmark requirement; no duplicate weights or all-combination GPU result retention.

## Constitution Check
Before research and after design: PASS. English artifacts, original scope retained; opt-in compatibility; train mathematics/RNG preserved; no generic framework; effective fail-fast config; evidence-based tests and README; no local weights/training/Pod; required remote F remains mandatory; two global correction rounds maximum. No constitution changes or exceptions.

## Project Structure
- `src/musubi_tuner/training/parser_common.py`, `dataset/config_utils.py`: effective options and opt-in paths.
- `training/krea2_validation_inputs.py`: fixed records, preflight, provenance/manifest, cache preparation using existing cache scripts.
- `dataset/cache_io.py`, `krea2_cache_latents.py`, `krea2_cache_text_encoder_outputs.py`: additional provenance, existing format/batching.
- `training/krea2_validation.py`: numerical computation, isolation and aggregation.
- `krea2_train_network.py`: Krea2 forward guard/integration; `training/trainer_base.py`: sole-owner scheduling/resume/hooks.
- `utils/train_utils.py`: limited opt-in checkpoint/retention integration if necessary.
- `tests/test_krea2_validation_inputs.py`, `tests/test_krea2_validation.py`, `tests/test_krea2_experiment_state.py`: focused fixtures plus existing cache/precision/training regressions.
- `config_for_krea2_raw_lora/`: main TOMLs/prompts, separate diagnostics; README/Krea2 docs and dependencies.
- This directory: spec, plan, research, data-model, contracts, quickstart, tasks, verification and checklists.

## Execution and ownership
A/B owner edits parser/config/cache/input files only. C/D owner alone edits trainer_base, Krea2 trainer and numerical/state helpers. E owner edits configs/docs/dependency only. Documents/journal owner exclusively edits feature documents and .specify/feature.json. Review is independent. No simultaneous file edits; utils/train_utils or a test shared with another owner requires explicit handoff.

A preflight → B cache/manifest → C isolated loss → D lifecycle/state → E full configs/local acceptance → F remote. Research/independent reference tests may run in parallel; implementation groups start only after required predecessor acceptance. Each group gets independent evidence review. Initial tasks cover all A–F. Later confirmed defects consume one of at most two global fix→verify passes, recorded before edits.

## Integration contracts
See contracts/validation.md. Preflight runs before any model; prepare caches after Accelerator creation and before dataset/model/sampling loading. Input object exists for experiment/cache modes even without val. Manifest checked before resume loading and each val. No val inputs means legacy path.

The completed optimizer hook is the `accelerator.sync_gradients` block after accumulation, immediately after successful global_step increment (current trainer_base ~2365–2369), with AMP skipped updates excluded. Existing on_post_optimizer_step (~2354) runs on every microbatch and receives the previous count; do not schedule validation there. Persist optimizer count separately from accelerator.step.

Use unwrapped DiT for uneven distributed forwards; records[rank::world] gives no padding. Reduce float64 per-image sums plus count, never equal-weight local means. Every rank participates even with zero records. Main process logs one triple.

Deterministic resume stores epoch, next consumed local microbatch, epoch-start dataloader generator state, topology/batch/accumulation and completed optimizer count. Reconstruct shuffled dataset by replaying required epoch transitions; reconstruct/skip iterator in RNG isolation then restore saved process RNG. Worker prefetch is not consumed position. Reject incompatible exact-resume topology/config.

At each completed state: validation → samples → single adapter/state save → completed metadata/marker. Serialize each rank's RNG; publish readiness only after all required files. Existing ordinary-mode save/load stays unchanged. For older-checkpoint continuation use a distinct TensorBoard continuation directory linked to resumed step, avoiding incompatible later events in one trajectory.

## Verification and delivery
CPU tests prove contracts, not real forward or GPU memory behavior. F uses supplied direct SSH/SCP, actual loaders and models, fixed diagnostic target5 with controlled stop after checkpoint2, and independent log/event/state review. Tolerances fixed before comparing: exact seed/noise; CPU float64 reference rtol=1e-12/atol=1e-12; FP32 tensor arithmetic rtol=1e-5/atol=1e-6; bf16 forward/trajectory tolerance must be justified and recorded before first GPU comparisons, never widened afterward. Actual multi-GPU is separately unverified if only one GPU exists. F must record exact installed environment, model origins, full-data cache integrity, verified commands and external TensorBoard endpoint. No full training.


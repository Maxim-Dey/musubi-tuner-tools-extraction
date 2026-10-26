# Implementation Plan: Deterministic validation loss

**Branch**: codex/deterministic-val-loss | **Date**: 2026-10-05 | **Spec**: [spec.md](spec.md)

## Summary
Extend existing Qwen original forward/loss and Accelerate boundaries. Strict source-linked records feed a streaming deterministic evaluator. The existing loop owns completed-update scheduling. One FP32 adapter serves export and resume. A–D are local CPU development/acceptance; E is a separately gated real-model RunPod acceptance.

## Technical Context
Python >=3.10,<3.13; PyTorch; Accelerate1.6.0, Diffusers0.32.1, Transformers4.57.6, safetensors0.4.5; TOML/voluptuous and TensorBoard. Windows CPU development, Linux CUDA remote. Tests use real readers, tensors, tiny Qwen/LoRA, Accelerate and event files. No local model GPU run/download. Storage uses existing safetensors cache names plus minimal provenance/manifest/state JSON. System Python3.13 is unsupported; use a supported CPU environment. Missing README.md referenced by pyproject is baseline: use existing README.ru.md, never recreate the deleted file.

## Constitution Check
Pre/post-design checks: language, fidelity, compatibility, training invariants, one model, strict effective config, factual evidence and separate remote stage pass by design; require runtime proof. The original autonomous allowance was two recorded fix→verify rounds across CPU/GPU. The explicit user continuation on2026-10-06 authorizes T032 correction and required single-GPU proof as round3; the prior rounds remain recorded and the constitution file is unchanged. No extension hooks file exists. Coordinator owns documents/shared trainer and sole GPU operations; independent agents review.

## Project Structure and Ownership
- Coordinator: qwen_image_train_network.py, training/trainer_base.py, loop tests, feature docs and integration.
- Data agent: training/experiment_config.py, training/validation_inputs.py, parser_common.py, dataset/config_utils.py, dataset/cache_io.py, both Qwen cache entrypoints and tests.
- Math agent: training/validation.py, training/timesteps.py, numerical/RNG/multiprocess tests.
- State agent: training/experiment_state.py, training/accelerator_setup.py, checkpoint/event/retention tests.
One writer per file. No generic experiment system or new training engine.

## A: Configuration, inputs and shared loss
Numeric val parser defaults are None to detect explicit settings without val data; effective defaults 50/42/10/2. Existing unknown-key/type checks and CLI precedence stay. experiment_mode=false by default; primary TOML supplies root. Training relative file paths including resume/models/dataset/prompts/output/logging resolve at primary TOML; dataset paths at each dataset TOML. Absolute/legacy paths unchanged. Both cache commands take explicit --experiment_mode and --validation for strict deterministic val preparation, not invented --config_file. Remove role from val TOML; val_dataset_config identifies purpose. Early reject lossy state precision/full_fp16/full_bf16 for state-bearing experiment checkpoints.

prepare_validation_inputs(args,train_group) runs before accelerator/model loading under RNG guard. Strict source enumeration, SHA-sorted records, complete cache tensor/schema checks, content/caption/geometry provenance, disjoint source SHA sets and nonoverlapping cache dirs. Reject invalid batch/repeats/augmentation, same-stem cache collision, empty sets and unexpected effective train caches. Existing VAE posterior.mode/center crop is deterministic. Cache naming unchanged; opted-in cache writes attach provenance. Fingerprint excludes paths/order but includes sources/captions/preprocessing/both cache contents/seed/N1/N2. verify_unchanged before evaluation and resume detects changes; training never repairs caches. load_batch(index) reuses current cache reader without random loader/shuffle.

Share NetworkTrainer.compute_loss, adding optional fixed_sigmas to it and existing SD3 weighting helper without changing train path. Val reuses scale_shift_latents/call_dit, x=(1-t)*latent+t*noise, FP32 timestep=1000*t, sigma=t, no shift/grid/+1. Qwen call_dit requires_grad only when autograd enabled.

Seed bytes: compact ASCII JSON [seed,lowercase_image_sha256,i,j], no whitespace; SHA-256 first8 unsigned big-endian modulo 2**63, indices1-based. Vectors: (42,64zeroes,1,1)=2528747074945454131; (42,64zeroes,1,2)=3513524646004844088; (42,64f,10,2)=8961402154011714235; (-17,'0123456789abcdef'*4,1,1)=491017634604846590.
Dedicated CPU FP32 torch.Generator per seed, then cast to latent device/dtype. Iterate SHA, i, j; stream one noise and use float64 scalar sums and equal image averages. Same-environment noise exact; repeated loss tolerance declared before checks: CPU float32 rtol1e-6/atol1e-7, GPU BF16 rtol1e-4/atol1e-5. Paired same-batch next-update tensors expected exact, or predeclared justified arithmetic tolerance before GPU scenario. No cross-device/kernel bitwise guarantee; never weaken tolerance after a failure.

## B: Evaluation, steps and events
Snapshot/restore Python, NumPy, CPU/all initialized CUDA RNG, exact per-module base+adapter training flags, block-swap mode/settings in finally. no_grad/eval, offloader forward_only temporarily, setup/read also guarded. No val DataLoader: stable records. Main-only unwrapped transformer avoids DDP forward collectives; broadcast result or structured failure to all ranks, no padded duplicates. Two-process CPU tests include failure propagation.
Loop alone owns optimizer_global_step: increment sync_gradients AND not optimizer_step_was_skipped. Preserve legacy branches. New-mode order: update, increment, existing train log once, val, samples, flush, save. Baseline before next batch; final deduplicated. Absolute target; stop inner and outer loops, S>=target makes zero updates. Loader-position precision beyond existing behavior excluded.
Read resume metadata before accelerator setup. Save run location relative to experiment where possible, reuse with SummaryWriter purge_step=S+1. Read actual three scalar events at S before dedup; metadata alone insufficient. Absent durable event requires evaluation or identity-checked re-emission. State records explicit step, input fingerprint/latest metrics, adapter identity and run path. Flush before complete checkpoint. New mode rejects legacy unknown step; legacy mode keeps legacy loading.

## C: Checkpoint, samples and retention
Canonical experiment save hook clears Accelerate weights after network.save_weights(model.safetensors,float32,metadata); strict load hook loads this file and clears model list. No duplicated LoRA/DiT. All ranks save_state to retain supported RNG files; main writes completeness manifest with actual inventory/digests after synchronization. Missing/partial/sample-only directories cannot resume. Val-only mode keeps old layout but persists explicit metadata.
Step dir output/<output_name>-step<step>; samples directly under samples/ at actual generation step. Dedup final and periodic save. State-bearing adapter must be FP32; reject lossy settings early. Exact escaped output-name scan; inclusive current-window cutoffs. Absent weight window unlimited; absent/zero state window inherits weight window. Keep weights while either window requires, remove only manifest-owned expired state, invalidate complete marker, preserve samples/other experiments. No fabricated missing steps.

## D/E: Verification matrix
| Area | Local evidence | Remote evidence |
|---|---|---|
| Config/data | real readers/CLI types/unknowns; overlap, corrupt/stale/missing/colliding caches; cwd and legacy | actual paths/four cache runs/fixed manifest |
| Math | N1=2/10,N2=1/2, vectors/new process, fixed sigma, count, unequal resolutions | real full10x2, finite/repeated losses |
| Isolation | tiny actual Qwen+LoRA dropout, checkpointing, exceptions, RNG/grads/next update/optimizer/scheduler | paired same batches/RNG and all used CUDA RNG |
| Loop/events | real loop fixture, accumulation2, overflow, targets4/5, resume3→5, S>=target, rollback/missing events | programmatic tags/steps/dedup; multi-GPU execution excluded by user |
| State | real Accelerate FP32 roundtrip/inventory, resume update, final dedup, windows/name isolation/sample-only | actual continued training, samples/retention |
| Delivery | command9 config/prompt preservation, separate short configs, README/help and full CPU suite/scoped lint | tested-tree identity, environment/config/commands/events/artifact paths |
Remote starts only after A–D gate. One GPU owner; isolated checkout/environment/output, independent real val, accumulation>1/dropout>0, target3→5 plus target4 interval2, full10x2 pass. Reduced acceptance resolution/batch explicitly recorded; never claim profile batch16 tested. Supplied SSH/key, no key output/new Pod/shutdown. Blocked checks remain not run. Full proof in validation.md and gpu-validation.md.

## Complexity Tracking
No violations. Small input/eval/state helpers support existing trainer with directly testable requirements.

Acceptance event clarification: stop3 then resume5 yields combined0,2,3,4,5; uninterrupted target5 yields0,2,4,5; target4 yields0,2,4. --validation enables provenance independently of path semantics; only --experiment_mode changes path resolution.

Computation identity refinement during initial B implementation: same validation series additionally persists the weighting scheme, precision/forward configuration and streaming SHA256 identity of frozen DiT/merged base adapters (content, not path). This prevents reusing events after a mathematically different loss or base model is selected. It implements FR005/011/016 rather than changing their meaning. The run-local validation_events.json ledger binds step/input/computation/adapter identity to durable scalar values; partial triplets may re-emit the saved values only when these identities match.

## T032 continuation plan — 2026-10-06

Scope: acceptance probe and its focused tests; change production training only if independently confirmed evidence requires it. Reproduce the real Accelerate prepared-loader lifecycle boundary using1/2batches with accumulation2 before implementation, retain the3-batch control and native CLI/resume tests. Use normal loader lifecycle cleanup, avoid replay under an ended-loader flag, and require exactly one completed warmup/control/val-branch update. Preserve full10x2 repeated seeds/noise/metrics and exact paired state comparisons; retain rtol1e-4/atol1e-5 for repeated BF16 loss and0/0 for paired tensors. Transfer verified local bytes to the existing isolated RunPod checkout, use the same real cached inputs and checkpoint3 weights, and run only the remaining single-GPU probe. Then independently review evidence and run append-only converge; mark completion only on actual pass. Do not run multi-GPU or batch16/1024.

Paired GPU proof refinement during authorized T026 continuation: establish a same-state no-validation replay first. If the ordinary CUDA execution is non-repeatable, retain its failure and use an explicitly recorded strict deterministic-algorithm acceptance environment (CUBLAS workspace configured before CUDA). Require exact0/0 baseline replay and val-on/off comparisons; preserve production defaults and distinguish this controlled proof from default-kernel bitwise reproducibility. The initial default-mode gradient failure remains evidence, not a relaxed tolerance.

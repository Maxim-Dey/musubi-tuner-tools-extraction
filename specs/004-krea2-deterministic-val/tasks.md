# Tasks: Deterministic Krea2 Raw validation

**Input**: spec.md, plan.md, research.md, data-model.md, contracts/validation.md.
All tasks reference original source commands1–10 through FR/SC IDs. CPU checks never close GPU acceptance. Document owner alone checks boxes after evidence and independent review.

## Phase 1: Setup
- [x] T001 [A; FR-020] Record baseline ownership, user diff and initial evidence in specs/004-krea2-deterministic-val/verification.md.

## Phase 2: Foundational contracts
- [x] T002 [A; FR-001,FR-020] Confirm sole-owner integration boundary and legacy opt-out guard in src/musubi_tuner/training/trainer_base.py against contracts/validation.md before integration.

## Phase 3: US1 — A effective configuration and preflight
Independent test: CPU synthetic configs/files fail before model loading and preserve old mode.
- [x] T003 [US1] [A; FR-002,SC-001] Add strict effective/explicit val-group parsing in src/musubi_tuner/training/parser_common.py: period integer >=1; seeds integer; N1 even integer >=2; N2 integer >=1; boolean is not integer; experiment_mode/auto_cache default false.
- [x] T004 [US1] [A; FR-003,SC-005] Resolve opt-in main local paths and dataset-relative paths, preserving absolute paths/module IDs/URLs and legacy behavior, in src/musubi_tuner/training/parser_common.py and src/musubi_tuner/dataset/config_utils.py.
- [x] T005 [US1] [A; FR-004,SC-001] Implement stable complete image/caption inventory, content IDs, leakage/duplicate conflicts, val batch1/repeat1 and deterministic input restrictions in src/musubi_tuner/training/krea2_validation_inputs.py.
- [x] T006 [US1] [A; FR-001–FR-004,SC-001] Test defaults/TOML/CLI/unknown keys, partial group/bools, missing/empty data/captions, arbitrary cwd/absolute paths and SHA duplicate/leakage/conflicts in tests/test_krea2_validation_inputs.py; independent review evidence in specs/004-krea2-deterministic-val/verification.md.

## Phase 4: US1 — B cache provenance and frozen inputs
Independent test: CPU mocked cache producers prove missing-only generation/reuse/failure without loading real models.
- [x] T007 [US1] [B; FR-005,SC-001] Extend existing Krea2 cache provenance via src/musubi_tuner/dataset/cache_io.py, src/musubi_tuner/krea2_cache_latents.py and src/musubi_tuner/krea2_cache_text_encoder_outputs.py without altering existing tensor/filename contracts; reject source/component/processing mismatch and unknown legacy provenance in new modes; detect stem collisions.
- [x] T008 [US1] [B; FR-005] Reuse existing cache entrypoints for missing train/val files only, main-rank writes/barrier, free VAE/encoder before DiT, in src/musubi_tuner/training/krea2_validation_inputs.py; missing with auto_cache=false fails.
- [x] T009 [US1] [B; FR-006] Freeze relocation-neutral semantic manifest with source/caption/bucket/cache/component/noise identities; validate resume before models and active inputs on each pass in src/musubi_tuner/training/krea2_validation_inputs.py. Missing or incompatible old manifest fails, never regenerated.
- [x] T010 [US1] [B; FR-005–FR-006,SC-001] Test compatible reuse, missing-only preparation, Qwen Image/legacy/cache drift, relocation, captions/buckets/model changes, race serialization and release seams in tests/test_krea2_validation_inputs.py; run tests/test_krea2_cache_contract.py and independent review in specs/004-krea2-deterministic-val/verification.md.

## Phase 5: US2 — C deterministic loss
Independent test: CPU known tensors plus independent reference and process tests; actual forward stays F.
- [x] T011 [US2] [C; FR-007,SC-002] Implement exact midpoint levels N1=2/10, canonical SHA seed test vector and isolated fixed device/dtype generator in src/musubi_tuner/training/krea2_validation.py; no step/path/rank/order inputs.
- [x] T012 [US2] [C; FR-008,SC-002] Implement direct interpolation/1000*t and target noise-latents with analytic fixed-t training-meaning weighting, normalized cached batches and existing call_dit in src/musubi_tuner/training/krea2_validation.py; add torch.is_grad_enabled guard only in src/musubi_tuner/krea2_train_network.py.
- [x] T013 [US2] [C; FR-009,SC-002] Implement equal-image means and unpadded rank slicing, float64 sum/count reduction including empty ranks, main-only triple in src/musubi_tuner/training/krea2_validation.py; never average local means equally.
- [x] T014 [US2] [C; FR-010,SC-003] Add finally-based RNG/mixed module modes/GC/block-swap restoration, no autograd/dropout or weight copies, in src/musubi_tuner/training/krea2_validation.py.
- [x] T015 [US2] [C; FR-007–FR-010,SC-002–SC-003] Test independent formulas/MSE/weighting, seed/noise reorder/resume, size-balanced means, unequal/empty distributed CPU aggregation, unchanged state and injected exceptions in tests/test_krea2_validation.py; independent numerical review and declared tolerances in specs/004-krea2-deterministic-val/verification.md.

## Phase 6: US3 — D schedule and coherent state
Independent test: CPU lifecycle fixtures exercise real integration boundaries; GPU trajectory proof stays F.
- [x] T016 [US3] [D; FR-011,SC-004] Integrate preflight before models, cache prep before DiT and validation 0/period/final on completed non-skipped optimizer updates in src/musubi_tuner/training/trainer_base.py; sampling independent, no duplicate val.
- [x] T017 [US3] [D; FR-012,SC-003–SC-004] Persist/restore optimizer count, exact consumed train cursor/epoch-generator/topology, per-rank RNG, manifest and validation metrics; reconstruct iteration in RNG isolation in src/musubi_tuner/training/trainer_base.py; incompatible exact resume fails.
- [x] T018 [US3] [D; FR-013,SC-004] Implement opt-in Accelerate save/load hooks producing one FP32 model.safetensors and complete staged state/manifest metadata, successful final actual-step save, adapter-only load and native resume in src/musubi_tuner/training/trainer_base.py and src/musubi_tuner/utils/train_utils.py; explicit incompatible save_precision fails.
- [x] T019 [US3] [D; FR-014,SC-004] Implement state-window precedence and existing boundary semantics deleting whole obsolete current-experiment folders; bind same-step samples or separate numbered unsaved-step samples in src/musubi_tuner/training/trainer_base.py and src/musubi_tuner/utils/train_utils.py.
- [x] T020 [US3] [D; FR-011–FR-012] Log exact three val tags at optimizer steps main-only and branch TensorBoard continuation when older checkpoint has later events in src/musubi_tuner/training/trainer_base.py.
- [x] T021 [US3] [D; FR-011–FR-014,SC-003–SC-004] Test accumulation>1/skipped update, initial/period/final/duplicate/resume cursor, sole FP32 adapter/reload, incomplete write, retention boundaries/precedence/nonmatching intervals, samples and legacy formats in tests/test_krea2_experiment_state.py plus tests/test_save_precision.py; independent resume review in specs/004-krea2-deterministic-val/verification.md.

## Phase 7: US4 — E main configurations and local acceptance
Independent test: actual parser/schema from different cwd, full-data inventory, existing regressions and independent A–E review.
- [x] T022 [US4] [E; FR-015–FR-016,SC-005] Generate exact source-command9 main settings and separate train-dataset.toml/val-dataset.toml in config_for_krea2_raw_lora/, preserving legacy dataset.toml and all source data; no active diagnostic limits/resume/REPLACE_WITH values.
- [x] T023 [US4] [E; FR-015,SC-005] Preserve ten user subjects with supported prompt flags and values, empty negative CFG behavior in config_for_krea2_raw_lora/sample_prompts.txt; put short-run configurations only under config_for_krea2_raw_lora/diagnostics/.
- [x] T024 [P] [US4] [E; FR-015,FR-020] Update README.md, docs/krea2.md (use existing relevant Krea2 document if differently named) and docs/sampling_during_training.md for options, paths, graphs, val/cache/isolation, checkpoints/window/resume and existing TensorBoard dependency installation; review pyproject.toml without redundant dependency.
- [x] T025 [US4] [E; FR-001–FR-016,SC-001–SC-005] Run actual parser/schema/other-cwd main configs, complete inventory/prompts and focused plus existing Krea2 regression suites; independently review all A–E and record exact commands/results/version in specs/004-krea2-deterministic-val/verification.md. CPU evidence does not close F.

## Phase 8: US4 — F remote operational acceptance
Independent test: actual supplied Pod artifacts/events/state and HTTP response, independently reviewed.
- [x] T026 [US4] [F; FR-017] Inspect supplied Pod /workspace/environment/models/processes/GPU via authorized SSH, preserve data, transfer actual implemented dirty working tree to safe workdir and record exact content/version in specs/004-krea2-deterministic-val/verification.md; inaccessible SSH stays open with exact error.
- [x] T027 [US4] [F; FR-016–FR-017] Prepare compatible project/TensorBoard/PyTorch environment, use current hf/primary sources and real loaders to verify/download only missing compatible DiT/VAE/Qwen3-VL/tokenizer under /workspace/models without renaming; sync real paths into config_for_krea2_raw_lora/train.toml and record origins/hashes in specs/004-krea2-deterministic-val/verification.md.
- [x] T028 [US4] [F; FR-005–FR-006,FR-017] Transfer and verify full107/20 file hashes/captions, prepare separate full caches and demonstrate reuse plus fail-fast negative tests on isolated copies; record /workspace actual paths and evidence in specs/004-krea2-deterministic-val/verification.md.
- [x] T029 [US4] [F; FR-018,SC-002–SC-004,SC-006] Run actual single-GPU isolated batch1/accum2/N1=2/N2=1/period2/final5 and separate N1=10/N2=2, verify points0,2,4,5/repeat/reference, target5 controlled-stop2/resume5 and val/no-val trajectories with predeclared tolerances; retain configs/logs under config_for_krea2_raw_lora/diagnostics/ and record evidence in specs/004-krea2-deterministic-val/verification.md.
- [x] T030 [US4] [F; FR-013–FR-014,FR-018,SC-004] Inspect actual safetensors FP32/sole adapter and separate reload/full state, retention/same-step samples/event tags/steps/no resume duplicates, GC/blockswap restoration and next train step, actually used quantization; record artifacts in specs/004-krea2-deterministic-val/verification.md.
- [ ] T031 [US4] [F; FR-009,FR-018,SC-006] If supplied multiple GPUs, compare real uneven/empty-rank distributed validation against one process; otherwise explicitly retain real multi-GPU criterion unverified in specs/004-krea2-deterministic-val/verification.md, without renting resources or treating CPU tests as proof.
- [x] T032 [US4] [F; FR-016,FR-019,SC-006] Independently verify main full-data preflight from other cwd, final path/config synchronization, release only created diagnostics/GPU, start persistent user-output TensorBoard0.0.0.0:6006 and verify local plus actual external HTTP; record verified activation/new/resume/TensorBoard commands and user-state absence handling in specs/004-krea2-deterministic-val/quickstart.md and verification.md; no full training or Pod shutdown.

## Phase 9: Cross-cutting completion
- [x] T033 [FR-020,SC-006] Independently review implementation against spec/plan/tasks and full commands1–10; execute append-only converge on specs/004-krea2-deterministic-val/tasks.md, record confirmed discrepancies and external blockers in specs/004-krea2-deterministic-val/verification.md; apply at most two separately logged global fix→verify rounds only if required and available.

## Dependencies and parallel work
T001→T002→A(T003–T006)→B(T007–T010)→C(T011–T015)→D(T016–T021)→E(T022–T025)→F(T026–T032)→T033. Each group's review is required before dependent implementation. No tasks are closed merely because their group starts. T024 may run beside T022/T023 because files differ; avoid shared docs ownership. Independent read-only/reference checks can run beside implementation, but checkpoint/trainer integration is serialized. A/B owner T003–T010; C/D owner T002,T011–T021; E owner T022–T025; one remote owner T026–T032; independent reviewer across groups. T001/T033 and journal are document-owner tasks. MVP for local demonstration is US1 fixed inputs, then US2; whole feature requires every A–F acceptance, not MVP only.


## Implementation handoff status (2026-10-27)
T003–T010: implementer work and verification complete (131 tests, Ruff, diff check); independent A/B acceptance pending. T022–T024: configs/docs/diagnostic artifacts implemented with actual parser/schema checks; independent E review pending. T025: final integrated local acceptance pending C/D and reviewer. Boxes above intentionally remain open where a task includes independent acceptance not yet completed. See verification.md for exact evidence and superseded preliminary results. F remains entirely unperformed; correction rounds0/2.

## Confirmed discrepancy correction — aggregate round 1
- [x] T034 [US1] [B; FR-004–FR-006,SC-001] Fix confirmed R1 in src/musubi_tuner/training/krea2_validation_inputs.py: retain all duplicate source records for cache provenance/compatibility while unique metrics count once; add combined compatible-copy+missing-auto-cache+repeat-reuse and incompatible nonselected-copy cache tests in tests/test_krea2_validation_inputs.py, run affected regressions and obtain independent verification recorded in specs/004-krea2-deterministic-val/verification.md. Depends on initial T003–T010; correction round1 is recorded OPEN before code edits. A–E gate stays failed until verified.


## Accepted local status after round 1
Independent reviewer verified R1/T034 closed and A–E LOCAL GATE PASS: affected suites144 passed, actual-loop/state suite17 passed, diff check0. T001–T025 and T034 complete; earlier provisional-open notes above are historical and superseded by this acceptance. F T026–T032 and T033 remain open. CUDA/multi-GPU/real-model and operational evidence is not implied by local completion. Correction rounds used1/2, round1 successfully closed.


## F partial acceptance (2026-10-27)
T026 complete by actual dirty-tree transfer and independent runtime-hash verification. T027/T028/T032 partial and open; T029/T030 blocked by official Raw DiT HTTP403; T031 real multi-GPU unverified because only one GPU supplied. Public254 caches, data hashes, parser inventory and externally verified persistent TensorBoard do not close actual DiT training/loss/resume/events/state criteria. See final F evidence and quickstart; no round2 opened.


## Command11 assessment completed
Converge found no newly confirmed code defect; existing external/operational gaps remain T027–T032. No duplicate tasks appended. During converge this file was byte-for-byte unchanged; T033 is marked complete only in separate subsequent bookkeeping as execution of review/converge, not feature completion. Command12 skipped; no round2. Whole feature incomplete until mandatory remote acceptance is fulfilled.

## Final aggregate correction — round 2 of 2
- [x] T035 [US3] [D/F; FR-011–FR-012,SC-003–SC-004] Fix confirmed R2 only in src/musubi_tuner/training/trainer_base.py: avoid empty consumed-epoch tracker loss evaluation on exact-boundary resume while preserving hooks/replay and actual-batch logging; add real-tracker boundary, partial7-step and AMP-skipped-real-batch regressions in tests/test_krea2_experiment_state.py; verify original unchanged checkpoint2 resume to target5 and deterministic split2/resume5 versus continuous5, and record independent evidence in specs/004-krea2-deterministic-val/verification.md. Round2 recorded OPEN before edits; no math/utility rewrite or third correction round.
- [x] T036 [US3] [D/F; FR-011–FR-014,SC-004] Fix confirmed R2b only in src/musubi_tuner/training/trainer_base.py: on actual resume retain restored sampled-step dedup state when checkpoint samples already exist, preventing orphan S2 resampling; add saved-sample/no-orphan and legitimate unmatched/new-sample regressions in tests/test_krea2_experiment_state.py; preserve frozen checkpoint2 file hashes and independently inspect actual resumed events/checkpoints/samples, recording evidence in specs/004-krea2-deterministic-val/verification.md. Shares final round2 with T035 and sole trainer owner.


## Final supplied-single-GPU acceptance
Independent remote round2 gate PASS: R2/R2b fixed and actual resume/sample paths verified on final patch, deterministic matched-control val/no-val and resume exact; production preflight/runtime identity/TensorBoard checked. T027–T030,T032,T035,T036 complete. T031 alone remains open because real multi-GPU was unavailable. Strict whole-feature completion is not claimed; main experiment is operationally ready on the supplied single GPU. Both allowed correction rounds consumed and successfully closed; no residual confirmed code defect or further correction pass.

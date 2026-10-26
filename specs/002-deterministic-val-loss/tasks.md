# Tasks: Deterministic validation loss

Coordinator owns documents/shared trainer and GPU processes. One writer per file. Groups execute A→B→C→D→E; [P] only independent files inside a group. Completion requires stated evidence, never just implementation.

## Phase 1: A — Foundation and deterministic loss (T001–T006)

Goal: strict inputs/effective configuration and exact shared math. Independent acceptance: real readers and CPU tensors with deterministic seed subprocesses.

- [x] T001 [A] Establish supported CPU runtime, baseline tests and validation.md (FR-025,028); no dependencies; done when environment/initial results/protected user prompt hash recorded.
- [x] T002 [P] [A] Implement effective flags/strict types/CLI priority/opt-in paths in training/experiment_config.py, parser_common.py, dataset/config_utils.py and both cache entrypoints; tests/test_validation_config.py (FR-001,002,006,019); depends T001; done on invalid/unknown settings, other cwd and legacy controls.
- [x] T003 [A] Implement strict source identity/provenance/cache contract in training/validation_inputs.py, dataset/cache_io.py and cache entrypoints, tests/test_validation_inputs.py (FR-003–005,020); depends T002 interface; done on nonempty/disjoint/immutable/colliding/missing/corrupt/stale inputs, deterministic encoding and all datasets equal population.
- [x] T004 [P] [A] Implement levels, seeds, streaming noise and means in training/validation.py and tests/test_validation_math.py (FR-007–011,014); depends T001; done on N1=2/10,N2=1/2 vectors/new-process/reordered/moved images, exact forward counts and unequal resolutions.
- [x] T005 [A] Extend fixed sigma weighting through timesteps.py and shared trainer_base.compute_loss, guard Qwen autograd setup (FR-008,011,013); depends T004; done on shared target/mixing/weighting plus unchanged train regressions.
- [x] T006 [A] Integrate strict inputs into Qwen preflight before model loading, verify all A checks and update validation.md (FR-001–011,019,025); depends T002–T005; done when CPU group A checks pass.

## Phase 2: B — Evaluation, scheduling and resume (T007–T012)

Goal: real loop integration and durable resumed events. Depends completed A. Independent acceptance: actual loop with small CPU model boundary, actual optimizer/Accelerate/TensorBoard.

- [x] T007 [P] [B] Implement evaluation context/main-rank unwrapped forward/error broadcast in training/validation.py and tests/test_validation_runtime.py (FR-010,013,014); depends T006; done on dropout/checkpointing/exceptions/modes/all RNG, two-process synchronization and next-update equivalence.
- [x] T008 [P] [B] Implement metadata/event helpers in training/experiment_state.py, accelerator_setup.py and tests/test_validation_events.py (FR-015–018); depends T006; done on same-run resume, actual event reconciliation/purge and missing-event dedup cases.
- [x] T009 [B] Connect existing trainer_base.py loop to evaluator, explicit completed updates and absolute target (FR-012,015–018); depends T007,T008; done on 0/2/4/5,0/2/4,accumulation2,overflow,resume3→5,S>=target and no synthetic train points.
- [x] T010 [B] Persist val/run/step metadata through existing legacy-layout state hooks when val enabled (FR-016–018); depends T008,T009; done on real Accelerate metadata roundtrip and legacy controls.
- [x] T011 [B] Add tests/test_validation_loop.py exercising real _run_training_loop and event files, paired next update/batch/noise/gradients/optimizer/scheduler (FR-010–018); depends T009,T010; done on mandatory loop/isolation scenarios including errors.
- [x] T012 [B] Review B independently and record evidence in validation.md (FR-025,028); depends T007–T011; done when all B checks pass.

## Phase 3: C — Unified experiment saves (T013–T017)

Goal: one resumable FP32 adapter checkpoint, samples and retention. Depends completed B. Independent acceptance: real CPU Accelerate artifact roundtrip/inventory.

- [x] T013 [P] [C] Add canonical FP32 save/load hooks, all-rank RNG and complete inventory in training/experiment_state.py; tests/test_experiment_state.py (FR-018,020–022); depends T012; done on one LoRA/no DiT, exact parameters/optimizer/scheduler/RNG/metadata and continued update, corrupt/sample-only rejection.
- [x] T014 [C] Add inclusive retention/name isolation in training/experiment_state.py and tests/test_experiment_retention.py (FR-023); serialized after T013 same-file edits; done on windows40/20,40/60,inheritance,unlimited/boundaries/final/missing steps.
- [x] T015 [C] Integrate save hooks/periodic-final dedup and step-linked samples into trainer_base.py (FR-019–023); depends T013,T014; done on mismatched sample/save periods and actual resume; legacy layout unchanged.
- [x] T016 [C] Verify early FP32 restrictions and new layout CLI/metadata integration in tests/test_validation_config.py and tests/test_validation_loop.py (FR-002,018,021,022); depends T015; done on lossless controls and lossy errors.
- [x] T017 [C] Independently audit tensor files, retention and legacy compatibility; record validation.md (FR-018–023,028); depends T013–T016; done when all C checks pass.

## Phase 4: D — Configurations and local acceptance (T018–T022)

Goal: usable example/instructions and complete local gate. Depends completed C.

- [x] T018 [P] [D] Update config_for_qwen_image_lora/train.toml, train-dataset.toml, val-dataset.toml preserving command9 values and sample_prompts.txt bytes; add separate acceptance configs (FR-024,025); depends T017; done with real parser/schema/effective setting checks.
- [x] T019 [P] [D] Update README.ru.md, relevant docs/links and quickstart with four cache commands/train/other cwd/resume/TensorBoard, inherited state window and readiness limits (FR-019,024); depends T017; done with CLI/help/docs audit and no restored README.md.
- [x] T020 [D] Run retained/full new CPU regressions, scoped lint, diff and real entrypoint checks; document baseline vs regressions (FR-025); depends T018,T019; done with passing required tests and preserved data.
- [x] T021 [D] Independent speckit-converge local A–D review; append confirmed tasks only, remote E stays pending (FR-028); depends T020; done when no confirmed local gaps; corrections separately recorded within total two-round cap.
- [x] T022 [D] Complete requirement→implementation→check→result matrix and mark local gate in validation.md (FR-025,028); depends T021; done only when A–D complete, GPU still not run.

## Phase 5: E — RunPod real-model acceptance (T023–T029)

Goal: operational proof on actual model. Depends mandatory local gate T022. Coordinator exclusively owns GPU; agents review code/evidence. SSH/model/independent-data blockers remain open.

- [x] T023 [E] Inspect supplied SSH/server/GPU/models and independent train/val; transfer tested working tree to isolated checkout and install compatible environment, verify hashes/effective configs (FR-026); depends T022; done with resource/code/environment evidence in gpu-validation.md, no source changes/new Pod.
- [x] T024 [E] Run actual latent/text caches for train and independent val, verify immutable provenance/manifest (FR-004,005,027); depends T023; done with four commands/artifacts, no synthetic substitution.
- [x] T025 [E] Run sequential real train/val schedule scenarios accumulation>1/dropout>0, full N1=10,N2=2 and repeated same-weight loss at fixed tolerance (FR-007–017,027); depends T024; done on finite/count/math/expected events0,2,4,5 and0,2,4 without duplicates.
- [x] T026 [E] Run paired real identical-batch/RNG val-on/off next-update isolation and checkpointing/RNG checks (FR-010,013,027); depends T025; done on weights/optimizer/scheduler equivalence, not inferred from launch success. Exact0/0 proof passed in the explicitly recorded deterministic acceptance environment; default-mode no-val replay failure is retained separately.
- [x] T027 [E] Verify FP32 one-copy inventory, real save/resume3→5/continued updates, event reader, samples/retention and user5000 profile effective paths without full run (FR-018–024,027); depends T025; done with actual artifacts/logs/events.
- [x] T028 [E] Record real multi-GPU execution as excluded by the explicit2026-10-06 user instruction; retain prior one-GPU topology evidence and existing CPU distributed tests (FR-014,022,027). No real multi-GPU pass is claimed.
- [x] T029 [E] Perform independent final speckit-converge A–E, factual evidence/tasks/data audit and concise Russian report (FR-025–028); depends T023–T028; done only if mandatory remote scenarios pass; any fixes consume shared cap and repeat affected CPU/GPU checks. Final user-authorized continuation convergence found zero gaps; tasks remained byte-for-byte unchanged during converge, then this completion was recorded separately.

## Dependencies and implementation strategy

A foundations precede B, then C, D and E. T002/T003 share an owner; T004/T005 share math agent plus coordinator's shared trainer. B evaluation and event helpers can proceed in parallel. C helper edits are serial despite independent tests. D config and documentation owners use disjoint files. E always sequential GPU processes. Complete initial implementation before counting post-implementation correction rounds; never regenerate existing tasks after implementation starts.





## Phase 6: Convergence

- [x] T030 [D] Reject invalid experiment output_name directory components in real pre-model Qwen argument validation, with source-specific guidance and a regression proving the model loader is not reached; retain legacy behavior (FR-002, Constitution VI; partial; HIGH). Evidence: output_name="../nested" passes validate_training_args with valid input paths and is rejected only later by checkpoint_directory. Verification: actual entrypoint early-error fixture plus config and checkpoint regressions.

## Phase 7: Remote correction round2

- [x] T031 [E] Delegate the acceptance probe CLI through native qwen.main so DiT/VAE defaults and model-version resolution occur before trainer.train; preserve report routing and restore temporary globals; test actual CLI initialization and rerun full real-model repeatability/paired isolation (FR-027; confirmed F2 in evidence/probe-initial-failure.log). The CLI correction consumed the second round under the original autonomous allowance; its remaining GPU proof passed in the2026-10-06 user-authorized continuation.


## Phase 8: Convergence

- [x] T032 [E] Repair the acceptance probe prepared-loader collection/replay lifecycle when loader length equals or is shorter than gradient_accumulation_steps; add actual Accelerate boundary regressions with 2 and 1 batches at accumulation2, then record the real full10x2 repeatability and exact paired next-update proof without changing tolerances (FR-027, SC-002/005; partial; HIGH; F3). Original evidence: evidence/probe-round2.log/json reports two warmup optimizer updates instead of one; last-yield iterator.close leaves Accelerate end_of_dataloader active. The original attempt stopped at exhausted autonomous budget2/2 with T025/T026/T031 proof incomplete. The explicit2026-10-06 user continuation authorized round3. Loader1/2 regressions reproduced the failure before repair; native end() cleanup and loader1/2/3 controls now pass. Evidence/probe-deterministic.json records full10x2 repeatability, four single-update stages, exact no-val replay and exact paired isolation. Prior failures and original tolerances remain unchanged; see gpu-validation.md for controlled-environment qualification.

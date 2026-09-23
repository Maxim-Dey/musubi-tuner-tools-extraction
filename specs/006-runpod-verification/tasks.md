# Tasks: Finite RunPod Verification

**Inputs**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), [quickstart.md](quickstart.md), and the fixed [G01–G10 protocol](verification-plan.md).

**Run paths**: `$R=/workspace/musubi-val-loss-tests/<unique_run_id>`, `$SRC=$R/code`, `$PY=$R/.venv/bin/python`, `$CASES=$R/experiments`. Assign these only after T002. All remote work uses the user's selected Pod. Preserve original models, datasets, results, user processes, and Pod lifecycle. An unmet prerequisite blocks dependent tasks and is recorded as `NOT RUN`; a measured failure is `FAIL`. Neither is a pass.

## Phase 1: Preparation — setup

**Goal**: Establish access and one separate run location before changing remote files.

- [X] T001 Inspect the selected Pod through host-key-checked interactive SSH and record observed `/workspace`, GPU count/type/VRAM, driver, Python, CUDA/PyTorch, space, model/data paths, process inventory, and transport constraints in `$R/evidence/environment.json` after T002; input: the user's selected Pod and [research.md](research.md); done: access and actual hardware are distinguished from assumptions, the private key stays local and is never printed, and unavailable access blocks all later tasks (FR-002, FR-017).
- [X] T002 Create one collision-free UTC `run_id` after checking the target does not exist; create only `$R/{code,.venv,fixtures,experiments,scripts,outputs,evidence}/` and `$R/evidence/run.json`; input: T001 access; done: selected Pod/run identity is recorded and earlier trees, source data, and user processes remain untouched (FR-001, FR-003).

## Phase 2: Preparation — foundational gates

**Goal**: Prove exact tested code and an isolated compatible environment before model work.

- [X] T003 Round-trip a small byte-preserving transfer and evidence-retrieval canary, fetch base commit `e8af43abc32d967153e3aa3c31d50c77d3f7c47e` into `$SRC`, transfer the finished local modified/untracked runtime and G08 test overlay, and compare archive and every file SHA-256 in `$R/evidence/source-manifest.json`; input: T002 run and finished local tree; done: remote bytes match the local manifest. If Basic SSH transport is unreliable, stop and request exposed-TCP SSH; do not test default HEAD or publish code/credentials for transfer (FR-004, FR-018).
- [X] T004 Create `$R/.venv` using Python 3.12 and declared project dependencies, compatible pinned CUDA 12.8 torch 2.8.0/torchvision 0.23.0, AdamW8bit, TensorBoard and pytest; record versions/imports, model headers/paths and free VRAM in `$R/evidence/environment.json`; input: verified `$SRC/pyproject.toml` and T001 inventory; done: isolated `PYTHONPATH=$SRC/src` execution is viable without global upgrades, model downloads or optimizer updates (FR-002, FR-005).

**Gate**: T003–T004 must pass. A source, model-format, memory, or environment failure is reported; do not weaken 1024/BF16/no-FP8/no-block-swap settings to manufacture a pass.

## Phase 3: Preparation — User Story 1 (P1)

**Goal**: Freeze real role-separated inputs, build exact caches, and reject an unsafe training budget.

**Independent test**: Review source/environment identity, fixture/cache manifests, effective configurations, and maximum update/event grid without loading the model.

- [X] T005 [US1] Choose at most three train, two familiar, and three unfamiliar existing user-designated image/caption pairs; prove familiar image hashes are in train and unfamiliar hashes are disjoint, copy exact bytes to `$R/fixtures/`, rehash them and write `$R/evidence/fixture-manifest.json`; input: original user data and T001 inventory; done: roles, source/copy SHA-256s, dimensions/buckets and model references are frozen. If roles are ambiguous, ask one focused question and stop; invent no split or synthetic real-model image (FR-006).
- [X] T006 [US1] Use existing `$SRC/qwen_image_cache_latents.py` and `$SRC/qwen_image_cache_text_encoder_outputs.py` with isolated canonical train/val TOMLs under `$CASES/cache/` to build both caches, or verify exact matching reusable caches. Before freezing validation inputs, prepare G03's renamed/reordered copies of the same fixture images/captions and their basename-dependent, source-bound latent/text cache pairs in an isolated tree; input: T003–T005; done: `$R/evidence/cache-manifest.json` records commands, exits, original and renamed per-role cache hashes/bindings and Stage 1 fingerprints, and both validation input variants are then fixed (FR-006, FR-010).
- [X] T007 [US1] Create separate `$CASES/<case>/{train.toml,train-dataset.toml,val-dataset.toml}` and one fixed four-step sample prompt from `qwen_image_lora_val_example/`; input: frozen T005–T006 paths and [verification-plan.md](verification-plan.md); done: `$R/evidence/case-configs.json` records every deviation from the 1,600-step example and effective Qwen/LoRA 16/16, BF16, 1024 buckets, SDPA, AdamW8bit, 5e-5, gradient checkpointing, FP32 adapter, no FP8/block swap, warmup 2, seed 42, and each case's budget/interval/retention/protocol. The G03-renamed TOML uses T006's verified renamed cache pairs; G05-off omits both `experiment_dir` and `val_dataset_config`, uses explicit absolute data/model/output paths without experiment-root rebasing, and exercises the legacy validation-disabled path (FR-007, FR-010, FR-011, FR-017).
- [ ] T008 [US1] Add only bounded one-off `$R/scripts/{run_case.sh,real_probe.py,inspect_evidence.py}` and their hashes to `$R/evidence/source-manifest.json`; input: fixed case/evidence protocol and T003 source; done: launcher records exact command/PID/start/end/stdout/stderr/exit, refuses duplicate case IDs and inspects an existing PID after disconnect; probe captures bounded G03–G05 evidence and inspector reads events/packages without a new trainer or full-model dumps (FR-010–FR-018).
- [X] T009 [US1] Run a zero-update static gate on `$R/evidence/{source-manifest,environment,fixture-manifest,cache-manifest,case-configs}.json` and every effective TOML plus CLI override, writing `$R/evidence/preflight.json`; input: T003–T008; done: source, roles/hashes, original and renamed caches, model compatibility, memory, 1024/BF16 settings, predeclared tolerances, event steps and total maximum 38 plus conditional 2 are verified and any inherited 1,600-step budget is rejected before trainer DiT/LoRA load and optimizer updates (FR-001, FR-004–FR-007, FR-017).

**US1 checkpoint**: T009 must pass before any G case. A plausible filename is not proof of the correct source, role, or cache.

## Phase 4: Verification — User Story 2, G01–G05 (P1)

**Goal**: Measure schedule, fixed checks, independent loss mathematics, and unchanged training continuation.

**Independent test**: Compare actual events, captured checks, state snapshots and paired updates with predeclared counts and tolerances.

- [X] T010 [US2] Launch G01 once via `$R/scripts/run_case.sh` with `$CASES/g01/train.toml`; input: passed T009; done: `$R/evidence/G01/` contains command/PID/log/exit, effective config, update/LR/weight-norm/GPU traces, TensorBoard events and package/sample/sidecar inventory proving or refuting at most 12 completed updates with accumulation 2, finite loss/gradients and LoRA change after nonzero LR, exactly six tags at each of `0,4,8,12`, coalesced step-12 output and inclusive step-4 retention (FR-008, SC-002/005).
- [X] T011 [US2] Launch G02 once via `$R/scripts/run_case.sh` with `$CASES/g02/train.toml`; input: passed T009 and no concurrent full-model process; done: `$R/evidence/G02/` command/log/exit, event dump and package/sample/sidecar inventory prove or refute at most 10 updates with accumulation 1, events `0,4,8,10` and one complete final step-10 state outside save cadence (FR-009, SC-002).
- [ ] T012 [US2] Run `$R/scripts/real_probe.py g03` against one immutable G01 checkpoint named in `$R/evidence/selected-checkpoint.txt` using `$CASES/g03-{4x2,renamed,10x1}/train.toml`; input: T010 checkpoint and both T006-frozen original and renamed fixture/cache variants; done: `$R/evidence/G03/` records checkpoint hash, five zero-update calls (two same-process, one fresh process, one renamed/reordered copy with verified source-bound caches, one separate weights-only 10×1), exact IDs/levels/seeds/noise keyed by `(image ID,i,j)`, six finite metric differences within G03 tolerance, and no incompatible history append (FR-010, SC-003).
- [ ] T013 [US2] Run `$R/scripts/real_probe.py g04` and an independent calculation in `$R/scripts/inspect_evidence.py` on bounded `$R/evidence/G04/forwards.jsonl`; input: T012 checkpoint and captured actual t/latent/epsilon/output/target/weight with unequal role counts and available buckets; done: zero updates, one validation call and a reference that reconstructs midpoint levels, mixing, `1000*t`, `epsilon-latent`, directed exact-sigma weighting, per-image loss, image-first set/low/high means and half-mean within G04 tolerance without calling the tool's aggregator (FR-010, SC-003).
- [ ] T014 [US2] Run G05's one evaluation snapshot and sequential enabled/disabled four-update branches through `$R/scripts/real_probe.py g05` with `$CASES/g05-{snapshot,on,off}/train.toml`; input: equal initial adapter, batches/order/seeds and T009; done: `$R/evidence/G05/` proves or refutes exact module modes, Python/NumPy/torch CPU/used CUDA RNG, gradients, LoRA, optimizer/scheduler around evaluation, exact paired noise/timesteps and predeclared loss/parameter tolerances. The G05-off branch uses the legacy path with neither `experiment_dir` nor `val_dataset_config`, no validation/samples, at most 4+4 total updates and three validation calls, and only one full base model resident (FR-011, SC-004).

**US2 checkpoint**: Record PASS/FAIL/NOT RUN per case. A missing prerequisite stops its dependent probe; never train longer or relax a tolerance.

## Phase 5: Verification — User Story 3, G06–G08 (P1)

**Goal**: Verify separate current/best resume, genuine best/event attribution, and controlled failure branches.

**Independent test**: Hash immutable input states before resume, compare loaded state before an update, then inspect real event files and complete package ownership.

- [ ] T015 [US3] Copy one actual current and one actual best G01 package as immutable resume inputs into separate `$CASES/g06-{current,best}/output/` trees, hash source/copy, then launch each G06 resume once through `$R/scripts/run_case.sh`. Also copy the actual G01 best package into the G06-current tree's best location so its prior best score/history remains authoritative; verify all source/copy hashes before launch; input: T010 packages and an actual saved `s` divisible by 4; done: `$R/evidence/G06-{current,best}/` records exact loaded adapter/optimizer/scheduler/rank RNG/fingerprint, initial validation at `s`, first new training loss at `s+1`, final `s+4`, prior-best attribution, at most 4+4 updates and 2+2 validation calls. If either real package is unavailable, mark NOT RUN; do not fabricate it with extra training or require data-loader position identity (FR-012, SC-004).
- [ ] T016 [US3] Inspect real G01/G02/G06 TensorBoard files and output trees with `$R/scripts/inspect_evidence.py g07` into `$R/evidence/G07/results.json` and scalar/package inventories; input: T010–T011 and T015 available evidence; done: check six exact finite tags at due absolute steps, at most one tag/step within each run, retained training graphs, strict finite real `val_loss_mean` minimum mapped to the sole complete best, one adapter/sample set per saved step, step-0 best sample where enabled, inclusive retention and older-best protection; immutable G06 inputs are not counted as duplicate outputs (FR-013, SC-005).
- [X] T017 [US3] Run G08 controlled CPU tests once via `$R/scripts/run_case.sh` on `$SRC/tests/test_qwen_image_{experiment_states,experiment_paths,validation_inputs,validation_training,val_example,dataset_cache}.py` and save `$R/evidence/G08/{stdout,stderr,exit,junit.xml}`; input: verified T003 source/T004 environment; done: strict best/tie, coalescing/off-cadence best, retention, interrupted write, evaluation restoration and invalid config/empty-role/changed-input paths have explicit test outcomes, including the empty-role preflight case in `test_qwen_image_dataset_cache.py`, with zero real-model updates and synthetic metrics excluded from real GPU graphs (FR-014).

**US3 checkpoint**: G08 controlled evidence does not substitute for a failed real-model state or event case.

## Phase 6: Verification — User Story 4, conditional G09 (P2)

**Goal**: Check two ranks only if this same Pod has two suitable GPUs.

**Independent test**: Fresh hardware evidence either supports the bounded two-rank run/load or documents a NOT RUN.

- [X] T018 [US4] Refresh `nvidia-smi` in `$R/evidence/G09/`; if two suitable GPUs exist on the selected Pod, launch `$CASES/g09/train.toml` once with two Accelerate ranks and perform `$R/scripts/real_probe.py load-two-rank` on its actual package; input: T009 and G09 2×1/accumulation-2 config; done: at most two updates, events `0,1,2`, global reduction/count, no deadlock, one publication per tag, both rank RNG files and no-update two-rank load are evidenced. Otherwise record NOT RUN with observed one-GPU reason and zero updates; never rent another Pod (FR-015, SC-006/007).

## Phase 7: Report — User Story 4, G10

**Goal**: Publish a complete, bounded result while leaving the user's Pod running.

**Independent test**: Every G row can be audited from source and fixture identities, commands/exits, actual versus expected observations and evidence locations.

- [X] T019 [US4] Generate `$R/evidence/G10/summary.json` through read-only `$R/scripts/inspect_evidence.py g10`, retrieve concise evidence through the verified channel, and write local `specs/006-runpod-verification/{verification.md,verification.json}`; input: T001–T018 evidence and [verification-plan.md](verification-plan.md); done: all G01–G10 rows have PASS/FAIL/NOT RUN, FR IDs, conditions, identities, exact commands/exits, expected/actual counts and numeric differences, stdout/stderr/event/state paths and retrieval status; actual updates total at most 38 or conditional 40, and unavailable mandatory checks prevent overall PASS (FR-016–FR-018, SC-001/006/007).
- [X] T020 [US4] Reconcile `README.ru.md` with observed limits and link `specs/006-runpod-verification/verification.md`; save its diff and final agent-owned PID/process inventory in `$R/evidence/G10/`; input: T019 report and T001 process baseline; done: only still-running agent test processes are stopped after confirming no further updates, user processes and Pod remain running, and defects remain reported without server-only product fixes or changed tolerances (FR-002, FR-016–FR-018, SC-007).

## Dependencies and stopping rules

**Observed Stage 5 stop:** The selected one-L40S Pod passed T009 and G08, but G01 and G02 both failed at the mandatory step-0 sample with CUDA OOM before any completed update or package. One G01 allocator-only retry also failed. T008 remains incomplete because its G05 probe was not built after the required G01 adapter became unavailable. T012–T016 remain unchecked and are reported as NOT RUN in [verification.md](verification.md); G09 is NOT RUN because this Pod has one GPU. T010/T011 are checked as executed failure investigations, not as case passes.

The conservative execution order is T001→T002→T003→T004→T005→T006→T007→T008→T009→G01/T010→G02/T011→G03/T012→G04/T013→G05/T014→G06/T015→G07/T016→G08/T017→G09/T018→G10/T019–T020. No GPU model processes run concurrently. Local overlay hashing and read-only remote environment inventory are independent preparation work, but both must be reconciled at T009. G03/G04 require an immutable actual G01 checkpoint; G06 requires actual current and best packages. G09 may be NOT RUN. After SSH loss, inspect the original PID, exit file, logs and output before any retry. Never repeat a successful case without a code change or concrete unresolved question.

| Case | Maximum real-model completed updates | Maximum real validation calls | Expected validation steps |
| --- | ---: | ---: | --- |
| G01 | 12 | 4 | 0, 4, 8, 12 |
| G02 | 10 | 4 | 0, 4, 8, 10 |
| G03 | 0 | 5 | Fixed-check calls, no training |
| G04 | 0 | 1 | Fixed-check call, no training |
| G05 | 4+4 | 3 | Snapshot, enabled 0/4, disabled none |
| G06 | 4+4 | 2+2 | Each resume at s and s+4 |
| G07, G08, G10 | 0 | 0 | Read-only or controlled CPU |
| G09, conditional | 2 | 3 | 0, 1, 2 |

G01+G02+G05+G06 = **38** real-model completed updates; conditional G09 adds **2** for a hard maximum of **40**. A skipped attempt is recorded separately from a completed update. Counts, formulas and tolerances were fixed in [verification-plan.md](verification-plan.md) before remote execution. `NOT RUN` is never PASS.

## Phase 8: Convergence

- [X] T021 Arrange a user-selected environment with sufficient memory for the mandatory 1024/BF16/no-FP8/no-block-swap step-0 sample, then repeat only bounded G01/G02 in a new isolated run with the same predeclared controls, preserving the failed L40S evidence; per FR-008, FR-009, SC-002 (partial). G01 and G02 passed on the user-selected RTX PRO 6000 Pod; see [new run](runs/20260923T0705Z-b8ed90/verification.md).
- [X] T022 Finish and verify only the minimal G05 state/paired-run probe, transfer the bounded G03/G04 probe with source hashes, and keep all probes outside the production trainer; per FR-010, FR-011, T008 (partial). The probes ran on the new Pod; G03/G04 passed and G05 found a paired-training difference beyond the fixed tolerance.
- [X] T023 After T021 provides a complete immutable G01 package and T022 is ready, execute the dependency-blocked G03–G07 cases without changing their fixtures, tolerances, algorithms or update caps; append a new evidence-backed report rather than rewriting the L40S result; per FR-010–FR-013, SC-003–SC-005 (partial). The new run used exactly 38 optimizer updates and preserved the old report. G06-current failed because the chosen step-8 resume reached an already occupied step-12 best-package path; G07 records that incomplete attribution.

## Phase 9: Convergence

- [ ] T024 Establish whether the G05 paired-run difference is attributable to validation using the recorded state snapshots and the separate off/off baseline; retain G05 FAIL and the original tolerance unless controlled evidence meets FR-011 and SC-004. Do not infer a product defect from the current difference alone; see `runs/20260923T0705Z-b8ed90/verification.md` and `runs/20260923T075021Z-offoff-f6ce8c/verification.md` (partial).
- [ ] T025 Correct the G06-current verification fixture so its prior-best package and saved current step form a coherent resume state without a future-step best-name collision; then verify G06-current and dependent G07 attribution only if the server-verification scope is expanded again. Until then, retain their recorded FAIL statuses and do not claim FR-012, FR-013, SC-004, or SC-005 as passed; see `runs/20260923T0705Z-b8ed90/verification.md` (partial).

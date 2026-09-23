# Feature Specification: Finite RunPod Verification of Qwen-Image Validation

**Feature Branch**: `001-scope-qwen-image-lora` (current branch; feature directory is independent)

**Created**: 2026-09-23

**Status**: Draft

**Input**: Stage 5.1 of `config_for_qwen_image_lora/val_loss_speckit_commands.md`: a separate, finite remote operational verification of the completed Stage 1–4 Qwen-Image original LoRA validation and experiment-state features on the user's existing RunPod.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Verify the exact completed tool on real hardware (Priority: P1)

The user receives evidence that the code completed locally is the code tested against the existing original Qwen-Image model and user data on the selected Pod. Source models, working datasets, prior results, and the Pod remain intact.

**Why this priority**: A successful test of an old commit, a different dataset split, or an altered shared environment would not verify the delivered tool.

**Independent Test**: Compare source identity, environment inventory, fixture manifest, actual commands, and isolated result locations before interpreting any model result.

**Acceptance Scenarios**:

1. **Given** the selected Pod is reachable, **when** preparation completes, **then** the tested source matches the finished local implementation by commit or verified patch identity, and the environment, models, data, GPU, and available space have recorded observed values.
2. **Given** user data with unambiguous train/familiar/unfamiliar roles, **when** small test fixtures and caches are prepared, **then** their exact members and source hashes are recorded before any validation run, all familiar members belong to train, and unfamiliar members share no content with them.
3. **Given** the selected Pod is unavailable or code transfer cannot be verified, **when** preparation stops, **then** no substitute Pod or stale source is tested and the blocking condition is reported.

---

### User Story 2 - Check validation mathematics and training isolation (Priority: P1)

The user sees finite real-model evidence for the specified validation schedule, deterministic fixed checks, independently reconstructed loss and aggregation, and unchanged training behavior around validation.

**Why this priority**: A falling loss or generated image alone does not establish that validation evaluates the intended data or leaves training untouched.

**Independent Test**: Run the bounded G01–G05 cases with fixed fixtures and predeclared numeric tolerances; inspect event logs, independent calculation evidence, and controlled updates.

**Acceptance Scenarios**:

1. **Given** the primary 12-update run, **when** it completes, **then** validation appears once at absolute steps `0,4,8,12`, the completed-update count excludes accumulation microbatches, and LoRA changes on nonzero updates.
2. **Given** the 10-update run with a final step outside save cadence, **when** it completes, **then** validation occurs at `0,4,8,10` and one complete final package belongs to step 10.
3. **Given** unchanged weights and fixed inputs, **when** validation repeats without optimizer updates, **then** discrete image IDs, levels, seeds, and noise agree exactly on the same backend/device, and six metrics meet tolerances chosen before results are seen.
4. **Given** validation between controlled training updates, **when** states and next updates are compared, **then** model modes, random state, gradients, optimizer/scheduler state, training noise/timesteps, loss, and LoRA changes meet their respective exact or predeclared numeric criteria.

---

### User Story 3 - Verify one resumable best state and honest evidence (Priority: P1)

The user can trace real TensorBoard events to one complete best package, verify whole-package retention, and resume separately from current and best without mistaking a new run's initial event for a duplicate in the old run.

**Why this priority**: Accurate state ownership and absolute steps are necessary for safe continuation and trustworthy best selection.

**Independent Test**: Inspect G01/G02/G06 event files and published packages, then run the bounded current/best resume checks and controlled G08 failure branches.

**Acceptance Scenarios**:

1. **Given** valid completed validation events, **when** their unfamiliar-set means are compared, **then** the single best package corresponds to the strict finite minimum and contains its attributed weights, metrics, and samples.
2. **Given** published current and best packages, **when** each is resumed in its own isolated test experiment for four completed updates, **then** initial validation uses the saved step `s`, the first new training loss is at `s+1`, and the final absolute step is `s+4` with restored state.
3. **Given** retention at G01 step 12 with window 8, **when** outputs are inspected, **then** the complete step-4 package remains in its legitimate current or best location, older owned current packages are removed whole, and best remains protected. If step 4 is best, controlled G08 establishes the inclusive current-boundary behavior.

---

### User Story 4 - Close the bounded matrix without overstating results (Priority: P2)

The user receives a per-scenario PASS, FAIL, or NOT RUN report with reproducible evidence and explicit limits. If two suitable GPUs are present on the selected Pod, the user also gets a short multi-rank result.

**Why this priority**: A precise report separates confirmed behavior from unavailable checks and avoids further training to chase favorable outcomes.

**Independent Test**: Reconcile every G01–G10 row with its planned bounds, actual commands, measurements, files, and final process inventory.

**Acceptance Scenarios**:

1. **Given** two suitable GPUs, **when** G09 runs, **then** two ranks finish two updates without deadlock, publish each tag once, save both rank random states, and load the package with no further updates.
2. **Given** only one suitable GPU, **when** G09 is reported, **then** it is NOT RUN with the observed reason; no second Pod is rented or multi-GPU claim made.
3. **Given** any failed or unavailable mandatory check, **when** the report is completed, **then** its status and evidence remain visible and the remote verification is not declared fully passed.

### Edge Cases

- The Basic SSH connection is lost or a command response is uncertain: inspect whether the original test process or evidence still exists before retrying; never start a duplicate training run blindly.
- The remote Git history lacks the finished local changes: use a verifiable transfer or stop and request suitable access; never quietly test an older checkout or publish changes or credentials for transfer.
- The existing user data cannot support the requested role relationship or fixture sizes: ask one specific role question and do not invent a split or synthetic images to claim real-model success.
- A validation event is tied, nonfinite, or fails, or package writing is interrupted: verify that the old best remains valid through a bounded controlled branch, not extra training for a desired result.
- G03 changes from `N1=4,N2=2` to `N1=10,N2=1`: treat it as a separate validation-only experiment on the same weights, not a compatible resume or added history for the earlier protocol.
- A current package and an immutable copy used as a separate resume input have matching content: the input copy is not a duplicate output within one experiment. Duplicate detection is based on event attribution, metadata, and location rather than equality of optimizer-state hashes.
- Only one suitable GPU is available, or the selected GPU is not H200: record the observed hardware and limit the conclusion; do not create another Pod or call another GPU an H200 result.
- A short run shows decreasing loss or produces a PNG: neither alone proves correctness, image quality, or long-run stability.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: This MUST be a separate remote operational verification after local completion of [Stage 1](../002-val-loss-core/spec.md), [Stage 2](../003-val-loss-training/spec.md), [Stage 3](../004-experiment-training-states/spec.md), and [Stage 4](../005-val-loss-configs/spec.md). The constitution's Local Stage limits still govern the local stage; this feature authorizes only the finite remote tests below. It MUST NOT change the original training algorithm, broaden model scope, run normal 1,600-step training, tune learning rate/rank/seeds, or train until loss or images improve.
- **FR-002**: Preparation MUST observe and record access to the user-selected Pod, `/workspace`, GPU model/count/VRAM, driver, Python, CUDA/PyTorch, free space, existing model/data paths, and model-file compatibility before claiming them. If access fails, request current credentials instead of replacing the Pod. The user stops the Pod; the agent MUST NOT stop, delete, restart, or replace it or stop user processes. The provided SSH key remains on the local computer, with neither key content printed nor copied remotely.
- **FR-003**: Each verification attempt MUST use a distinct `/workspace/musubi-val-loss-tests/<run_id>/` for tested code, environment, test configurations, new caches, and outputs. Source models, user datasets, previous results, and previous evidence MUST remain unchanged. A repeat MUST preserve earlier evidence.
- **FR-004**: The tested source MUST match the finished local implementation by a recorded commit or a verified base commit plus patch/source hash. A remote checkout missing local changes MUST NOT be tested as the delivered tool. If transfer cannot be verified over available SSH, request suitable access rather than publishing code or credentials merely to proceed.
- **FR-005**: A separate environment MUST use project-declared dependencies and compatible Python, PyTorch/torchvision/CUDA, AdamW8bit, and TensorBoard; actual versions MUST be recorded. The user's global environment MUST NOT be upgraded, and alternative attention or quantization modes MUST NOT be installed without a concrete test need. Existing models in `/workspace` MUST NOT be downloaded again.
- **FR-006**: Test fixtures MUST use only existing user-designated train, `val_familiar`, and `val_unfamiliar` members in a separate test experiment, targeting 3 train, 2 familiar, and 3 unfamiliar where available. Every familiar fixture MUST be in train and unfamiliar fixtures MUST have no content overlap. The exact membership and SHA-256 hashes MUST be frozen before execution. If role assignment is ambiguous, ask one targeted question; never invent the split or synthesize missing real-model images. Prepare both latent and text caches with the changed cache commands, or reuse only verified matching caches; validation inputs remain fixed thereafter.
- **FR-007**: Short-run configurations MUST retain original Qwen-Image, LoRA rank/alpha 16, BF16, SDPA, AdamW8bit, learning rate `5e-5`, gradient checkpointing, no FP8 or block swap, and target training resolution 1024 with current bucket rules. They MUST enumerate every deviation from the 1,600-step example, including two-step warmup, `val_seed_noise=42`, `N1=4,N2=2` for short runs, and a separate `N1=10,N2=1` validation-only check. Sample verification MUST use one fixed prompt/seed and four inference steps without judging image quality.
- **FR-008 (G01)**: The primary real-model case MUST stop after 12 completed optimizer updates with accumulation 2 and validation/save/sample intervals 4 plus retention window 8. Its expected validation steps are exactly `0,4,8,12`; periodic/final/new-best coincidence at 12 MUST NOT duplicate events or outputs. Record actual GPU execution, finite loss/gradients and LoRA changes after nonzero updates, distinguishing completed updates from microbatches and allowing the initial zero learning rate of warmup.
- **FR-009 (G02)**: The off-cadence final case MUST stop after 10 completed updates with accumulation 1, validation interval 4, save/sample intervals 8, and retention 8. It MUST validate at exactly `0,4,8,10`, publish a complete final state at 10, and apply the agreed sample/best policy independently of save cadence.
- **FR-010 (G03–G04)**: With no optimizer updates, validation MUST repeat twice on unchanged weights and once after loading the same checkpoint in a new process. On the same backend/device, image IDs, levels, seeds and noise MUST match exactly, and metrics MUST meet numeric tolerances fixed and justified before measurement. Renaming/reordering unchanged fixture copies MUST preserve the protocol and results. A separate validation-only `N1=10,N2=1` case MUST load the same weights without resuming or appending to the `4×2` history. Independent evidence from real forwards MUST reconstruct levels, mixing, `timestep=1000*t`, target, elementwise weighted loss, per-image reduction, set/low/high means, their half-mean relation, and the absence of a second training shift; expected values MUST NOT call the implementation's aggregation function. Use the unequal role sizes and different resolutions/buckets where the existing data permit them. The weighted-sigma path requires one directed check, not another training run.
- **FR-011 (G05)**: A real evaluation MUST leave module modes, Python/NumPy/torch CPU and used CUDA random states, existing gradients, LoRA weights, optimizer and scheduler state unchanged. Two controlled four-update runs with and without validation MUST start from equal weights, batches, seeds and order, with samples disabled; training noise/timesteps, losses and LoRA changes MUST meet exact or predeclared numeric criteria. The comparison MUST NOT load two full base models concurrently.
- **FR-012 (G06)**: Resume MUST be checked separately from one actual current package and one actual best package, preserving each original as an immutable input. Each branch runs exactly four additional completed updates. It MUST verify restored weights, optimizer, scheduler and random state, LR/count continuity, unchanged validation protocol and prior best, initial validation at saved absolute `s`, first new training loss at `s+1`, and final step `s+4`. It MUST NOT demand data-loader-position identity absent from the contract.
- **FR-013 (G07)**: Real event files and G01/G02/G06 output trees MUST show the six exact Stage 2 scalar tags, finite values, expected absolute steps, at most one value per tag/step within each run, and retained training graphs. A separate resume process may have its own initial event at `s`. The strict finite minimum of real `val_loss_mean` MUST identify the sole complete best package with its own step, metrics, model and samples, including step-0 best samples when enabled despite `sample_at_first=false`. Within one experiment output, each saved step MUST have one package, one `model.safetensors`, and at most one sample set, with no detached/final duplicate. Retention MUST remove older owned packages whole, keep the inclusive boundary (G01 step 4 at step 12), and protect best.
- **FR-014 (G08)**: Finite controlled tests of the real decision/save paths MUST cover strict best and tie, coincident triggers, best outside periodic cadence, inclusive retention and older-best protection, interrupted write, restoration after validation failure, and early errors for invalid configuration, empty roles and changed inputs. Synthetic metrics are permitted only for these labelled controlled branches and MUST NOT appear in real GPU event graphs or best conclusions. Extra training MUST NOT be used to wait for accidental ties, NaNs or improvements.
- **FR-015 (G09)**: If the selected Pod has two suitable GPUs, a bounded two-process case MUST perform two completed updates with accumulation 2, validation interval 1 and `N1=2,N2=1`; it MUST verify no deadlock, global counts/reduction, one publication per tag, both rank random-state files, and two-rank state loading without further updates. Otherwise G09 MUST be NOT RUN with observed reason; no second Pod may be rented or multi-GPU success claimed.
- **FR-016 (G10)**: The final report MUST give PASS/FAIL/NOT RUN for every G01–G10, associated requirements, conditions, expected and actual counts/steps, numeric differences, commands and exit codes, source/environment/config/model/fixture identities, and links to stdout/stderr, event files and state metadata. It MUST check README against actual behavior and link the report with its limits. Image presence, falling loss, short runs or testing another GPU MUST NOT be presented as proof of image quality, long-term stability, algorithmic correctness by themselves, or H200 verification.
- **FR-017**: Across the initial real-model matrix, at most 40 completed optimizer updates are allowed: `12+10+4+4+4+4`, plus conditional G09 `2`. Validation-only, cache and sample inference do not update LoRA. Each scenario's event count and numeric tolerances MUST be set in the plan before execution; successful tests MUST NOT be repeated without a code change or concrete unresolved question. After uncertain connection loss, inspect the existing process and evidence before retrying. At completion, stop only the agent's test processes after verifying no further updates, while leaving the Pod and user processes running.
- **FR-018**: A functional failure MUST be reported as a confirmed discrepancy, not hidden by longer training, relaxed post-hoc tolerances, changed fixtures, or server-only code edits. Product-code fixes after local implementation require a separately approved discrepancy list and remain within the constitution's correction-round limits. An unavailable mandatory case leaves remote verification incomplete.

### Key Entities

- **Selected Pod**: The existing user-owned server and observed hardware/environment, never a disposable test resource.
- **Isolated verification run**: One uniquely identified test tree containing exact tested source, environment, configurations, caches, outputs, and evidence without modifying user originals.
- **Frozen fixture manifest**: Named train/familiar/unfamiliar members and hashes, role relationship, model references, and cache bindings established before validation.
- **Finite test case**: One G01–G09 scenario with a bounded update count, declared validation events and tolerance, recorded result, and no hidden retries.
- **Verification report**: G10's auditable PASS/FAIL/NOT RUN matrix and links to measured evidence and limits.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All ten G rows have an explicit status and traceable evidence or NOT RUN reason; 100% of executed real-model cases identify the tested source, GPU/environment, fixture hashes, command, and exit code.
- **SC-002**: G01 and G02 complete no more than 12 and 10 optimizer updates respectively and show exactly their four specified validation steps, with zero duplicate event/output attribution within a run.
- **SC-003**: On one backend/device, G03 has zero optimizer updates, exact equality of IDs/levels/seeds/noise across three fixed-check executions, and metrics within predeclared tolerances; G04 independently reconstructs the specified loss and image-first aggregates within predeclared tolerances.
- **SC-004**: G05 records unchanged state around real validation and four-update paired runs whose training noise/timesteps and numerical outputs satisfy their declared comparisons; G06's two four-update branches each end at `s+4` from a verified current or best state.
- **SC-005**: G07 identifies exactly one complete best matching the strict minimum of real unfamiliar-set means and finds no extra package, adapter, or sample for any exercised step; G01 step 4 remains at the inclusive retention boundary when step 12 is published.
- **SC-006**: Initial real-model updates total at most 38 on one GPU or 40 with conditional two-GPU G09; no normal 1,600-step run, hidden extra update, duplicate launch, model redownload, user-data mutation, or Pod stop occurs.
- **SC-007**: Any failed or unavailable mandatory case is visibly reported as incomplete or failed, and the agent's test processes finish without affecting the user's Pod or processes. G09 may be NOT RUN only with a recorded lack of two suitable GPUs.

## Assumptions

- Stages 1–4 are locally complete before this separate remote stage. The user supplied `ssh yxd2b6cm3e8gfa-64412491@ssh.runpod.io -i ~/.ssh/id_ed25519` (local Windows key path `C:\Users\inbox\.ssh\id_ed25519`), repository `https://github.com/Maxim-Dey/musubi-tuner-tools-extraction`, and `/workspace` as the expected remote root. Their current availability and contents remain unverified until later inventory. This specification performs no connection or deployment.
- The user-provided Basic SSH mode has no SCP/SFTP. Planning will choose a verifiable source-transfer method after access inventory; no particular transport is assumed successful in advance.
- The target fixture sizes are conditional on existing user-designated roles and data. A role ambiguity requires one focused user question at preparation time; it is not permission to invent data or split.
- Numeric tolerances depend on the observed precision/backend and must be declared and justified in the plan before any results are collected. Exact discrete and random-input comparisons remain exact on one backend/device; cross-GPU bitwise equality is not promised.
- G08 controlled metrics are diagnostic only; real-model best and loss conclusions derive exclusively from G01/G02/G06 event evidence. The user alone stops or changes the Pod.

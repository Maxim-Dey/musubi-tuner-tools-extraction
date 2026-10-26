# Real-model RunPod acceptance

Dates: 2026-10-05–06. **Status: required single-GPU acceptance PASSED.** Local A–D, real training/save/resume, recorded validation repeatability and exact paired GPU isolation passed. The paired proof uses an explicitly recorded strict deterministic acceptance environment, after ordinary no-val replay proved non-repeatable at the bit level. The user explicitly authorized round3 after the original autonomous2/2 stop. T032 is repaired and verified. Multi-GPU and batch16/1024 execution are excluded by the user. Earlier failures below are retained as history.

## Environment and resources

Coordinator alone ran GPU processes, sequentially, in `/workspace/codex-val-loss-20261005-6da0266c`. The supplied direct SSH endpoint accepted the existing key; no secret was printed. No Pod was created or stopped. All acceptance processes have exited; the final GPU compute-process inventory is empty.

- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 595.91.07, capability 12.0. **Real two-GPU acceptance: excluded by the user; not run.** Local two-process CPU tests are separate evidence.
- Linux 6.17.0-20, Python 3.12.3; torch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8.
- Accelerate 1.6.0, bitsandbytes 0.50.2, Diffusers 0.32.1, Transformers 4.57.6, TensorBoard 2.21.0, safetensors 0.4.5, NumPy 2.5.2.
- Isolated installation: `python -m venv .venv`; `.venv/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128`; `.venv/bin/pip install -e . tensorboard pytest`. All 12 declared pyproject requirements satisfy their specifiers; pip check passed. Installation logs remain at the remote checkout root as `install-torch.log` and `install-project.log`.

| Model | Actual bytes |
|---|---:|
| `/workspace/models/qwen_image_2512_bf16.safetensors` | 40,861,031,488 |
| `/workspace/models/qwen_image_vae.safetensors` | 253,806,246 |
| `/workspace/models/qwen_2.5_vl_7b.safetensors` | 16,584,415,576 |

Actual loaders reported all keys matched. Frozen DiT SHA256 is `cbf55390fff27dbc785046d7007b04e0c5dd7421e7ef128f2831eacb53a8e075` in the saved computation contract. Existing tokenizer cache is `/workspace/.cache/huggingface`; offline mode prevented model downloads.

[Resource inventory](evidence/resources.json) records 107 original train and 20 independent val images, captions and SHA256 identities; source-content sets are disjoint. Acceptance symlinks select train `a1fa_k1b_med1um_0001.png`, `0002.png` and val `0003.png`. No automatic split or synthetic replacement. Original resources were preserved. The main profile's dataset paths also link to the complete original directories.

All acceptance configs use 256-pixel buckets, batch1, accumulation2, LoRA rank/alpha32, dropout0.05, gradient checkpointing, BF16 and AdamW8bit. `train.toml`/`resume.toml` use N1=10/N2=2; `periodic-final.toml` and its later CLI variants use N1=2/N2=1. The user profile remains **1024px, batch16, 5000 steps**; its full training and 1024px caches were **not run**. Actual full effective settings for the four TOMLs and four dataset blueprints, model/source paths and dependency versions passed independent review: [effective environment/configs](evidence/environment-effective.json).

## Executed code identity

The initial transfer included 180 individually SHA256-verified files from the modified working tree, including untracked implementation, tests and configurations. It was not a clone of old HEAD. Base HEAD: `f23da34159c56c16b230603b44fae4a2c1c74e34`. [Initial transfer](transfer.json), [initial file manifest](evidence/transferred-tree-sha256.json).

Two acceptance-only helpers were added with [supplemental hashes](evidence/acceptance-helpers-sha256.json): native resume proof and artifact audit. Round2 corrected only the probe CLI and its tests; [round2 hashes](evidence/round2-source-sha256.json) supersede those initial entries. Production code did not change remotely.

The original round2 [executed-source manifest](evidence/executed-source-sha256.json) covers89 runtime source files, entrypoints, acceptance helpers, tests, pyproject, TOMLs and prompts. Its SHA256 is `95416f21caec9d97f6117f17b8e9737c1c9886e7821c37a0acbcaf53f7d4ba0c`. [Remote verification](evidence/executed-source-verification.json) confirmed exact local/remote bytes then. Round3 changes only the probe and its focused tests; the current [89-file source manifest](evidence/round3-isolation-source-sha256.json), SHA256 `ae35cfa9c5821cf1104ece7cdfc4590ad54da9811ee7a494083bd11b5b9d2034`, supersedes those two entries and [matches remote bytes](evidence/round3-isolation-source-verification.json). Current full CPU suite: **398 passed in112.83s**; independent focused probe suite:23passed in12.96s. Those CPU results are separate from GPU evidence.

## Commands

All commands below ran from the isolated checkout. Training/probe environment:

```sh
export HF_HOME=/workspace/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
```

Four cache processes completed with exit0. Exact argv, durations and logs are in [cache commands](evidence/cache-commands.json):

```sh
.venv/bin/python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/acceptance/train-dataset.toml --experiment_mode --model_version original --batch_size 1 --num_workers 1 --device cuda --vae /workspace/models/qwen_image_vae.safetensors
.venv/bin/python qwen_image_cache_latents.py --dataset_config config_for_qwen_image_lora/acceptance/val-dataset.toml --experiment_mode --model_version original --batch_size 1 --num_workers 1 --device cuda --vae /workspace/models/qwen_image_vae.safetensors --validation
.venv/bin/python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/acceptance/train-dataset.toml --experiment_mode --model_version original --batch_size 1 --num_workers 1 --device cuda --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors
.venv/bin/python qwen_image_cache_text_encoder_outputs.py --dataset_config config_for_qwen_image_lora/acceptance/val-dataset.toml --experiment_mode --model_version original --batch_size 1 --num_workers 1 --device cuda --text_encoder /workspace/models/qwen_2.5_vl_7b.safetensors --validation
```

The following actual training scenarios completed with exit0, in separate outputs as shown:

```sh
.venv/bin/accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no qwen_image_train_network.py --config_file config_for_qwen_image_lora/acceptance/train.toml
.venv/bin/python -u specs/002-deterministic-val-loss/gpu_train_acceptance.py --resume_report output-resume/resume-3-to-5.json --config_file config_for_qwen_image_lora/acceptance/resume.toml
.venv/bin/python -u qwen_image_train_network.py --config_file config_for_qwen_image_lora/acceptance/periodic-final.toml
.venv/bin/python -u qwen_image_train_network.py --config_file config_for_qwen_image_lora/acceptance/periodic-final.toml --max_train_steps 5 --save_every_n_steps 3 --output_dir output-uninterrupted --logging_dir output-uninterrupted/tensorboard --log_prefix qwen_image_val_acceptance_uninterrupted
.venv/bin/python -u specs/002-deterministic-val-loss/gpu_train_acceptance.py --resume_report output-periodic/resume-4-to-7.json --config_file config_for_qwen_image_lora/acceptance/periodic-final.toml --resume output-periodic/qwen_image_val_acceptance-step4 --max_train_steps 7 --save_last_n_steps 3 --save_last_n_steps_state 1
```

The probe command below ran initially and again after round2. Both exited1 for different documented helper defects; neither is a passed acceptance run. The corrected probe ran between uninterrupted5 and retention7.

```sh
.venv/bin/python -u specs/002-deterministic-val-loss/gpu_acceptance_probe.py --config_file config_for_qwen_image_lora/acceptance/train.toml --network_weights output-resume/qwen_image_val_acceptance-step3/model.safetensors --output_dir output-probe --logging_dir output-probe/tensorboard --probe_report output-probe/probe.json
```

## Actual event and artifact results

`gpu_acceptance_audit.py` read native TensorBoard event files and checkpoint manifests with CUDA disabled, without loading the model. It required exactly `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`; finite values; mean=(low+high)/2 within1e-4/1e-5; train `loss/current` steps1..target; no duplicate val steps; one continued TensorBoard run. It checked actual tensor dtypes/keys, native manifest hashes and sample paths.

| Scenario | Expected and actual val steps | Weights / complete states / sample directories | Evidence |
|---|---|---|---|
| Initial3, full10x2 | 0,2,3 | 3 / 3 / 2 | [audit](evidence/train-3-audit.json), [log](evidence/train-3.log) |
| Native resume3→5, full10x2 | 0,2,3,4,5 | 3,5 / 3,5 / 2,4 | [audit](evidence/resume-5-audit.json), [log](evidence/resume-5.log) |
| Periodic/final4, short2x1 | 0,2,4 exactly once | 2,4 / 2,4 / 2,4 | [pre-retention snapshot](evidence/periodic-4-audit.json), [log](evidence/periodic-4.log) |
| Uninterrupted5, short2x1 | 0,2,4,5 | 3,5 / 3,5 / 2,4 | [audit](evidence/uninterrupted-5-audit.json), [log](evidence/uninterrupted-5.log) |
| Native resume4→7, retention3/1 | 0,2,4,6,7 | 4,6,7 / 6,7 / 2,4,6 | [audit](evidence/retention-7-audit.json), [log](evidence/retention-7.log) |

All five event/artifact audits passed. Initial full10x2 mean losses were0.1426313072,0.1423138678,0.1419153064 at0/2/3. Samples are actual decoded PNGs; sample and save periods differ in the3→5 scenario. At the periodic/final boundary the log contains exactly one adapter write for step4. The periodic4 snapshot precedes retention extension; it is historical evidence, not a claim that state2/4 still exists after step7.

Each remaining adapter has exactly2520 FP32 `lora_` tensors in one `model.safetensors` (about1.18GB), with no base DiT or second adapter tensor artifact. Complete checkpoints contain native optimizer, scheduler, per-rank RNG and explicit trainer metadata plus the hashed completeness manifest. [Initial native metadata](evidence/train-3-state.json).

The thin resume wrapper delegates to native qwen.main and verifies state immediately after native Accelerate load_state, before the next update. Both [3→5 load proof](evidence/resume-3-to-5.json) and [4→7 load proof](evidence/resume-4-to-7.json) passed: exact adapter identity, optimizer, scheduler, explicit step and Python/NumPy/CPU/one-CUDA-device RNG. Subsequent real updates and changed saved adapter identities were observed. This does not claim exact restoration of training-loader position beyond native capability. Step3 metadata predates its final val3; the durable identity ledger reconciles the existing event without duplication.

Retention passed inclusive boundaries: at7 with weights_window3/state_window1, adapter4 remains at the lower weight boundary, state4 is expired, states6/7 remain and step2 samples survive after its owned weights/state expire. Resuming at divisible step4 can produce another timestamped PNG in the same correct step4 directory; the sample-directory requirement does not require one PNG per step.

## Original probe failures and stop (historical)

Tolerances were declared before execution and never changed: repeated BF16 loss rtol1e-4/atol1e-5; seeds/noise hashes exact; paired adapter/gradients/optimizer/scheduler/scaler/RNG/noise/timestep comparison exact0/0.

The initial probe bypassed native qwen.main initialization and failed before model loading with missing args.dit_dtype: [initial failure](evidence/probe-initial-failure.log). Recorded correction round2 delegates to native qwen.main, preserving native dtype defaults/model-version resolution and restoring temporary argv/trainer-factory substitutions. Focused and full CPU regressions passed, and the corrected GPU run loaded the real model and saved adapter successfully.

The corrected run then failed during optimizer warmup: **expected one completed update, observed2**. [Failed JSON](evidence/probe-round2.json), [full log](evidence/probe-round2.log). Independent math and checkpoint reviewers confirmed F3: the probe prefetches exactly the two batches in the real prepared loader, then closes its iterator at the final yield. Accelerate1.6 leaves its end-of-dataloader flag active because normal end() cleanup is after that yield. Both replayed accumulation microbatches consequently force an update. The original tiny CPU probe uses three loader batches and missed this boundary; an independent CPU lifecycle reproduction confirmed replay sync flags `[True, True]` after closing the two-batch loader.

The paired branches were **never reached**. Earlier repeat assertions precede warmup in control flow, but the failure report preserves no seeds/noise/counts/metrics, so a complete auditable repeatability pass is **not claimed**. This is a confirmed acceptance-helper defect; it does not establish a defect in production validation isolation. CPU production-loop isolation passes remain separate.

The original final speckit-converge appended **T032**, with prior task bytes preserved. At that stop, loader repair,1/2-batch boundary regressions and full10x2 repeatability/paired isolation remained incomplete. The agent stopped corrections at the original autonomous2/2 cap. T025/T026/T029/T031/T032 were open then. The explicit user continuation below supersedes that stop without erasing its evidence or resetting the counter.

## Original integrity snapshot and scope

[Original integrity proof](evidence/final-integrity.json) confirmed all127 original source images and captions, all6 generated caches and all89 then-executed source files unchanged. Validation fingerprints remain identical across each save/resume series; short2x1 and full10x2 intentionally have different fingerprints. Original user prompts and constitution retain their initial SHA256 values. [Cache hashes](evidence/cache-sha256.json).

Independent configuration/documentation review passed all46 mandatory user settings, dataset schemas, prompt preservation and command/link consistency. Independent checkpoint review covered both load proofs and actual schedule/retention artifacts. Real multi-GPU and parent1024px/batch16/5000-step execution are explicitly excluded by the user and are not claimed tested.


## Authorized continuation evidence — 2026-10-06 (passed)

The user explicitly requested T032 correction, remaining single-GPU proof and final SpecKit completion after the original2/2 stop. Multi-GPU and batch16/1024 execution are excluded by that request. This continuation retains the old failed evidence and does not reset the historical correction counter.

The loader lifecycle fix passed real tiny Qwen/LoRA/native Accelerate tests at loader lengths1/2/3 and accumulation2; native end() cleanup restores the previous active-loader references. The first corrected GPU run confirms exactly one update per warmup/control/measured replay but fails later at exact gradients: [report](evidence/probe-round3-first.json), [log](evidence/probe-round3-first.log). The report is a failed isolation attempt, not a failed repeatability measurement.

**Repeated GPU validation passed.** Both20-forward passes have the same seeds/noise hashes and allthree metrics exactly equal: mean0.14191530235111713, low0.12941182255744935, high0.15441878214478494. An independent CPU calculation in the same torch2.8 environment reconstructs all20seeds and BF16 noise tensors directly from the real image SHA and cache header, without importing the probe or validation code: [audit](evidence/round3-repeatability-audit.json). The mean half-sum error is2.7755575615628914e-17. T025 is complete.

The remaining exact-gradient comparison now includes a same-state no-validation replay before the val branch. It records pre/post-clipping differences, trace/loss/state equality and exact restoration. Any baseline failure blocks an isolation pass. An explicit --probe_deterministic mode enables strict PyTorch deterministic algorithms and restores prior flags in finally; CUBLAS_WORKSPACE_CONFIG=:4096:8 will be supplied before CUDA when that mode is used. No backend is forced, no production/default/TOML changes are made, and tolerance remains0/0. A controlled-mode pass will not be presented as bitwise reproducibility of ordinary CUDA kernels.

This diagnostic implementation passed owner23focused tests in13.84s, independent23tests in12.96s and the full **398-test CPU suite in112.83s**; scoped lint and whitespace checks passed. Current execution identity is round3-isolation-source-sha256.json, SHA256 ae35cfa9c5821cf1104ece7cdfc4590ad54da9811ee7a494083bd11b5b9d2034, with89files verified remotely. Only the probe and its test changed; production source and user configurations remain the previously accepted bytes.

### Default replay diagnosis and exact controlled proof

The default-mode diagnostic exited1: [JSON](evidence/probe-default-replay.json), [log](evidence/probe-default-replay.log). Restored state, training inputs/noise/timesteps and both training losses matched exactly, but the no-validation replay gradients differed before clipping. The first FP32 tensor lora_unet_transformer_blocks_0_img_mod_1.lora_down.weight, shape[32,3072], had98094 unequal entries, maximum absolute difference2.6106834411621094e-05 and RMS4.6151706680904544e-07. The updated adapter also differed. This run correctly stopped before the val-insertion branch and contains no isolation pass. It establishes that default GPU training replay was not bitwise repeatable even without validation; the selected kernel and precise cause are not established by these traces.

The controlled run exited0: [passed JSON](evidence/probe-deterministic.json), [full log](evidence/probe-deterministic.log). It enables strict PyTorch deterministic algorithms with warn_only=false and CUBLAS_WORKSPACE_CONFIG=:4096:8 set before CUDA initialization. No attention backend is forced; all reported SDPA backends remain enabled. The actual selected backend is not recorded. BF16, dropout0.05, gradient checkpointing, accumulation2 and AdamW8bit remain the same.

Exact continuation commands used the environment above and the same checkpoint3 adapter. The first command exited1 as expected by the diagnostic; the second passed:

```sh
.venv/bin/python -u specs/002-deterministic-val-loss/gpu_acceptance_probe.py --config_file config_for_qwen_image_lora/acceptance/train.toml --network_weights output-resume/qwen_image_val_acceptance-step3/model.safetensors --output_dir output-probe-default-replay --logging_dir output-probe-default-replay/tensorboard --probe_report output-probe-default-replay/probe.json
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -u specs/002-deterministic-val-loss/gpu_acceptance_probe.py --probe_deterministic --config_file config_for_qwen_image_lora/acceptance/train.toml --network_weights output-resume/qwen_image_val_acceptance-step3/model.safetensors --output_dir output-probe-deterministic --logging_dir output-probe-deterministic/tensorboard --probe_report output-probe-deterministic/probe.json
```

| Required comparison | Actual result |
|---|---|
| Repeated full10x2 validation |20forwards each; all seeds/noise hashes and allthree metrics exactly equal; independent seed/noise audit matches |
| Accumulation lifecycle | Warmup, control, control replay and val branch each complete1update with microbatch sync[false,true] |
| Restored branch starting state | Exact before both branches; complete snapshot unchanged by inserted validation |
| Control versus no-val replay | Exact loss, batches/noise/timesteps, pre/post-clipping gradients and complete next state |
| Control versus inserted val | Same exact comparisons pass, including adapter/optimizer/scheduler/Python/NumPy/CPU/all-used-CUDA RNG, scaler, module modes and accelerator step |
| Numerical thresholds | Repeated loss rtol1e-4/atol1e-5; paired comparisons0/0; unchanged from declaration |

Both training losses are0.07530149817466736 and0.08943668752908707 in all compared branches. Repeated validation metrics retain mean0.14191530235111713, low0.12941182255744935 and high0.15441878214478494. Independent math and checkpoint reviewers verified the raw report, comparison code and previous actual save/resume/event/retention evidence; no remaining implementation gap was confirmed.

This proves validation isolation for the recorded strict environment and tested scenario. It does not promise bitwise reproducibility of default GPU training or exact native data-loader position on resume. The default-mode discrepancy alone is not evidence of validation interference or a production-training defect; no production change was justified by it.

### Final integrity

[Round3 final integrity](evidence/round3-final-integrity.json) verifies127 original images and captions,6 caches and all89 current executable/config/test files, unchanged checkpoint3 adapter SHA256 `47326f162000823741ca12ebe54d43eb81f82a9a945f5ce6c77b205a2a1469c3`, the original full10x2 validation fingerprint, constitution and user prompt hashes. No GPU compute process remains. T025/T026/T031/T032 are complete on actual evidence; T029 records the final SpecKit convergence separately. User-excluded multi-GPU and batch16/1024 runs are not represented as passes.

Final independent speckit-converge is clean:28functional requirements,5success criteria, all5user stories,5plan areas and7constitution principles checked; zero confirmed gaps. It left tasks.md byte-for-byte unchanged and appended nothing. Subsequent completion bookkeeping closes T029; all32tasks are complete. See [convergence and requirement matrix](validation.md). No further production change or acceptance run is required within the agreed scope.

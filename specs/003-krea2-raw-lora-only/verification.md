# Verification Record

## Baseline — 2026-10-26, before source edits

Branch `codex/krea2-raw-lora-only`; initial untracked user folder `config_for_krea2_raw_lora/` contains the preserved prompt. Constitution unchanged. No tracked/ignored project lock file or standalone LICENSE/COPYING/NOTICE file found; license information exists in README and retained source headers. Other feature `specs/002-deterministic-val-loss` is protected.

Environment: `.venv/Scripts/python.exe`, Python 3.12.11, torch 2.7.1+cpu, torchvision 0.22.1+cpu, pytest 9.1.1. `av`, `ftfy`, `easydict`, `sentencepiece`, `wandb` absent. Set `PYTHONPATH=src`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `CUDA_VISIBLE_DEVICES=''` for checks. No installs or model calls.

Baseline command:

```text
.venv/Scripts/python.exe -m pytest -q tests/test_krea2_convrot_int8.py tests/test_convrot_int8_prequantized.py tests/test_krea2_gather_valid_text.py tests/test_krea2_gradient_checkpointing.py tests/test_save_precision.py tests/test_lora_dtype_bridging.py
```

Result: **59 passed, 7 skipped, 4 failed**. Skips need CUDA/Triton; failures import Krea2 trainer through dataset/audio_utils and fail on absent `av`. Adding `test_krea2_timesteps.py`, `test_grad_metrics.py`, `test_datasource_item_extras.py` yields three collection errors for the same import.

Real imports of `networks.lora_krea2`, `krea2.krea2_utils`, `training.parser_common` pass. Both Krea2 caches, trainer and dataset.cache_io fail on `av`. Actual parser accepts Krea2 network/SDPA/bf16/krea2_shift. Temporary TOML through actual read_config_from_file accepts unknown_parameter, mixed_precision='garbage', learning_rate='not-a-number'; CLI override to 0.02 works and unknown CLI exits 2. These are baseline defects, not successful validation.

## Correction counter

Post-implementation correction rounds used: **1 / 2** (closed; final local checks and independent convergence passed). Research/document preparation and initial implementation are not correction rounds. Changing agents or commands does not reset this counter.

## Stage status

Specification, plan and 16 tasks prepared. Independent read-only speckit-analyze assessed 12 FR + 5 SC: 100% task coverage, no unmapped tasks, no confirmed contradictions or blocking findings. No document correction needed. Requirements checklist: 13 checked, 0 unchecked. Implementation prerequisite passed; extension hooks file absent. Implementation, final local verification and independent convergence completed with fixed file ownership.

## Initial implementation evidence

Three GPT-6.1 Sol agents handled training, data/cache, and documentation/independent verification with exclusive file ownership. Coordinator preserved VAE helpers/LoRA and removed 295 tracked excluded-workflow files; no untracked user files were selected for deletion. Fifty retained source modules imported successfully. Static internal import inventory found zero missing modules.

Data/cache targeted suite: 19 passed, with real image directory/JSONL → preprocessing → latent/text writers → safetensors → training dataset/collator round trips, repeats/cache lifecycle and early field errors. The final area-based bucket regression accepts resolution 1000 with buckets and preserves `(992,1008)`; unaligned fixed-resolution errors remain early. Three shared helper regressions passed: VAE key conversion, real tiny LoRA CPU step/save/load, streaming base-LoRA merge. Metadata/dtype suite: 13 passed. All three root command --help subprocess checks passed after specifying UTF-8 output encoding on Windows; the earlier cp1252 capture failure was recorded and resolved at the subprocess boundary, without removing multilingual help.

Independent numerical comparison executed the baseline and current Raw `do_inference` on the same tiny real CPU SingleStreamDiT with variable-length CFG, three Euler steps and identical seed; only the VAE decode boundary used a small identity slice. Results matched exactly (rtol=0, atol=0), finite, shape `(1,3,1,4,4)`. Training loss/noise/timestep and Krea2 call_dit ASTs were also compared; retained encoder/DiT bodies and core state utilities remained unchanged.

Independent documentation acceptance exercised README cache/train/resume arguments through real parsers, real temporary image dataset and starter consumers; text/JSON/TOML sample examples and relative file links passed. One incorrect attention backend recommendation remains in the round-1 list below. Prompt SHA256 after implementation: `79f9aa5d27fe8ff4857e0db30c104f6ad952ce83d24a7a7dbb3e33f0ec508170`; the initial prompt was untracked and no before-hash was recorded, so preservation is supported by unchanged content/ownership, not a claimed hash comparison.

## Round 1 — confirmed defects and correction criteria

These are in-scope gaps already covered by open T010, T011 and T014; do not duplicate them as new tasks. No code changes occurred during the independent assessment.

| ID | Requirement / evidence | Correction and verification |
|---|---|---|
| R1-1 | FR-008, Constitution VI: both cache main functions with a valid image dataset and nonexistent checkpoint path reached sentinel model loaders. | Validate local VAE/text-encoder file existence before loaders; actual parse/dataset chain must fail with source/flag/remedy and never reach the boundary. Preserve image debug mode without a VAE. |
| R1-2 | FR-008: malformed existing log_tracker_config TOML passes `_validate_args_and_init`, then `toml.load` fails only in `_run_training_loop` after model loading. | Parse tracker TOML during early validation; malformed CLI/TOML-referenced configs fail with source and filename before models. |
| R1-3 | FR-008: metadata_reso='garbage' passes validation, then actual `sai_model_spec.build_metadata` fails during model saving. | Validate actual supported metadata resolution syntax early, preserving valid forms; invalid values fail before model boundary. |
| R1-4 | FR-008: cosine_with_min_lr without min_lr_ratio passes validation; actual get_lr_scheduler raises missing min_lr/min_lr_rate after models. | Validate required supported scheduler values using existing consumers/constraints before models; test missing ratio and valid configured scheduler. |
| R1-5 | FR-008: custom torch StepLR with step_size='bad' passes signature validation/construction; actual first scheduler update raises TypeError. | Check supported torch scheduler parameter values before models, preserving valid dynamic optimizer/scheduler support; test invalid type and valid step progression on tiny CPU optimizer. |
| R1-6 | FR-009: docs/krea2.md recommends sage_attn although actual trainer always rejects it for training. | Remove unsupported recommendation; verify all documented backends against actual validation. |

Round-1 changes are separate from assessment. Full retained tests, imports, lint/format, starter/document checks and independent convergence will be rerun after this grouped pass; no unperformed check counts as passed.


## Round 1 results and final local checks

R1-1 through R1-6 are fixed. Cache file validation now precedes model loading and preserves latent debug mode without a VAE (22 targeted tests passed). Trainer validation parses tracker TOML before loading, validates metadata resolution through the actual metadata builder, requires a valid cosine minimum unless its scheduler is overridden, and checks supported torch scheduler arithmetic on a tiny CPU optimizer; valid dynamic StepLR progression remains supported (83 targeted tests passed). The unsupported SageAttention training recommendation was removed. Attention documentation was checked against actual validation; absent optional backends fail early, and backend execution is not claimed.

Final full-suite command with PYTHONPATH=src, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, CUDA_VISIBLE_DEVICES='' and PYTHONIOENCODING=utf-8:

```text
.venv/Scripts/python.exe -m pytest -q --disable-warnings
```

Result: **167 passed, 7 skipped in 30.78s**. The seven skips require CUDA/Triton. The suite includes the three real root --help subprocesses, actual starter/config consumers, CLI/TOML precedence and early rejections, image/cache round trips, tiny real LoRA updates and Accelerator save/resume regressions.

Additional final checks passed: real imports plus AST syntax for all 50 retained source Python modules; exact three-command root inventory; relative file links in README, retained docs, contributing documents and feature quickstart; Ruff check and format (64 files according to the existing exclusions); git diff --check. Existing license information and protected-file ownership were preserved. No package installation/build, model-weight download, remote verification or real training/GPU run was performed.

Implementation tasks T001-T015 are complete from recorded evidence. The separate independent convergence result below completes T016.


## Independent final convergence

Outcome: **converged** within the authorized local stage. A separate GPT-6.1 Sol agent assessed the current code against 12 functional requirements, 5 success criteria, 9 acceptance scenarios, 14 component decisions, all configuration/execution constraints and 7 constitutional principles plus local-stage/governance rules. Findings: missing 0, partial 0, contradicts 0, unrequested 0. No tasks appended and no second correction round needed.

Independent consumer probes confirmed all round-1 defects fail early. Valid dynamic SGD/StepLR reached learning rate 0.05 after two tiny CPU updates; configured cosine minimum, tracker initialization values and 768x1024 metadata remained supported. Validation probes preserved Torch RNG state. The final coordinator suite and import/lint/document evidence were assessed without duplicating the complete suite.

The assessment used the current implementation without history/diff comparison and changed no source, documentation or tasks. tasks.md SHA256 before and after convergence was identical: `9DB7844610144A9B085DF9A68594141C7FA7F64A53761AD95A227940268C9E9A`. Only after the independent report did the coordinator mark T016 complete and append this result. All 16 tasks and all 13 requirements-checklist entries are complete. No extension hooks were configured.

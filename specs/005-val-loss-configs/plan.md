# Implementation Plan: Qwen-Image Validation Example and Run Guide

**Branch**: 001-scope-qwen-image-lora | **Date**: 2026-09-23 | **Spec**: [spec.md](spec.md)

**Input**: Stage 4 example and guide specification in this feature directory.

**Revision**: 2026-09-24 single-command cache preparation, added after the code was initially written. The original four-command design below is superseded where explicitly revised.

## Summary

Deliver four physical example files in a separate portable experiment folder, plus one training command that prepares required caches, and monitor, move, and resume instructions. Use the existing Qwen-Image original LoRA entrypoints and Stage 1–3 path, validation, and complete-state contracts. Verify the example through real parsers and readers, cache orchestration checks, and controlled CPU integration. No training-math change is planned.

## Technical Context

**Language/Version**: Python 3.12 for local acceptance; project supports Python 3.10–3.12.

**Primary Dependencies**: Existing Qwen trainer/cache commands, TOML and prompt readers, PyTorch/Accelerate, safetensors, Pillow, pytest; TensorBoard and bitsandbytes are operational dependencies of the example.

**Storage**: Four example text files, user-owned image and cache directories, and Stage 3 complete state packages under the experiment output tree.

**Testing**: Real parser/reader checks, existing small CPU fixtures and focused pytest integration; no downloaded weights, GPU run, or real-model training.

**Target Platform**: Local Windows CPU verification; later Linux/H200 operation is a separate server stage.

**Project Type**: Python CLI training and caching tool with documentation and a portable example folder.

**Performance Goals**: None in this local documentation/configuration stage; H200 performance is not measured.

**Constraints**: Do not overwrite user files, fabricate training images, alter prior stage contracts, run a model/GPU/download, or claim a server result. Keep integer warmup, exact FP32 resumable adapter saving, and unchanged legacy behavior.

**Scale/Scope**: One Qwen-Image original LoRA example, three dataset roles, four physical files, one ordinary training invocation and two resume examples; targeted CPU checks only.

## Constitution Check

*Gate before design and again after design.*

| Principle | Gate |
| --- | --- |
| I. Language | PASS: Spec Kit artifacts remain English; user communication remains Russian. |
| II. Task Fidelity | PASS for this revision: the user explicitly replaced the four manual cache commands with one training command. The original numerical example contract remains unchanged pending convergence review. |
| III. Minimal, Compatible Changes | PASS: Reuse existing Qwen cache entrypoints from the training command; no generic cache framework or change to other model families. |
| IV. Training Invariants | PASS: The example selects existing loss/noise/step behavior. CPU controlled integration checks metrics and subsequent update/state without changing training mathematics. |
| V. Memory and Performance | PASS: Cache encoders run before training-model loading and leave no retained copy in the training process; fixtures stay small. No H200 memory or speed claim. |
| VI. Configuration and Errors | PASS: Effective TOML and CLI values go through real readers and preflight before model weights; invalid source/correction cases remain visible. |
| VII. Verification and Documentation | PASS: Physical files, README, run guide, parser checks, and controlled CPU integration are required; unrun GPU work is reported separately. |
| Local Stage and Workflow | PASS: Small CPU fixture models are allowed for controlled tests; no full Qwen model weights/loading, GPU training, download, or server access in this local revision. |

**Post-design gate**: PASS after the path/CLI, example-file, and CPU acceptance design below. No exception or complexity waiver is needed.

## Project Structure

### Documentation and deliverables

    specs/005-val-loss-configs/
      spec.md
      plan.md
      research.md
      data-model.md
      contracts/example-cli.md
      quickstart.md
      tasks.md                 # generated in the next phase
    qwen_image_lora_val_example/
      train.toml
      train-dataset.toml
      val-dataset.toml
      sample_prompts.txt
      dataset/train/           # user supplies images and .txt captions
      dataset/val_familiar/   # user supplies images and .txt captions
      dataset/val_unfamiliar/ # user supplies images and .txt captions
      cache/train/            # training command fills these after data preparation
      cache/val_familiar/
      cache/val_unfamiliar/
      output/                 # trainer creates packages/logs
    README.md
    docs/qwen_image.md
    tests/test_qwen_image_experiment_paths.py
    tests/test_qwen_image_validation_training.py
    tests/test_qwen_image_experiment_states.py

**Structure Decision**: Add the example as a sibling folder, update existing Qwen documentation, and extend the existing path/training/state CPU tests where the scenarios already fit. A focused composition test may be added only if those modules cannot express the config-to-resume chain without duplication.

## Phase 0: Research and Decisions

Use [research.md](research.md) for confirmed parser names, path anchors, package names, and decisions. Cross-check the real Qwen trainer and both cache entrypoints, rather than copying a legacy command. Key decisions already fixed by the spec:

1. The user selects root/train.toml with --config_file. The Qwen trainer first checks both source declarations without loading weights, then uses the effective train and validation dataset declarations plus VAE/text-encoder paths to invoke the existing cache entrypoints in separate processes. Existing train caches are skipped by path existence; missing caches are created. Strict validation preflight triggers a validation-cache rebuild if it finds stale content. Cache subprocesses must not join the training process group, and concurrent ranks must not race over cache files.
2. The three model references in train.toml are clearly replaceable absolute server paths to original BF16 DiT, VAE and text encoder. The example is a template until user data, captions, and model paths exist.
3. Keep BF16 computation but save the single complete adapter in FP32. Do not introduce a second adapter file or state retention knob. Remove both sample_prompts and sample_every_n_steps to disable sampling; an empty prompt path is invalid.
4. Use established Stage 1 fixed validation grid independent of training shift, Stage 2 six-tag event and absolute-step semantics, and Stage 3 best/coalesced package rules.

## Phase 1: Design and Acceptance

### Example file to reader matrix

| Field or file | Real reader/entrypoint | Required interpretation |
| --- | --- | --- |
| train.toml, selected through --config_file | Qwen trainer configuration reader and opt-in preflight | defaults, then TOML, then explicit CLI; root is selected train.toml directory because experiment_dir is dot |
| dataset_config and train-dataset.toml | Training dataset reader and both Qwen cache commands | canonical root/train-dataset.toml, one unroled train source, root-relative image/cache paths |
| val_dataset_config and val-dataset.toml | Strict validation reader and both Qwen cache commands | canonical root/val-dataset.toml, exactly two roles with separate sources/caches |
| dit, vae, text_encoder | Trainer preflight and model readers; Qwen cache preparation receives effective VAE/text-encoder paths | replaceable absolute server files; original BF16 DiT, not FP8 conversion |
| sample_prompts.txt | Existing prompt reader | two exact TOK lines, valid 1024 dimensions/options; replacement explained |
| output_dir, logging_dir, save_precision, save_last_n_steps | Trainer experiment preflight and package policy | root/output, root/output/tensorboard, FP32 sole adapter, inclusive 1000-step current window |
| val_every_n_steps, val_seed_noise, val_level_noise_n, val_seed_noise_n | Stage 1/2 validation controls | 200/42/10/1, fixed 10 checks per image and six exact tags |
| --resume current/best package | Stage 3 complete-state loader | restore package-local state and absolute step; no exact data-loader position promise |

### Work sequence

1. Write the four example files with exact values from FR-002–FR-007 and create only the agreed directory structure. Keep real images, caption files, and model paths user-supplied; the training command creates caches. Verify no role-bearing entry in train-dataset.toml and no random transform in val-dataset.toml.
2. Exercise the real training TOML parser, dataset readers, prompt reader and cache CLI parsers. Check relative/absolute path resolution from another CWD and after moving/renaming a temporary copy of the root. Verify model path placeholders are recognizable as required substitutions, and all other effective values are accepted. Preserve the existing no-experiment mode.
3. Extend existing CPU fixtures, with the example settings adapted to small temporary images/caches and a small model: use two val sets of different sizes; verify the Stage 1 SHA-256-seeded 10x1 grid and exact t with no training shift; the Stage 2 common loss/image-first low/high means and six tags; step-0/periodic/final dedup and a strict best; a complete single-adapter state and nonzero-step resume. Exercise the 0–1600 due-step grid through the existing scheduling seam, not 1,600 optimizer updates; use a separate short controlled loop for the composed chain. Check unchanged next controlled update/RNG where the existing invariant tests provide the seam.
4. Update README.md and docs/qwen_image.md with the one training command, automatic cache behavior, TensorBoard, current/best resume, relocation, required replacements and samples-disable instruction. Include FP32 save explanation, inclusive retention, absolute resume offset and per-run max_train_steps budget.
5. Run focused CPU tests and affected legacy regression tests. Record exact command, result, unavailable checks, and requirement coverage. Full H200 operation is only a later private-server procedure; do not execute or claim it locally.

### Acceptance gates

- **Files and parser gate**: Four files exist and every key/line parses. No unknown key, boolean-as-integer warmup, invalid path, role collision, or accidental use of val_unfamiliar for training.
- **Path and CLI gate**: One `--config_file` invocation selects both configured dataset TOMLs and model paths; existing caches skip encoder loading, missing caches encode, and invalid validation caches rebuild before model loading. Root relocation and other-CWD invocation preserve effective paths and Stage 1 identity.
- **Controlled integration gate**: Small CPU fixtures prove fixed grid, six tags, 0/periodic/final event coalescing, one strict best, one complete package without duplicate adapter/sample, inclusive retention and nonzero-step resume.
- **Compatibility and documentation gate**: Existing opt-out behavior and relevant tests remain passing; README/quickstart agree with actual commands, FP32 state policy, and run limits. Any contradiction to approved requirements is reported explicitly, not hidden by editing example values.

## Complexity Tracking

No constitution violations or additional production abstraction are planned.

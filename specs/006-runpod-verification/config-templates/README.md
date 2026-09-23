# Stage 5 short-case configuration templates

These are local preparation inputs for T007, not ready-to-run configs. The case files inherit the Stage 4 Qwen-Image `original` LoRA settings: rank/alpha 16, BF16, SDPA, AdamW8bit, learning rate `5e-5`, 1024 bucket rules, gradient checkpointing, no FP8 or block swap, and FP32 adapter saves. Every case has an explicit short `max_train_steps` and two-step warmup. Do not run a file containing `__MARKER__` or `TOK`.

For each case, copy `<case>.train.toml.in` to `$CASES/<case>/train.toml`, `train-dataset.toml.in` to the same directory as `train-dataset.toml`, and `val-dataset.toml.in` as `val-dataset.toml` where validation is enabled. Replace all model markers with verified absolute server paths. Replace dataset directory markers with the frozen absolute image/cache directories, and `TOK` in the one-line `sample_prompts.txt.in` with the chosen trigger. The G03-renamed case uses its separately prepared renamed image/caption copies and basename-matched source-bound caches; record their hashes before evaluation. G05-off omits `experiment_dir` and `val_dataset_config`; replace `__CASE_ROOT__` with its absolute case directory because legacy mode does not rebase paths. Keep its output separate.

| Case | Completed updates | Accumulation | Validation steps | Save / sample cadence | Retention |
| --- | ---: | ---: | --- | --- | ---: |
| G01 | 12 | 2 | 0, 4, 8, 12 | 4 / 4 | 8 |
| G02 | 10 | 1 | 0, 4, 8, 10 | 8 / 8 | 8 |
| G03 4x2, renamed, 10x1 | 0 | none | probe calls only | none | none |
| G04 | 0 | none | one probe call | none | none |
| G05 snapshot | 0 | none | one probe call | none | none |
| G05 on / off | 4 + 4 | 1 | on: 0, 4; off: none | 4 / disabled | on: 8 |
| G06 current / best | 4 + 4 | 1 | each: s, s+4 | 4 / 4 | 8 |
| G09, conditional | 2 | 2 | 0, 1, 2 | 2 / disabled | 2 |

The validation-only templates set `max_train_steps=1` solely because the Qwen CLI rejects zero; `real_probe.py` must never enter the training loop and must verify **zero** optimizer updates. The G03 `10x1` case loads adapter weights only in a separate validation protocol, never `--resume` into the `4x2` history. The G05 comparison disables samples in both branches; G05-off uses the legacy validation-disabled path. G06 supplies `--resume` on the command line with an actual copied package, preserving the latest G01 best package in the G06-current tree even when its step is later than the chosen current state.

With `R`, `SRC`, `CASES`, and `PY` set as in `verification-plan.md`, the planned one-shot commands are:

```bash
export PYTHONPATH="$SRC/src${PYTHONPATH:+:$PYTHONPATH}"
"$R/scripts/run_case.sh" G01 accelerate launch --mixed_precision bf16 "$SRC/qwen_image_train_network.py" --config_file "$CASES/g01/train.toml"
"$R/scripts/run_case.sh" G02 accelerate launch --mixed_precision bf16 "$SRC/qwen_image_train_network.py" --config_file "$CASES/g02/train.toml"
"$R/scripts/run_case.sh" G03 "$PY" "$R/scripts/real_probe.py" g03 --checkpoint "$R/evidence/selected-checkpoint.txt" --config "$CASES/g03-4x2/train.toml" --renamed-config "$CASES/g03-renamed/train.toml" --ten-by-one-config "$CASES/g03-10x1/train.toml"
"$R/scripts/run_case.sh" G04 "$PY" "$R/scripts/real_probe.py" g04 --checkpoint "$R/evidence/selected-checkpoint.txt" --config "$CASES/g04/train.toml" --out "$R/evidence/G04/forwards.jsonl"
"$R/scripts/run_case.sh" G05 "$PY" "$R/scripts/real_probe.py" g05 --snapshot-config "$CASES/g05-snapshot/train.toml" --enabled "$CASES/g05-on/train.toml" --disabled "$CASES/g05-off/train.toml" --initial-adapter "$R/evidence/selected-adapter.txt"
"$R/scripts/run_case.sh" G06-current accelerate launch --mixed_precision bf16 "$SRC/qwen_image_train_network.py" --config_file "$CASES/g06-current/train.toml" --resume "$CASES/g06-current/output/current_training_states/<actual-package>"
"$R/scripts/run_case.sh" G06-best accelerate launch --mixed_precision bf16 "$SRC/qwen_image_train_network.py" --config_file "$CASES/g06-best/train.toml" --resume "$CASES/g06-best/output/val_training_states/val-loss/<actual-package>"
"$R/scripts/run_case.sh" G09 accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 "$SRC/qwen_image_train_network.py" --config_file "$CASES/g09/train.toml"
```

G09 runs only after fresh inventory proves two suitable GPUs on the same Pod. The G09 two-rank no-update load, G04 independent inspection, G07/G08/G10 commands and all PASS criteria remain in [verification-plan.md](../verification-plan.md). Before any trainer launch, T009 must parse effective TOML and CLI settings, reject unresolved markers or a 1600-step fallback, and reconcile the total maximum of 38 updates plus conditional G09's 2. These templates are not included in the earlier 38-file runtime overlay; transfer them separately with recorded hashes if used on the Pod.

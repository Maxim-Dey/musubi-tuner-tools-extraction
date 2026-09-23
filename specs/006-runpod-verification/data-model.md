# Data Model: RunPod Verification Evidence

| Entity | Required fields / location | Integrity rule |
| --- | --- | --- |
| Run | `run_id`, selected Pod identity, creation UTC, `/workspace/musubi-val-loss-tests/<run_id>/` | New unique directory for each attempt; previous attempts remain intact. |
| Source identity | Exact base commit, local overlay file list and SHA-256s, archive SHA-256, remote checked hashes in `evidence/source-manifest.json` | Remote tree must match finished local bytes before model execution. |
| Environment | GPU count/type/VRAM, driver, Python/CUDA, package versions, free disk/VRAM, model paths and header/hash observations in `evidence/environment.json` | Separate observed values from inferred compatibility; no global changes. |
| Frozen fixture | Role, source and copied path, image/caption SHA-256, dimensions/bucket, source-bound cache path, protocol fingerprint in `evidence/fixture-manifest.json`; G03 renamed/reordered copies and their caches in a separate manifest | Familiar image hashes are a subset of train; unfamiliar hashes are disjoint; copied bytes match originals; both variants freeze before first evaluation. |
| Case | G ID, requirement IDs, effective train/val TOMLs, exact command, maximum completed updates/validation calls, expected steps, numeric criteria | One case ID owns one launch; configuration is checked against [verification-plan.md](verification-plan.md) before updating weights. |
| Case execution | Start/end UTC, PID, stdout/stderr/exit, completed and attempted update counts, validation event IDs, output package paths | A lost SSH connection leads to inspection of the original PID and files before any retry. No hidden extra launch. |
| Case result | `PASS`, `FAIL`, or `NOT RUN`; actual counts/steps, measured differences, reason, evidence paths and retrieval status | `FAIL` preserves observed discrepancy; `NOT RUN` names missing prerequisite. Controlled G08 evidence remains separate from real-model metrics. |
| Report | Local `verification.md` and `verification.json`, one G01–G10 row each | Links to concise retrieved evidence and remote paths for bulky items; no unsupported H200 or long-run claim. |

The remote run tree holds `code/`, `.venv/`, `fixtures/`, `experiments/<case>/train.toml` with each opt-in case's `output/`, `scripts/`, and `evidence/<case>/`. G05-off is the isolated legacy exception: its TOML has no `experiment_dir` or `val_dataset_config`. Model files and user datasets stay at their original `/workspace` paths. G06-current receives immutable copies of both the chosen current package and the prior best package, preserving its best score; G06-best receives the chosen best. These inputs are not counted as extra G01 outputs. A saved package owns its adapter, metadata, state, and samples; retention or rollback operates on that package as a unit.

State sequence: **planned → prepared → running → PASS/FAIL/NOT RUN → reported**. Preparation cannot advance until source transfer, environment, role/hash, cache, and effective-budget gates pass. A case blocked by a missing prerequisite becomes `NOT RUN`; an executed check that violates its criterion becomes `FAIL`.

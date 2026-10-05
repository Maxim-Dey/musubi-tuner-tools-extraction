# Validation evidence: Automatic cache preparation

## Status and governance

2026-10-06: Completed: implementation, local verification and independent convergence passed. Feature003 has its own original two-round correction allowance; post-implementation rounds used0/2. No model download, GPU, remote transfer, training run or packaging is part of this feature's acceptance.

Prior feature002 committed separately as f838ebc (deterministic validation and resumable experiment checkpoints). All88 other files in its89-file execution manifest still match the accepted bytes. The main training profile has three user additions since acceptance: min_timestep=0, max_timestep=1000, preserve_distribution_shape=true; these remain in the working tree and were not included in the prior feature commit. Commit stages only the known earlier configuration additions. Historical raw logs are retained unchanged; whitespace checking excludes raw .log evidence.

Branch: codex/auto-cache-training; active SpecKit feature: specs/003-auto-cache-training. Existing supported .venv: Python3.12.11, torch2.7.1+cpu and project dependencies. No new package is needed. Constitution and user prompts are protected; original SHA256 values remain142c40f1d23875e13a91c26271abd5e4035c9b04366548e9ccd0a48512fec626 and ebc005d4eeb644d57bd716cd9459feeefcc253975ca30730114f4a7b5dc13711.

Baseline tests/test_qwen_image_config.py:154passed,1failed in8.41s. The failure is the exact expected-profile assertion missing the three user-added timestep settings. T012 will update expectations while preserving those settings, not remove user values.

## SpecKit design and analysis

Resolved spec/plan/tasks templates using project scripts; prerequisites passed. No .specify/extensions.yml exists; no before/after hooks. Requirements checklist16/16 reviewed. Three independent agents inspected cache discovery, native launch/encoder lifetimes and artifact consistency before implementation. Source-only inventory and prepare_missing_caches interfaces agreed.

Read-only analyze assessed12FR,4SC,3stories and16tasks:100% task mapping;0unmapped tasks,0duplications,0task-order contradictions,0constitution conflicts. Finding I1/HIGH: all-resource preflight omitted the native Qwen tokenizer required by TE. In a separate pre-implementation editing step, plan/contracts/T004/T006/T009 now require lightweight tokenizer preflight before any cache stage when text work is planned and bypass it for complete caches. Verification: missing tokenizer means no subprocess; complete caches mean no tokenizer load. This is design completion, not a post-implementation correction round.

Single-process children receive a copied environment without inherited rank/world/master rendezvous settings; parent environment/device visibility are preserved. No new distributed coordinator is added.

## Implementation evidence

Initial implementation is complete. Data agent owns validation_inputs.py/test_auto_cache_inputs.py; orchestration agent owns auto_cache.py/test_auto_cache.py; coordinator owns native launch/config/docs and test_auto_cache_launch.py. One writer per file.

Before any launch-test edits, ownership of test_auto_cache_launch.py was delegated to the independent math/runtime agent for T007/T011; coordinator retains native parser/hook/config/docs. Implementation owners do not edit tasks or evidence.

T003 RED before source edits:10failed in0.44s because inspect_cache_inputs is absent. Real source cases cover train-only/train+val, missing train/val captions, overlaps/collisions/shared/nested cache directories and invalid val batch/repeats. T004 RED before orchestrator creation:1collection error in7.38s because training.auto_cache is absent; planned cases cover scope/ranks, all-source/all-resource/tokenizer preflight, native argv ordering, isolated child env, complete bypass, failed/incomplete/invalid postcheck and RNG. These are initial implementation checks, rounds0/2.


T007/T011 native launch RED before parser/hook implementation: 16 failed, 2 passed in 11.31s. GREEN after integration: 18 passed in 13.11s. Tests use actual Qwen main/config parsing and native cache mains with expensive encoders substituted; training reaches real dataset construction, validation and resume checks before a sentinel at the Accelerator/model boundary. Valid saved state with unchanged validation fingerprint passes; changed cache bytes after re-preparation fail the retained strict resume guard.

Final source-inventory GREEN: 35 new plus 20 retained validation-input checks, 55 passed in 17.81s. Real images, captions, JSONL, safetensors and metadata cover complete/partial populations, invalid/stale/provenance/shape/nonfinite caches, file-leaf directories/broken links, orphan train latents, isolation, duplicate destinations, byte preservation and fixed validation identity. On Windows without symlink privilege, the broken cache-file leaf metadata boundary is substituted; normal cache files/metadata remain real. Native cache-directory path resolution is unchanged; no extra raw-path ancestor walker was introduced.

Final orchestration GREEN: 26 passed in 10.67s. Native cache mains are exercised with expensive encoders and subprocess execution substituted. None/partial/complete, train-only/train+val, retry, unrelated-file preservation, all-resource/tokenizer preflight, incomplete output, scope, child environment and RNG checks pass. Complete caches run zero preparation stages and load no tokenizer; partial preparation preserves existing bytes.

Independent initial implementation review: zero confirmed in-scope defects in inventory, orchestration or native launch. No changes to training loss/noise/precision/optimizer/scheduler algorithms. Main profile now has auto_cache=true in addition to all 50 existing fields; the three user timestep values are retained. README and Qwen guide document automatic launch, manual disable/recovery and scope.

Scoped Ruff check and format check: all 7 new/modified Python files pass. Native Qwen main --help exits successfully and exposes both flags. Working diff whitespace check passes. Local Markdown file targets: 31 links checked across 11 documents, all exist. Constitution and sample prompt hashes match the protected values above. No extension hooks are installed.

## Requirement evidence matrix

| Requirement / acceptance | Implementation | Local evidence | Result |
|---|---|---|---|
| FR-001, FR-008; US3/AC1-2 | Qwen parser/preflight, effective configuration, main profile | test_auto_cache_launch boolean/CLI/native entrypoint cases; test_qwen_image_config profile; JSONL/other-CWD integration | PASS (including full profile regression) |
| FR-002; US1/AC1-3 | validation_inputs.inspect_cache_inputs enumerates source datasets before cached buckets | test_auto_cache_inputs source-complete train/val/no-val, captions/collision/overlap/destination checks | PASS |
| FR-003; US2/AC1-2; SC-002 | Missing-stage planner plus skip_existing/keep_cache | Real cache-main reuse/partial/retry cases, existing/unrelated byte assertions, zero complete-cache stages/tokenizer loads | PASS |
| FR-004; US2/AC3-4 | Existing source/provenance/tensor checks; retained native validation fingerprint | Invalid-cache inventory tests and native saved-state resume guard | PASS |
| FR-005; US1/AC1-2; SC-001 | Preflight hook before session/datasets/Accelerator; sequential child processes; RNG guard | Native launch ordering, child argv/order and parent RNG assertions | PASS (encoder execution substituted) |
| FR-006; US1/AC3 | All required model files and native tokenizer preflight | Missing later resource/tokenizer means zero children; complete caches bypass preparation resources | PASS |
| FR-007; US3/AC4; SC-003 | Contextual subprocess/postcheck failures; retain completed files | Failure/incomplete/invalid output and retry integration cases | PASS |
| FR-009; US3/AC3 | Disabled bypass; experiment and rank guards | Unsupported modes fail before encoding; manual launch bypass; copied child env and unchanged parent | PASS |
| FR-010 | README.ru.md, docs/qwen_image.md, unchanged manual entrypoints | Documentation review, native cache-main integration and local link checks | PASS |
| FR-011; SC-004 | Full SpecKit workflow, independent agents and local gates | Red/green evidence above; full suite and converge below | PASS; final converge clean |
| FR-012 | Separate validation commit f838ebc, new feature/branch | Prior commit recorded before implementation; retained user values, prompts and historical evidence | PASS |


## Final local gate

Full retained CPU suite after integration: **477 passed in 132.67s**, comprising the prior 398 tests and 79 new cases. This includes the updated exact main-profile assertion, all current user values, native config/launch/path behavior and prior validation/resume regressions. No skip or unresolved failure was reported.

Reproduction from the repository root uses the existing virtual environment, PYTHONPATH=src, PYTHONIOENCODING=utf-8, HF_HUB_OFFLINE=1 and CUDA_VISIBLE_DEVICES empty. Import torch before invoking pytest to use the established local DLL initialization order:

```python
import torch, pytest
raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", "tests"]))
```

The pre-existing exact-profile baseline mismatch is resolved by preserving and asserting the three user settings, not changing their values. Seven-file scoped Ruff/format, native help, whitespace and file-link checks also pass. No confirmed initial implementation defect remains. Post-implementation correction rounds used: **0/2**.


## Final convergence and closure

Independent `speckit-converge` outcome: **converged**. The sole intent sources were spec.md, plan.md and tasks.md, governed by the constitution. No git/history/diff, file writes or test reruns were used during the assessment. No extension hooks exist before or after convergence.

| Inventory | Checked | Result |
|---|---:|---|
| Functional requirements | 12 | Satisfied |
| Success criteria | 4 | Satisfied within local scope |
| Acceptance scenarios | 11 | Satisfied |
| Plan decisions | 14 | Satisfied |
| Constitution principles | 7 | Compliant |
| Tasks | 16 | Implementation and verification evidenced; audit/report closure recorded separately afterward |

The 14 assessed plan decisions were: existing preflight placement; opt-in/strict bool/CLI precedence; single-process experiment scope and disabled compatibility; source-complete inventory; native validity/destination/isolation checks; sequential native missing-stage processes; complete checkpoint/tokenizer preflight; native arguments and role-specific absolute paths; copied child environment with parent argv/cwd/environment preservation; parent RNG restoration; complete postcheck/contextual failure/retry preservation; retained validation/resume identity; minimal compatible implementation without dependencies or format changes; meaningful CPU verification/documentation and bounded independent workflow.

All seven constitution principles pass: Language, Task Fidelity, Minimal Compatible Changes, Training Invariants, Memory and Performance, Configuration and Errors, Verification and Documentation. Findings by type: missing 0, partial 0, contradicts 0, unrequested 0. Findings by severity: CRITICAL 0, HIGH 0, MEDIUM 0, LOW 0. No new convergence tasks or corrective pass is needed.

During read-only convergence tasks.md remained byte-for-byte unchanged, SHA256 **18a8566c67ae5fff60c92283fb19b2c2293a2cd2f0aa4225774c3ed7f7740460**, independently confirmed by reviewer and coordinator. Only after the clean result, a separate administrative step marked T015/T016 and the specification complete. Correction rounds remain **0/2**. All16 tasks are complete.

Delivery: the ordinary main-profile launch prepares missing train/val caches before training, reuses valid caches and stops on invalid files. `--no_auto_cache` restores manual preparation. Prior validation is committed as **f838ebc**; this new feature is implemented in the working tree on **codex/auto-cache-training**, without a new feature commit or push. The three pre-existing user timestep settings remain present.

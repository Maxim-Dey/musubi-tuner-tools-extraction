# Tasks: Automatic cache preparation

Read spec.md, plan.md, research.md, data-model.md and contracts/launch.md. One writer per file. Tests target meaningful source/cache and launch boundaries; no GPU/download/remote work.

## Phase 1: Setup
- [x] T001 Record prior validation commit, preserved user edits, local environment and baseline results in specs/003-auto-cache-training/validation.md (FR-011,012).

## Phase 2: Foundation
- [x] T002 Verify native preflight/cache boundaries and agree CacheInventory/inspect_cache_inputs and prepare_missing_caches interfaces in specs/003-auto-cache-training/plan.md and contracts/launch.md before parallel code work (FR-002,005,008,009).

## Phase 3: US1 — First launch (P1)
Goal: uncached train/val launch prepares all required files before training. Independent check: real source/cache fixtures and native training boundary with expensive encoders replaced.
- [x] T003 [P] [US1] Write failing source-inventory tests in tests/test_auto_cache_inputs.py for missing train/val/no-val, source/caption/cache collision and isolation checks before encoding (FR-002,004; SC-001,003).
- [x] T004 [P] [US1] Write failing orchestration tests in tests/test_auto_cache.py for required-stage ordering, all-resource including native-tokenizer preflight, child failure and incomplete output using the agreed inventory contract (FR-005–007; SC-001,003; analyze I1).
- [x] T005 [P] [US1] Implement source-only CacheInventory/inspect_cache_inputs in src/musubi_tuner/training/validation_inputs.py, factoring existing source-pair checks without weakening validation/resume; missing_latents/missing_text are immutable tuples and present invalid paths are errors; depends T003 (FR-002,004).
- [x] T006 [P] [US1] Implement sequential native subprocess preparation in src/musubi_tuner/training/auto_cache.py with all needed VAE/TE/tokenizer resources checked first, preserved parent RNG/environment, isolated child launcher environment, skip_existing+keep_cache and complete postcheck; depends T004 (FR-003,005–009; analyze I1).
- [x] T007 [US1] Add Qwen-only auto_cache/no_auto_cache parser and existing-preflight hook in src/musubi_tuner/qwen_image_train_network.py plus failing-first native-entrypoint tests in tests/test_auto_cache_launch.py; depends T005,T006 (FR-001,005,008; SC-001).

## Phase 4: US2 — Reuse and recovery (P1)
Goal: no unnecessary encoding or overwrite; partial work can resume. Independent check: complete/partial real cache inventories and byte-preservation assertions.
- [x] T008 [P] [US2] Extend tests/test_auto_cache_inputs.py with complete/partial caches, corrupt/stale provenance, directory/broken-link paths, unexpected training cache and existing validation identity regressions; depends T005 (FR-003,004; SC-002,003).
- [x] T009 [P] [US2] Exercise native cache entrypoints in tests/test_auto_cache.py using substituted expensive encoders; prove missing-only writes, keep_cache preservation, zero complete-cache subprocesses/tokenizer loads and retry after failure; depends T006 (FR-003,006,007; SC-002,003; analyze I1).
- [x] T010 [US2] Verify all source/encoding reuse checks and existing strict validation/resume tests; resolve initial-implementation defects only in owned validation_inputs.py/auto_cache.py files and record evidence in specs/003-auto-cache-training/validation.md; depends T008,T009 (FR-003–007,011).

## Phase 5: US3 — Configuration and compatibility (P2)
Goal: supplied profile works automatically and old/manual launches remain controllable. Independent check: real config/parser and launch boundaries from different directories.
- [x] T011 [US3] Test strict bool/CLI enable-disable priority, disabled bypass, single-process/experiment guards, paths with spaces/other cwd, fail-before-model ordering and parent RNG in tests/test_auto_cache_launch.py and tests/test_auto_cache.py; depends T007,T010 (FR-001,005,008,009; SC-003,004).
- [x] T012 [US3] Enable auto_cache in config_for_qwen_image_lora/train.toml while preserving all50 current fields/prompts; update current profile expectations in tests/test_qwen_image_config.py including the three pre-existing user timestep settings; depends T011 (FR-001,008,012).
- [x] T013 [US3] Update README.ru.md and docs/qwen_image.md for automatic existing launch, reuse, disable, scope and manual recovery, keeping manual cache commands; depends T012 (FR-010).

## Phase 6: Final verification
- [x] T014 Run focused/new and full retained CPU regressions, scoped lint/format, CLI help and diff checks; record requirement→implementation→check→result in specs/003-auto-cache-training/validation.md; depends T013 (FR-011; SC-001–004).
- [x] T015 Perform independent read-only/append-only speckit-converge against spec.md/plan.md/tasks.md and record its result separately in validation.md; never rewrite tasks during convergence and use the original two-round correction cap for confirmed post-implementation gaps; depends T014 (FR-011; SC-004).
- [x] T016 Mark factual completion and deliver the launch/limitations/commit summary using specs/003-auto-cache-training/validation.md and tasks.md; depends T015 (FR-010–012; SC-004).

## Dependencies and parallel work

T001→T002→US1→US2→US3→final gate. T003/T004 may run together; T005/T006 follow their own failing tests and may run together against the agreed interface. T008/T009 likewise own separate files. Coordinator serializes launch/parser/config/docs. No task regenerations after implementation starts; convergence appends only confirmed new work. MVP is US1, but the requested complete delivery includes all stories.

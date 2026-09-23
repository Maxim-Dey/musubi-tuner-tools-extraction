# RunPod Verification Report

**Overall: FAIL.** The selected L40S Pod could run the original Qwen-Image-2512 BF16 model and initial validation, but it could not complete the required step-0 sample. G01 and G02 each ended in CUDA out-of-memory before any optimizer update or complete state package. The retry with PyTorch expandable CUDA segments also ended at step 0. No precision, model, image size, LoRA rank, or sampling requirement was reduced to claim a pass.

## Identity and preparation

| Item | Observed result |
| --- | --- |
| Pod / run | yxd2b6cm3e8gfa / 20260923T0243Z-e8af43a |
| Remote root | /workspace/musubi-val-loss-tests/20260923T0243Z-e8af43a |
| Source | Base commit e8af43abc32d967153e3aa3c31d50c77d3f7c47e plus 38 byte-verified local overlay files; transfer archive SHA-256 1fd5305e03d064165d2f5434a420ef05278e0d48be753954d0e2457ce46c8485 |
| Hardware | One NVIDIA L40S, 46,068 MiB VRAM, driver 580.178.04; Python 3.12.3 and isolated torch 2.8.0+cu128 |
| Models | Qwen-Image-2512 BF16 DiT, Qwen-Image VAE, Qwen2.5-VL text encoder; all three SHA-256s and headers recorded in remote evidence. DiT, VAE and text encoder loaded with matching keys. The observed checkpoint is the 2512 release in original model mode. |
| Fixtures | Three train, two familiar, three unfamiliar existing 1024×1024 image/caption pairs copied byte for byte. Familiar image hashes are a subset of train; unfamiliar hashes are disjoint. All available selected images have the same bucket, so bucket-diversity coverage was unavailable. |
| Caches | All 26 canonical and renamed source-bound latent/text cache files hashed; Stage 1 preflight accepted both role sets. Canonical fingerprint f719d57d2feb785f7c74d1fbcecd7852f03180792e92377f02531a31ddba8894. |
| Static gate | PASS: source, fixtures, caches, model hash evidence, 12 effective short configurations and parser preflight. The 1,600-step example budget was rejected. Maximum planned updates: 38 on this Pod, 40 only with two suitable GPUs. |

The exact manifest paths and SHA-256s are in the remote evidence tree and in [verification.json](verification.json). Source data, models, user processes, and Pod lifecycle were left untouched.

## G01–G10 results

| Case | Status | Expected | Actual and reason |
| --- | --- | --- | --- |
| G01 | **FAIL** | 12 updates; six tags at steps 0, 4, 8, 12; complete packages and samples | Exit 1. Six finite tags at step 0, 0 updates and 0 packages. Step-0 sample failed when CUDA requested 1.13 GiB with 1.03 GiB free. A separate allocator retry also exited 1 at step 0, requesting 192 MiB with 63 MiB free. |
| G02 | **FAIL** | 10 updates; tags at 0, 4, 8, 10; final step-10 package | Exit 1. Six finite tags at step 0, 0 updates and 0 packages; the same 1.13 GiB sample allocation failed. |
| G03 | **NOT RUN** | Five fixed-check calls, zero updates | No complete G01 checkpoint. |
| G04 | **NOT RUN** | One real-forward reference call, zero updates | No complete G01 checkpoint. |
| G05 | **NOT RUN** | One state snapshot and paired 4+4 updates | No selected G01 adapter. |
| G06 | **NOT RUN** | Current and best resume, 4+4 updates | No complete current or best G01 package. |
| G07 | **NOT RUN** | Full event, best, package and retention inspection | Partial inventory found six step-0 tags in each failed G01/G02 run and no packages. Full attribution and retention cannot be assessed. |
| G08 | **PASS** | Controlled CPU regression cases pass; zero real-model updates | Exit 0; 228 passed, 0 failed or skipped, one PyTorch scheduler warning. |
| G09 | **NOT RUN** | Two-rank run only with two suitable GPUs on this Pod | Fresh inventory showed one L40S; zero updates. |
| G10 | **PASS** | Ten honest case records, reconciled budget and process state | This report and [verification.json](verification.json) contain ten statuses. Total completed real-model updates: 0, including the failed retry. No agent test process remains; Pod is running. |

G01/G02 step-0 TensorBoard values are finite and identical in both runs: familiar mean 0.0364812, low 0.0373929, high 0.0355695; unfamiliar mean 0.0288093, low 0.0293125, high 0.0283061. These are initial evaluations only. They do not establish training invariance, convergence, best selection, or image quality.

## Evidence and limits

The verified interactive SSH channel retrieved the concise remote G10 summary byte for byte as [verification.json](verification.json), SHA-256 a7a86a7be56d3c58182f06fd179df04341500647719a3a2c61f6f28a41df0e2b. The remote evidence root contains:

- evidence/G01, evidence/G02 and evidence/G01-alloc-retry: exact command, PID, UTC start/end, stdout, stderr and exit.
- evidence/G07/results.json: scalar and package inventory; evidence/G08: pytest output, exit and JUnit; evidence/G09: fresh GPU inventory and NOT RUN reason.
- evidence/G10/summary.json and final-processes.json: reconciled case records and final process inventory; evidence/preflight.json and the source, fixture, cache and configuration manifests.

The local JSON carries the exact remote evidence paths. The 2512 checkpoint and one L40S are the observed setup; this run makes no H200, long-run, resume-success, or image-quality claim. The immediate blocking condition is insufficient L40S memory for the mandatory 1024-pixel step-0 sample under the agreed BF16/no-FP8/no-block-swap configuration. The expandable-segments attempt did not resolve it. Further GPU cases require a suitable environment and a new bounded verification run; no product algorithm was changed on the Pod.

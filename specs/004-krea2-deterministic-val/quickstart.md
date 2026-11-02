# Prepared single-GPU experiment and verification

## Current deployment — 2026-10-30

The previous Pod was lost. Replacement Pod `2n39ofwfmx7rks` is ready for the user on its supplied single GPU: current GPU diagnostics, independent verification and requested post-test cleanup passed on 2026-10-30. Both feature correction rounds remain consumed and closed; this operational redeployment made no source-code change or new correction pass. Real multi-GPU T031 remains unverified on one supplied GPU.

Direct connection (key stays local):

```sh
ssh root@81.27.69.177 -p 36627 -i ~/.ssh/id_ed25519
```

Fallback connection:

```sh
ssh 2n39ofwfmx7rks-64411d9e@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Working directory remains `/workspace/musubi-krea2-val-004`; model paths and main configuration are unchanged. Current independent post-cleanup gate passed. Models, original data,254 prepared train/val caches and the main manifest are retained; main user output has no diagnostic checkpoints/events. Full1000-step user training was not started. The commands below are ready for the user.

## Activate the prepared environment

```sh
cd /workspace/musubi-krea2-val-004
source .venv/bin/activate
```

Current replacement-Pod environment manifest verifies Python 3.12.3 (GCC 13.3.0), torch 2.8.0+cu128/CUDA 12.8, Accelerate 1.6.0, transformers 4.57.6, diffusers 0.32.1, huggingface-hub 0.34.3, safetensors 0.4.5, bitsandbytes 0.50.2 and TensorBoard 2.21.0. Isolated current hf CLI is 2.4. These are actual replacement-Pod values from deployment-evidence/environment.json.

## Existing verified models and complete data

Main TOML contains actual model paths:

- DiT: `/workspace/models/Krea-2/diffusion_models/krea2_raw_bf16.safetensors`.
- VAE: `/workspace/models/Qwen-Image/split_files/vae/qwen_image_vae.safetensors`.
- Encoder: `/workspace/models/Qwen3-VL/text_encoders/qwen3vl_4b_bf16.safetensors`.
- Tokenizer: `/workspace/models/Qwen3-VL-4B-Instruct`.

The Raw DiT is the public Comfy-Org/Krea-2 revision 1171f4dc7e325bf3734586942e737872506c73e1 distribution, byte-identical to the official Raw LFS object. Actual SHA256 `f99bb0ff8e362b77342bc4994e0c50906fe7ef7074864b181b7d48d2fa6d03d7`, size 26,283,332,608 bytes; strict native loading passed. Historical official-source 403 did not require accepting terms or copying tokens: the identical public copy is present and verified, so no download guard is needed. Exact revisions/hashes are recorded in verification.md and `/workspace/model_sources.json`.

All 107 train and 20 independent val images retain their names/captions. Separate train/val caches contain 254 files, produced with actual VAE/encoder entrypoints and reused without rewriting. Full production preflight from another cwd and the immutable 20-record main manifest passed. Diagnostic outputs/caches are separate from main `output/`.

## Turbo sampling enabled — native BF16 verified

The main config now sets `turbo_dit="/workspace/models/Krea-2/diffusion_models/krea2_turbo_bf16.safetensors"` and `turbo_dit_cache=true`; the ten existing prompts change only steps52→8 and CFG4.5→1. Raw train/val, seeds, resolutions, input split, LR/batch/save settings are unchanged. Turbo sampling uses fixed mu1.15 and the current trained LoRA, with guaranteed Raw rollback; explicit prompt overrides remain honored. This is a separate narrow restoration request, not a reset of feature004's completed2/2 correction counter or its historical acceptance.

Full Turbo SHA256, actual parser, independent review and native actual main() BF16 acceptance passed on the replacement Pod. Resident M1, streaming M2 and Raw control each completed two updates with identical training losses, Raw val metrics, both792-key FP32 adapters and optimizer/scheduler/RNG/data-position state. Resident/streaming Turbo images matched exactly. Cached mode keeps both Raw+Turbo CPU states (about52.6GB) plus loader/conversion overhead; streaming still needs a transient Raw rollback snapshot. The actual Pod cgroup limit is about233.76GiB. Gradient checkpointing is supported; full Turbo swapping rejects block swap/ConvRot INT8. The earlier heavy observer diagnostic FAILED its next gradient-checkpoint recomputation; exact cause remains unresolved. Its specifically successful swap/restore invariants are retained separately from its failed continuation. Native production acceptance passed without those observers, with no production fix or tolerance change. GPU FP8/compiled/multi-GPU are unverified; no blanket all-diagnostics-pass claim is made. See docs/krea2.md and verification.md.

## Start the user's full experiment

After activation, this command starts a new full-data experiment using the verified model paths already in TOML:

```sh
python krea2_train_network.py --config_file config_for_krea2_raw_lora/train.toml
```

The agent did not launch this 1000-step run. Main train batch 16, accumulation 1 (effective 16), bf16, SDPA, gradient checkpointing, swap 0 and quantization disabled remain unchanged. A full-data batch 16 optimizer-step diagnostic passed without OOM, peak allocated about 82.08GiB. Fixed val N1=10/N2=2, save/sample/val periods 200, FP32 adapters and the single save_last_n_steps=1000 retention window are retained. This window applies to the complete checkpoint folder, including adapter weights, optimizer/scheduler/RNG state, metadata and samples. Paths in main TOML resolve from its directory; dataset paths resolve from their respective TOMLs, including an unrelated cwd when using absolute script/config paths.

## Resume only a real user checkpoint

No user checkpoint exists now. This command searches only the user experiment's completed state folders and reports absence instead of selecting diagnostic state:

```sh
state=$(find config_for_krea2_raw_lora/output -maxdepth 2 -type f -path '*/krea2_raw_lora-step*/checkpoint_complete' -printf '%h\n' | sort | tail -n 1)
if [ -z "$state" ]; then
  echo 'No complete checkpoint exists in the user experiment.'
else
  python krea2_train_network.py --config_file config_for_krea2_raw_lora/train.toml --resume "$state"
fi
```

`--network_weights` initializes an adapter only; full continuation uses `--resume`. Preserve compatible models/data/topology. Each complete checkpoint contains one FP32 model.safetensors, native state, train cursor, validation metadata and only same-step samples. Incomplete folders do not qualify.

## Current deployment checks and reproducibility

Current replacement-Pod GPU checks passed: controlled continuous5 versus stop2/resume5 and no-val5 had exactly equal10 training records, all792 FP32 adapter tensors and native optimizer/scheduler/RNG/data positions; metadata, manifest and step4 sample matched. All three val tags occurred at0,2,4,5 once, with no restored-step sample orphan or false epoch-loss entry. Standalone adapter reload had no missing/unexpected keys.

A combined main N1=10/N2=2, swap2/gradient-checkpointing run passed40 actual forwards per validation, repeated metrics, independent reference, exception restoration and next train update. The full107-image batch16/accumulation1 optimizer-step diagnostic passed without OOM; main settings stayed unchanged. These are newly completed replacement-Pod checks, independently verified before test cleanup, rather than assumed from the lost Pod.

Default GPU backend same-mode repeats differed numerically despite identical input/RNG snapshots. The original val/no-val tolerance failure is preserved in the journal; no tolerance was enlarged and main backend/training math was not changed. Equal deterministic diagnostic controls (SDPA-MATH, deterministic algorithms/cuDNN/CUBLAS settings, TF32 disabled) gave exactly equal training losses and all 792 adapter tensors for val/no-val and continuous/resumed comparisons. These controls demonstrate isolation and resume under matched deterministic execution; they do not promise bitwise repeatability of the default backend. Real multi-GPU validation is not verified on this single-GPU Pod.

Final post-cleanup inspection independently confirmed unchanged343 runtime files, original image/caption data and all254 cache hashes/mtime; main manifest SHA256 `111024c4ca233d17a8cedcac7960a24ad308a60fc5135235922625e8db3a348c`. All34 authorized test roots were removed (11,535,860,530 bytes); user output contains no test state/events. Model files, environment, original data, prepared caches, main manifest and TensorBoard remain. Final GPU use was0MiB/0% with no compute jobs.

Heavy diagnostic folders were removed after preserving the verified compact evidence archive locally at `remote-evidence/deployment-20261030/compact-gpu-provisioning-evidence.tar.gz` (136 members,1,410,954 bytes; SHA256 `5e1d62b6b949168570639fa47bfc0aedcdb2a411570aebe86af9420a59943908`). Compact current reports remain under remote `deployment-evidence/`. Historical diagnostic paths in verification.md are evidence history, not a promise deleted directories remain on the Pod.

## TensorBoard

Final replacement-Pod internal and independent external HTTP200 with TensorBoard title verified; persistent service PID1598, PPID1, SID1598. Address: [TensorBoard](https://2n39ofwfmx7rks-6006.proxy.runpod.net/). It binds 0.0.0.0:6006 and reads `/workspace/musubi-krea2-val-004/config_for_krea2_raw_lora/output/tensorboard`. It was left running independently of SSH, using `deployment-evidence/tensorboard.pid` and `deployment-evidence/tensorboard.log`. User training events remain absent until the user starts training; diagnostic events stay in their own output.

Status:

```sh
curl -I http://127.0.0.1:6006/
ps -p "$(cat deployment-evidence/tensorboard.pid)" -o pid,comm
```

Start only if the recorded service is not running:

```sh
if ! kill -0 "$(cat deployment-evidence/tensorboard.pid 2>/dev/null)" 2>/dev/null; then
  nohup .venv/bin/tensorboard --host 0.0.0.0 --port 6006 --logdir /workspace/musubi-krea2-val-004/config_for_krea2_raw_lora/output/tensorboard > deployment-evidence/tensorboard.log 2>&1 &
  echo $! > deployment-evidence/tensorboard.pid
fi
```

Stop only this created service if desired:

```sh
kill "$(cat deployment-evidence/tensorboard.pid)"
```

Replacement-Pod diagnostics and requested cleanup are complete. User output is clean of diagnostic states/events, prepared train/val caches are preserved and TensorBoard remains running. Full user training has not been launched.


## Turbo acceptance cleanup and evidence

After independent native BF16 acceptance,16 own diagnostic roots were removed:9,884,294,243 bytes,100 files and8 symlinks without following links. All345 runtime identities, original data,254 cache SHA256/size/mtime and main manifest bytes/mtime remained unchanged; manifest SHA256111024c4ca233d17a8cedcac7960a24ad308a60fc5135235922625e8db3a348c. Main user output contains no diagnostic state/events. Raw/Turbo/public models and virtual environment remain. Final GPU0MiB/no compute jobs; existing TensorBoard PID1598 remained untouched and returned internal/external HTTP200/title at the current proxy.

The verified local archive `remote-evidence/turbo-restoration-20261030/compact-turbo-acceptance-evidence.tar.gz` preserves88 members/6,631,441 bytes, SHA256 `126eab0220e45a91cb8c31afefb1af627f86e348a6f9c6d61df38748641bf102`, including the failed heavy diagnostic and successful native cases. Deleted remote diagnostic paths are historical; compact final reports remain under `deployment-evidence/turbo-restoration-20261030/`. Full user training has not started. Main command above already enables Turbo through TOML; Raw training and validation remain Raw. Old feature004 correction counter2/2 and real-multi-GPU T031 limitation are unchanged.

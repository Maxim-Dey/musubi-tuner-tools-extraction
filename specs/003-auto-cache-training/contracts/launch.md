# Launch Contract

Qwen original training accepts auto_cache=true/false in TOML; defaultfalse. --auto_cache enables and --no_auto_cache disables, with CLI priority and mutual exclusion. Only strict booleans are accepted.

Enabled mode requires experiment_mode=true and a single training process. Invalid combinations fail before cache writes. The supplied main config enables both. Existing manual cache commands remain unchanged.

Before training models, inventory effective train plus optional val sources. For each incomplete dataset role, run latent then text preparation as required, train before val. Each subprocess uses the current Python and native module, --dataset_config, --model_version original, --experiment_mode, --skip_existing, --keep_cache, --batch_size 1, --num_workers 1, plus the appropriate absolute --vae or --text_encoder. --validation is val-only; --fp8_vl follows the selected TE setting.

All needed resources are checked before any subprocess. No jobs means no subprocess and no added encoder requirement. Child failure/incomplete output prevents training and reports role/stage/dataset. Existing valid and unrelated files are never automatically replaced or removed. Existing strict validation/resume fingerprint checks still run.

No guarantee of detecting encoder-checkpoint changes is added; cache validity follows existing source/geometry/caption/provenance contracts.

Resource preflight also loads/releases the native Qwen tokenizer before any cache child when text preparation is required; it does not load TE weights. Complete caches bypass the tokenizer check. Children receive an environment copy with launcher rendezvous variables removed; parent environment/device visibility remains unchanged.

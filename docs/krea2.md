# Krea2 Raw LoRA

The three supported commands cache image latents, cache Qwen3-VL representations, and train a Krea2 Raw LoRA. Follow the [root quickstart](../README.md) for commands and the supplied experiment configuration.

## Representation and model contracts

Krea2 uses a single-stream MMDiT, Qwen3-VL-4B-Instruct conditioning and the Qwen-Image VAE. Supply safetensors files to `--dit`, `--text_encoder` and `--vae`. Qwen3-VL weights accept supported official/ComfyUI layouts; the tokenizer uses the Qwen3-VL repository ID and must already be cached for offline operation.

Image latents are normalized by the Qwen VAE's existing mean/std. Cache tensors retain channel/frame/spatial layout, with one image frame. Text caches contain selected encoder hidden-state layers for valid tokens only; training pads these variable-length stacks and preserves their masks. Do not substitute caches produced for another architecture.

## LoRA targets

`network_module = "networks.lora_krea2"` (also accepted as `musubi_tuner.networks.lora_krea2`) selects the Krea2 adapter. The default targets all DiT Linear layers, including text fusion and projections. Raw modulation/RMSNorm parameters are not Linear layers. Rank, alpha, neuron/rank/module dropout, LoRA+ and target patterns remain configurable through the existing network options.

For example, `network_args = ["loraplus_lr_ratio=4"]` enables the existing LoRA+ learning-rate ratio. `exclude_patterns` and `include_patterns` can narrow the target modules; see [advanced configuration](advanced_config.md).

## Timesteps and precision

The starter uses `timestep_sampling = "krea2_shift"` and `weighting_scheme = "none"`. `krea2_shift` follows the existing resolution-dependent Raw sampling shift. Fixed `shift` with `discrete_flow_shift`, and the other retained mathematical distributions, remain available; their names do not load another model. `max_train_epochs`, if supplied, takes precedence over the example step limit.

The base DiT uses bf16; the starter requests bf16 mixed precision. Adapter save precision is independently configurable. Supported mixed-precision choices and optional optimizers require a compatible runtime; the example is not a hardware guarantee.

## Memory options

- `--gradient_checkpointing` recomputes activations; `--gradient_checkpointing_cpu_offload` additionally offloads saved activations and requires checkpointing.
- `--fp8_base --fp8_scaled` stores the eligible main-block Linear weights in scaled FP8. Text fusion, norms and raw modulation tensors remain outside that scope.
- `--convrot_int8` is an alternative to FP8. `--convrot_int8_bwd bf16` is the standard backward mode; `int8` requests the CUDA/Triton path. Prequantized supported ConvRot safetensors can be used with this option. Base-LoRA merging into prequantized INT8 weights is rejected.
- `--blocks_to_swap N` offloads main blocks; the 28-block Raw model supports up to 26 swapped blocks. See [block swap](block_swap.md) for H2D-only and pinned transfers.
- `--compile` applies the existing model-block compilation path. See [compile settings](torch_compile.md).

Choose one training attention backend: `--sdpa`, `--flash_attn`, `--xformers` or `--flash3`; optional backends need their corresponding packages. `--split_attn` remains available. Unsupported combinations fail before models are loaded.

## Outputs and continuation

The training hooks generate Raw samples by default or opt-in Turbo samples, log progress/optional tracker metrics, save adapter safetensors and save Accelerate state. Sample prompts, save intervals and retention are configurable. Resume with `--resume STATE_DIRECTORY`; adapter-only initialization uses `--network_weights`. See [sampling](sampling_during_training.md) and [advanced settings](advanced_config.md).

## Fixed validation loss

Enable validation with the complete explicit group: `val_dataset_config`, `val_every_n_steps`, `val_seed_noise`, `val_level_noise_n` and `val_seed_noise_n`. The main example uses period 200, seed 42, N1=10 and N2=2. Validation is absent only when all five options are absent. Validation data must be nonempty with batch_size=1 and num_repeats=1, without random preprocessing/caption changes. Compatible duplicate image content counts once; conflicting duplicates and train/val content overlap fail before model loading.

For each unique image and each i=1..N1, j=1..N2, t=0.05+(i-0.5)*0.90/N1. For N1=10 the levels are 0.095,0.185,0.275,0.365,0.455,0.545,0.635,0.725,0.815,0.905. Low has t<0.5, high t>=0.5. Standard normal noise comes from an isolated generator seeded by SHA-256 of the common seed, original-image content SHA-256 and one-based level/realization indices. Filenames, absolute locations, rank, traversal order and optimizer step do not enter the seed.

The cached VAE latent is already normalized. Validation forms x_t=(1-t)*latent+t*noise, calls the existing Krea2 forward with timesteps=1000*t and predicts noise-latent. It uses no training sampler/shift/+1, CFG or sample generator. Loss is latent-element MSE with the selected training-meaning weighting evaluated at the actual fixed t; `none` gives unit weight. Average combinations within each image, then images equally, regardless of image size. Distributed validation partitions unique images without padding, reduces sums/counts and permits empty ranks. It never averages unequal local means as equally weighted values.

The three main-process TensorBoard tags are `val_loss_mean`, `val_loss_low_noise`, `val_loss_high_noise`. The mean equals half the low-plus-high means within numerical tolerance because N1 is even. X is completed optimizer updates. Measurements occur at new step 0, every period and an unduplicated final step; resume uses the restored step and persisted validation metadata. Sampling intervals do not enable or disable validation.

Validation temporarily uses eval/no autograd, disables dropout and restores Python/NumPy/Torch CPU/CUDA RNG, individual module modes, gradient checkpointing and block swap in finally, including exceptions. It must preserve existing gradients, weights, optimizer and scheduler. Seeds/noise are reproducible under the frozen generator contract; forward/reduction precision depends on dtype, device, kernels and hardware. Identical seeds alone do not establish identical training trajectories. See feature verification for actual tested numerical tolerances and the actual GPU evidence and separately unverified real multi-GPU checks.

## Experiment inputs and caches

`experiment_mode` and `auto_cache` are opt-in and default false. Main configuration local paths resolve from its TOML directory; dataset paths resolve from the corresponding dataset TOML. Ordinary mode retains old working-directory rules. The supplied full-data train/val configs have separate cache/train and cache/val directories.

Automatic preparation reuses existing Krea2 cache producers and only fills missing compatible files before loading DiT. Cached latent names preserve original-image dimensions: `<stem>_<width:04d>x<height:04d>_kr2.safetensors`; text names are `<stem>_kr2_te.safetensors`, architecture is `krea2` and embeddings use `varlen_krea2_vl_embed_<dtype>`. Provenance must match source bytes/caption, model/tokenizer identities, preprocessing/buckets, representation, shape/dtype and normalization. Equal-stem collisions fail before writes. Existing incompatible files and legacy files without sufficient evidence are rejected rather than overwritten silently; use a separate empty cache location or explicitly regenerate outside resume after correcting inputs. `auto_cache=false` requires complete compatible files.

A frozen val manifest records semantic input/cache/component/noise identities. Relocation does not alter that identity; image/caption/bucket/cache changes do. The manifest is checked before resume/model loading and again at each validation, never replaced to hide drift.

## Unified experiment state

Only experiment_mode with save_state uses one exact-step checkpoint folder with sole FP32 `model.safetensors`, native optimizer/scheduler/RNG state, completed optimizer count, exact consumed train cursor, frozen manifest and validation state. Explicit incompatible save_precision is an error. Resume restores the full state; network_weights only initializes the adapter. The accumulation counter is not the completed optimizer count. A completed checkpoint is published only after required state is written; final and periodic saves of one state are deduplicated.

In experiment mode, save_last_n_steps is the single optimizer-step retention window for the whole unified checkpoint folder, including adapter weights, optimizer/scheduler/RNG state, metadata and samples. Omit it for unlimited retention. The main value is 1000; the existing save-interval boundary determines which complete obsolete folders are removed. A sample belongs only to its actual weight step: matching sample/save intervals place it under that checkpoint's samples; unmatched sampling and sample_at_first have separate explicitly step-numbered output. Ordinary mode keeps its previous format.

The feature [quickstart](../specs/004-krea2-deterministic-val/quickstart.md) separates local tests, short GPU diagnostics and the user's future full training. Required GPU/resume/TensorBoard evidence is recorded in [verification](../specs/004-krea2-deterministic-val/verification.md); local CPU success is not full operational acceptance.


## Turbo samples during Raw training

`--turbo_dit PATH` temporarily uses a full distilled Turbo base for sample generation with the current trained LoRA. Training and deterministic validation always use Raw. The same model object is restored to Raw in finally, including sampling exceptions, with modes/RNG/gradient-checkpointing state restored. Omit both new options for the previous Raw behavior. This restoration supports only `turbo_dit` and `turbo_dit_cache`; it adds no Turbo training or separate Turbo LoRA feature.

Turbo sampling fixes mu=1.15. Missing prompt steps/CFG default to8/1; explicit prompt overrides remain respected. Raw fallback retains its former defaults. The supplied main prompts explicitly select `--s 8 --l 1`, preserving all ten subjects, dimensions and seeds. This follows [upstream Musubi guidance](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/krea2.md). Relative Turbo checkpoint paths follow the main experiment TOML directory; absolute paths remain unchanged. A missing checkpoint or invalid combination fails before model loading. `turbo_dit_cache` requires `turbo_dit`.

The local implementation's memory contract includes rollback safety: streaming mode loads Turbo per sampling pass and temporarily snapshots Raw on CPU (about26.3GB for the BF16 base), with additional transient loader/conversion payload. Cached mode keeps both Raw and Turbo CPU states (about52.6GB combined), plus transient loader/conversion overhead; it is not a single-base CPU budget. The supplied Pod has a cgroup RAM limit about233.76GiB, rather than its much larger host RAM figure. The main config chooses caching for this machine after that capacity inspection. Native BF16 GPU acceptance passed on the supplied Pod; these CPU storage figures are not a peak-memory guarantee.

Gradient checkpointing is compatible. Full-base Turbo swapping rejects block swap and ConvRot INT8 before model loading. FP8 behavior must be reported at the actually tested level; CPU helper tests do not establish real FP8 GPU compatibility. Full model integrity and native actual main() BF16 sampling→Raw validation/next-train acceptance passed for both cached and streaming modes. Existing dataset/caches and Raw val noise parameters are not changed by these options.


Native production verification compared two optimizer updates each for resident Turbo, streaming Turbo and Raw control: training losses, Raw val metrics, both saved FP32 adapters and native state matched exactly; Turbo resident/streaming images matched pixel-for-pixel. An earlier heavily instrumented observer diagnostic failed on the next gradient-checkpoint recomputation; its exact cause remains unresolved and that diagnostic is not reported as passed. Successful native runs used the unmodified production entrypoint without observers. Detailed failed/successful evidence is retained in verification.md and the compact local archive. Actual GPU FP8, compiled execution and multi-GPU remain unverified.

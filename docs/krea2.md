# Krea2 Raw LoRA

The three supported commands cache image latents, cache Qwen3-VL representations, and train a Krea2 Raw LoRA. Follow the [root quickstart](../README.md) for complete commands and the three starter files.

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

The training hooks generate Raw samples, log progress/optional tracker metrics, save adapter safetensors and save Accelerate state. Sample prompts, save intervals and retention are configurable. Resume with `--resume STATE_DIRECTORY`; adapter-only initialization uses `--network_weights`. See [sampling](sampling_during_training.md) and [advanced settings](advanced_config.md).

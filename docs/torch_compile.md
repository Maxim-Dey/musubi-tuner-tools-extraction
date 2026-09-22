# Compile Qwen training blocks

`compile` uses the existing Qwen transformer compilation hook. It does not add a new training loop. Available settings are `compile_backend`, `compile_mode`, `compile_dynamic`, `compile_fullgraph` and `compile_cache_size_limit`. See the actual training command's `--help` for defaults.

```toml
compile = true
compile_backend = "inductor"
compile_mode = "default"
compile_dynamic = "true"
compile_fullgraph = false
```

Inductor requires a compatible Triton/PyTorch/platform setup. The first invocation can compile slowly; multiple resolutions/batch shapes can cause recompilation. No speed or memory improvement is guaranteed. With block swap, the retained helper excludes the relevant linear layers from compilation.

The separate Accelerate Dynamo settings remain `dynamo_backend`, `dynamo_mode`, `dynamo_fullgraph`, `dynamo_dynamic`; these configure the existing Accelerator plugin. Additional CUDA settings are `cuda_allow_tf32` and `cuda_cudnn_benchmark`. Select compatible precision/backends for the actual device. The local extraction checks exercise parser/import and small CPU numerical boundaries, not compilation on a training GPU.

See [Qwen workflow](qwen_image.md), [block swap](block_swap.md) and [advanced settings](advanced_config.md).

## 日本語

既存のQwenブロックに`torch.compile`を適用します。Inductorには互換性のあるTriton環境が必要です。初回のコンパイル時間と形状変更時の再コンパイルに注意し、実機で性能と互換性を確認してください。

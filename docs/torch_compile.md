# torch.compile for Dev LoRA

Compilation is optional and is not enabled in the supplied template. It needs an operational PyTorch/compiler/backend environment; local CPU contract checks do not compile a model or establish performance. First-use compilation and shape changes can be expensive.

The retained training options are `--compile`, `--compile_backend` (default `inductor`), `--compile_mode` (default `default`; also `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`), `--compile_dynamic` (`true`, `false`, `auto`, with unset equivalent to auto), `--compile_fullgraph`, and `--compile_cache_size_limit`. The existing Dev double/single block compilation path is retained. Block swapping disables compilation of linear layers through the existing wrapper; test the chosen combination only in the separate operational stage.

CUDA compiler prerequisites vary by platform; retain the compatible toolchain for the installed PyTorch. Windows dynamic compilation can require a C++ build toolchain. `--cuda_allow_tf32` and `--cuda_cudnn_benchmark` are separate opt-in precision/performance controls; the cleanup does not enable them. Accelerate's `dynamo_backend`/mode options also remain available.

There is no standalone generation or multi-architecture compile workflow. See [Dev training](flux_2.md) and [block swap](block_swap.md).

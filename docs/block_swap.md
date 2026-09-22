# Qwen block swap

`blocks_to_swap=N` offloads N transformer blocks to CPU and streams them for computation, trading VRAM for host memory and transfer time. The existing idea is based on the implementation by 2kpr. For original Qwen, `0`/omission disables swap; the maximum is `num_layers-1` (59 with the default 60-layer model).

```toml
blocks_to_swap = 20
use_pinned_memory_for_block_swap = true
block_swap_h2d_only = true
block_swap_ring_size = 2
```

Pinned memory can improve transfer speed and consumes additional host/shared memory. `block_swap_h2d_only` is the experimental frozen-base/adapter path: keep CPU master weights and copy host→device using `block_swap_ring_size` positive buffers. Size 1 minimizes buffers; 2 allows transfer/compute overlap. This does not offload trainable adapters or change updates. The original sample boundary switches the offloader for inference and back to training. Compatibility/performance must be checked on the actual hardware.

Combine with `gradient_checkpointing`; `gradient_checkpointing_cpu_offload` requires checkpointing. With [compilation](torch_compile.md), the existing Qwen helper keeps swapped linear layers outside compilation. See [Qwen training](qwen_image.md).

## 日本語

`blocks_to_swap`はCPUへ移すブロック数です。既定の60層では最大59、0で無効です。メモリ節約と転送時間のトレードオフがあり、実機で確認してください。実装の着想は2kpr氏によるものです。

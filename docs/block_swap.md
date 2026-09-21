# Block swap for Dev LoRA

`--blocks_to_swap N` moves frozen transformer blocks between CPU RAM and GPU to reduce device memory use. Transfer costs and RAM requirements depend on the runtime; no throughput or hardware capacity is established by local CPU checks. The supplied `# blocks_to_swap = 20` remains commented.

`--use_pinned_memory_for_block_swap` uses pinned host memory. `--block_swap_h2d_only` keeps CPU master weights and transfers only host-to-device during frozen-base LoRA training. It requires gradient checkpointing. `--block_swap_ring_size` defaults to 2, must be >=1, and controls streamed GPU buffers (1 reduces overlap). The Dev implementation distributes requested swaps across double/single blocks and must retain at least two blocks in each stack. Do not use another model's numeric limits.

Training samples temporarily switch the existing offload mode and restore it afterward. No standalone inference command remains. See [workflow](flux_2.md) and [compile](torch_compile.md).

The block-swap idea is based on 2kpr's implementation; upstream attribution and common offloading code are retained. CPU helper tests cover selection/layout/hook order only, not CUDA streaming equivalence.

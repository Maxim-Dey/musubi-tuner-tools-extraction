# Krea2 block swap

`--blocks_to_swap N` moves selected main DiT blocks between CPU memory and the training device. The Raw model has 28 main blocks and permits up to 26 swapped blocks. Zero or an omitted value leaves block swap disabled. CPU memory and transfer requirements depend on the selected count and precision.

The classic path exchanges weights in both directions. `--use_pinned_memory_for_block_swap` uses pinned CPU memory for transfers; it increases page-locked RAM usage and may also consume shared GPU memory on Windows.

## H2D-only mode

`--block_swap_h2d_only` keeps frozen base weights in CPU master storage and streams them Host→Device, avoiding the return transfer. It requires positive `blocks_to_swap`, CUDA, frozen base weights and `--gradient_checkpointing` during training.

`--block_swap_ring_size` controls reusable device buffers. The default is 2 for overlapping transfer and computation; 1 uses one buffer without that overlap. Enable only supported ring sizes reported by the parser.

For example, append these settings to the README training command if appropriate for your environment:

```text
--blocks_to_swap 8 --gradient_checkpointing --block_swap_h2d_only --block_swap_ring_size 2
```

Block swap can be combined with the retained scaled FP8 or ConvRot base-weight paths. Compilation excludes affected Linear paths where the existing implementation requires it. See [torch.compile](torch_compile.md). No throughput or memory-fit guarantee is implied.

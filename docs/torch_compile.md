# Compilation options

Krea2 retains its model-block `torch.compile` path. It is disabled in the starter. Optional compilation needs a compatible PyTorch/compiler/backend environment; select settings for that environment without assuming a speedup.

| Training option | Meaning |
|---|---|
| `--compile` | Compile eligible model blocks |
| `--compile_backend` | PyTorch compile backend; default `inductor` |
| `--compile_mode` | `default`, `reduce-overhead`, `max-autotune` or `max-autotune-no-cudagraphs` |
| `--compile_dynamic` | `true`, `false` or `auto`; omitted uses automatic shape handling |
| `--compile_fullgraph` | Request full-graph capture |
| `--compile_cache_size_limit` | Override the Dynamo cache-size limit |

Append options to the README training command, for example `--compile --compile_mode default`. Bucketing can change tensor shapes; recompilation or graph breaks depend on those shapes and the backend. Full-graph capture and autotuning are optional and may be incompatible with a particular runtime.

Quantized or swapped Linear modules are excluded where required by the existing model/offload code. Block swap remains configurable; see [block swap](block_swap.md).

The existing Accelerate Dynamo controls (`--dynamo_backend`, `--dynamo_mode`, `--dynamo_fullgraph`, `--dynamo_dynamic`) are a separate configuration path. Use one compilation path rather than applying both. `--cuda_allow_tf32` and `--cuda_cudnn_benchmark` remain explicit runtime options; neither is enabled in the starter.

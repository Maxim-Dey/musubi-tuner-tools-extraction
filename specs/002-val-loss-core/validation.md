# Local Validation: Fixed Inputs and Deterministic Noise

Date: 2026-09-23

After convergence round 1, the Stage-1 CPU acceptance ran with Python 3.12.14 from the existing project-compatible virtual environment and `src` on `PYTHONPATH`:

```text
pytest tests/test_qwen_image_validation_inputs.py tests/test_qwen_image_dataset_cache.py tests/test_qwen_image_config.py tests/test_qwen_image_training_invariants.py -q -k 'not test_prompt_templates_and_all_readers'
268 passed, 1 deselected
```

The checks cover the two explicit roles, strict source/cache binding, portable input identity, later-read change detection, the midpoint grid, SHA-256 seeds, isolated RNG, effective configuration, and legacy training/cache behavior. Round 1 additionally checked missing roles, shared latent-cache collisions, nonfinite cache tensors, and equivalent path spellings. The one excluded existing test expects two sample prompts while the unchanged tracked `sample_prompts.txt` has ten; running the full selection produced `222 passed, 1 failed` before the final US3 additions. This mismatch does not exercise validation and is not counted as a pass.

Training, GPU execution, model weights, downloads, and server verification were not run. Stage 1 prepares validation inputs and noise only; loss calculation, TensorBoard, resume state persistence, and the result hierarchy remain for later stages.

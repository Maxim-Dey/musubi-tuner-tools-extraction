# Local Implementation Validation: Qwen-Image Original LoRA

Date: 2026-09-22

## Status

Approved correction round 1 is complete: T051–T057 are checked, with 207 retained tests passing. Ruff lint, changed-file formatting, AST and local link checks pass. Repository-wide `ruff format --check --no-cache .` remains nonzero on five unchanged protected `.specify/scripts/python/` files identified by converge; it is not reported as a pass. The round-1 evidence below supersedes the initial blanket completion/formatting statements. Specification, plan and constitution are unchanged.

## T001: Pre-change baseline

- Revision: `3157a911d75805f7bad8c2e5747f81655407b5bd`
- Branch: `001-scope-qwen-image-lora`
- Existing working-tree changes (before this report):

```text
M config_for_qwen_image_lora/train.toml
?? config_for_qwen_image_lora/speckit_prompts.md
?? specs/001-scope-qwen-image-lora/checklists/requirements.md
?? specs/001-scope-qwen-image-lora/contracts/cli-and-config.md
?? specs/001-scope-qwen-image-lora/data-model.md
?? specs/001-scope-qwen-image-lora/plan.md
?? specs/001-scope-qwen-image-lora/quickstart.md
?? specs/001-scope-qwen-image-lora/research.md
?? specs/001-scope-qwen-image-lora/spec.md
?? specs/001-scope-qwen-image-lora/tasks.md
```

### Protected input hashes (SHA-256)

| File | SHA-256 |
| --- | --- |
| `config_for_qwen_image_lora/train.toml` | `1f199a8e56375433db4cf72acc56798e4ac061ec6bb28dc948126651242338b1` |
| `config_for_qwen_image_lora/dataset.toml` | `832516bc8788930bcb23d195f768420612f920869b86e1bdaf88a8b0ba49fe15` |
| `config_for_qwen_image_lora/sample_prompts.txt` | `c32ddf8bb7f28309e5f0ae5236534a8c3a6d1fc584b9a0b4f7cff0b663440db0` |
| `.specify/memory/constitution.md` | `651a05349939b3156db15e7eb278368b20cf7f3e46dca5f5027d29c5061cedd8` |
| `specs/001-scope-qwen-image-lora/spec.md` | `ecc8d5711e36f91c57637e1256a545e55b9b500d15af3872373dc872149c8b6a` |
| `specs/001-scope-qwen-image-lora/plan.md` | `4c3920840e878eb514e32b55cf3a94e44ae4be1d5daeb0e1407e4f3ba9870a6f` |
| `config_for_qwen_image_lora/speckit_prompts.md` | `4599df0668b223524c77482e20113c431b9f3bc85fb9a5f81022426de542a77c` |

### Existing training-template diff

```diff
diff --git a/config_for_qwen_image_lora/train.toml b/config_for_qwen_image_lora/train.toml
index 03cdee8..fc081e4 100644
--- a/config_for_qwen_image_lora/train.toml
+++ b/config_for_qwen_image_lora/train.toml
@@ -38,7 +38,7 @@ seed = 42
 # === Оптимизатор и скорость обучения ===
 optimizer_type = "adamw8bit"
 learning_rate = 5e-5
-lr_scheduler = "constant"
+lr_scheduler = "constant_with_warmup"
 lr_warmup_steps = 200
 max_grad_norm = 1.0
```

### Protected scope

- `.specify/`, `.agents/`, `.cursor/` are protected feature inputs; hashes follow. The constitution will not be edited.
- `.git/` metadata/history is protected; no branch, index or history mutation.
- Existing user-authored `config_for_qwen_image_lora/speckit_prompts.md`, all datasets, weights, caches, outputs and unrelated user files are protected. No weight contents were read or copied.
- Standalone tracked license/notice files: none; embedded source/documentation notices and attribution remain protected.
- Dependency lock files from repository filesystem scan: none. Do not create one or resolve dependencies incidentally.

<details>
<summary>Protected infrastructure file hashes</summary>

| File | SHA-256 |
| --- | --- |
| `.agents/skills/speckit-analyze/SKILL.md` | `3c33ddda853d6cf68f65d3c557a8ee0df6bf245609afcccb3996c1b77bfa4f2a` |
| `.agents/skills/speckit-checklist/SKILL.md` | `190e421f62b9e5cfb63749463b7c077d71ccaf6b07bc2a0b69e4a2236bf4a99c` |
| `.agents/skills/speckit-clarify/SKILL.md` | `babd8bdab8d2778dda2aa64798c867ac7b401fc4eb47e9f34294726b847860d1` |
| `.agents/skills/speckit-constitution/SKILL.md` | `a85bc4e6a0e81acc38cee85e43c0eaa7e16de3368b39771c7ba3623e34c6131a` |
| `.agents/skills/speckit-converge/SKILL.md` | `aeed04d7960d811f88cb29f7050fecf766bdeb18d2536f32a4f6d5478a14219c` |
| `.agents/skills/speckit-implement/SKILL.md` | `4f268265d8610150d9d1f1d359d98cea686936d5b7243a0f61801d8a0d33a7fe` |
| `.agents/skills/speckit-plan/SKILL.md` | `d61f53c6dbcad378c352e4574d0956042dcdf2bdf2c6cd0b539f40ce2a83f057` |
| `.agents/skills/speckit-specify/SKILL.md` | `e6b880a4ed6af8d8a9de01f3f771f5f633b09f951f05e4ba5bd9ee9d94b6a3e7` |
| `.agents/skills/speckit-tasks/SKILL.md` | `d21acf4fd5fdaff4cb0bde1f3278a13523035153f5410a33e6a0127a5f5a15b3` |
| `.agents/skills/speckit-taskstoissues/SKILL.md` | `46bbb5bf14ddc2d5bef9340c67a457b1cb2ae00b9b657b09997cba476d522420` |
| `.cursor/skills/speckit-analyze/SKILL.md` | `6cf2a51c3c7d15c01652b95350233be2d6bb7719eca9860f79451eaf209a878d` |
| `.cursor/skills/speckit-checklist/SKILL.md` | `db45d7b0e6aaceeb4f0a7cf68e7cb3af01d2ccf8ec837e765d854cc3ad2b88a1` |
| `.cursor/skills/speckit-clarify/SKILL.md` | `3e53a6c39c0f209488c8918683ac1fab24c63e52a31c06bef1f998940357284a` |
| `.cursor/skills/speckit-constitution/SKILL.md` | `321403307a303fe8e46cb1152d3fbb15a983485dbf36ac82a6a769c1aff7c96a` |
| `.cursor/skills/speckit-converge/SKILL.md` | `a06a4b89cfa35a17340e15c8fb3a6a39837c491ea38be9af455a1df8b497cef8` |
| `.cursor/skills/speckit-implement/SKILL.md` | `9dba12832d7755258a983df92eb2ca0784f60f893632c8b67a9660d82d62412f` |
| `.cursor/skills/speckit-plan/SKILL.md` | `1b5bbb79f17b3f4d3822343e430bceb49207980a81c946299283bbe22f9a819c` |
| `.cursor/skills/speckit-specify/SKILL.md` | `6cfd398a6665809b7c838f4f705d39ccffedc836cb9218efdc03c2ddce26655f` |
| `.cursor/skills/speckit-tasks/SKILL.md` | `68f4a93aa65038919ceee88e17aa3b3790fa3216b9adbeea92246786e6822106` |
| `.cursor/skills/speckit-taskstoissues/SKILL.md` | `9085508151595021f7cf168cda98f3b53c9865d2baba755116294821a7f2edac` |
| `.specify/.gitignore` | `05c2054a5f5ca103ed62daa73191675013ee1d3d44eafb8e49891898b5368878` |
| `.specify/feature.json` | `beef69b2996a730bcbf53f765c1359cd5dcee45902b5d6f20111ab73a0533b48` |
| `.specify/init-options.json` | `b384c08477d3cdf8f55a754e58c2688a86f28d8c5c76f0f7b0676dcdc250251e` |
| `.specify/integration.json` | `42270051a204c8ffc26ec50a5c22e827744770b70cac4ce98370385eb8413e54` |
| `.specify/integrations/codex.manifest.json` | `9739994efbf06310ce812c338a04bba1cb12e95e2eaeee3db4fb6b947fdae4b0` |
| `.specify/integrations/cursor-agent.manifest.json` | `b42c2c20b4b135c7621dce5f79497f82bcd42ead5f21dc327d37a31137d7a21f` |
| `.specify/integrations/speckit.manifest.json` | `d544fae75729b68db4d642211301b52cae5bc54bca102c6f2b4c3e7787d8c469` |
| `.specify/memory/.constitution-template.json` | `36e9c61ac9797211aba27a5dc6dcc20d11419406448c387f7fb2e7a407610d60` |
| `.specify/memory/constitution.md` | `651a05349939b3156db15e7eb278368b20cf7f3e46dca5f5027d29c5061cedd8` |
| `.specify/scripts/powershell/check-prerequisites.ps1` | `e71c1273020aa9888cf15cd7e895c3c6133041db590fe34324ac6b8b94b93b79` |
| `.specify/scripts/powershell/common.ps1` | `dbb07fc929287e9c7e03a53f4b95d2de3c88e75524b618136d7413aac40a83bd` |
| `.specify/scripts/powershell/create-new-feature.ps1` | `79dfefe26fe532f2b703dddb543cac4f9e5d69132c4eaa4bb032d5ffd82a1024` |
| `.specify/scripts/powershell/resolve-template.ps1` | `99a4589963895d07146ccc195873832c67141ee7d9dd4cb2c1e47bf39f933778` |
| `.specify/scripts/powershell/setup-plan.ps1` | `dda307ce74dcdf7a72d438746e67df488a40f202cb9ec670031c1527908e9e12` |
| `.specify/scripts/powershell/setup-tasks.ps1` | `7b0c10d7b0fa1856689114724141375fcc2ca8b7accaae32bb46d1e64b707335` |
| `.specify/scripts/python/__pycache__/common.cpython-313.pyc` | `b63f9de17a5e11cd2d8cc4c9b3537930174952d2211abfa6df56bc6b929e3204` |
| `.specify/scripts/python/check_prerequisites.py` | `ab61da7a850fac67a34612f4f0cd7eeee196f753bd9c41c753fbf0392f2284cb` |
| `.specify/scripts/python/common.py` | `5ed4b02bafc513d68034c31c525749f072b3da62195ad256b0e7a03c876d4f98` |
| `.specify/scripts/python/create_new_feature.py` | `31788bdae6869d8e468bffbf8bfbc19b7b5f30964c8c418e03f66f40d361a27f` |
| `.specify/scripts/python/resolve_template.py` | `c5c81fdec418bcfdb1da0055fdd32a2713b1784b5e50c1db4e6837d2fffdc9ed` |
| `.specify/scripts/python/setup_plan.py` | `06e3bec3f1d715973712fa3cbd959ba005d8029af4b1c66b3725c71160a4ffec` |
| `.specify/scripts/python/setup_tasks.py` | `8826ab45ddf5337219ab1d8b78ffb4c3f0bd7c82b6eaca446c104b75481f11dd` |
| `.specify/templates/checklist-template.md` | `0915ed5d49a922cbf5a23415b13a56adc47145eb489486d3877d48438d043dca` |
| `.specify/templates/constitution-template.md` | `29fc1837b20f493d302203554923322a245123b4be9ed5734d9e1af96cd5ef0b` |
| `.specify/templates/plan-template.md` | `eddcad4c5b6fcefb81a29b0495d060a87764ed89cc3d393f04aabcc40ad3e688` |
| `.specify/templates/spec-template.md` | `cfc2f4605b4366d798d184956ca848a8301ad701d548e0c2c1a33ad7aeda7289` |
| `.specify/templates/tasks-template.md` | `85d2141d85f50f124874a9579b4410f4e6c0049e36a478e972b451e43bec8fca` |
| `.specify/workflows/speckit/workflow.yml` | `ea636ff77e01aafdd5d44097e16d47ddb62264ea66b09b63f560aa4d543af082` |
| `.specify/workflows/workflow-registry.json` | `d6b68e227fea1fc6681636ab3e3598afaf013d62451eedac9dce4d509e8158db` |

</details>

## Pre-execution gates

- `python .specify/scripts/python/check_prerequisites.py --json --require-tasks --include-tasks`: passed; selected this feature directory.
- `checklists/requirements.md`: 16 checked, 0 unchecked; checklist unchanged.
- `.specify/extensions.yml`: absent; no pre-implementation hooks.
- No separate saved analysis report found in this feature. `spec.md` records incorporated analysis findings; this is not runtime verification.

## T002: Existing CPU environment and runtime baseline

Selected interpreter: `C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe`, Python 3.12.14, torch 2.7.1+cpu, torchvision 0.22.1+cpu, Accelerate 1.6.0, Diffusers 0.32.1, Transformers 4.57.6, safetensors 0.4.5, pytest 9.1.1. This existing environment supplies all checked runtime/test prerequisites (including TOML, voluptuous, NumPy, Pillow, OpenCV, einops, Hub, av, bitsandbytes, TensorBoard); no dependency installation or environment modification was needed. All commands use this checkout's absolute `src` as PYTHONPATH, not the neighboring project's sources.

Environment for checks: `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `WANDB_MODE=disabled`, `PYTHONIOENCODING=utf-8`.

| Actual command/check | Result |
| --- | --- |
| Real imports of `musubi_tuner.qwen_image_train_network`, `musubi_tuner.qwen_image_cache_latents`, `musubi_tuner.qwen_image_cache_text_encoder_outputs` in separate processes | All exit 0 |
| `python -B qwen_image_cache_latents.py --help` | Exit 0 |
| `python -B qwen_image_cache_text_encoder_outputs.py --help` | Exit 0 |
| `python -B qwen_image_train_network.py --help` | First redirected run failed with cp1252 UnicodeEncodeError while printing Japanese help; repeated with UTF-8 output: exit 0. No code change |
| `python -B -m pytest -p no:cacheprovider -q tests/test_save_precision.py tests/test_lora_dtype_bridging.py tests/test_grad_metrics.py tests/test_krea2_timesteps.py tests/test_ideogram4_timesteps.py tests/test_datasource_item_extras.py tests/test_sai_model_spec.py` | 38 passed, 1 dependency SyntaxWarning, 14.71 seconds |

Optional SageAttention is absent and prints an import diagnostic; it does not prevent imports. GPU/optional accelerator backends are not validated by the CPU baseline. Bare uv Python 3.10/3.12 installations lack ML dependencies; the existing virtual environment above resolves that prerequisite. Baseline revision and branch still match T001. No dependency lock exists in the repository. Existing Python ignore rules were supplemented with bytecode/build/distribution/egg-info exclusions.


## T003: Real-reader pre-change characterization

All 40 training TOML values equal the effective parser values (before model-specific derived defaults). Source consumers are mapped in research.md section 3.

| Key | Effective value | Python type |
| --- | --- | --- |
| `model_version` | `'original'` | `str` |
| `dit` | `'D:/AI/models/qwen_image_bf16.safetensors'` | `str` |
| `vae` | `'D:/AI/models/qwen_image_vae.safetensors'` | `str` |
| `text_encoder` | `'D:/AI/models/qwen_2.5_vl_7b.safetensors'` | `str` |
| `dataset_config` | `'qwen_image_lora/dataset.toml'` | `str` |
| `max_data_loader_n_workers` | `0` | `int` |
| `persistent_data_loader_workers` | `False` | `bool` |
| `network_module` | `'networks.lora_qwen_image'` | `str` |
| `network_dim` | `16` | `int` |
| `network_alpha` | `16` | `int` |
| `mixed_precision` | `'bf16'` | `str` |
| `fp8_base` | `False` | `bool` |
| `fp8_scaled` | `False` | `bool` |
| `fp8_vl` | `False` | `bool` |
| `blocks_to_swap` | `0` | `int` |
| `sdpa` | `True` | `bool` |
| `gradient_checkpointing` | `True` | `bool` |
| `max_train_steps` | `1600` | `int` |
| `gradient_accumulation_steps` | `1` | `int` |
| `seed` | `42` | `int` |
| `optimizer_type` | `'adamw8bit'` | `str` |
| `learning_rate` | `5e-05` | `float` |
| `lr_scheduler` | `'constant_with_warmup'` | `str` |
| `lr_warmup_steps` | `200` | `int` |
| `max_grad_norm` | `1.0` | `float` |
| `timestep_sampling` | `'shift'` | `str` |
| `discrete_flow_shift` | `2.2` | `float` |
| `weighting_scheme` | `'none'` | `str` |
| `output_dir` | `'qwen_image_lora/output'` | `str` |
| `output_name` | `'qwen_image_lora'` | `str` |
| `save_precision` | `'bf16'` | `str` |
| `save_every_n_steps` | `200` | `int` |
| `save_last_n_steps` | `1000` | `int` |
| `save_state` | `True` | `bool` |
| `save_last_n_steps_state` | `1000` | `int` |
| `logging_dir` | `'qwen_image_lora/logs'` | `str` |
| `log_with` | `'tensorboard'` | `str` |
| `sample_prompts` | `'qwen_image_lora/sample_prompts.txt'` | `str` |
| `sample_every_n_steps` | `200` | `int` |
| `sample_at_first` | `False` | `bool` |

Explicit CLI rank=8, learning_rate=0.002 and fp8_base=true override TOML; omitted max_train_steps retains 1600.
Defaults: network_dim=None, network_alpha=1, lr_scheduler='constant', model_version=None.

Dataset template read by the actual TOML/schema/blueprint chain:
```json
{
  "dataset_group": {
    "datasets": [
      {
        "is_image_dataset": true,
        "params": {
          "resolution": [
            1024,
            1024
          ],
          "enable_bucket": true,
          "bucket_no_upscale": true,
          "caption_extension": ".txt",
          "batch_size": 1,
          "num_repeats": 1,
          "cache_directory": "qwen_image_lora/cache/train",
          "debug_dataset": false,
          "architecture": "qi",
          "image_directory": "D:/AI/datasets/qwen_image/train",
          "image_jsonl_file": null,
          "control_directory": null,
          "multiple_target": false,
          "fp_latent_window_size": 9,
          "fp_1f_clean_indices": null,
          "fp_1f_target_index": null,
          "fp_1f_no_post": false,
          "no_resize_control": false,
          "control_resolution": null
        }
      }
    ]
  }
}
```
Dataset fallback: dataset batch_size=3 beats general=2; general repeats=4 beats CLI=9; runtime architecture=qi fills omission; dataclass resolution is (960,544).

Both real prompt records:
```json
[
  {
    "prompt": "TOK, a portrait of a woman in soft daylight, detailed background",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "sample_steps": 30,
    "cfg_scale": 4.0,
    "discrete_flow_shift": 2.2,
    "enum": 0
  },
  {
    "prompt": "TOK, a small house beside a lake, mountains in the background",
    "width": 1024,
    "height": 1024,
    "seed": 43,
    "sample_steps": 30,
    "cfg_scale": 4.0,
    "discrete_flow_shift": 2.2,
    "enum": 1
  }
]
```
Baseline gap `unknown`: {'unknown_setting': 123}.
Baseline gap `mixed`: {'edit': True, 'edit_plus': True, 'model_version': 'original'}.
Baseline gap `wrong_type`: {'max_train_steps': '1600'}.
Baseline malformed prompt: `{'prompt': 'example', 'width': 512, 'sample_steps': 1}`; unknown option ignored, trailing width junk accepted, zero steps clamped.
Existing constant/200 scheduler construction rejects: `SchedulerType.CONSTANT does not require `num_warmup_steps`. Set None or 0.`.

Actual cache parsers are distinct from training TOML:
- latent: `{'dataset_config': 'dataset.toml', 'vae': None, 'vae_dtype': None, 'device': None, 'batch_size': None, 'num_workers': None, 'skip_existing': False, 'keep_cache': False, 'debug_mode': None, 'console_width': 80, 'console_back': None, 'console_num_images': None, 'disable_cudnn_backend': False, 'vae_tiling': False, 'vae_chunk_size': None, 'vae_spatial_tile_sample_min_size': None, 'edit': False, 'edit_plus': False, 'model_version': None}`.
- text: `{'dataset_config': 'dataset.toml', 'device': None, 'batch_size': None, 'num_workers': None, 'skip_existing': False, 'keep_cache': False, 'text_encoder': 'example.safetensors', 'fp8_vl': False, 'edit': False, 'edit_plus': False, 'model_version': None}`.

Both shipped input references currently do not resolve: `qwen_image_lora/dataset.toml`, `qwen_image_lora/sample_prompts.txt`. No template was edited.

Actual `_load_dit_and_swap` with a sentinel only at `load_transformer`: SDPA+xformers -> torch; FlashAttention+xformers -> flash; SDPA+Flash3 -> torch. No weights loaded.

Actual tiny Qwen block factories for musubi_tuner.networks.lora_qwen_image, musubi_tuner.networks.loha, musubi_tuner.networks.lokr: unknown `rank_dropuot=0.1` is silently ignored (rank_dropout=None); valid `rank_dropout=0.1` is consumed. This is a baseline validation gap, not target acceptance.


## T004: Exact reviewed file inventory

This inventory enumerates existing tracked files, not future filename masks. Deletions are pending the stated consumer migration and T035 runtime gate; no user data is included. Import edges include AST imports and literal dynamic module names. Inheritance and runtime network resolution are separately reviewed below.

| Path | Decision | Reason / prerequisite |
| --- | --- | --- |
| `.agents/skills/speckit-analyze/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.agents/skills/speckit-checklist/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.agents/skills/speckit-clarify/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.agents/skills/speckit-constitution/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.agents/skills/speckit-converge/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.agents/skills/speckit-implement/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.agents/skills/speckit-plan/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.agents/skills/speckit-specify/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.agents/skills/speckit-tasks/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.agents/skills/speckit-taskstoissues/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-analyze/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-checklist/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-clarify/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-constitution/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-converge/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-implement/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-plan/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-specify/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-tasks/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.cursor/skills/speckit-taskstoissues/SKILL.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.github/FUNDING.yml` | keep | Protected generic project asset/infrastructure |
| `.github/workflows/ruff-lint.yml` | keep | Protected generic project asset/infrastructure |
| `.gitignore` | adapt | Repository documentation/dependencies |
| `.python-version` | keep | Protected generic project asset/infrastructure |
| `.specify/.gitignore` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/init-options.json` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/integration.json` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/integrations/codex.manifest.json` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/integrations/cursor-agent.manifest.json` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/integrations/speckit.manifest.json` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/memory/.constitution-template.json` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/memory/constitution.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/powershell/check-prerequisites.ps1` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/powershell/common.ps1` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/powershell/create-new-feature.ps1` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/powershell/resolve-template.ps1` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/powershell/setup-plan.ps1` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/powershell/setup-tasks.ps1` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/python/check_prerequisites.py` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/python/common.py` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/python/create_new_feature.py` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/python/resolve_template.py` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/python/setup_plan.py` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/scripts/python/setup_tasks.py` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/templates/checklist-template.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/templates/constitution-template.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/templates/plan-template.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/templates/spec-template.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/templates/tasks-template.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/workflows/speckit/workflow.yml` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `.specify/workflows/workflow-registry.json` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `CONTRIBUTING.ja.md` | adapt | Repository documentation/dependencies |
| `CONTRIBUTING.md` | adapt | Repository documentation/dependencies |
| `README.ja.md` | adapt | Repository documentation/dependencies |
| `README.md` | adapt | Repository documentation/dependencies |
| `README.ru.md` | adapt | Repository documentation/dependencies |
| `cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `caption_images_by_qwen_vl.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `config_for_qwen_image_lora/dataset.toml` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `config_for_qwen_image_lora/sample_prompts.txt` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `config_for_qwen_image_lora/speckit_prompts.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `config_for_qwen_image_lora/train.toml` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `convert_lora.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `docs/advanced_config.md` | adapt | Shared original workflow guide/distribution figure |
| `docs/betas_for_sigma_rel.png` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/block_swap.md` | adapt | Shared original workflow guide/distribution figure |
| `docs/dataset_config.md` | adapt | Shared original workflow guide/distribution figure |
| `docs/flux_2.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/flux_kontext.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/framepack.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/framepack_1f.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/hidream_o1.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/hunyuan_video.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/hunyuan_video_1_5.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/ideogram4.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/kandinsky5.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/kisekaeichi_ref.png` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/kisekaeichi_ref_mask.png` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/kisekaeichi_result.png` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/kisekaeichi_start.png` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/kisekaeichi_start_mask.png` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/krea2.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/logsnr_distribution.png` | keep | Shared original workflow guide/distribution figure |
| `docs/loha_lokr.md` | adapt | Shared original workflow guide/distribution figure |
| `docs/minimax_h3.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/minimax_h3_1f.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/minimax_h3_advanced.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/qinglong_distribution.png` | keep | Shared original workflow guide/distribution figure |
| `docs/qwen_image.md` | adapt | Shared original workflow guide/distribution figure |
| `docs/sampling_during_training.md` | adapt | Shared original workflow guide/distribution figure |
| `docs/shift_3.png` | keep | Shared original workflow guide/distribution figure |
| `docs/shift_3_500_1000.png` | keep | Shared original workflow guide/distribution figure |
| `docs/shift_3_500_1000_preserve.png` | keep | Shared original workflow guide/distribution figure |
| `docs/tools.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/torch_compile.md` | adapt | Shared original workflow guide/distribution figure |
| `docs/wan.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/wan_1f.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `docs/zimage.md` | delete | Exclusive guide/Edit/tool asset; T043 after retained guide migration |
| `flux_2_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `flux_2_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `flux_2_generate_image.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `flux_2_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `flux_kontext_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `flux_kontext_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `flux_kontext_generate_image.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `flux_kontext_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `fpack_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `fpack_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `fpack_generate_video.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `fpack_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hidream_o1_cache_pixel.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hidream_o1_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hidream_o1_generate_image.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hidream_o1_train.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hidream_o1_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hv_1_5_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hv_1_5_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hv_1_5_generate_video.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hv_1_5_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hv_generate_video.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hv_train.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `hv_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `ideogram4_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `ideogram4_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `ideogram4_generate_image.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `ideogram4_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `images/logo_aihub.png` | keep | Protected generic project asset/infrastructure |
| `kandinsky5_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `kandinsky5_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `kandinsky5_generate_video.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `kandinsky5_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `krea2_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `krea2_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `krea2_generate_image.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `krea2_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `lora_post_hoc_ema.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `merge_lora.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `minimax_h3_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `minimax_h3_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `minimax_h3_generate_video.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `minimax_h3_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `pyproject.toml` | adapt | Repository documentation/dependencies |
| `qwen_extract_lora.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `qwen_image_cache_latents.py` | keep | Public Qwen wrapper |
| `qwen_image_cache_text_encoder_outputs.py` | keep | Public Qwen wrapper |
| `qwen_image_generate_image.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `qwen_image_train.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `qwen_image_train_network.py` | keep | Public Qwen wrapper |
| `specs/001-scope-qwen-image-lora/checklists/requirements.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `specs/001-scope-qwen-image-lora/contracts/cli-and-config.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `specs/001-scope-qwen-image-lora/data-model.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `specs/001-scope-qwen-image-lora/plan.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `specs/001-scope-qwen-image-lora/quickstart.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `specs/001-scope-qwen-image-lora/research.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `specs/001-scope-qwen-image-lora/spec.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `specs/001-scope-qwen-image-lora/tasks.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `specs/001-scope-qwen-image-lora/validation.md` | protect | Inputs/infrastructure; only approved two template references and implementation evidence/tasks may change |
| `src/musubi_tuner/__init__.py` | keep | Retained Qwen entry/shared orchestration |
| `src/musubi_tuner/cache_latents.py` | adapt | Retained Qwen entry/shared orchestration |
| `src/musubi_tuner/cache_text_encoder_outputs.py` | adapt | Retained Qwen entry/shared orchestration |
| `src/musubi_tuner/caption_images_by_qwen_vl.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/convert_lora.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/dataset/__init__.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/dataset/architectures.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/dataset/audio_utils.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/dataset/bucket.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/dataset/cache_io.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/dataset/config_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/dataset/dataset_config.md` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/dataset/datasources.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/dataset/image_video_dataset.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/dataset/media_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/flux/flux_models.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/flux/flux_utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/flux_2/flux2_models.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/flux_2/flux2_utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/flux_2_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/flux_2_generate_image.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/flux_2_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/flux_2_train_network_self_flow.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/flux_kontext_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/flux_kontext_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/flux_kontext_generate_image.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/flux_kontext_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/fpack_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/fpack_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/fpack_generate_video.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/fpack_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/frame_pack/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/bucket_tools.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/clip_vision.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/framepack_utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/hunyuan.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/hunyuan_video_packed.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/hunyuan_video_packed_inference.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/k_diffusion_hunyuan.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/uni_pc_fm.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/frame_pack/wrapper.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/gui/config_manager.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/gui/gui.ja.md` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/gui/gui.md` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/gui/gui.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/gui/gui_implementation_plan.md` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/gui/i18n_data.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hidream_o1/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hidream_o1/flash_scheduler.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hidream_o1/hidream_o1_utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hidream_o1/pipeline.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hidream_o1/qwen3_vl_transformers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hidream_o1/utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hidream_o1_cache_pixel.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hidream_o1_generate_image.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hidream_o1_train.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hidream_o1_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hunyuan_model/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/activation_layers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/attention.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/autoencoder_kl_causal_3d.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/embed_layers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/fp8_optimization.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/helpers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/mlp_layers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/models.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/modulate_layers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/norm_layers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/pipeline_hunyuan_video.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/posemb_layers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/text_encoder.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/token_refiner.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_model/vae.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_models.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_modules.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_text_encoder.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_vae.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/hv_1_5_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hv_1_5_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hv_1_5_generate_video.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hv_1_5_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hv_generate_video.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hv_train.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/hv_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/ideogram4/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/caption_verifier.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/constants.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/ideogram4_autoencoder.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/ideogram4_model.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/ideogram4_quantized_loading.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/ideogram4_scheduler.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/ideogram4_utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/latent_norm.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/sampler_configs.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4/sampling_policy.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/ideogram4_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/ideogram4_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/ideogram4_generate_image.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/ideogram4_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/kandinsky5/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5/configs.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5/generation_utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5/models/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5/models/attention.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5/models/dit.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5/models/nn.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5/models/text_embedders.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5/models/utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5/models/vae.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/kandinsky5_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/kandinsky5_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/kandinsky5_generate_video.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/kandinsky5_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/krea2/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/krea2/krea2_encoder.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/krea2/krea2_mmdit.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/krea2/krea2_sampling.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/krea2/krea2_utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/krea2_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/krea2_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/krea2_generate_image.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/krea2_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/lora_post_hoc_ema.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/merge_lora.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/minimax_h3/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/args.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/audio_vae.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/cache_plan.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/checkpoint.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/generation_inputs.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/media.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/model.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/packing.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/sampling.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/text_encoder.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3/video_vae.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/minimax_h3_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/minimax_h3_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/minimax_h3_generate_video.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/minimax_h3_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/modules/__init__.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/modules/adafactor_fused.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/modules/attention.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/modules/comfy_quant_utils.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/modules/convrot_int8_kernels.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/modules/convrot_int8_utils.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/modules/custom_offloading_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/modules/fp8_optimization_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/modules/lr_schedulers.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/modules/nvfp4_utils.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/modules/scheduling_flow_match_discrete.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/modules/unet_causal_3d_blocks.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/__init__.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/networks/convert_hunyuan_video_1_5_lora_to_comfy.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/loha.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/networks/lokr.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/networks/lora.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/networks/lora_flux.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_flux_2.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_framepack.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_hidream_o1.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_hv_1_5.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_ideogram4.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_kandinsky.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_krea2.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_minimax_h3.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_qwen_image.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/networks/lora_wan.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/lora_zimage.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/networks/network_arch.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/qwen_extract_lora.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/qwen_image/__init__.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/qwen_image/qwen_image_autoencoder_kl.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/qwen_image/qwen_image_model.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/qwen_image/qwen_image_modules.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/qwen_image/qwen_image_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/qwen_image_cache_latents.py` | adapt | Retained Qwen entry/shared orchestration |
| `src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py` | adapt | Retained Qwen entry/shared orchestration |
| `src/musubi_tuner/qwen_image_generate_image.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/qwen_image_train.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/qwen_image_train_network.py` | adapt | Retained Qwen entry/shared orchestration |
| `src/musubi_tuner/training/__init__.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/training/accelerator_setup.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/training/audio_loss.py` | delete | Exclusive model/adapter/audio/quantization helper; prune retained imports first, T021-T024/T039 |
| `src/musubi_tuner/training/parser_common.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/training/sampling_prompts.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/training/timesteps.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/training/trainer_base.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/utils/__init__.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/utils/device_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/utils/huggingface_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/utils/image_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/utils/lora_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/utils/model_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/utils/safetensors_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/utils/sai_model_spec.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/utils/train_utils.py` | adapt | Retained shared/original implementation; consumer-first pruning, T012-T024/T039 |
| `src/musubi_tuner/wan/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/configs/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/configs/shared_config.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/configs/wan_i2v_14B.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/configs/wan_i2v_A14B.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/configs/wan_t2v_14B.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/configs/wan_t2v_1_3B.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/configs/wan_t2v_A14B.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/modules/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/modules/attention.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/modules/clip.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/modules/model.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/modules/t5.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/modules/tokenizers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/modules/vae.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/modules/xlm_roberta.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/utils/__init__.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/utils/fm_solvers.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/utils/fm_solvers_unipc.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan/utils/utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/wan_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/wan_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/wan_generate_video.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/wan_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/zimage/zimage_autoencoder.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/zimage/zimage_config.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/zimage/zimage_model.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/zimage/zimage_utils.py` | delete | Excluded implementation; detach retained consumers before owner deletion (T006-T024, T035) |
| `src/musubi_tuner/zimage_cache_latents.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/zimage_cache_text_encoder_outputs.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/zimage_generate_image.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/zimage_train.py` | delete | Excluded command implementation; T035-T038 |
| `src/musubi_tuner/zimage_train_network.py` | delete | Excluded command implementation; T035-T038 |
| `tests/test_audio_dataset_seam.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_convrot_int8_prequantized.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_datasource_item_extras.py` | adapt | Shared regression protection |
| `tests/test_grad_metrics.py` | adapt | Shared regression protection |
| `tests/test_ideogram4_autoencoder.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_ideogram4_fp8_loading.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_ideogram4_lora_sampling.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_ideogram4_synthetic.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_ideogram4_te_fp8_loading.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_ideogram4_timesteps.py` | adapt | Shared regression protection |
| `tests/test_krea2_convrot_int8.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_krea2_gather_valid_text.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_krea2_gradient_checkpointing.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_krea2_timesteps.py` | adapt | Shared regression protection |
| `tests/test_krea2_turbo_lora.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_lora_dtype_bridging.py` | adapt | Shared regression protection |
| `tests/test_minimax_h3_attention_extent.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_cache_contract.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_cache_plan.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_convrot_int8.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_convrot_runtime.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_dataset.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_generation_modes.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_generation_request.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_layering.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_model.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_nvfp4.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_packing.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_provenance.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_sampling.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_te_streaming.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_temporal_stretch.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_text_encoder.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_training.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_minimax_h3_vae.py` | delete | Exclusive foreign-model/audio/quantization assertions; T040 after owner removal |
| `tests/test_sai_model_spec.py` | adapt | Shared regression protection |
| `tests/test_save_precision.py` | adapt | Shared regression protection |
| `tests/test_top_level_entrypoints.py` | adapt | Shared regression protection |
| `wan_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `wan_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `wan_generate_video.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `wan_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `zimage_cache_latents.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `zimage_cache_text_encoder_outputs.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `zimage_generate_image.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `zimage_train.py` | delete | Excluded wrapper; package consumer detachment and T035 required |
| `zimage_train_network.py` | delete | Excluded wrapper; package consumer detachment and T035 required |

### Retained callers of proposed deletions

| Proposed owner | Current retained callers to migrate |
| --- | --- |
| `src/musubi_tuner/dataset/audio_utils.py` | `src/musubi_tuner/dataset/config_utils.py`, `src/musubi_tuner/dataset/datasources.py`, `src/musubi_tuner/dataset/image_video_dataset.py` |
| `src/musubi_tuner/flux/flux_utils.py` | `src/musubi_tuner/qwen_image/qwen_image_utils.py` |
| `src/musubi_tuner/hunyuan_model/attention.py` | `src/musubi_tuner/qwen_image/qwen_image_model.py` |
| `src/musubi_tuner/hunyuan_model/autoencoder_kl_causal_3d.py` | `src/musubi_tuner/cache_latents.py` |
| `src/musubi_tuner/hunyuan_model/text_encoder.py` | `src/musubi_tuner/cache_text_encoder_outputs.py` |
| `src/musubi_tuner/hunyuan_model/vae.py` | `src/musubi_tuner/cache_latents.py` |
| `src/musubi_tuner/hv_generate_video.py` | `src/musubi_tuner/training/trainer_base.py` |
| `src/musubi_tuner/hv_train_network.py` | `src/musubi_tuner/qwen_image_train_network.py` |
| `src/musubi_tuner/ideogram4_cache_latents.py` | `tests/test_top_level_entrypoints.py` |
| `src/musubi_tuner/ideogram4_cache_text_encoder_outputs.py` | `tests/test_top_level_entrypoints.py` |
| `src/musubi_tuner/ideogram4_generate_image.py` | `tests/test_top_level_entrypoints.py` |
| `src/musubi_tuner/ideogram4_train_network.py` | `tests/test_top_level_entrypoints.py` |
| `src/musubi_tuner/minimax_h3/packing.py` | `src/musubi_tuner/dataset/cache_io.py` |
| `src/musubi_tuner/minimax_h3/text_encoder.py` | `src/musubi_tuner/dataset/cache_io.py` |
| `src/musubi_tuner/networks/lora_flux.py` | `src/musubi_tuner/networks/network_arch.py` |
| `src/musubi_tuner/networks/lora_framepack.py` | `src/musubi_tuner/networks/network_arch.py` |
| `src/musubi_tuner/networks/lora_hv_1_5.py` | `src/musubi_tuner/networks/network_arch.py` |
| `src/musubi_tuner/networks/lora_wan.py` | `src/musubi_tuner/networks/network_arch.py` |
| `src/musubi_tuner/networks/lora_zimage.py` | `src/musubi_tuner/networks/network_arch.py` |

### Consumer and migration review

- Qwen inherits the existing NetworkTrainer via hv_train_network re-exports. T009 changes only import ownership. Sampling is retained through Qwen do_inference; the Hunyuan generator owns its PNG saver until T008.
- Qwen model attention and text-loading FP8 predicate currently reach Hunyuan/Flux; T006/T007 detach exactly those helpers. Cache modules eagerly import Hunyuan encoders; T015/T016 prune them before deletion.
- Dynamic adapter selection in NetworkTrainer._build_network and base-weight merging calls the selected factories. Preserve short and qualified Qwen/LoHa/LoKr modules; network_arch must lose foreign dispatch, and generic lora must lose Hunyuan factories.
- During dataset pruning keep architecture symbols still imported by Qwen train/cache and sai_model_spec until their migration. Do not delete exports/signatures solely because foreign implementations are scheduled for removal. T039 finishes this transition.
- Shared test imports/assertions were classified by content: timestep tests cover retained formulas; dtype/save/gradient tests cover shared training behavior; datasource and metadata tests mix retained and excluded cases. Preserve retained assertions. Other listed tests exercise the excluded model packages, audio seam or exclusive quantization.
- Dependency consumers and exact versions follow research.md section 2: keep torch/torchvision (PNG grid), OpenCV (image resizing), bitsandbytes (existing optimizer), Transformers/Accelerate/Diffusers/Hub/safetensors/TOML/voluptuous/einops/Pillow/NumPy. av/ftfy/easydict/sentencepiece and GUI/HiDream/prompt-toolkit may go only after their consumers. Keep original CUDA extras and conditional backend/optimizer/logging/Hub support.
- Retained package __init__.py files were inspected; they are empty and introduce no additional initializer dependencies.
- Notice-bearing original Qwen model/VAE sources retain their copyright/license text and Wan provenance. Extracted helpers must retain their original project provenance. Root README contribution/license text is retained during documentation narrowing.
- The current revision became dd728d4bdb3ea2422c403c5157138db3b8179e5e during preflight through an external commit of specification/prompts. No source implementation changed between it and T001; this agent did not commit, switch branches or alter Git history.

## T005-T010: Shared-helper extraction

Before edits, 12 real CPU tests in test_qwen_image_training_invariants.py passed: SDPA outputs and gradients for list/separate QKV and padded/split layouts, all four FP8 dtypes, exact PNG pixels/layout, Qwen packing order/round trip, original forward timestep/padding/target and weighted loss. Only the model-forward boundary is replaced in the last fixture. CPU Accelerator, tensor math and image IO are real. Baseline command: python -B -m pytest -p no:cacheprovider -q tests/test_qwen_image_training_invariants.py.

Moved Qwen-used attention branches to qwen_image_modules, unchanged is_fp8 to model_utils, unchanged save_images_grid to image_utils. Redirected Qwen imports to existing training modules, and replaced the trainer's generator dependency with the image helper. Removed its video output branch. Source owners remain pending T035.

After extraction: the 12 helper tests plus test_save_precision, test_lora_dtype_bridging, test_grad_metrics, test_krea2_timesteps, test_ideogram4_timesteps: **44 passed**, exit 0, 25.60 seconds. Existing environment emitted Windows access-violation diagnostics during imports but continued and completed the suite; an additional baseline invocation importing torch before pytest completed 12/12 without that diagnostic. Do not represent those diagnostics as test failures or hide them.

Remaining incoming Hunyuan references from retained code are cache_latents VAE imports and cache_text_encoder_outputs text imports, assigned to T015/T016. Dataset architecture exports and excluded cache helpers remain transitional until their retained consumers migrate. No original source owner was deleted.

## T011-T017: Image data and cache preservation

Before edits, test_qwen_image_dataset_cache.py had 11 preservation cases passing and 14 intended rejection cases failing. Those failures covered excluded declaration/item fields, bool-as-integer and missing JSONL cache location; requirements were not weakened. Schema changes then passed 12 declaration/fallback cases.

Narrowed schema/blueprints, directory/JSONL image sources, media helpers, ImageDataset, Qwen buckets, cache writers, shared cache orchestration and both Qwen entry modules. Existing dataset/general/CLI/runtime/default fallback, caption association, OpenCV resize, repeat references, varlen batching, format_version 1.0.1 and qi names/metadata are retained. Both cache encoders were tested with substitution only at the image/text encoding boundary; actual normalization, mask trimming, safetensors writes/reads, chunking, existence-only skip and keep/default-cleanup ran in temporary directories.

Consumer-first adjustment: T013 revealed cache_io's eager MiniMax packing import indirectly required media_utils.load_video. Its exclusive writers/imports were pruned as the necessary T014 prerequisite before completing T013. The intermediate import error and a leftover MiniMax constant were fixed before the gate passed. Foreign architecture constants remain transitional for Qwen trainer/model metadata; round_down_frame_count remains until the trainer migration. Shared Hunyuan cache writer exports stayed until T015/T016 callers were detached, then were removed.

After source pruning: 28 passed (24.68s). After cache entry/orchestration pruning: 28 passed (20.39s), exit 0, one existing einops SyntaxWarning. Command uses the T002 environment: python -B -c 'import torch; import pytest; raise SystemExit(pytest.main(["-p","no:cacheprovider","-q","tests/test_qwen_image_dataset_cache.py","tests/test_datasource_item_extras.py"]))'. Baseline numerical training checks remain separately recorded. Missing text caches retain warning/skip; zero usable training data is covered at the later entry validation task.


## T018–T019: observation and serialization baseline

`tests/test_qwen_image_training_invariants.py`: **22 passed** on the real CPU/offline environment (17.66s). Includes original Qwen denoising with real scheduler and substituted model-forward/VAE-decode boundaries; CFG norm scaling, dimensions, timesteps, positive/negative embeddings; actual sample trigger, PNG, swap/mode and state preservation; actual adapter gradients, BF16 safetensors metadata/reload/frozen base; actual Accelerate save hooks and adapter/optimizer/scheduler/Python/NumPy/Torch RNG restoration; independent step/epoch state retention and final names. Unit optimizer updates only; no training loop. Initial new-test failures were fixture expectations (integer alpha buffer loaded from BF16 and a string supplied instead of torch.device); corrected fixture contracts, with no implementation change. No exceptional-exit RNG restoration guarantee is added: existing sampling restores normal-exit state only. Trainer epoch/global-step/data cursor are still initialized by the training loop; Accelerate state resume does not preserve exact progress.


## T020–T025: original model and trainer closure

Removed Edit/Layered RoPE, timestep/control embeddings, processor/caption generation, layered packing, RGBA VAE configuration and model dispatch; retained RGB VAE causal calculations and original scheduler/packing. Updated consumers first, then narrowed helper signatures. Removed foreign adapter detection/Hunyuan factories, shared sample control/video branches and unsupported guidance/offloading CLI. Preserved all applicable historical timestep methods, loss/optimizer/training-loop algorithms, dtype bridges, save hooks and optional Hub support. Removed exclusively called runtime `attach_lora_weights`, control preprocessing and the optional foreign quantizer route; reviewed remaining model/device/Hub/safetensors/FP8/offloading/scheduler helpers: no incoming excluded-owner dependency is required by the retained paths.

Eight suites (training invariants, save precision, LoRA dtype bridging, grad metrics, Krea/Ideogram timestep formulas, adapted metadata, dataset/cache): **86 passed**, 24.96s, CPU/offline. Real Qwen/LoHa/LoKr factory gradients and frozen base, sample embedding memoization/negative default, Qwen metadata and previous numerical/serialization boundaries pass. A new factory assertion initially tested flattened adapter names rather than actual dotted names: baseline code applies `_mod_` regex to dotted original names. Replaced the fixture assertion with the exact expected target set plus a synthetic matching module; no targeting algorithm or exclusion pattern changed. This preserves existing `img_mod.1`/`txt_mod.1` targeting rather than silently changing trained parameters. Short aliases receive their real import/factory check at T035. Existing dependency warning only; no external run.


## T026–T035: strict readers and deletion gate

New reader baseline: **2 passed, 39 failed**. Failures exposed 38 documented rejection gaps and the existing TOML prompt default/subset duplicate-key crash. Fixed subset merging according to the agreed override contract; added strict full-value parsing and source labels. After the first reader changes: **41 passed, 18 failed**, the latter documenting missing early consumer/cache checks. After boundary integration: **110 passed**. Extended positive non-template option coverage initially had six fixture errors (`log_grad_norm` rather than the actual `log_grad_metrics`); corrected spelling without adding an option. Combined US1/US2/US3 regression: **165 passed**, 17.82s. Added two concrete unknown optimizer/scheduler argument cases, checked against their real signatures.

T035 actual root/module help (six invocations), all **45** retained package imports and six short/qualified tiny adapter factories run in clean subprocesses with a meta-path guard for the exact excluded inventory (condensed to **92** owner names/prefixes). Gate plus configuration suite: **85 passed**, 117.13s. No fake dependency modules. Static incoming-edge review of all retained Python files found zero excluded imports before deletion.

Additional pre-deletion numerical check loaded the unchanged model source at `dd728d4` in a temporary namespace and compared two-layer CPU Qwen original models using identical real weights, padded text and SDPA: outputs and all parameter gradients were **bitwise equal**, both with and without gradient checkpointing. No training loop/model weights/downloads were used.

Positive coverage includes all 40 training-template values, CLI override and suffix/section behavior, all prompt formats/defaults, dataset precedence, all six adapter selectors, non-template memory/compile/metadata/Hub/optimizer/loss/timestep options, integer and ratio/percentage scheduler semantics with real LR progression, and ignored warmup on custom/schedule-free paths. Invalid matrix covers unknown/prohibited raw fields, effective TOML types/choices, network argument names/patterns, backend selection/requirements, persistent workers, rank/dropout/block/interval values, missing/invalid sources, empty training caches and sample fields. Actual main boundaries use loader/tracker/cache/training sentinels; invalid tested inputs reach none.

## T036–T041: reviewed source removal

Immediately rechecked retained static imports, then removed only the following exact inventoried Python files (root/package owners and exclusive tests). No directory recursion, prefix deletion, user data or infrastructure removal was used. Finalized image-only records/metadata and removed transitional foreign constants. Corrected the two approved training-template references; verification follows removal.

- `cache_latents.py`
- `cache_text_encoder_outputs.py`
- `caption_images_by_qwen_vl.py`
- `convert_lora.py`
- `flux_2_cache_latents.py`
- `flux_2_cache_text_encoder_outputs.py`
- `flux_2_generate_image.py`
- `flux_2_train_network.py`
- `flux_kontext_cache_latents.py`
- `flux_kontext_cache_text_encoder_outputs.py`
- `flux_kontext_generate_image.py`
- `flux_kontext_train_network.py`
- `fpack_cache_latents.py`
- `fpack_cache_text_encoder_outputs.py`
- `fpack_generate_video.py`
- `fpack_train_network.py`
- `hidream_o1_cache_pixel.py`
- `hidream_o1_cache_text_encoder_outputs.py`
- `hidream_o1_generate_image.py`
- `hidream_o1_train.py`
- `hidream_o1_train_network.py`
- `hv_1_5_cache_latents.py`
- `hv_1_5_cache_text_encoder_outputs.py`
- `hv_1_5_generate_video.py`
- `hv_1_5_train_network.py`
- `hv_generate_video.py`
- `hv_train.py`
- `hv_train_network.py`
- `ideogram4_cache_latents.py`
- `ideogram4_cache_text_encoder_outputs.py`
- `ideogram4_generate_image.py`
- `ideogram4_train_network.py`
- `kandinsky5_cache_latents.py`
- `kandinsky5_cache_text_encoder_outputs.py`
- `kandinsky5_generate_video.py`
- `kandinsky5_train_network.py`
- `krea2_cache_latents.py`
- `krea2_cache_text_encoder_outputs.py`
- `krea2_generate_image.py`
- `krea2_train_network.py`
- `lora_post_hoc_ema.py`
- `merge_lora.py`
- `minimax_h3_cache_latents.py`
- `minimax_h3_cache_text_encoder_outputs.py`
- `minimax_h3_generate_video.py`
- `minimax_h3_train_network.py`
- `qwen_extract_lora.py`
- `qwen_image_generate_image.py`
- `qwen_image_train.py`
- `src/musubi_tuner/caption_images_by_qwen_vl.py`
- `src/musubi_tuner/convert_lora.py`
- `src/musubi_tuner/dataset/audio_utils.py`
- `src/musubi_tuner/flux/flux_models.py`
- `src/musubi_tuner/flux/flux_utils.py`
- `src/musubi_tuner/flux_2/flux2_models.py`
- `src/musubi_tuner/flux_2/flux2_utils.py`
- `src/musubi_tuner/flux_2_cache_latents.py`
- `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py`
- `src/musubi_tuner/flux_2_generate_image.py`
- `src/musubi_tuner/flux_2_train_network.py`
- `src/musubi_tuner/flux_2_train_network_self_flow.py`
- `src/musubi_tuner/flux_kontext_cache_latents.py`
- `src/musubi_tuner/flux_kontext_cache_text_encoder_outputs.py`
- `src/musubi_tuner/flux_kontext_generate_image.py`
- `src/musubi_tuner/flux_kontext_train_network.py`
- `src/musubi_tuner/fpack_cache_latents.py`
- `src/musubi_tuner/fpack_cache_text_encoder_outputs.py`
- `src/musubi_tuner/fpack_generate_video.py`
- `src/musubi_tuner/fpack_train_network.py`
- `src/musubi_tuner/frame_pack/__init__.py`
- `src/musubi_tuner/frame_pack/bucket_tools.py`
- `src/musubi_tuner/frame_pack/clip_vision.py`
- `src/musubi_tuner/frame_pack/framepack_utils.py`
- `src/musubi_tuner/frame_pack/hunyuan.py`
- `src/musubi_tuner/frame_pack/hunyuan_video_packed.py`
- `src/musubi_tuner/frame_pack/hunyuan_video_packed_inference.py`
- `src/musubi_tuner/frame_pack/k_diffusion_hunyuan.py`
- `src/musubi_tuner/frame_pack/uni_pc_fm.py`
- `src/musubi_tuner/frame_pack/utils.py`
- `src/musubi_tuner/frame_pack/wrapper.py`
- `src/musubi_tuner/gui/config_manager.py`
- `src/musubi_tuner/gui/gui.py`
- `src/musubi_tuner/gui/i18n_data.py`
- `src/musubi_tuner/hidream_o1/__init__.py`
- `src/musubi_tuner/hidream_o1/flash_scheduler.py`
- `src/musubi_tuner/hidream_o1/hidream_o1_utils.py`
- `src/musubi_tuner/hidream_o1/pipeline.py`
- `src/musubi_tuner/hidream_o1/qwen3_vl_transformers.py`
- `src/musubi_tuner/hidream_o1/utils.py`
- `src/musubi_tuner/hidream_o1_cache_pixel.py`
- `src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py`
- `src/musubi_tuner/hidream_o1_generate_image.py`
- `src/musubi_tuner/hidream_o1_train.py`
- `src/musubi_tuner/hidream_o1_train_network.py`
- `src/musubi_tuner/hunyuan_model/__init__.py`
- `src/musubi_tuner/hunyuan_model/activation_layers.py`
- `src/musubi_tuner/hunyuan_model/attention.py`
- `src/musubi_tuner/hunyuan_model/autoencoder_kl_causal_3d.py`
- `src/musubi_tuner/hunyuan_model/embed_layers.py`
- `src/musubi_tuner/hunyuan_model/fp8_optimization.py`
- `src/musubi_tuner/hunyuan_model/helpers.py`
- `src/musubi_tuner/hunyuan_model/mlp_layers.py`
- `src/musubi_tuner/hunyuan_model/models.py`
- `src/musubi_tuner/hunyuan_model/modulate_layers.py`
- `src/musubi_tuner/hunyuan_model/norm_layers.py`
- `src/musubi_tuner/hunyuan_model/pipeline_hunyuan_video.py`
- `src/musubi_tuner/hunyuan_model/posemb_layers.py`
- `src/musubi_tuner/hunyuan_model/text_encoder.py`
- `src/musubi_tuner/hunyuan_model/token_refiner.py`
- `src/musubi_tuner/hunyuan_model/vae.py`
- `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_models.py`
- `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_modules.py`
- `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_text_encoder.py`
- `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_utils.py`
- `src/musubi_tuner/hunyuan_video_1_5/hunyuan_video_1_5_vae.py`
- `src/musubi_tuner/hv_1_5_cache_latents.py`
- `src/musubi_tuner/hv_1_5_cache_text_encoder_outputs.py`
- `src/musubi_tuner/hv_1_5_generate_video.py`
- `src/musubi_tuner/hv_1_5_train_network.py`
- `src/musubi_tuner/hv_generate_video.py`
- `src/musubi_tuner/hv_train.py`
- `src/musubi_tuner/hv_train_network.py`
- `src/musubi_tuner/ideogram4/__init__.py`
- `src/musubi_tuner/ideogram4/caption_verifier.py`
- `src/musubi_tuner/ideogram4/constants.py`
- `src/musubi_tuner/ideogram4/ideogram4_autoencoder.py`
- `src/musubi_tuner/ideogram4/ideogram4_model.py`
- `src/musubi_tuner/ideogram4/ideogram4_quantized_loading.py`
- `src/musubi_tuner/ideogram4/ideogram4_scheduler.py`
- `src/musubi_tuner/ideogram4/ideogram4_utils.py`
- `src/musubi_tuner/ideogram4/latent_norm.py`
- `src/musubi_tuner/ideogram4/sampler_configs.py`
- `src/musubi_tuner/ideogram4/sampling_policy.py`
- `src/musubi_tuner/ideogram4_cache_latents.py`
- `src/musubi_tuner/ideogram4_cache_text_encoder_outputs.py`
- `src/musubi_tuner/ideogram4_generate_image.py`
- `src/musubi_tuner/ideogram4_train_network.py`
- `src/musubi_tuner/kandinsky5/__init__.py`
- `src/musubi_tuner/kandinsky5/configs.py`
- `src/musubi_tuner/kandinsky5/generation_utils.py`
- `src/musubi_tuner/kandinsky5/models/__init__.py`
- `src/musubi_tuner/kandinsky5/models/attention.py`
- `src/musubi_tuner/kandinsky5/models/dit.py`
- `src/musubi_tuner/kandinsky5/models/nn.py`
- `src/musubi_tuner/kandinsky5/models/text_embedders.py`
- `src/musubi_tuner/kandinsky5/models/utils.py`
- `src/musubi_tuner/kandinsky5/models/vae.py`
- `src/musubi_tuner/kandinsky5_cache_latents.py`
- `src/musubi_tuner/kandinsky5_cache_text_encoder_outputs.py`
- `src/musubi_tuner/kandinsky5_generate_video.py`
- `src/musubi_tuner/kandinsky5_train_network.py`
- `src/musubi_tuner/krea2/__init__.py`
- `src/musubi_tuner/krea2/krea2_encoder.py`
- `src/musubi_tuner/krea2/krea2_mmdit.py`
- `src/musubi_tuner/krea2/krea2_sampling.py`
- `src/musubi_tuner/krea2/krea2_utils.py`
- `src/musubi_tuner/krea2_cache_latents.py`
- `src/musubi_tuner/krea2_cache_text_encoder_outputs.py`
- `src/musubi_tuner/krea2_generate_image.py`
- `src/musubi_tuner/krea2_train_network.py`
- `src/musubi_tuner/lora_post_hoc_ema.py`
- `src/musubi_tuner/merge_lora.py`
- `src/musubi_tuner/minimax_h3/__init__.py`
- `src/musubi_tuner/minimax_h3/args.py`
- `src/musubi_tuner/minimax_h3/audio_vae.py`
- `src/musubi_tuner/minimax_h3/cache_plan.py`
- `src/musubi_tuner/minimax_h3/checkpoint.py`
- `src/musubi_tuner/minimax_h3/generation_inputs.py`
- `src/musubi_tuner/minimax_h3/media.py`
- `src/musubi_tuner/minimax_h3/model.py`
- `src/musubi_tuner/minimax_h3/packing.py`
- `src/musubi_tuner/minimax_h3/sampling.py`
- `src/musubi_tuner/minimax_h3/text_encoder.py`
- `src/musubi_tuner/minimax_h3/video_vae.py`
- `src/musubi_tuner/minimax_h3_cache_latents.py`
- `src/musubi_tuner/minimax_h3_cache_text_encoder_outputs.py`
- `src/musubi_tuner/minimax_h3_generate_video.py`
- `src/musubi_tuner/minimax_h3_train_network.py`
- `src/musubi_tuner/modules/adafactor_fused.py`
- `src/musubi_tuner/modules/attention.py`
- `src/musubi_tuner/modules/comfy_quant_utils.py`
- `src/musubi_tuner/modules/convrot_int8_kernels.py`
- `src/musubi_tuner/modules/convrot_int8_utils.py`
- `src/musubi_tuner/modules/nvfp4_utils.py`
- `src/musubi_tuner/modules/unet_causal_3d_blocks.py`
- `src/musubi_tuner/networks/convert_hunyuan_video_1_5_lora_to_comfy.py`
- `src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py`
- `src/musubi_tuner/networks/lora_flux.py`
- `src/musubi_tuner/networks/lora_flux_2.py`
- `src/musubi_tuner/networks/lora_framepack.py`
- `src/musubi_tuner/networks/lora_hidream_o1.py`
- `src/musubi_tuner/networks/lora_hv_1_5.py`
- `src/musubi_tuner/networks/lora_ideogram4.py`
- `src/musubi_tuner/networks/lora_kandinsky.py`
- `src/musubi_tuner/networks/lora_krea2.py`
- `src/musubi_tuner/networks/lora_minimax_h3.py`
- `src/musubi_tuner/networks/lora_wan.py`
- `src/musubi_tuner/networks/lora_zimage.py`
- `src/musubi_tuner/qwen_extract_lora.py`
- `src/musubi_tuner/qwen_image_generate_image.py`
- `src/musubi_tuner/qwen_image_train.py`
- `src/musubi_tuner/training/audio_loss.py`
- `src/musubi_tuner/wan/__init__.py`
- `src/musubi_tuner/wan/configs/__init__.py`
- `src/musubi_tuner/wan/configs/shared_config.py`
- `src/musubi_tuner/wan/configs/wan_i2v_14B.py`
- `src/musubi_tuner/wan/configs/wan_i2v_A14B.py`
- `src/musubi_tuner/wan/configs/wan_t2v_14B.py`
- `src/musubi_tuner/wan/configs/wan_t2v_1_3B.py`
- `src/musubi_tuner/wan/configs/wan_t2v_A14B.py`
- `src/musubi_tuner/wan/modules/__init__.py`
- `src/musubi_tuner/wan/modules/attention.py`
- `src/musubi_tuner/wan/modules/clip.py`
- `src/musubi_tuner/wan/modules/model.py`
- `src/musubi_tuner/wan/modules/t5.py`
- `src/musubi_tuner/wan/modules/tokenizers.py`
- `src/musubi_tuner/wan/modules/vae.py`
- `src/musubi_tuner/wan/modules/xlm_roberta.py`
- `src/musubi_tuner/wan/utils/__init__.py`
- `src/musubi_tuner/wan/utils/fm_solvers.py`
- `src/musubi_tuner/wan/utils/fm_solvers_unipc.py`
- `src/musubi_tuner/wan/utils/utils.py`
- `src/musubi_tuner/wan_cache_latents.py`
- `src/musubi_tuner/wan_cache_text_encoder_outputs.py`
- `src/musubi_tuner/wan_generate_video.py`
- `src/musubi_tuner/wan_train_network.py`
- `src/musubi_tuner/zimage/zimage_autoencoder.py`
- `src/musubi_tuner/zimage/zimage_config.py`
- `src/musubi_tuner/zimage/zimage_model.py`
- `src/musubi_tuner/zimage/zimage_utils.py`
- `src/musubi_tuner/zimage_cache_latents.py`
- `src/musubi_tuner/zimage_cache_text_encoder_outputs.py`
- `src/musubi_tuner/zimage_generate_image.py`
- `src/musubi_tuner/zimage_train.py`
- `src/musubi_tuner/zimage_train_network.py`
- `tests/test_audio_dataset_seam.py`
- `tests/test_convrot_int8_prequantized.py`
- `tests/test_ideogram4_autoencoder.py`
- `tests/test_ideogram4_fp8_loading.py`
- `tests/test_ideogram4_lora_sampling.py`
- `tests/test_ideogram4_synthetic.py`
- `tests/test_ideogram4_te_fp8_loading.py`
- `tests/test_krea2_convrot_int8.py`
- `tests/test_krea2_gather_valid_text.py`
- `tests/test_krea2_gradient_checkpointing.py`
- `tests/test_krea2_turbo_lora.py`
- `tests/test_minimax_h3_attention_extent.py`
- `tests/test_minimax_h3_cache_contract.py`
- `tests/test_minimax_h3_cache_plan.py`
- `tests/test_minimax_h3_convrot_int8.py`
- `tests/test_minimax_h3_convrot_runtime.py`
- `tests/test_minimax_h3_dataset.py`
- `tests/test_minimax_h3_generation_modes.py`
- `tests/test_minimax_h3_generation_request.py`
- `tests/test_minimax_h3_layering.py`
- `tests/test_minimax_h3_model.py`
- `tests/test_minimax_h3_nvfp4.py`
- `tests/test_minimax_h3_packing.py`
- `tests/test_minimax_h3_provenance.py`
- `tests/test_minimax_h3_sampling.py`
- `tests/test_minimax_h3_te_streaming.py`
- `tests/test_minimax_h3_temporal_stretch.py`
- `tests/test_minimax_h3_text_encoder.py`
- `tests/test_minimax_h3_training.py`
- `tests/test_minimax_h3_vae.py`
- `wan_cache_latents.py`
- `wan_cache_text_encoder_outputs.py`
- `wan_generate_video.py`
- `wan_train_network.py`
- `zimage_cache_latents.py`
- `zimage_cache_text_encoder_outputs.py`
- `zimage_generate_image.py`
- `zimage_train.py`
- `zimage_train_network.py`


Post-deletion real guarded import/factory and US1/US2/US3 tests: **133 passed**, 34.16s. Both corrected template references exist from repository CWD; warmup remains integer 200 and constant_with_warmup. Git diff has three value changes relative to HEAD because the scheduler correction was pre-existing; this implementation contributes only the two references.

## T042–T044: guides and dependencies

Aligned ten retained guides/contribution files and quickstart with actual original-only consumers, preserving timestep formulas, distribution figures and contributor attribution. Removed these remaining exact exclusive guide/assets:
- `docs/betas_for_sigma_rel.png`
- `docs/flux_2.md`
- `docs/flux_kontext.md`
- `docs/framepack.md`
- `docs/framepack_1f.md`
- `docs/hidream_o1.md`
- `docs/hunyuan_video.md`
- `docs/hunyuan_video_1_5.md`
- `docs/ideogram4.md`
- `docs/kandinsky5.md`
- `docs/kisekaeichi_ref.png`
- `docs/kisekaeichi_ref_mask.png`
- `docs/kisekaeichi_result.png`
- `docs/kisekaeichi_start.png`
- `docs/kisekaeichi_start_mask.png`
- `docs/krea2.md`
- `docs/minimax_h3.md`
- `docs/minimax_h3_1f.md`
- `docs/minimax_h3_advanced.md`
- `docs/tools.md`
- `docs/wan.md`
- `docs/wan_1f.md`
- `docs/zimage.md`
- `src/musubi_tuner/gui/gui.ja.md`
- `src/musubi_tuner/gui/gui.md`
- `src/musubi_tuner/gui/gui_implementation_plan.md`

Parsed the edited manifest and asserted retained dependency versions/CUDA extras are unchanged. Removed av/ftfy/easydict/sentencepiece, GUI/HiDream extras, dev prompt-toolkit and only inventoried obsolete Ruff paths. No lock exists and none was generated. No install/resolution/build was performed. Final parser/link and closure audits follow.


## T045–T046: final retained dependency graph

After owner and manifest cleanup: all six actual root/module help invocations and all guarded retained imports/factories passed: **7 passed**, 99.48s. The 133 post-deletion behavioral cases already exercised the affected source/cache/model consumers. Final static review found no incoming excluded imports, remaining inventoried removed files or alternate root commands (exactly three remain). Retained dynamic factory resolution is limited to Qwen/LoHa/LoKr; optimizer/scheduler custom classes and optional Hub/logging remain by contract. Removed unused model-level scheduler duplicates, generation example/config constants and stale comments; numerical scheduler implementation remains in its existing Qwen utility owner.

Reviewed all 45 package files across training/dataset/Qwen/networks/modules/utils. Remaining historical names are intentional: image_video_dataset is the retained filename, original VAE and original positional math use 3D/singleton axes, VL's original config includes multimodal token IDs, timestep names denote retained shared mathematical samplers, and notices credit original authors. None adds a foreign loader, dataset, adapter dispatch or training mode. No excluded implementation was copied under a neutral name. Generic FP8/adapter/tensor IO primitives remain where the original engine uses them. Ruff cleanup removed 24 newly unused imports; only changed retained files were formatted.

### T047 — final retained suite and static checks

- Full retained suite in the existing Python 3.12 CPU environment, with CUDA disabled and Hugging Face offline: **174 passed, 1 warning in 114.17s**. The warning is the installed einops invalid-escape SyntaxWarning, not a test failure. Torch was imported before pytest to avoid the previously observed Windows native initialization diagnostics.
- Existing Ruff configuration: `ruff check .` passed; `ruff format --check .` passed (**58 files already formatted**). No lint rule changes.
- AST parse passed for all **48** retained runtime Python files (45 package modules and three root wrappers). No excluded imports or inventoried removed files remain. Exactly three root Python commands remain.
- Baseline retained suite was 38 passing cases; final coverage includes the substantive dataset/cache, configuration, numerical/gradient and state/observation boundary cases recorded above.

## T048–T049: final README and example verification

The three README files now describe the complete retained workflow, required inputs/packages, actual options/precedence, both cache passes, training samples/logs, artifact naming and independent retention, and the existing resume counter/cursor limitation. No fixed GPU/rank/resolution requirement or real-model verification claim is introduced. Retained source attribution and upstream notices remain in the READMEs; related contributor credits remain in the guides.

Actual offline audit of **14 documents** (three READMEs, seven retained guides, the packaged dataset guide, two contribution guides and quickstart):

| Exercised documentation contract | Actual result |
| --- | --- |
| Repository-relative document/configuration/image links | **91 resolved** |
| Root command examples, including help | **22 accepted** by the actual retained parsers |
| Accelerate launcher portions | **9 accepted** by the installed Accelerate launch parser, preserving bf16 |
| Fenced TOML examples | **7 accepted** by the corresponding training/dataset/prompt readers |
| Prompt examples | **2 accepted**, using real TXT/TOML readers |
| Image JSONL examples | **2 accepted**, using the real datasource with a temporary image |
| Complete inline training assignments | **14 accepted** by the actual effective-config resolver |
| Three compatibility templates together | All **40 training keys**, **1 dataset declaration**, and **2 prompts** accepted; original effective values asserted before temporary path substitution |

External weight/image/resume paths were replaced only in temporary fixtures, with real tiny images/captions and empty checkpoint placeholders used solely for existence checks. Output/log/cache fixture paths stayed in the temporary directory. No model loader, cache writer/cleanup, tracker, generation or training command was called. Dataset declarations used the real schema, blueprint, datasource and preflight. Training examples used the actual resolver and preflight; the Inductor example used actual type/choice parsing only because its documented operational Triton prerequisite is unavailable here. This does not claim compilation was verified. Test command paths and the PowerShell/POSIX source-path setup were also checked against the repository.

The first documentation audit caught a documentation-only type error: `compile_dynamic` consumes the string `"true"`, not TOML boolean `true`. Corrected the example to the existing parser contract; no parser or test was weakened. Fixed both relative links in the packaged dataset guide and replaced quickstart's stale planning-only statements. The final audit above passed. Remaining inline identifiers outside the training parser were checked against dataset/cache/prompt fields, adapter factory arguments, optimizer arguments and timestep/scheduler choices; none is an unsupported executable command or missing shipped reference. External attribution links were preserved, not fetched as operational checks.

## T050: final requirement and protection review

| Agreed requirement / success criterion | Evidence and conclusion |
| --- | --- |
| FR-001, FR-013–015; SC-004 | T004 inventory, T010 shared-owner extraction, T035 guarded runtime gate, T036–T046 deletion and final graph review: **300 inventoried tracked files removed** (274 Python, 26 guides/assets), no remaining inventoried exclusions or incoming dependency on removed owners. Exactly three root commands and 45 package Python files remain. Dynamic adapter names resolve only to retained Qwen/LoHa/LoKr factories. Manifest versions and CUDA choices retained; exclusive dependencies/extras removed; no lock existed. |
| FR-002–003; SC-001 | T011–T017/T031 actual image/caption directory and JSONL readers, strict declarations, fallback, buckets/repeats/batches, real cache serialization/consumption and reuse/retention checks. Original singleton latent axes, variable text lengths, cache names/dtypes/metadata and association preserved. |
| FR-004, FR-008; SC-001, SC-007 | T005–T010/T018–T025 baseline comparisons and tiny CPU invariants cover attention, packing, original forward/noise/target/loss, dtype/gradients, adapters and optimizers/schedulers. Identical tiny original Qwen models before/after extraction had bitwise-equal outputs and gradients, with and without checkpointing. Existing optimizer/accumulation/checkpoint/resume loop retained; removed only excluded routes and their consumers. No hyperparameter repair, new optimization or tensor/weight duplication. |
| FR-005–006; SC-001 | Real prompt formats/trigger rules/PNG saver and logging metrics tested; Qwen memoization/default negative/CFG path preserved. Normal sampling preserves weights, training mode and torch RNG through the exercised boundary. Tracker initialization remains after preflight. Exceptional sample-exit restoration is an unchanged baseline limitation, documented rather than claimed verified. |
| FR-007; SC-001 | Real adapter safetensors/dtype checks and CPU Accelerate adapter/optimizer/scheduler/Python-NumPy-torch RNG round trip; independent state/checkpoint retention boundaries and filter hooks checked. Source review confirms unchanged local counter reset/no consumed-batch skip; README states this limitation. |
| FR-009–010; SC-002 | All 40 template values and explicit CLI precedence tested; all three templates parsed together. Short/qualified network names, original selection, selected-only attention prerequisites and existing flag precedence verified. The pre-existing approved warmup remains integer 200 with constant_with_warmup. |
| FR-011–012; SC-003 | T026–T034 actual invalid CLI/TOML/dataset/prompt/argument cases fail before loader/tokenizer/cache/tracker/training sentinels, with source/key/cause/correction. Effective validation preserves supported non-template settings. Template meanings were not changed to obtain passes. |
| FR-016; SC-005 | T048–T049 parser/reader audit above; coherent full workflow in all three READMEs; 91 shipped links resolve. User-replaceable external paths are distinguished from repository inputs. |
| FR-017; SC-007 | Final **54 unchanged T001 SHA-256 hashes**, including constitution, specification, plan, both other templates, user prompt notes and protected infrastructure. Protected paths have no Git diff. Reversing the two approved train-template references exactly reproduces its T001 SHA-256, including its pre-existing mixed newline at the scheduler line; all other bytes are preserved. **11 original retained source copyright/license lines** match, with source notices and README attribution preserved. All deletions exactly equal the reviewed inventory. No user dataset, weight, cache or output was deleted; check fixtures were temporary. |
| FR-018; SC-007 | Narrow existing readers/helpers and validators retained; no framework, new workflow, benchmarks, performance budgets or training-engine replacement. Extracted attention/FP8-predicate/PNG helpers keep their characterized behavior. |
| FR-019–020; SC-006 | T047 full retained suite **174 passed**, Ruff lint and format checks passed, all retained runtime files parsed. Required local import/help/factory/readers/numerical/serialization/documentation evidence is present. Real-model operation remains outside these checks. |
| FR-021 / phase authorization | This separate user invocation explicitly authorizes initial implementation. Specification/constitution unchanged. No analyze/converge or subsequent correction round was started. |

Protection audit: branch remains `001-scope-qwen-image-lora`; HEAD remains the externally created `dd728d4bdb3ea2422c403c5157138db3b8179e5e` recorded earlier. This implementation did not commit, change branch, index or history. Task text compared with HEAD differs only in completion markers. `.specify/extensions.yml` is still absent; no post-implementation hooks apply. `git diff --check` passed.

### Initial local stage result and limits (superseded by convergence and round 1)

All 50 initial implementation tasks are complete with the evidence above. No confirmed in-scope discrepancy or missing required local check remains. The single test warning is from the installed einops package. Existing CPU dependency versions were used without installation; optional GPU/backend behavior and real model quality/performance are not established. Resume retains the existing local counter reset/no data-cursor restoration, and exceptional-exit sample restoration is not guaranteed.

Real-model encoding/training/sampling, optional backend execution and operational compatibility remain separate server-stage work; none is counted as a local pass. No converge or additional correction cycle was launched.


## Approved correction round 1: T051–T057

**Authorization**: The user explicitly approved only T051–T057 from the 2026-09-22 convergence assessment. This execution adds no requirements, changes no constitution/design documents, and starts no new converge or correction round.

### Baseline and regression evidence

Used the existing T002 Python 3.12.14 CPU environment without installing dependencies. Captured SHA-256 hashes of 143 existing tracked/test files before changes, plus copies of the files edited in this round; protected infrastructure is also checked against the recorded T001 hashes. The existing working-tree implementation was preserved. `requirements.md` had 16/16 checked items and was not edited. Existing Python ignore rules were sufficient; no ignore-file change or dependency lock was introduced. `.specify/extensions.yml` was absent before and after execution, so no implementation hooks applied.

Before edits, the three Qwen regression files passed **129 tests** (16.15s). After adding the regression cases, but before code fixes, configuration/training-invariant tests produced **19 expected failures, 118 passes** (12.88s): the failures reproduced the approved early-validation, abbreviated-selector, default-optimizer and adapter-initialization gaps. No dependency or real reader was substituted to obtain a pass.

| Task / finding | Implemented correction and verified boundary | Targeted result |
| --- | --- | --- |
| T051 / F1 | Full required-argument binding for optimizer/custom scheduler and derived scheduler arguments; AdamW beta/epsilon/weight-decay checks; Adafactor optimizer/scheduler compatibility; piecewise rule syntax checked against the existing consumer grammar. Negative entry-point cases reject before resource boundaries. Real CPU optimizer/scheduler controls preserve StepLR and piecewise progression, integer/ratio warmup, and Adafactor relative-step/warmup-init derivation; preflight does not mutate arguments or torch RNG. | **12 passed** (9.83s) |
| T052 / F2 | Resolve only the selected compile backend with PyTorch's resolver; require nonnegative logit standard deviation only for distributions that consume it; read supplied tracker TOML before loaders; reject a file at the output directory or an existing parent. Valid eager compilation selection, zero standard deviation, new output directories and unused-option cases remain accepted. No compilation is executed. | **18 passed** (9.93s) |
| T053 / F3 | Existing source preflight uses the existing BucketSelector on each image header. An 8×8 no-upscale input rejects with dataset/item/path/bucket and correction before VAE loading; 8×8 with upscaling and 32×32 without upscaling reach only the loader sentinel. Existing cache entry call already precedes the loader and needed no duplicate validation. Bucket/resize algorithms are unchanged. | **5 passed** (9.61s) |
| T054 / F4 | Raw selector scanning uses argparse option resolution, including separate/equals abbreviations, before a masked value is lost. Accommodates the tuple and candidate-list result shapes used by supported Python maintenance versions. Existing supported overrides and template values remain valid. | **10 passed** (9.42s) |
| T055 / F5 | AdamW is the parser's effective default; preflight consumes the same value without a local fallback. The existing optimizer factory now receives AdamW on omission. Real one-parameter factory checks confirm omission and explicit SGD selection, learning rate and parameter identity. No optimizer calculation changed. | **7 passed** (9.87s), including shared T051 controls |
| T056 / F6 | `_build_network` reads `network_weights` and consumes the single network returned by the retained factories. The subsequent existing weight-loading call remains. Tiny real serialized LoRA/LoHa/LoKr adapters restore inferred rank 1, alpha 3 and every saved tensor despite conflicting explicit rank/alpha arguments; base tensors remain unchanged and resume is unused. | **8 passed** (10.39s), including retained adapter cases |
| T057 / F7 | Both dataset guides now document persisted `varlen_vl_embed_<dtype>` and distinguish the reader's `vl_embed` batch key. Writer, reader and cache format were not changed. | **25 dataset/cache tests passed** (14.10s); **91 local links resolved** across 14 documents |

The first T054 test run exposed an argparse maintenance-version return-shape difference (6 failures); the implementation was corrected before proceeding, and the unchanged assertions passed. This was development verification within the approved round, not a new correction cycle.

### Final local verification

Commands used this checkout's sources and the same offline environment:

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -c 'import torch; import pytest; raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", "tests"]))'
ruff check --no-cache .
ruff format --check --no-cache .
git diff --check
```

- Full retained suite: **207 passed, 1 warning, 115.53s**. The warning is the existing installed-einops SyntaxWarning. This includes real root/module help and guarded import/factory checks, configuration/dataset/cache checks, numerical/gradient checks and actual adapter/Accelerate serialization.
- Additional real CPU preflight/factory audit accepted all ten existing schedules checked: constant, constant_with_warmup, linear, cosine, cosine_with_restarts, polynomial, inverse_sqrt, cosine_with_min_lr, warmup_stable_decay and rex. Each used a one-parameter AdamW optimizer, max_train_steps=10, lr_decay_steps=2 and lr_scheduler_min_lr_ratio=0.1; no training loop was invoked.
- `ruff check --no-cache .`: **PASS**. `ruff format --check --no-cache` with the six changed Python paths: **PASS**, six files already formatted. All **48 runtime Python files** parsed with AST. `git diff --check`: **PASS**; Git emitted its existing line-ending notices for the previously changed `.gitignore` and training template.
- Repository-wide `ruff format --check --no-cache .`: **FAIL**, five protected files would be reformatted, 59 files already formatted. The unchanged paths are `.specify/scripts/python/check_prerequisites.py`, `common.py`, `create_new_feature.py`, `resolve_template.py` and `setup_tasks.py`. This is the same out-of-scope formatting result recorded by converge; protected infrastructure and lint rules were not changed to hide it.
- README review and local-link audit: all **91 links** in **14 documents** resolve. README workflow, optimizer selection and adapter-initialization descriptions agree with this round; only the two approved cache-key descriptions required documentation changes.
- Scope audit: only four runtime Python files, two regression files, two dataset guides, task completion markers and this validation record changed in the round. All **49 recorded protected-infrastructure hashes** match; the three templates, user prompt notes, specification and plan retain their round-entry hashes. No source/license notice was removed. No new runtime file, dependency or abstraction was added. Branch remains `001-scope-qwen-image-lora`, HEAD remains `dd728d4bdb3ea2422c403c5157138db3b8179e5e`; no commit/index/history change was made.

### Round 1 result and limits

T051–T057 are complete with the evidence above. The seven approved findings are corrected; no further converge or correction round was started. This is completion of the authorized round, not a blanket claim that the repository-wide format gate passed. The known protected-file formatting result remains as recorded and outside this authorization. Real-model/backend execution remains outside the local evidence, as do changes to the existing resume counter/data-cursor and exceptional sampling-restoration behavior.

**No training, GPU run, weight download, package build, server transfer or server verification was performed.**

## Approved correction round 2: T058–T060

**Date and authorization**: 2026-09-23. The user explicitly approved only T058–T060 from the latest convergence assessment. This is the second and final permitted correction round. No requirements, specification, plan, constitution or template values were changed; no subsequent convergence or correction round was started.

### Baseline and regression evidence

Used the existing T002 Python 3.12.14 CPU environment and the same offline settings, without installing dependencies. Captured round-entry SHA-256 hashes of 143 existing tracked/test files and copies of the files to be edited. Existing working-tree changes were preserved. The requirements checklist remained read-only and passed with 16/16 checked items. Existing Python ignore rules were sufficient; no ignore-file or dependency change was needed. `.specify/extensions.yml` was absent before and after implementation, so no hooks applied.

Before edits, `tests/test_qwen_image_config.py` and `tests/test_qwen_image_dataset_cache.py` passed **133 tests** (16.13s). Added regressions before changing runtime code: the configuration suite then produced **13 expected failures, 123 passes** (13.11s). These exposed the six missing training preflight checks, rejected valid `REX` spelling, four invalid cache-parent boundary crossings, and two existing-file errors lacking path context. Passing controls used real readers and tiny CPU optimizer/scheduler factories; no real dependency or reader was replaced.

| Task / finding | Correction and verified behavior | Focused result |
| --- | --- | --- |
| T058 / F8 | Training preflight checks the existing H2D-only checkpointing prerequisite when `blocks_to_swap` is active. The diagnostic identifies the config, swap/checkpointing settings, reused-weight backward constraint and corrections. Invalid input stops before resource boundaries. Checkpointed H2D, disabled swap and ordinary non-H2D swap remain accepted; active controls use the real `BlockSwapConfig.from_args` on CPU without constructing an offloader. | **4 passed** (9.95s) |
| T059 / F9 | Preflight validates SGD momentum, polynomial initial/end LR ordering, the required cosine minimum, inverse-sqrt timescale and WSD decay type using the existing consumers/defaults. Rex spelling is normalized only in the validator's local selection; resolved arguments remain unchanged. Real CPU factories verify valid SGD, polynomial default/custom end LR, inverse-sqrt progression, cosine ratio/absolute minimum and WSD decay. `rex`/`REX` produce identical LR sequences. Existing custom/schedule-free early returns, unused options and derived Adafactor scheduling remain intact. Preflight preserves the tested Namespace and torch RNG. | **46 passed** (11.10s), including existing early-error and scheduler controls |
| T060 / F10 | Dataset preflight checks the selected cache path and every existing ancestor, reporting dataset/config, requested cache path, blocking file and correction. Both actual cache entry points reject a file at the target, immediate parent or more distant ancestor before resources/cache preparation. New nested directories reach only loader sentinels and remain absent; blocking fixture files retain their contents. The valid text-cache control keeps the real read-only cache-path enumeration. | **8 passed** (10.69s) |

The runtime changes are confined to `qwen_image_train_network.py` and `dataset/config_utils.py`; tests extend the existing `test_qwen_image_config.py`. Offloader, checkpointing, optimizer/scheduler calculations, cache writers/naming/cleanup and training-loop code were not edited.

### Final local verification

Commands used the repository sources and existing CPU environment:

```powershell
$env:PYTHONPATH = "$PWD/src"
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$env:PYTHONDONTWRITEBYTECODE = '1'
& 'C:/Users/inbox/Desktop/musubi-tuner-flux2dev-lora/.venv/Scripts/python.exe' -B -c 'import torch; import pytest; raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", "tests"]))'
ruff check --no-cache .
ruff format --check --no-cache src/musubi_tuner/qwen_image_train_network.py src/musubi_tuner/dataset/config_utils.py tests/test_qwen_image_config.py
ruff format --check --no-cache .
git diff --check
```

Focused runs used the same pytest driver with `tests/test_qwen_image_config.py` and, respectively, `-k h2d_block_swap`, `-k "training_early_errors or optimizer_scheduler_preflight or rex_scheduler or scheduler_float_ratio"`, and `-k cache_directory_ancestors`.

- Full retained CPU/offline suite: **235 passed, 1 warning, 121.02s**. The warning is the existing installed-einops SyntaxWarning. This includes six real root/module help invocations, retained imports/factories, template/parser checks, image/cache serialization, numerical/gradient checks and adapter/Accelerate state checks.
- Ruff lint: **PASS**. Format check of the three edited Python files: **PASS**. Runtime AST parsing: **48 files passed**. `git diff --check`: **PASS**, with the existing Git line-ending notices for `.gitignore` and the training template.
- Repository-wide format check: **FAIL** on the same five unchanged protected `.specify/scripts/python/` files: `check_prerequisites.py`, `common.py`, `create_new_feature.py`, `resolve_template.py`, `setup_tasks.py`; 59 other files are formatted. This required failing check remains unresolved and is not counted as a pass. No protected file or lint configuration was changed to hide it.
- README review confirmed that the affected configuration, optimization and cache-path descriptions remain consistent with this round. All **91 local links across 14 documents** resolve; no README/documentation change was required.
- Scope audit: changes since round entry are exactly the two runtime files, one regression file, T058–T060 completion markers and this appended evidence. All **47 recorded protected-infrastructure hashes** match. The specification, plan, three templates, other documentation and user prompt notes retain their round-entry hashes. Branch remains `001-scope-qwen-image-lora`, HEAD remains `dd728d4bdb3ea2422c403c5157138db3b8179e5e`; no commit, index or history change was made.

### Round 2 result and limits

T058–T060 are complete and marked `[X]`. The three approved findings are corrected with local evidence. The correction cycle stops here. The local stage as a whole is **not declared complete**, because the known required repository-wide format check still fails outside this round's authorized scope. These results do not establish real-model or optional GPU/backend operation.

**No training, GPU run, weight download, package build, server transfer or server verification was performed in this round.**

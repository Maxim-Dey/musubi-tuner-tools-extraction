"""Small real-file CPU fixtures for Qwen-Image validation input tests."""

import hashlib
import json
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
import pytest
import torch
from torch.utils.data import DataLoader

from musubi_tuner.training.validation_inputs import (
    iter_noise_checks,
    iter_validation_checks,
    make_validation_loader,
    noise_levels,
    prepare_validation_inputs,
    read_validation_cache_pair,
    stable_hash,
)


def make_validation_item(
    root: Path,
    role: str,
    *,
    stem: str = "sample",
    color: tuple[int, int, int] = (128, 64, 32),
    caption: str = "caption sample",
    size: tuple[int, int] = (64, 64),
) -> dict:
    """Write one captioned image and its real Qwen latent/text safetensors."""
    image_dir = root / role / "images"
    cache_dir = root / role / "cache"
    image_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"{stem}.png"
    caption_path = image_dir / f"{stem}.txt"
    Image.new("RGB", size, color).save(image_path)
    caption_path.write_text(f"{caption}\n", encoding="utf-8")
    image_id = hashlib.sha256(image_path.read_bytes()).hexdigest()

    latent_path = cache_dir / f"{stem}_{size[0]:04d}x{size[1]:04d}_qi.safetensors"
    text_path = cache_dir / f"{stem}_qi_te.safetensors"
    common_metadata = {"architecture": "qwen_image", "format_version": "1.0.1", "source_image_sha256": image_id}
    save_file(
        {"latents_1x8x8_float32": torch.ones((2, 1, 8, 8), dtype=torch.float32)},
        str(latent_path),
        metadata={**common_metadata, "width": str(size[0]), "height": str(size[1])},
    )
    save_file(
        {"varlen_vl_embed_float32": torch.ones((3, 4), dtype=torch.float32)},
        str(text_path),
        metadata={**common_metadata, "caption1": caption},
    )
    return {
        "role": role,
        "image": image_path,
        "caption": caption_path,
        "cache_dir": cache_dir,
        "latent_cache": latent_path,
        "text_cache": text_path,
        "image_id": image_id,
    }


def make_two_role_validation_files(root: Path, *, same_image: bool = False) -> tuple[Path, dict[str, dict]]:
    """Write one item per role and the matching val-dataset.toml."""
    items = {
        "val_familiar": make_validation_item(root, "val_familiar", color=(255, 0, 0), caption="familiar caption"),
        "val_unfamiliar": make_validation_item(
            root,
            "val_unfamiliar",
            color=(255, 0, 0) if same_image else (0, 0, 255),
            caption="unfamiliar caption",
        ),
    }
    declaration = [
        "[general]",
        "resolution = [64, 64]",
        "enable_bucket = true",
        "bucket_no_upscale = true",
        'caption_extension = ".txt"',
        "batch_size = 1",
        "num_repeats = 1",
    ]
    for role, item in items.items():
        declaration.extend(
            [
                "",
                "[[datasets]]",
                f"role = {json.dumps(role)}",
                f"image_directory = {json.dumps(item['image'].parent.as_posix())}",
                f"cache_directory = {json.dumps(item['cache_dir'].as_posix())}",
            ]
        )
    config_path = root / "val-dataset.toml"
    config_path.write_text("\n".join(declaration) + "\n", encoding="utf-8")
    return config_path, items


def _prepare(config_path: Path, *, expected_fingerprint=None, **controls):
    from types import SimpleNamespace

    return prepare_validation_inputs(
        SimpleNamespace(val_dataset_config=str(config_path), **controls),
        expected_fingerprint=expected_fingerprint,
    )


def test_unroled_validation_config_fails_before_dataset_or_model_setup(tmp_path, monkeypatch):
    config_path, _ = make_two_role_validation_files(tmp_path)
    config_path.write_text(
        "\n".join(line for line in config_path.read_text(encoding="utf-8").splitlines() if not line.startswith("role =")) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "musubi_tuner.training.validation_inputs.generate_dataset_group_by_blueprint",
        lambda *_args, **_kwargs: pytest.fail("dataset setup was reached before role validation"),
    )
    with pytest.raises(ValueError, match=r"val-dataset\.toml.*role.*val_familiar.*val_unfamiliar"):
        _prepare(config_path)


@pytest.mark.parametrize("cache_name", ["latent_cache", "text_cache"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_nonfinite_real_cache_tensor_fails_preflight(tmp_path, cache_name, nonfinite):
    config_path, sources = make_two_role_validation_files(tmp_path)
    path = sources["val_unfamiliar"][cache_name]
    with safe_open(str(path), framework="pt", device="cpu") as cache:
        metadata = dict(cache.metadata())
        tensors = {key: cache.get_tensor(key).clone() for key in cache.keys()}
    key = next(iter(tensors))
    tensors[key].flatten()[0] = nonfinite
    save_file(tensors, str(path), metadata=metadata)
    with pytest.raises(ValueError, match=r"val_unfamiliar.*(nonfinite|non-finite|finite)"):
        _prepare(config_path)


def test_manifest_reads_every_declared_source_and_exact_cache_pair(tmp_path):
    config_path, sources = make_two_role_validation_files(tmp_path)
    extra = make_validation_item(tmp_path, "val_familiar", stem="second", color=(0, 255, 0))

    manifest = _prepare(config_path)
    assert set(manifest.items_by_role) == {"val_familiar", "val_unfamiliar"}
    assert len(manifest.items_by_role["val_familiar"]) == 2
    assert len(manifest.items_by_role["val_unfamiliar"]) == 1
    assert len(manifest.items) == 3
    expected = [*sources.values(), extra]
    for source in expected:
        records = [item for item in manifest.items if item.image_path == source["image"]]
        assert len(records) == 1
        item = records[0]
        assert item.role == source["role"]
        assert item.image_id == hashlib.sha256(source["image"].read_bytes()).hexdigest()
        assert item.caption == source["caption"].read_text(encoding="utf-8").strip()
        assert item.original_size == (64, 64)
        assert item.bucket_size == (64, 64)
        assert item.latent_cache_path == source["latent_cache"]
        assert item.text_cache_path == source["text_cache"]


def test_equal_image_bytes_share_id_across_roles_and_empty_caption_is_allowed(tmp_path):
    config_path, sources = make_two_role_validation_files(tmp_path, same_image=True)
    sources["val_unfamiliar"]["caption"].write_text("\n", encoding="utf-8")
    text_path = sources["val_unfamiliar"]["text_cache"]
    save_file(
        {"varlen_vl_embed_float32": torch.ones((3, 4), dtype=torch.float32)},
        str(text_path),
        metadata={
            "architecture": "qwen_image",
            "format_version": "1.0.1",
            "source_image_sha256": sources["val_unfamiliar"]["image_id"],
            "caption1": "",
        },
    )

    manifest = _prepare(config_path)
    familiar = manifest.items_by_role["val_familiar"][0]
    unfamiliar = manifest.items_by_role["val_unfamiliar"][0]
    assert familiar.image_id == unfamiliar.image_id
    assert unfamiliar.caption == ""


@pytest.mark.parametrize("cache_name", ["latent_cache", "text_cache"])
def test_missing_cache_fails_instead_of_skipping_item(tmp_path, cache_name):
    config_path, sources = make_two_role_validation_files(tmp_path)
    sources["val_unfamiliar"][cache_name].unlink()
    with pytest.raises(ValueError, match=r"val_unfamiliar.*(missing|not found).*(cache|safetensors)|val_unfamiliar.*cache.*(missing|not found)"):
        _prepare(config_path)


@pytest.mark.parametrize("cache_name", ["latent_cache", "text_cache"])
def test_corrupt_cache_fails_instead_of_skipping_item(tmp_path, cache_name):
    config_path, sources = make_two_role_validation_files(tmp_path)
    sources["val_familiar"][cache_name].write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match=r"val_familiar.*(invalid|corrupt|unreadable|safetensors)"):
        _prepare(config_path)


@pytest.mark.parametrize("cache_name", ["latent_cache", "text_cache"])
def test_cache_bound_to_other_image_fails(tmp_path, cache_name):
    config_path, sources = make_two_role_validation_files(tmp_path)
    source = sources["val_unfamiliar"]
    path = source[cache_name]
    if cache_name == "latent_cache":
        tensors = {"latents_1x8x8_float32": torch.ones((2, 1, 8, 8))}
        metadata = {"width": "64", "height": "64"}
    else:
        tensors = {"varlen_vl_embed_float32": torch.ones((3, 4))}
        metadata = {"caption1": "unfamiliar caption"}
    save_file(
        tensors,
        str(path),
        metadata={
            **metadata,
            "architecture": "qwen_image",
            "format_version": "1.0.1",
            "source_image_sha256": sources["val_familiar"]["image_id"],
        },
    )
    with pytest.raises(ValueError, match=r"val_unfamiliar.*(source_image_sha256|image|stale|mismatch)"):
        _prepare(config_path)


def test_mismatched_text_caption_and_latent_dimensions_fail(tmp_path):
    config_path, sources = make_two_role_validation_files(tmp_path)
    source = sources["val_familiar"]
    save_file(
        {"varlen_vl_embed_float32": torch.ones((3, 4))},
        str(source["text_cache"]),
        metadata={
            "architecture": "qwen_image",
            "format_version": "1.0.1",
            "source_image_sha256": source["image_id"],
            "caption1": "wrong caption",
        },
    )
    with pytest.raises(ValueError, match=r"val_familiar.*caption"):
        _prepare(config_path)

    save_file(
        {"varlen_vl_embed_float32": torch.ones((3, 4))},
        str(source["text_cache"]),
        metadata={
            "architecture": "qwen_image",
            "format_version": "1.0.1",
            "source_image_sha256": source["image_id"],
            "caption1": "familiar caption",
        },
    )
    save_file(
        {"latents_1x8x8_float32": torch.ones((2, 1, 8, 8))},
        str(source["latent_cache"]),
        metadata={
            "architecture": "qwen_image",
            "format_version": "1.0.1",
            "source_image_sha256": source["image_id"],
            "width": "32",
            "height": "64",
        },
    )
    with pytest.raises(ValueError, match=r"val_familiar.*(width|dimensions)"):
        _prepare(config_path)


def test_missing_caption_fails_even_if_directory_reader_would_filter_image(tmp_path):
    config_path, sources = make_two_role_validation_files(tmp_path)
    sources["val_unfamiliar"]["caption"].unlink()
    with pytest.raises(ValueError, match=r"val_unfamiliar.*caption"):
        _prepare(config_path)


def test_latent_shape_must_match_selected_bucket(tmp_path):
    config_path, sources = make_two_role_validation_files(tmp_path)
    source = sources["val_familiar"]
    save_file(
        {"latents_1x4x4_float32": torch.ones((2, 1, 4, 4))},
        str(source["latent_cache"]),
        metadata={
            "architecture": "qwen_image",
            "format_version": "1.0.1",
            "source_image_sha256": source["image_id"],
            "width": "64",
            "height": "64",
        },
    )
    with pytest.raises(ValueError, match=r"val_familiar.*(bucket|shape)"):
        _prepare(config_path)


def _reference_seed(base_seed: int, image_id: str, i: int, j: int) -> int:
    message = f"qwen-image-val-noise-v1\n{base_seed}\n{image_id}\n{i}\n{j}\n".encode("ascii")
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "big") & ((1 << 63) - 1)


def test_noise_grid_indices_hash_and_final_mixing(tmp_path):
    config_path, _ = make_two_role_validation_files(tmp_path)
    item = _prepare(config_path).items_by_role["val_familiar"][0]
    latent = torch.arange(2 * 1 * 8 * 8, dtype=torch.float32).reshape(2, 1, 8, 8)

    assert noise_levels(2) == pytest.approx((0.275, 0.725))
    assert len(noise_levels(10)) == 10
    assert sum(t < 0.5 for t in noise_levels(10)) == 5
    assert sum(t >= 0.5 for t in noise_levels(10)) == 5
    checks = list(iter_noise_checks(item, latent, val_seed_noise=-7, val_level_noise_n=2, val_seed_noise_n=2))
    assert [(check.i, check.j) for check in checks] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert [check.group for check in checks] == ["low", "low", "high", "high"]
    for check in checks:
        expected_t = 0.05 + (check.i - 0.5) * 0.90 / 2
        assert check.t == expected_t
        assert check.timestep == 1000 * expected_t
        expected_seed = _reference_seed(-7, item.image_id, check.i, check.j)
        assert check.seed == expected_seed == stable_hash(-7, item.image_id, check.i, check.j)
        expected_epsilon = torch.randn(latent.shape, generator=torch.Generator(device="cpu").manual_seed(expected_seed))
        torch.testing.assert_close(check.epsilon, expected_epsilon, rtol=0, atol=0)
        torch.testing.assert_close(check.noisy_latent, (1 - expected_t) * latent + expected_t * expected_epsilon)


def test_check_count_and_same_noise_across_roles_and_repeated_events(tmp_path):
    config_path, _ = make_two_role_validation_files(tmp_path, same_image=True)
    manifest = _prepare(config_path)
    loader = make_validation_loader(manifest)
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == 1
    assert loader.num_workers == 0
    assert len(list(loader)) == 2
    latent = torch.zeros((2, 1, 8, 8))
    observed = [
        list(iter_noise_checks(item, latent, val_seed_noise=42, val_level_noise_n=4, val_seed_noise_n=3))
        for item in make_validation_loader(manifest)
    ]
    assert [len(checks) for checks in observed] == [12, 12]
    for familiar, unfamiliar in zip(*observed):
        assert (familiar.i, familiar.j, familiar.seed) == (unfamiliar.i, unfamiliar.j, unfamiliar.seed)
        torch.testing.assert_close(familiar.epsilon, unfamiliar.epsilon, rtol=0, atol=0)
    repeated = list(iter_noise_checks(manifest.items_by_role["val_familiar"][0], latent, val_seed_noise=42, val_level_noise_n=4, val_seed_noise_n=3))
    assert [check.seed for check in repeated] == [check.seed for check in observed[0]]
    sections = config_path.read_text(encoding="utf-8").split("[[datasets]]")
    assert len(sections) == 3
    config_path.write_text(sections[0] + "[[datasets]]" + sections[2] + "[[datasets]]" + sections[1], encoding="utf-8")
    reordered = _prepare(config_path)
    reordered_checks = list(
        iter_noise_checks(
            reordered.items_by_role["val_familiar"][0],
            latent,
            val_seed_noise=42,
            val_level_noise_n=4,
            val_seed_noise_n=3,
        )
    )
    assert [check.seed for check in reordered_checks] == [check.seed for check in observed[0]]


def _rng_snapshot():
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (numpy_state[0], numpy_state[1].copy(), *numpy_state[2:]),
        torch.get_rng_state().clone(),
        [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    )


def _assert_rng_unchanged(before, after):
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])
    if before[3] is None:
        assert after[3] is None
    else:
        assert after[3] is not None and len(before[3]) == len(after[3])
        assert all(torch.equal(a, b) for a, b in zip(before[3], after[3]))


def test_manifest_loader_iteration_and_noise_preserve_training_rng(tmp_path):
    config_path, _ = make_two_role_validation_files(tmp_path)
    before = _rng_snapshot()
    manifest = _prepare(config_path)
    loader = make_validation_loader(manifest)
    records = list(loader)
    latent = torch.zeros((2, 1, 8, 8))
    for item in records:
        list(iter_noise_checks(item, latent, val_seed_noise=42, val_level_noise_n=2, val_seed_noise_n=1))
    list(iter_validation_checks(manifest, val_seed_noise=42, val_level_noise_n=2, val_seed_noise_n=1))
    after = _rng_snapshot()
    _assert_rng_unchanged(before, after)


def test_epsilon_is_generated_one_check_at_a_time(tmp_path, monkeypatch):
    config_path, _ = make_two_role_validation_files(tmp_path)
    item = _prepare(config_path).items[0]
    real_randn = torch.randn
    calls = []

    def counted_randn(*args, **kwargs):
        calls.append(args)
        return real_randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", counted_randn)
    checks = iter_noise_checks(item, torch.zeros((2, 1, 8, 8)), val_seed_noise=42, val_level_noise_n=4, val_seed_noise_n=2)
    assert calls == []
    next(checks)
    assert len(calls) == 1
    next(checks)
    assert len(calls) == 2


def test_bfloat16_latent_uses_float32_cpu_noise_then_cast(tmp_path):
    config_path, _ = make_two_role_validation_files(tmp_path)
    item = _prepare(config_path).items[0]
    latent = torch.ones((2, 1, 8, 8), dtype=torch.bfloat16)
    check = next(iter_noise_checks(item, latent, val_seed_noise=42, val_level_noise_n=2, val_seed_noise_n=1))
    reference = torch.randn(
        latent.shape,
        dtype=torch.float32,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(_reference_seed(42, item.image_id, 1, 1)),
    ).to(dtype=torch.bfloat16)
    assert check.epsilon.dtype == torch.bfloat16
    torch.testing.assert_close(check.epsilon, reference, rtol=0, atol=0)
    torch.testing.assert_close(check.noisy_latent, (1 - check.t) * latent + check.t * reference, rtol=0, atol=0)


def test_validation_checks_read_one_cache_pair_then_yield(tmp_path, monkeypatch):
    from musubi_tuner.training import validation_inputs

    config_path, _ = make_two_role_validation_files(tmp_path)
    manifest = _prepare(config_path)
    real_read = validation_inputs.read_validation_cache_pair
    reads = []

    def counted_read(item):
        reads.append(item.image_id)
        return real_read(item)

    monkeypatch.setattr(validation_inputs, "read_validation_cache_pair", counted_read)
    checks = iter_validation_checks(manifest, val_seed_noise=42, val_level_noise_n=2, val_seed_noise_n=2)
    assert reads == []
    first = next(checks)
    assert len(reads) == 1
    assert first.latent.shape == (2, 1, 8, 8)
    assert first.vl_embed.shape == (3, 4)
    assert first.noisy_latent.shape == first.latent.shape
    for _ in range(3):
        next(checks)
    assert len(reads) == 1
    next(checks)
    assert len(reads) == 2


def test_manifest_preflight_rejects_unusable_resolution_before_cache_read(tmp_path):
    config_path, _ = make_two_role_validation_files(tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("resolution = [64, 64]", "resolution = [8, 8]"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"resolution.*(at least 16|increase)"):
        _prepare(config_path)


def test_manifest_preflight_rejects_cache_directory_file(tmp_path):
    config_path, sources = make_two_role_validation_files(tmp_path)
    unavailable = tmp_path / "cache-is-a-file"
    unavailable.write_bytes(b"not a directory")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            sources["val_familiar"]["cache_dir"].as_posix(), unavailable.as_posix()
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"cache_directory.*is a file"):
        _prepare(config_path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fingerprint_matches_independent_canonical_records_and_survives_move_and_reorder(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    config_path, sources = make_two_role_validation_files(original)
    controls = dict(val_every_n_steps=7, val_seed_noise=-3, val_level_noise_n=4, val_seed_noise_n=2)
    manifest = _prepare(config_path, **controls)
    expected_sets = {}
    for role, source in sources.items():
        expected_sets[role] = [{
            "image_id": _file_sha256(source["image"]),
            "caption_sha256": _file_sha256(source["caption"]),
            "caption": source["caption"].read_text(encoding="utf-8").strip(),
            "original_size": [64, 64],
            "resolution": [64, 64],
            "enable_bucket": True,
            "bucket_no_upscale": True,
            "bucket_size": [64, 64],
            "latent_sha256": _file_sha256(source["latent_cache"]),
            "text_sha256": _file_sha256(source["text_cache"]),
        }]
    identity = {"version": "qwen-image-validation-input-v1", "protocol": controls, "sets": expected_sets}
    expected = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    assert manifest.fingerprint == expected
    with pytest.raises(TypeError):
        manifest.items_by_role["val_familiar"] = ()

    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    moved_config = moved / config_path.name
    moved_config.write_text(
        moved_config.read_text(encoding="utf-8").replace(original.as_posix(), moved.as_posix()), encoding="utf-8"
    )
    assert _prepare(moved_config, expected_fingerprint=expected, **controls).fingerprint == expected
    sections = moved_config.read_text(encoding="utf-8").split("[[datasets]]")
    moved_config.write_text(sections[0] + "[[datasets]]" + sections[2] + "[[datasets]]" + sections[1], encoding="utf-8")
    assert _prepare(moved_config, **controls).fingerprint == expected


def _replace_cache_metadata(path: Path, **new_metadata):
    with safe_open(str(path), framework="pt") as cache:
        metadata = dict(cache.metadata())
        tensors = {key: cache.get_tensor(key).clone() for key in cache.keys()}
    metadata.update(new_metadata)
    save_file(tensors, str(path), metadata=metadata)


@pytest.mark.parametrize("change", ["image", "caption", "cache", "resolution", "controls"])
def test_fingerprint_changes_with_frozen_input_or_protocol(tmp_path, change):
    config_path, sources = make_two_role_validation_files(tmp_path)
    original = _prepare(config_path).fingerprint
    familiar = sources["val_familiar"]
    if change == "image":
        make_validation_item(tmp_path, "val_familiar", color=(2, 3, 4), caption="familiar caption")
    elif change == "caption":
        make_validation_item(tmp_path, "val_familiar", color=(255, 0, 0), caption="revised caption")
    elif change == "cache":
        _replace_cache_metadata(familiar["latent_cache"], extra="changed")
    elif change == "resolution":
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace("resolution = [64, 64]", "resolution = [128, 128]"),
            encoding="utf-8",
        )
        assert _prepare(config_path).items_by_role["val_familiar"][0].bucket_size == (64, 64)
    else:
        assert _prepare(config_path, val_seed_noise=43).fingerprint != original
        return
    assert _prepare(config_path).fingerprint != original


def test_jsonl_occurrences_are_preserved_and_invalid_caption_is_rejected(tmp_path):
    config_path, sources = make_two_role_validation_files(tmp_path)
    familiar = sources["val_familiar"]
    jsonl = tmp_path / "familiar.jsonl"
    record = {"image_path": str(familiar["image"]), "caption": "familiar caption"}
    jsonl.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            f'image_directory = {json.dumps(familiar["image"].parent.as_posix())}',
            f'image_jsonl_file = {json.dumps(jsonl.as_posix())}',
        ),
        encoding="utf-8",
    )
    manifest = _prepare(config_path)
    assert len(manifest.items_by_role["val_familiar"]) == 2
    assert manifest.items_by_role["val_familiar"][0].image_id == manifest.items_by_role["val_familiar"][1].image_id
    duplicate_fingerprint = manifest.fingerprint
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert _prepare(config_path).fingerprint != duplicate_fingerprint
    jsonl.write_text(json.dumps({"image_path": str(familiar["image"]), "caption": None}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"caption"):
        _prepare(config_path)


@pytest.mark.parametrize("change", ["image", "caption", "cache", "config", "removed_cache"])
def test_later_read_rejects_changed_or_removed_frozen_input(tmp_path, change):
    config_path, sources = make_two_role_validation_files(tmp_path)
    manifest = _prepare(config_path)
    frozen_fingerprint = manifest.fingerprint
    item = manifest.items_by_role["val_familiar"][0]
    familiar = sources["val_familiar"]
    if change == "image":
        Image.new("RGB", (64, 64), (3, 4, 5)).save(familiar["image"])
    elif change == "caption":
        familiar["caption"].write_text("changed caption\n", encoding="utf-8")
    elif change == "cache":
        _replace_cache_metadata(familiar["text_cache"], extra="changed")
    elif change == "config":
        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace("resolution = [64, 64]", "resolution = [128, 128]"),
            encoding="utf-8",
        )
    else:
        familiar["latent_cache"].unlink()
    with pytest.raises(ValueError, match=r"val_familiar.*(changed|missing|restore)"):
        read_validation_cache_pair(item)
    assert manifest.fingerprint == frozen_fingerprint


def test_comment_only_config_change_does_not_change_frozen_input(tmp_path):
    config_path, _ = make_two_role_validation_files(tmp_path)
    item = _prepare(config_path).items[0]
    config_path.write_text(config_path.read_text(encoding="utf-8") + "# comment only\n", encoding="utf-8")
    latent, text = read_validation_cache_pair(item)
    assert latent.shape == (2, 1, 8, 8)
    assert text.shape == (3, 4)


def test_later_read_accepts_equivalent_source_and_cache_paths_but_rejects_resolution_change(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path, sources = make_two_role_validation_files(data_dir)
    monkeypatch.chdir(tmp_path)
    config = config_path.read_text(encoding="utf-8").replace(tmp_path.as_posix() + "/", "")
    config_path.write_text(config, encoding="utf-8")
    item = _prepare(config_path).items_by_role["val_familiar"][0]
    for source in sources.values():
        for directory in (source["image"].parent, source["cache_dir"]):
            relative = directory.relative_to(tmp_path).as_posix()
            config = config.replace(f'"{relative}"', f'"data/./{directory.relative_to(data_dir).as_posix()}"')
    config_path.write_text(config, encoding="utf-8")
    latent, text = read_validation_cache_pair(item)
    assert latent.shape == (2, 1, 8, 8)
    assert text.shape == (3, 4)

    config_path.write_text(config.replace("resolution = [64, 64]", "resolution = [128, 128]"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"val_familiar.*config changed"):
        read_validation_cache_pair(item)


def test_later_read_rejects_new_directory_member(tmp_path):
    config_path, _ = make_two_role_validation_files(tmp_path)
    item = _prepare(config_path).items_by_role["val_familiar"][0]
    Image.new("RGB", (64, 64), (1, 2, 3)).save(item.image_path.parent / "added.png")
    (item.image_path.parent / "added.txt").write_text("new item\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"val_familiar.*(membership|changed)"):
        read_validation_cache_pair(item)


def test_expected_fingerprint_rejects_changed_valid_input_without_mutating_snapshot(tmp_path):
    config_path, _ = make_two_role_validation_files(tmp_path)
    frozen = _prepare(config_path)
    assert _prepare(config_path, expected_fingerprint=frozen.fingerprint).fingerprint == frozen.fingerprint
    make_validation_item(tmp_path, "val_familiar", color=(255, 0, 0), caption="new caption")
    with pytest.raises(ValueError, match=r"fingerprint.*changed"):
        _prepare(config_path, expected_fingerprint=frozen.fingerprint)
    assert frozen.fingerprint != _prepare(config_path).fingerprint

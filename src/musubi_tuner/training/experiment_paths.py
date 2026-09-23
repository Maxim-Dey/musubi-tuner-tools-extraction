"""Path rules shared by the opt-in Qwen-Image experiment commands."""

from pathlib import Path


def resolve_experiment_root(experiment_dir: str, selected_train_config_path: str | None = None) -> Path:
    root = Path(experiment_dir).expanduser()
    if not root.is_absolute():
        if not selected_train_config_path:
            raise ValueError(
                "experiment_dir: a relative path requires --config_file/--train_config; "
                "select <root>/train.toml or provide an absolute experiment_dir"
            )
        root = Path(selected_train_config_path).resolve().parent / root
    return root.resolve()


def rebase_path(value: str | None, root: Path) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else root / path).resolve())

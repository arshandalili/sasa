"""Shared helpers: cache setup, SAE-class registration, small IO utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from paths import CACHE_ROOT, CHECKPOINTS, RESULTS

GPT2_MODEL_NAME = "gpt2"
GPT2_D_IN = 768
TOKENIZED_OPENWEBTEXT_DATASET = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
DEFAULT_CONTEXT_SIZE = 128
DEFAULT_TRAIN_BATCH_SIZE_TOKENS = 4096

ARTIFACTS = RESULTS / "artifacts"
REPORTS = RESULTS / "reports"


class _Paths:
    artifacts_root = ARTIFACTS
    reports_root = REPORTS
    rank_analysis_root = ARTIFACTS / "rank_analysis"
    autointerp_root = ARTIFACTS / "autointerp"
    checkpoints_root = CHECKPOINTS
    trained_saes_root = CHECKPOINTS
    cache_root = CACHE_ROOT


PATHS = _Paths()


def configure_runtime_environment(cache_root: Path | None = None) -> None:
    cache_root = cache_root or CACHE_ROOT
    wandb_root = cache_root / "wandb"
    tmp_root = cache_root / "tmp"
    for directory in (cache_root, wandb_root, tmp_root):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_root / "hf")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "hf_datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "hf_transformers")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hf_hub")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["WANDB_DIR"] = str(wandb_root)
    os.environ["WANDB_CACHE_DIR"] = str(wandb_root / "cache")
    os.environ["WANDB_CONFIG_DIR"] = str(wandb_root / "config")
    os.environ["WANDB_DATA_DIR"] = str(wandb_root / "data")
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(tmp_root)
    for name in ("hf", "hf_datasets", "hf_transformers", "hf_hub"):
        (cache_root / name).mkdir(parents=True, exist_ok=True)
    for name in ("cache", "config", "data"):
        (wandb_root / name).mkdir(parents=True, exist_ok=True)


def ensure_workspace_directories() -> None:
    for path in (ARTIFACTS, REPORTS, CHECKPOINTS, PATHS.rank_analysis_root,
                 PATHS.autointerp_root):
        path.mkdir(parents=True, exist_ok=True)


def resolve_device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def slugify_label(text: str) -> str:
    return text.replace("/", "_").replace(" ", "_")


def get_cfg_attr(cfg: Any, name: str, default: Any = None) -> Any:
    value = getattr(cfg, name, None)
    if value is not None:
        return value
    metadata = getattr(cfg, "metadata", None)
    if metadata is None:
        return default
    value = getattr(metadata, name, None)
    if value is not None:
        return value
    if isinstance(metadata, dict):
        return metadata.get(name, default)
    return default


def infer_hook_name(obj: Any, default: str = "blocks.7.hook_resid_pre") -> str:
    return str(get_cfg_attr(getattr(obj, "cfg", obj), "hook_name", default))


def infer_model_name(obj: Any, default: str = "gpt2") -> str:
    model_name = get_cfg_attr(getattr(obj, "cfg", obj), "model_name", default)
    if isinstance(model_name, str) and model_name.startswith("gpt2"):
        return "gpt2"
    return str(model_name)


def register_topk_sasa_classes() -> None:
    from sae_lens import register_sae_class, register_sae_training_class

    from sasa.model import (
        TopKSASA,
        TopKSASAConfig,
        TopKSASAInference,
        TopKSASAInferenceConfig,
    )

    for register, name, cls, cfg in (
        (register_sae_training_class, "topk_sasa", TopKSASA, TopKSASAConfig),
        (register_sae_class, "topk_sasa", TopKSASAInference, TopKSASAInferenceConfig),
    ):
        try:
            register(name, cls, cfg)
        except ValueError:
            pass


def resolve_sae_dir(name_or_path: str | Path) -> tuple[str, Path]:
    candidate = Path(name_or_path)
    if candidate.exists():
        resolved = candidate.resolve()
        return resolved.name, resolved

    candidate = CHECKPOINTS / str(name_or_path)
    if candidate.exists():
        return str(name_or_path), candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve SAE target '{name_or_path}'. "
        f"Pass a path, or a directory name under {CHECKPOINTS}."
    )

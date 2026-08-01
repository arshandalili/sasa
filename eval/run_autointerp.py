from __future__ import annotations

import sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import os
from pathlib import Path
from typing import Any

from common import (
    configure_runtime_environment,
    ensure_workspace_directories,
    register_topk_sasa_classes,
    resolve_device,
    slugify_label,
    write_json,
)
from common import GPT2_MODEL_NAME, PATHS
from common import resolve_sae_dir

configure_runtime_environment()
ensure_workspace_directories()

import torch
from sae_lens import SAE


def _require_local_sae_dir(sae_dir: Path, *, role: str) -> None:
    cfg_path = sae_dir / "cfg.json"
    weights_path = sae_dir / "sae_weights.safetensors"
    if cfg_path.exists() and weights_path.exists():
        return
    raise FileNotFoundError(
        f"{role} SAE checkpoint is missing a saved SAE at {sae_dir}. "
        f"Expected both {cfg_path.name} and {weights_path.name}."
    )


def resolve_target(target: str | None, sae_dir: str | None, label: str | None) -> tuple[str, Path]:
    if sae_dir:
        resolved = Path(sae_dir).resolve()
        return slugify_label(label or resolved.name), resolved
    if not target:
        raise ValueError("Provide either --target or --sae-dir.")
    default_label, resolved = resolve_sae_dir(target)
    return slugify_label(label or default_label), resolved


def resolve_api_key(api_key_file: str | None) -> str:
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    if api_key_file:
        return Path(api_key_file).read_text().strip()
    raise ValueError("Provide OPENAI_API_KEY or --api-key-file.")


def _load_local_autointerp_sae(
    *,
    sae_dir: Path,
    device: str,
    feature_mode: str,
) -> Any:
    from eval._common import SAEBenchAdapter, _canonicalize_model_name_for_saebench

    register_topk_sasa_classes()
    _require_local_sae_dir(sae_dir, role=f"AutoInterp {feature_mode}")
    sae_raw = SAE.load_from_disk(str(sae_dir), device=device)
    with torch.no_grad():
        sae_raw.W_dec.data = sae_raw.W_dec.data / sae_raw.W_dec.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    _canonicalize_model_name_for_saebench(sae_raw)

    adapter_kwargs: dict[str, Any] = {"match_token_norm": False}
    n_groups = getattr(sae_raw, "n_groups", None) or getattr(getattr(sae_raw, "cfg", None), "n_groups", None)
    group_rank = getattr(sae_raw, "group_rank", None) or getattr(
        getattr(sae_raw, "cfg", None), "group_rank", None
    )
    group_sizes = getattr(sae_raw, "group_sizes", None) or getattr(getattr(sae_raw, "cfg", None), "group_sizes", None)
    if n_groups is not None:
        adapter_kwargs["n_groups"] = int(n_groups)
    if group_rank is not None:
        adapter_kwargs["group_rank"] = int(group_rank)
    if group_sizes is not None:
        adapter_kwargs["group_sizes"] = [int(value) for value in group_sizes]
    if feature_mode == "group_norm":
        if group_sizes is None and (n_groups is None or group_rank is None):
            raise ValueError(
                "group_norm view requires grouped metadata: either n_groups/group_rank or explicit group_sizes."
            )
        adapter_kwargs["use_group_norm"] = True

    return SAEBenchAdapter(sae_raw, **adapter_kwargs)


def _build_panel_entries(
    *,
    sasa_target: str,
    relu_target: str,
    jumprelu_target: str,
    include_topk_group_norm: bool,
    include_topk_atoms: bool,
) -> list[dict[str, Any]]:
    sasa_label, sasa_dir = resolve_target(sasa_target, None, None)
    relu_label, relu_dir = resolve_target(relu_target, None, None)
    jumprelu_label, jumprelu_dir = resolve_target(jumprelu_target, None, None)

    entries: list[dict[str, Any]] = []
    if include_topk_group_norm:
        entries.append(
            {
                "result_name": "topk_sasa_group_norm",
                "model_key": "topk_sasa",
                "display_name": "TopK-SASA",
                "feature_mode": "group_norm",
                "target_label": sasa_label,
                "sae_dir": str(sasa_dir),
            }
        )
    if include_topk_atoms:
        entries.append(
            {
                "result_name": "topk_sasa_atoms",
                "model_key": "topk_sasa",
                "display_name": "TopK-SASA",
                "feature_mode": "atoms",
                "target_label": sasa_label,
                "sae_dir": str(sasa_dir),
            }
        )

    entries.extend(
        [
            {
                "result_name": "relu_atoms",
                "model_key": "relu",
                "display_name": "ReLU",
                "feature_mode": "atoms",
                "target_label": relu_label,
                "sae_dir": str(relu_dir),
            },
            {
                "result_name": "jumprelu_atoms",
                "model_key": "jumprelu",
                "display_name": "JumpReLU",
                "feature_mode": "atoms",
                "target_label": jumprelu_label,
                "sae_dir": str(jumprelu_dir),
            },
        ]
    )
    return entries


def build_conciseness_panel_entries(
    *,
    model_configs: list[dict[str, Any]],
    include_topk_group_norm: bool = True,
    include_topk_atoms: bool = True,
) -> list[dict[str, Any]]:
    configs_by_key = {str(config["model_key"]): config for config in model_configs}
    required = {"topk_sasa", "relu", "jumprelu"}
    missing = sorted(required.difference(configs_by_key))
    if missing:
        raise ValueError(
            f"Conciseness AutoInterp panel requires model configs for {sorted(required)}; "
            f"missing {missing}."
        )

    topk_config = configs_by_key["topk_sasa"]
    relu_config = configs_by_key["relu"]
    jumprelu_config = configs_by_key["jumprelu"]

    entries: list[dict[str, Any]] = []
    if include_topk_group_norm:
        entries.append(
            {
                "result_name": "topk_sasa_group_norm",
                "model_key": "topk_sasa",
                "display_name": str(topk_config.get("display_name", "TopK-SASA")),
                "feature_mode": "group_norm",
                "target_label": str(topk_config.get("target_label", topk_config.get("eval_label", "topk_sasa"))),
                "sae_dir": str(topk_config["sae_dir"]),
            }
        )
    if include_topk_atoms:
        entries.append(
            {
                "result_name": "topk_sasa_atoms",
                "model_key": "topk_sasa",
                "display_name": str(topk_config.get("display_name", "TopK-SASA")),
                "feature_mode": "atoms",
                "target_label": str(topk_config.get("target_label", topk_config.get("eval_label", "topk_sasa"))),
                "sae_dir": str(topk_config["sae_dir"]),
            }
        )

    for config, result_name, model_key, default_name in (
        (relu_config, "relu_atoms", "relu", "ReLU"),
        (jumprelu_config, "jumprelu_atoms", "jumprelu", "JumpReLU"),
    ):
        entries.append(
            {
                "result_name": result_name,
                "model_key": model_key,
                "display_name": str(config.get("display_name", default_name)),
                "feature_mode": "atoms",
                "target_label": str(config.get("target_label", config.get("eval_label", model_key))),
                "sae_dir": str(config["sae_dir"]),
            }
        )
    return entries


def _build_selected_saes_for_panel(
    entries: list[dict[str, Any]],
    *,
    device: str,
) -> list[tuple[str, Any]]:
    selected_saes: list[tuple[str, Any]] = []
    for entry in entries:
        sae = _load_local_autointerp_sae(
            sae_dir=Path(entry["sae_dir"]),
            device=device,
            feature_mode=str(entry["feature_mode"]),
        )
        selected_saes.append((str(entry["result_name"]), sae))
    return selected_saes


def _compact_summary_from_results(
    *,
    label: str,
    entries: list[dict[str, Any]],
    n_latents: int,
    total_tokens: int,
    random_seed: int,
    results: dict[str, Any],
) -> dict[str, Any]:
    entry_by_result_name = {str(entry["result_name"]): entry for entry in entries}
    compact_summary = {
        "label": label,
        "n_latents": n_latents,
        "total_tokens": total_tokens,
        "random_seed": random_seed,
        "results": {},
    }

    for sae_name, payload in results.items():
        metrics = payload.get("eval_result_metrics", {}).get("autointerp", {})
        result_name = sae_name.removesuffix("_custom_sae")
        entry = entry_by_result_name.get(result_name, {})
        compact_summary["results"][sae_name] = {
            "result_name": result_name,
            "model_key": entry.get("model_key"),
            "display_name": entry.get("display_name"),
            "feature_mode": entry.get("feature_mode"),
            "target_label": entry.get("target_label"),
            "autointerp_score": metrics.get("autointerp_score"),
            "autointerp_std_dev": metrics.get("autointerp_std_dev"),
        }
    return compact_summary


def run_autointerp_panel(
    *,
    label: str,
    entries: list[dict[str, Any]],
    device: str,
    api_key: str,
    n_latents: int,
    total_tokens: int,
    llm_batch_size: int,
    llm_dtype: str,
    random_seed: int,
    force_rerun: bool,
) -> Path:
    from sae_bench.evals.autointerp.eval_config import AutoInterpEvalConfig
    from sae_bench.evals.autointerp.main import run_eval

    output_dir = PATHS.autointerp_root / slugify_label(label)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENAI_API_KEY"] = api_key

    selected_saes = _build_selected_saes_for_panel(entries, device=device)
    config = AutoInterpEvalConfig(
        model_name=GPT2_MODEL_NAME,
        n_latents=n_latents,
        llm_dtype=llm_dtype,
        llm_batch_size=llm_batch_size,
        llm_context_size=128,
        total_tokens=total_tokens,
        random_seed=random_seed,
    )

    results = run_eval(
        config=config,
        selected_saes=selected_saes,
        device=device,
        api_key=api_key,
        output_path=str(output_dir),
        force_rerun=force_rerun,
        save_logs_path=str(output_dir / "autointerp_logs.txt"),
        artifacts_path=str(PATHS.artifacts_root / "autointerp_cache"),
    )

    compact_summary = _compact_summary_from_results(
        label=slugify_label(label),
        entries=entries,
        n_latents=n_latents,
        total_tokens=total_tokens,
        random_seed=random_seed,
        results=results,
    )
    write_json(output_dir / "summary.json", compact_summary)
    return output_dir


def run_single_target(
    *,
    label: str,
    sae_dir: Path,
    device: str,
    api_key: str,
    feature_mode: str,
    n_latents: int,
    total_tokens: int,
    llm_batch_size: int,
    llm_dtype: str,
    random_seed: int,
    force_rerun: bool,
) -> Path:
    entries = [
        {
            "result_name": f"{label}_{feature_mode}",
            "model_key": "topk_sasa",
            "display_name": label,
            "feature_mode": feature_mode,
            "target_label": label,
            "sae_dir": str(sae_dir),
        }
    ]
    return run_autointerp_panel(
        label=f"{label}__{feature_mode}",
        entries=entries,
        device=device,
        api_key=api_key,
        n_latents=n_latents,
        total_tokens=total_tokens,
        llm_batch_size=llm_batch_size,
        llm_dtype=llm_dtype,
        random_seed=random_seed,
        force_rerun=force_rerun,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAEBench AutoInterp for GPT-2 SAEs, including the matched-budget panel."
    )
    parser.add_argument("--mode", choices=["single", "panel"], default="single")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--sae-dir", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--api-key-file", type=str, default=None)
    parser.add_argument("--feature-mode", type=str, choices=["atoms", "group_norm"], default="group_norm")
    parser.add_argument("--n-latents", type=int, default=256)
    parser.add_argument("--total-tokens", type=int, default=500_000)
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--llm-dtype", type=str, default="float32")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--force-rerun", action="store_true")

    parser.add_argument("--panel-label", type=str, default="matched_budget_autointerp_300m")
    parser.add_argument(
        "--sasa-target",
        type=str,
        default="matched_budget_topk_sasa_layer7_r6_300m",
    )
    parser.add_argument(
        "--relu-target",
        type=str,
        default="matched_budget_relu_layer7_300m",
    )
    parser.add_argument(
        "--jumprelu-target",
        type=str,
        default="matched_budget_jumprelu_layer7_300m",
    )
    parser.add_argument("--skip-topk-group-norm", action="store_true")
    parser.add_argument("--skip-topk-atoms", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    api_key = resolve_api_key(args.api_key_file)
    register_topk_sasa_classes()
    torch.set_grad_enabled(False)

    if args.mode == "panel":
        entries = _build_panel_entries(
            sasa_target=args.sasa_target,
            relu_target=args.relu_target,
            jumprelu_target=args.jumprelu_target,
            include_topk_group_norm=not args.skip_topk_group_norm,
            include_topk_atoms=not args.skip_topk_atoms,
        )
        out_dir = run_autointerp_panel(
            label=args.panel_label,
            entries=entries,
            device=device,
            api_key=api_key,
            n_latents=args.n_latents,
            total_tokens=args.total_tokens,
            llm_batch_size=args.llm_batch_size,
            llm_dtype=args.llm_dtype,
            random_seed=args.random_seed,
            force_rerun=args.force_rerun,
        )
        print(f"Wrote AutoInterp outputs to {out_dir}")
        return

    label, sae_dir = resolve_target(args.target, args.sae_dir, args.label)
    out_dir = run_single_target(
        label=label,
        sae_dir=sae_dir,
        device=device,
        api_key=api_key,
        feature_mode=args.feature_mode,
        n_latents=args.n_latents,
        total_tokens=args.total_tokens,
        llm_batch_size=args.llm_batch_size,
        llm_dtype=args.llm_dtype,
        random_seed=args.random_seed,
        force_rerun=args.force_rerun,
    )
    print(f"Wrote AutoInterp outputs to {out_dir}")


if __name__ == "__main__":
    main()

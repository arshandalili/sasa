from __future__ import annotations

import sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    configure_runtime_environment,
    ensure_workspace_directories,
    infer_hook_name,
    resolve_device,
    write_json,
)
from common import DEFAULT_CONTEXT_SIZE, PATHS, TOKENIZED_OPENWEBTEXT_DATASET
from common import resolve_sae_dir

configure_runtime_environment()
ensure_workspace_directories()

import torch
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer


def resolve_target(target: str | None, sae_dir: str | None, label: str | None) -> tuple[str, Path]:
    if sae_dir:
        resolved = Path(sae_dir).resolve()
        return label or resolved.name, resolved
    if not target:
        raise ValueError("Provide either --target or --sae-dir.")
    default_label, resolved = resolve_sae_dir(target)
    return label or default_label, resolved


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def spectral_metrics_from_values(values: np.ndarray, energy_threshold: float = 0.9) -> dict[str, float]:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0]
    if positive.size == 0:
        return {"rank90": 0.0, "stable_rank": 0.0, "participation_ratio": 0.0}

    total = float(positive.sum())
    descending = np.sort(positive)[::-1]
    cumulative = np.cumsum(descending)
    rank90 = float(np.searchsorted(cumulative, energy_threshold * total, side="left") + 1)
    stable_rank = float(total / max(float(descending[0]), 1e-12))
    participation_ratio = float((total**2) / max(float(np.square(positive).sum()), 1e-12))
    return {
        "rank90": rank90,
        "stable_rank": stable_rank,
        "participation_ratio": participation_ratio,
    }


def iter_context_batches(
    dataset_name: str,
    *,
    context_size: int,
    batch_size_prompts: int,
    total_tokens: int,
) -> tuple[int, torch.Tensor]:
    stream = load_dataset(dataset_name, split="train", streaming=True)
    batch: list[list[int]] = []
    tokens_seen = 0

    for sample in stream:
        token_ids = sample.get("input_ids")
        if not isinstance(token_ids, list):
            continue

        for start in range(0, len(token_ids) - context_size + 1, context_size):
            chunk = token_ids[start : start + context_size]
            if len(chunk) != context_size:
                continue
            batch.append(chunk)
            tokens_seen += context_size
            if len(batch) == batch_size_prompts:
                yield tokens_seen, torch.tensor(batch, dtype=torch.long)
                batch = []
            if tokens_seen >= total_tokens:
                break
        if tokens_seen >= total_tokens:
            break

    if batch:
        yield tokens_seen, torch.tensor(batch, dtype=torch.long)


def analyze_decoder_geometry(sae: SAE) -> list[dict[str, Any]]:
    n_groups = int(getattr(sae, "n_groups"))
    group_rank = int(getattr(sae, "group_rank"))
    w_dec = sae.W_dec.detach().float().cpu().numpy().reshape(n_groups, group_rank, -1)

    rows: list[dict[str, Any]] = []
    for group_idx in range(n_groups):
        singular_values = np.linalg.svd(w_dec[group_idx], compute_uv=False)
        metrics = spectral_metrics_from_values(np.square(singular_values))
        rows.append(
            {
                "group": group_idx,
                "decoder_rank90": metrics["rank90"],
                "decoder_stable_rank": metrics["stable_rank"],
                "decoder_participation_ratio": metrics["participation_ratio"],
            }
        )
    return rows


def analyze_activation_geometry(
    sae: SAE,
    *,
    model: HookedTransformer,
    hook_name: str,
    device: str,
    dataset_name: str,
    total_tokens: int,
    context_size: int,
    batch_size_prompts: int,
    active_threshold: float,
    min_alive_hits: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_groups = int(getattr(sae, "n_groups"))
    group_rank = int(getattr(sae, "group_rank"))

    group_sum = np.zeros((n_groups, group_rank), dtype=np.float64)
    group_second_moment = np.zeros((n_groups, group_rank, group_rank), dtype=np.float64)
    group_active_counts = np.zeros(n_groups, dtype=np.int64)
    atom_active_counts = np.zeros((n_groups, group_rank), dtype=np.int64)
    tokens_processed = 0

    for tokens_seen, batch_tokens in iter_context_batches(
        dataset_name,
        context_size=context_size,
        batch_size_prompts=batch_size_prompts,
        total_tokens=total_tokens,
    ):
        input_ids = batch_tokens.to(device)
        with torch.inference_mode():
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
            hidden = cache[hook_name]
            acts = sae.encode(hidden)

        groups = acts.reshape(-1, n_groups, group_rank)
        abs_groups = groups.abs()
        group_norms = abs_groups.norm(dim=-1)
        active_mask = group_norms > active_threshold
        masked_groups = groups * active_mask.unsqueeze(-1)

        group_sum += masked_groups.sum(dim=0).double().cpu().numpy()
        group_second_moment += torch.einsum(
            "tgr,tgs->grs",
            masked_groups.double(),
            masked_groups.double(),
        ).cpu().numpy()
        group_active_counts += active_mask.sum(dim=0).cpu().numpy()
        atom_active_counts += (abs_groups > active_threshold).sum(dim=0).cpu().numpy()
        tokens_processed = max(tokens_processed, tokens_seen)

    rows: list[dict[str, Any]] = []
    for group_idx in range(n_groups):
        active_count = int(group_active_counts[group_idx])
        if active_count >= 2:
            mean = group_sum[group_idx] / active_count
            covariance = group_second_moment[group_idx] / active_count - np.outer(mean, mean)
            eigenvalues = np.linalg.eigvalsh(covariance)
            eigenvalues = np.clip(eigenvalues, 0.0, None)
            spectral = spectral_metrics_from_values(eigenvalues)
        else:
            spectral = {"rank90": 0.0, "stable_rank": 0.0, "participation_ratio": 0.0}

        rows.append(
            {
                "group": group_idx,
                "activation_rank90": spectral["rank90"],
                "activation_stable_rank": spectral["stable_rank"],
                "activation_participation_ratio": spectral["participation_ratio"],
                "group_active_tokens": active_count,
                "group_active_fraction": active_count / max(tokens_processed, 1),
                "alive_atoms": int((atom_active_counts[group_idx] >= min_alive_hits).sum()),
                "alive_atom_fraction": float(
                    (atom_active_counts[group_idx] >= min_alive_hits).sum() / max(group_rank, 1)
                ),
            }
        )

    summary = {
        "tokens_processed": tokens_processed,
        "alive_group_fraction": float((group_active_counts >= min_alive_hits).mean()),
        "alive_atom_fraction": float((atom_active_counts.reshape(-1) >= min_alive_hits).mean()),
        "median_group_active_fraction": float(np.median(group_active_counts / max(tokens_processed, 1))),
    }
    return rows, summary


def summarize_combined_rows(
    decoder_rows: list[dict[str, Any]],
    activation_rows: list[dict[str, Any]],
    activation_summary: dict[str, Any],
) -> dict[str, Any]:
    by_group = {
        row["group"]: dict(row)
        for row in decoder_rows
    }
    for row in activation_rows:
        by_group[row["group"]].update(row)

    combined_rows = [by_group[key] for key in sorted(by_group)]

    def mean_for(key: str) -> float | None:
        values = [float(row[key]) for row in combined_rows if row.get(key) is not None]
        return (sum(values) / len(values)) if values else None

    return {
        "group_count": len(combined_rows),
        "decoder_rank90_mean": mean_for("decoder_rank90"),
        "decoder_stable_rank_mean": mean_for("decoder_stable_rank"),
        "decoder_participation_ratio_mean": mean_for("decoder_participation_ratio"),
        "activation_rank90_mean": mean_for("activation_rank90"),
        "activation_stable_rank_mean": mean_for("activation_stable_rank"),
        "activation_participation_ratio_mean": mean_for("activation_participation_ratio"),
        **activation_summary,
    }


def run_rank_analysis_for_target(
    *,
    label: str,
    sae_dir: Path,
    device: str,
    dataset: str,
    total_tokens: int,
    context_size: int,
    batch_size_prompts: int,
    active_threshold: float,
    min_alive_hits: int,
) -> Path:
    output_dir = PATHS.rank_analysis_root / label
    output_dir.mkdir(parents=True, exist_ok=True)

    from common import register_topk_sasa_classes

    register_topk_sasa_classes()
    torch.set_grad_enabled(False)

    sae = SAE.load_from_disk(str(sae_dir), device=device)
    sae.eval()
    if not hasattr(sae, "n_groups") or not hasattr(sae, "group_rank"):
        raise ValueError("rank_analysis.py expects a TopK-SASA checkpoint with group structure.")

    hook_name = infer_hook_name(sae)
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()

    decoder_rows = analyze_decoder_geometry(sae)
    activation_rows, activation_summary = analyze_activation_geometry(
        sae,
        model=model,
        hook_name=hook_name,
        device=device,
        dataset_name=dataset,
        total_tokens=total_tokens,
        context_size=context_size,
        batch_size_prompts=batch_size_prompts,
        active_threshold=active_threshold,
        min_alive_hits=min_alive_hits,
    )

    combined_rows = {
        row["group"]: dict(row)
        for row in decoder_rows
    }
    for row in activation_rows:
        combined_rows[row["group"]].update(row)

    group_rows = [combined_rows[group] for group in sorted(combined_rows)]
    summary = summarize_combined_rows(decoder_rows, activation_rows, activation_summary)
    summary.update(
        {
            "label": label,
            "sae_dir": str(sae_dir),
            "hook_name": hook_name,
            "dataset": dataset,
            "total_tokens_requested": total_tokens,
            "context_size": context_size,
            "batch_size_prompts": batch_size_prompts,
        }
    )

    write_csv(output_dir / "group_level.csv", group_rows)
    write_json(output_dir / "summary.json", summary)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate effective subspace dimension and alive-feature usage for TopK-SASA groups."
    )
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--sae-dir", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=TOKENIZED_OPENWEBTEXT_DATASET)
    parser.add_argument("--total-tokens", type=int, default=20_000)
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--batch-size-prompts", type=int, default=8)
    parser.add_argument("--active-threshold", type=float, default=1e-6)
    parser.add_argument("--min-alive-hits", type=int, default=10)
    args = parser.parse_args()

    label, sae_dir = resolve_target(args.target, args.sae_dir, args.label)
    device = resolve_device(args.device)
    output_dir = run_rank_analysis_for_target(
        label=label,
        sae_dir=sae_dir,
        device=device,
        dataset=args.dataset,
        total_tokens=args.total_tokens,
        context_size=args.context_size,
        batch_size_prompts=args.batch_size_prompts,
        active_threshold=args.active_threshold,
        min_alive_hits=args.min_alive_hits,
    )
    print(f"Wrote rank-analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()

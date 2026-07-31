"""Run SAEBench Core eval comparing:
  1) Control SAE: SAE Lens pretrained `gpt2-small-res-jb` at a given hook
  2) Local TopK-SASA SAE loaded from disk
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sae_lens import SAE, register_sae_class

import sae_bench.evals.core.main as core_main
import sae_bench.sae_bench_utils.general_utils as sb_general_utils

from sasa import TopKSASAInference, TopKSASAInferenceConfig
from eval._common import (
    SAEBenchAdapter,
    _canonicalize_model_name_for_saebench,
    _postprocess_saebench_core_jsons,
    _set_hf_cache_defaults,
    _write_core_summary,
    _write_extra_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hook",
        type=str,
        default="blocks.7.hook_resid_pre",
        help="Hook name / SAE id to use for the control SAE.",
    )
    parser.add_argument(
        "--topk-sasa-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "checkpoints"
            / "topk_sasa_gpt2_l7_n2048_r6_k10"
        ),
        help="Path to the local TopK-SASA SAE folder (cfg.json + weights).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "saebench_results_gpt2_topk_sasa"),
        help="Where to write SAEBench JSON results.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument(
        "--dataset",
        type=str,
        default="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        help="HF dataset name used by SAEBench core eval.",
    )
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--n-recon-batches", type=int, default=2)
    parser.add_argument("--n-sparsity-batches", type=int, default=10)
    parser.add_argument("--batch-size-prompts", type=int, default=16)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[1]
    _set_hf_cache_defaults(workspace_root)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Control SAE directly through SAE Lens.
    control_sae = SAE.from_pretrained("gpt2-small-res-jb", args.hook, device=device)
    _canonicalize_model_name_for_saebench(control_sae)

    # Local TopK-SASA from disk (requires local registry).
    register_sae_class("topk_sasa", TopKSASAInference, TopKSASAInferenceConfig)
    topk_dir = Path(args.topk_sasa_dir).resolve()
    topk_sae = SAE.load_from_disk(str(topk_dir), device=device)
    _canonicalize_model_name_for_saebench(topk_sae)

    # Wrap SAEs so SAEBench sees unit-normalized decoder weights without changing recon.
    control_adapter = SAEBenchAdapter(control_sae, match_token_norm=False)
    topk_adapter = SAEBenchAdapter(topk_sae, match_token_norm=False)

    selected_saes = [
        (f"control_gpt2-small-res-jb_{args.hook}", control_adapter),
        (f"topk_sasa_local_{args.hook}", topk_adapter),
    ]

    out_dir = Path(args.output_dir).resolve() / "core"
    out_dir.mkdir(parents=True, exist_ok=True)

    core_main.multiple_evals(
        selected_saes=selected_saes,
        n_eval_reconstruction_batches=args.n_recon_batches,
        n_eval_sparsity_variance_batches=args.n_sparsity_batches,
        eval_batch_size_prompts=args.batch_size_prompts,
        compute_featurewise_density_statistics=True,
        compute_featurewise_weight_based_metrics=True,
        exclude_special_tokens_from_reconstruction=True,
        dataset=args.dataset,
        context_size=args.context_size,
        output_folder=str(out_dir),
        verbose=True,
        dtype=args.dtype,
        device=device,
        force_rerun=args.force_rerun,
    )

    _write_extra_metrics(out_dir, control_adapter=control_adapter, sasa_adapter=topk_adapter)
    _postprocess_saebench_core_jsons(out_dir, control_adapter=control_adapter, sasa_adapter=topk_adapter)
    _write_core_summary(out_dir)
    print(f"\nWrote SAEBench core results to: {out_dir}")


if __name__ == "__main__":
    _ = sb_general_utils.setup_environment()
    main()


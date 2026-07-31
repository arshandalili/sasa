"""Matched-budget GPT-2 layer-7 training"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (  # noqa: E402
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_TRAIN_BATCH_SIZE_TOKENS,
    GPT2_D_IN,
    GPT2_MODEL_NAME,
    TOKENIZED_OPENWEBTEXT_DATASET,
    configure_runtime_environment,
    ensure_workspace_directories,
    register_topk_sasa_classes,
    resolve_device,
    write_json,
)
from paths import CHECKPOINTS, SCRATCH  # noqa: E402

configure_runtime_environment()
ensure_workspace_directories()

from sae_lens import (  # noqa: E402
    BatchTopKTrainingSAEConfig,
    GatedTrainingSAEConfig,
    JumpReLUTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
    MatryoshkaBatchTopKTrainingSAEConfig,
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
)

from sasa.model import TopKSASAConfig  # noqa: E402

HOOK = "blocks.7.hook_resid_pre"
WIDTH = 12_288
TRAINING_TOKENS = 300_000_000
MATRYOSHKA_WIDTHS = [3_072, 6_144, 9_216, WIDTH]

ARCHITECTURES = ("relu", "jumprelu", "gated", "topk", "batchtopk", "matryoshka", "topk_sasa")


def run_name(arch: str, l0: int, group_rank: int, nuclear_coefficient: float) -> str:
    if arch == "topk_sasa":
        return f"topk_sasa_gpt2_l7_r{group_rank}_nuc{nuclear_coefficient:g}_l0{l0}"
    return f"{arch}_gpt2_l7_l0{l0}"


def build_sae_config(arch: str, l0: int, group_rank: int, nuclear_coefficient: float,
                     normalize: str | None):
    shared = {"d_in": GPT2_D_IN, "d_sae": WIDTH, "apply_b_dec_to_input": True}
    scalar_norm = normalize or "none"

    if arch == "relu":
        return StandardTrainingSAEConfig(
            **shared, l1_coefficient=8e-5, normalize_activations=scalar_norm)
    if arch == "gated":
        return GatedTrainingSAEConfig(
            **shared, l1_coefficient=8e-5, l1_warm_up_steps=0,
            normalize_activations=scalar_norm)
    if arch == "jumprelu":
        return JumpReLUTrainingSAEConfig(
            **shared, jumprelu_init_threshold=0.3, jumprelu_bandwidth=0.05,
            l0_coefficient=0.1, l0_warm_up_steps=5_000,
            normalize_activations=scalar_norm)
    if arch == "topk":
        return TopKTrainingSAEConfig(
            **shared, k=int(l0), aux_loss_coefficient=1.0,
            rescale_acts_by_decoder_norm=True, normalize_activations=scalar_norm)
    if arch == "batchtopk":
        return BatchTopKTrainingSAEConfig(
            **shared, k=float(l0), topk_threshold_lr=0.01, aux_loss_coefficient=1.0,
            rescale_acts_by_decoder_norm=True, normalize_activations=scalar_norm)
    if arch == "matryoshka":
        return MatryoshkaBatchTopKTrainingSAEConfig(
            **shared, k=float(l0), topk_threshold_lr=0.01, aux_loss_coefficient=1.0,
            rescale_acts_by_decoder_norm=True, matryoshka_widths=MATRYOSHKA_WIDTHS,
            normalize_activations=scalar_norm)
    if arch == "topk_sasa":
        if l0 % group_rank:
            raise SystemExit(f"l0={l0} must be a multiple of group_rank={group_rank}.")
        if WIDTH % group_rank:
            raise SystemExit(f"width={WIDTH} must be a multiple of group_rank={group_rank}.")
        return TopKSASAConfig(
            **shared, n_groups=WIDTH // group_rank, group_rank=group_rank,
            k_groups=l0 // group_rank, k_aux=512, aux_coefficient=1.0,
            nuclear_coefficient=nuclear_coefficient,
            normalize_activations=normalize or "layer_norm")
    raise SystemExit(f"Unknown architecture: {arch}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCHITECTURES, required=True)
    ap.add_argument("--l0", type=int, default=60,
                    help="Exact per-token l0 for the hard-k arms; ignored by the l1 family.")
    ap.add_argument("--nuclear-coefficient", type=float, default=100.0,
                    help="lambda_dim; topk_sasa only. 0 turns the trace-norm penalty off.")
    ap.add_argument("--group-rank", type=int, default=6,
                    help="topk_sasa only; n_groups is width // group_rank, so width is fixed.")
    ap.add_argument("--normalize-activations", default=None,
                    choices=["none", "expected_average_only_in", "layer_norm"],
                    help="Overrides the per-architecture default (layer_norm for SASA, "
                         "none otherwise). Used for the normalization control.")
    ap.add_argument("--training-tokens", type=int, default=TRAINING_TOKENS)
    ap.add_argument("--lr", type=float, default=None,
                    help="Default 2e-4 for topk_sasa, 4e-4 for the baselines.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--name", default=None, help="Output directory name.")
    ap.add_argument("--use-cached-activations", action="store_true",
                    help=f"Read the stream cached by scripts.cache_activations under {SCRATCH}.")
    ap.add_argument("--cached-activations-path", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="sasa-matched-budget")
    args = ap.parse_args()

    register_topk_sasa_classes()
    device = resolve_device(args.device)
    lr = args.lr if args.lr is not None else (2e-4 if args.arch == "topk_sasa" else 4e-4)
    name = args.name or run_name(args.arch, args.l0, args.group_rank,
                                 args.nuclear_coefficient)
    out_dir = CHECKPOINTS / name

    if out_dir.exists():
        raise SystemExit(f"Refusing to overwrite {out_dir}.")

    cache_path = args.cached_activations_path
    if args.use_cached_activations and cache_path is None:
        cache_path = str(SCRATCH / f"gpt2_l7_{args.training_tokens // 1_000_000}m")

    total_steps = args.training_tokens // DEFAULT_TRAIN_BATCH_SIZE_TOKENS
    cfg = LanguageModelSAERunnerConfig(
        model_name=GPT2_MODEL_NAME,
        hook_name=HOOK,
        dataset_path=TOKENIZED_OPENWEBTEXT_DATASET,
        streaming=True,
        is_dataset_tokenized=True,
        use_cached_activations=cache_path is not None,
        cached_activations_path=cache_path,
        sae=build_sae_config(args.arch, args.l0, args.group_rank,
                             args.nuclear_coefficient, args.normalize_activations),
        lr=lr,
        lr_warm_up_steps=0,
        lr_decay_steps=total_steps // 5,
        adam_beta1=0.9,
        adam_beta2=0.999,
        train_batch_size_tokens=DEFAULT_TRAIN_BATCH_SIZE_TOKENS,
        context_size=DEFAULT_CONTEXT_SIZE,
        n_batches_in_buffer=64,
        training_tokens=args.training_tokens,
        store_batch_size_prompts=16,
        eval_batch_size_prompts=8,
        n_eval_batches=10,
        logger=LoggingConfig(
            log_to_wandb=args.wandb,
            wandb_project=args.wandb_project,
            wandb_log_frequency=30,
            eval_every_n_wandb_logs=20,
        ),
        device=device,
        seed=args.seed,
        n_checkpoints=0,
        checkpoint_path=str(CHECKPOINTS / "_runs" / name),
        output_path=str(CHECKPOINTS / "_runs" / name / "runner_output"),
        dtype=args.dtype,
    )

    print(f"Training {name}  arch={args.arch}  l0={args.l0}  lr={lr}  device={device}")
    sae = LanguageModelSAETrainingRunner(cfg).run()
    sae.save_inference_model(str(out_dir))
    write_json(out_dir / "train_args.json", vars(args) | {"lr": lr, "width": WIDTH,
                                                          "hook": HOOK})
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

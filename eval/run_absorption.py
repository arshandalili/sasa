"""First-letter feature absorption"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import CHECKPOINTS, RESULTS  # noqa: E402
from eval import _common as adapter_mod  # noqa: E402

REPORT_DIR = RESULTS / "absorption"
PROBE_CACHE = REPO_ROOT / "artifacts" / "absorption" / "probes"


def point_saebench_at_repo_probes(probes_dir: Path) -> None:
    """SAEBench threads probes_dir through as a default argument, so redirecting it
    means rebinding those defaults as well as the module constant."""
    import sae_bench.evals.absorption.common as sb_common
    import sae_bench.evals.absorption.feature_absorption as sb_fa
    import sae_bench.evals.absorption.k_sparse_probing as sb_ksp

    old = str(sb_common.PROBES_DIR)
    sb_common.PROBES_DIR = probes_dir
    rebound = 0
    for module in (sb_common, sb_fa, sb_ksp):
        for name in dir(module):
            fn = getattr(module, name, None)
            if not callable(fn):
                continue
            for slot in ("__defaults__", "__kwdefaults__"):
                current = getattr(fn, slot, None)
                if not current:
                    continue
                if slot == "__defaults__":
                    new = tuple(probes_dir if str(d) == old else d for d in current)
                else:
                    new = {k: (probes_dir if str(v) == old else v) for k, v in current.items()}
                if new != current:
                    setattr(fn, slot, new)
                    rebound += 1
    print(f"[probes] using {probes_dir} ({rebound} SAEBench entry points rebound)")


def patch_adapter_encode_to_raw_scale() -> None:
    if getattr(adapter_mod.SAEBenchAdapter, "_raw_scale_patched", False):
        return
    original_encode = adapter_mod.SAEBenchAdapter.encode

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = original_encode(self, x)
        if str(getattr(self.sae.cfg, "normalize_activations", "none")) != "layer_norm":
            return z
        ln_std = getattr(self.sae, "ln_std", None)
        if ln_std is None:
            raise RuntimeError("layer_norm SAE did not expose ln_std after encode.")
        rows = int(torch.tensor(z.shape[:-1]).prod().item())
        if ln_std.numel() != rows:
            raise RuntimeError(
                f"ln_std has {ln_std.numel()} entries but encode returned {rows} rows."
            )
        scale = ln_std.reshape(*z.shape[:-1], 1).to(device=z.device, dtype=z.dtype)
        return z * scale

    adapter_mod.SAEBenchAdapter.encode = encode
    adapter_mod.SAEBenchAdapter._raw_scale_patched = True


def patch_adapter_encode_to_fixed_l0(k: int) -> None:
    previous = getattr(adapter_mod.SAEBenchAdapter, "_fixed_l0_original", None)
    original_encode = previous or adapter_mod.SAEBenchAdapter.encode

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = original_encode(self, x)
        flat = z.reshape(-1, z.shape[-1])
        if k >= flat.shape[-1]:
            return z
        idx = flat.topk(k, dim=-1).indices
        gated = torch.zeros_like(flat)
        gated.scatter_(-1, idx, flat.gather(-1, idx))
        return gated.view_as(z)

    adapter_mod.SAEBenchAdapter.encode = encode
    adapter_mod.SAEBenchAdapter._fixed_l0_original = original_encode


def load_adapter(sae_dir: Path, device: str):
    from sae_lens import SAE, register_sae_class

    from sasa import TopKSASAInference, TopKSASAInferenceConfig

    try:
        register_sae_class("topk_sasa", TopKSASAInference, TopKSASAInferenceConfig)
    except ValueError:
        pass

    try:
        sae = SAE.load_from_disk(str(sae_dir), device=device)
    except KeyError:
        # BatchTopK/Matryoshka save as jumprelu; convert if given a training checkpoint.
        from sae_lens.saes.sae import TrainingSAE

        training_sae = TrainingSAE.load_from_disk(str(sae_dir), device=device)
        converted = Path(tempfile.mkdtemp(prefix="sasa_infer_"))
        training_sae.save_inference_model(str(converted))
        del training_sae
        sae = SAE.load_from_disk(str(converted), device=device)
        print(f"[convert] {sae_dir.name}: training checkpoint -> {sae.cfg.architecture()} inference form")

    adapter_mod._canonicalize_model_name_for_saebench(sae)

    n_groups = getattr(sae, "n_groups", None) or getattr(sae.cfg, "n_groups", None)
    group_rank = getattr(sae, "group_rank", None) or getattr(sae.cfg, "group_rank", None)
    return adapter_mod.SAEBenchAdapter(
        sae,
        match_token_norm=False,
        n_groups=int(n_groups) if n_groups is not None else None,
        group_rank=int(group_rank) if group_rank is not None else None,
    )


def run_one(sae_dir: Path, label: str, device: str, dtype: str, batch_size: int,
            f1_jump: float, max_k: int) -> dict:
    import sae_bench.evals.absorption.main as absorption_main
    from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig

    adapter = load_adapter(sae_dir, device)
    hook = adapter.cfg.hook_name

    out_dir = REPORT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    config = AbsorptionEvalConfig(
        model_name="gpt2", random_seed=42, llm_dtype=dtype, llm_batch_size=batch_size,
        f1_jump_threshold=f1_jump, max_k_value=max_k,
    )
    results = absorption_main.run_eval(
        config=config,
        selected_saes=[(f"{label}_{hook}", adapter)],
        device=device,
        output_path=str(out_dir),
        force_rerun=True,
    )
    gc.collect()
    torch.cuda.empty_cache()

    if not results:
        raise SystemExit(
            "The absorption eval returned no results. SAEBench refuses to score a model "
            "when fewer than `min_feats_for_eval` (20) letters reach ground-truth probe "
            "F1 >= 0.6. Those probes do not depend on the SAE: they are read from "
            "`<sae_bench>/artifacts/absorption/probes/gpt2/`, and retraining them from "
            "scratch on GPT-2 lands near the threshold (we measured 18 of 26). Populate "
            "that probe cache, or lower the gate deliberately and say so when reporting."
        )

    metrics = results[next(iter(results))]["eval_result_metrics"]["mean"]
    row = {
        "label": label,
        "sae": str(sae_dir),
        "mean_absorption_fraction_score": metrics["mean_absorption_fraction_score"],
        "mean_full_absorption_score": metrics["mean_full_absorption_score"],
        "mean_num_split_features": metrics["mean_num_split_features"],
        "std_dev_absorption_fraction_score": metrics["std_dev_absorption_fraction_score"],
    }
    (out_dir / "summary.json").write_text(json.dumps(row, indent=2))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True,
                    help="Checkpoint directory name (under SASA_CHECKPOINTS) or an absolute path.")
    ap.add_argument("--label", required=True, help="Output label; must be unique per run.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--f1-jump", type=float, default=0.03)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--no-patch", action="store_true",
                    help="Run the eval exactly as shipped, for a side-by-side control.")
    ap.add_argument("--force-l0", type=int, default=None,
                    help="Gate the encoder with a per-token top-k so every architecture is "
                         "scored at the same l0 on the same prompts.")
    ap.add_argument("--probes-dir", default=str(PROBE_CACHE),
                    help="Ground-truth probe cache. Defaults to the copy shipped here; "
                         "pass SAEBench's own path to use that instead.")
    args = ap.parse_args()

    adapter_mod._set_hf_cache_defaults(REPO_ROOT)
    probes_dir = Path(args.probes_dir)
    if probes_dir.exists():
        point_saebench_at_repo_probes(probes_dir)
    else:
        print(f"[probes] {probes_dir} not found; SAEBench will use its own cache or retrain")

    if not args.no_patch:
        patch_adapter_encode_to_raw_scale()
        print("[scale fix] encoder output rescaled to raw-activation units")
    else:
        print("[control] eval running as shipped")

    if args.force_l0 is not None:
        patch_adapter_encode_to_fixed_l0(args.force_l0)
        print(f"[fixed l0] per-token top-{args.force_l0} gate applied")

    sae_dir = Path(args.sae)
    if not sae_dir.is_absolute():
        sae_dir = CHECKPOINTS / args.sae

    print(json.dumps(run_one(sae_dir, args.label, args.device, args.dtype,
                             args.batch_size, args.f1_jump, args.max_k), indent=2))


if __name__ == "__main__":
    main()

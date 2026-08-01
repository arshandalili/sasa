"""Rebuild data.npz for the absorption eval from the prompts shipped in this repo.

The array is the residual stream at token position -6 of every prompt in
{train,test}_df.csv, so it is reproducible rather than committed. One GPU pass, a couple
of minutes. Compare against a reference copy with --check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import configure_runtime_environment  # noqa: E402
from scripts.install_probe_cache import DEST, probe_subdir  # noqa: E402

POSITION_IDX = -6


def build_split(model, df: pd.DataFrame, hook: str, batch_size: int) -> np.ndarray:
    import torch

    prompts = df["prompt"].tolist()
    out = np.empty((len(prompts), model.cfg.d_model), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            cache = model.run_with_cache(batch, names_filter=[hook])[1]
            out[i:i + len(batch)] = (
                cache[hook][:, POSITION_IDX, :].to(torch.float32).cpu().numpy()
            )
            if (i // batch_size) % 50 == 0:
                print(f"    {i + len(batch)}/{len(prompts)}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--hook", default="blocks.7.hook_resid_pre")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--check", default=None,
                    help="Path to a reference data.npz to compare against instead of writing.")
    args = ap.parse_args()

    configure_runtime_environment()
    import torch
    from transformer_lens import HookedTransformer

    probe_dir = DEST / probe_subdir(args.model, args.layer)
    if not (probe_dir / "train_df.csv").exists():
        raise SystemExit(f"{probe_dir} has no train_df.csv; nothing to rebuild from.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(
        args.model, device=device, center_writing_weights=False
    )

    arrays = {}
    for split in ("train", "test"):
        df = pd.read_csv(probe_dir / f"{split}_df.csv", keep_default_na=False, na_values=[""])
        print(f"  {split}: {len(df)} prompts")
        arrays[f"X_{split}"] = build_split(model, df, args.hook, args.batch_size)
        arrays[f"y_{split}"] = df["answer_class"].to_numpy(dtype=np.int64)

    if args.check:
        ref = np.load(args.check)
        for key in sorted(arrays):
            a, b = arrays[key], ref[key]
            same = a.shape == b.shape
            err = float(np.abs(a - b).max()) if same else float("nan")
            print(f"  {key:9s} shape_match={same} max_abs_diff={err:.3e}")
        return

    out = probe_dir / "data.npz"
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

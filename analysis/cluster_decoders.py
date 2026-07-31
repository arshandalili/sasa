"""Redundancy ratio of a standard SAE's decoder clusters."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import configure_runtime_environment, resolve_device

configure_runtime_environment()

from sae_lens import SAE  # noqa: E402

DEFAULT_CLUSTERS = REPO_ROOT / "artifacts" / "clusters" / "gpt-2_layer_7_clusters_spectral_n1000.pkl"


def pca_dimension(Xg: np.ndarray, var_threshold: float, center: bool) -> int:
    """Smallest k whose top-k eigenvalues explain var_threshold of the variance."""
    if Xg.shape[0] <= 1:
        return 1
    Xc = Xg - Xg.mean(axis=0, keepdims=True) if center else Xg
    evals = np.clip(np.linalg.eigvalsh(Xc @ Xc.T), 0.0, None)
    total = float(evals.sum())
    if total <= 1e-12:
        return 1
    evals = np.sort(evals)[::-1]
    return int(np.searchsorted(np.cumsum(evals) / total, var_threshold, side="left") + 1)


def plot(sizes, dims, ratios, out_path, title):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax1.scatter(sizes, dims, s=9, alpha=0.45, color="#1f77b4", edgecolors="none")
    lim = np.array([max(sizes.min(), 1), sizes.max()], dtype=float)
    ax1.plot(lim, lim, "k--", lw=1.2, label="pca_dim = size")
    ax1.plot(lim, lim / 2.0, "r--", lw=1.2, label="pca_dim = size/2")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Cluster size")
    ax1.set_ylabel("PCA dim (80% var)")
    ax1.set_title("Cluster Size vs PCA Dim")
    ax1.legend(fontsize=8, loc="upper left")

    ax2.hist(ratios, bins=30, color="#1f77b4", edgecolor="white")
    ax2.axvline(float(np.median(ratios)), color="red", ls="--", lw=1.2,
                label=f"median:{np.median(ratios):.2f}")
    ax2.axvline(1.0, color="black", ls="--", lw=1.2, label="ratio:1.0")
    ax2.set_xlabel("size / pca_dim")
    ax2.set_ylabel("Count")
    ax2.set_title("Redundancy Ratio")
    ax2.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-release", default="gpt2-small-res-jb")
    ap.add_argument("--hook", default="blocks.7.hook_resid_pre")
    ap.add_argument("--clusters", default=str(DEFAULT_CLUSTERS),
                    help="Pickled {cluster_id: [atom indices]} from Engels et al.")
    ap.add_argument("--pca-var", type=float, default=0.8)
    ap.add_argument("--min-size", type=int, default=3,
                    help="Keep clusters strictly larger than this.")
    ap.add_argument("--center", action="store_true",
                    help="Center within each cluster first; the appendix does not.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or REPO_ROOT / "results" / "decoder_clusters")
    out_dir.mkdir(parents=True, exist_ok=True)

    sae = SAE.from_pretrained(args.sae_release, args.hook, device=resolve_device(args.device))
    X = sae.W_dec.detach().cpu().float().numpy().astype(np.float64)
    clusters_in = pickle.loads(Path(args.clusters).read_bytes())
    print(f"{args.sae_release} {args.hook}: {X.shape[0]} atoms, "
          f"{len(clusters_in)} clusters", flush=True)

    rows = []
    for cid, idx in clusters_in.items():
        idx = np.asarray(idx, dtype=int)
        if idx.size <= args.min_size:
            continue
        dim = pca_dimension(X[idx], args.pca_var, args.center)
        rows.append({"cluster_id": int(cid), "size": int(idx.size),
                     "pca_dim": dim, "ratio": float(idx.size / dim)})
    if not rows:
        raise SystemExit("No cluster larger than --min-size.")

    sizes = np.array([r["size"] for r in rows])
    dims = np.array([r["pca_dim"] for r in rows])
    ratios = sizes / dims
    plot(sizes, dims, ratios, out_dir / "redundancy_ratio.pdf",
         f"{args.sae_release} {args.hook}")

    summary = {
        "sae_release": args.sae_release, "hook": args.hook,
        "clusters_file": str(args.clusters), "pca_var": args.pca_var,
        "min_size": args.min_size, "center": args.center,
        "n_atoms": int(X.shape[0]), "n_clusters_kept": len(rows),
        "median_ratio": float(np.median(ratios)), "mean_ratio": float(ratios.mean()),
        "frac_ratio_gt_1": float((ratios > 1.0).mean()),
    }
    (out_dir / "cluster_stats.json").write_text(
        json.dumps({"summary": summary, "clusters": rows}, indent=1))
    print(f"clusters kept (size > {args.min_size}): {len(rows)}")
    print(f"median redundancy ratio: {summary['median_ratio']:.3f}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

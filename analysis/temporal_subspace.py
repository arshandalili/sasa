"""Figure 3: the temporal subspace of one SASA group (GPT-2 layer 7, group 1473).

Three panels, following Appendix F.2.2-F.2.4:

  (a) token activation profiles -- ||a_{t,g}||_2 against token index on three fixed prompts
  (b) the subspace -- PCA of the group coordinates collected at temporal tokens in
      OpenWebText, coloured by temporal category
  (c) cyclic topology -- month centroids mapped to R^2 by a ridge-fit linear map that
      sends month m to (cos(2*pi*m/12), sin(2*pi*m/12)), with season centroids on top

    python -m analysis.temporal_subspace \\
      --sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 --group-id 1473
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import configure_runtime_environment, register_topk_sasa_classes, resolve_device

configure_runtime_environment()

import torch  # noqa: E402
from sae_lens import SAE  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ALIASES = {
    "Monday": ["Mon", "Mondays"], "Tuesday": ["Tue", "Tues", "Tuesdays"],
    "Wednesday": ["Wed", "Weds", "Wednesdays"], "Thursday": ["Thu", "Thur", "Thurs", "Thursdays"],
    "Friday": ["Fri", "Fridays"], "Saturday": ["Sat", "Saturdays"], "Sunday": ["Sun", "Sundays"],
}
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
MONTH_ALIASES = {
    "January": ["Jan"], "February": ["Feb"], "March": ["Mar"], "April": ["Apr"],
    "May": [], "June": ["Jun"], "July": ["Jul"], "August": ["Aug"],
    "September": ["Sep", "Sept"], "October": ["Oct"], "November": ["Nov"], "December": ["Dec"],
}
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_OF_MONTH = {m: SEASONS[((i + 1) % 12) // 3] for i, m in enumerate(MONTHS)}

PROMPTS = [
    "On Monday, March 3, 1997, the committee met in private session.",
    "On Friday, September 21, 2001, the city was marked by heavy rain.",
    "In the summer of 2012, the team traveled to Europe for training.",
]
CATEGORY_COLOR = {"Day": "#1f77b4", "Month": "#ff7f0e", "Season": "#2ca02c",
                  "Year": "#d62728", "Number": "#9467bd"}
YEAR_RE = re.compile(r"^(18|19|20)\d{2}$")


def single_token_ids(tokenizer, terms):
    """{token id: term} for terms that are one token with a leading space."""
    out = {}
    for term in terms:
        for variant in (" " + term, " " + term.lower()):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                out[ids[0]] = term
    return out


def temporal_vocabulary(tokenizer, year_min, year_max):
    """{token id: (label, category)} over days, months, seasons and years."""
    vocab = {}
    for canonical, aliases, category in (
        *[(d, DAY_ALIASES[d], "Day") for d in DAYS],
        *[(m, MONTH_ALIASES[m], "Month") for m in MONTHS],
        *[(s, [], "Season") for s in SEASONS],
    ):
        for tid in single_token_ids(tokenizer, [canonical, *aliases]):
            vocab[tid] = (canonical, category)
    for year in range(year_min, year_max + 1):
        for tid in single_token_ids(tokenizer, [str(year)]):
            vocab[tid] = (str(year), "Year")
    return vocab


@torch.no_grad()
def group_coords(sae, model, toks, hook_name, layer, group_id):
    """(B, T, r) coordinates of one group at every token position."""
    _, cache = model.run_with_cache(toks, names_filter=hook_name, stop_at_layer=layer + 1)
    acts = sae.encode(cache[hook_name])
    groups = acts.view(*acts.shape[:-1], sae.n_groups, sae.group_rank)
    return groups[..., group_id, :].float().cpu()


@torch.no_grad()
def collect(sae, model, hook_name, layer, group_id, *, max_docs, max_tokens,
            max_per_label, min_norm, year_min, year_max, batch_size=8):
    """Group coordinates at temporal tokens in OpenWebText."""
    from datasets import load_dataset

    vocab = temporal_vocabulary(model.tokenizer, year_min, year_max)
    want = torch.tensor(sorted(vocab), device=model.cfg.device)
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                      trust_remote_code=True)

    vectors, labels, categories, counts = [], [], [], {}
    texts, seen = [], 0
    for record in ds:
        texts.append(record["text"])
        seen += 1
        if len(texts) == batch_size or seen >= max_docs:
            toks = model.to_tokens(texts, truncate=True)[:, :max_tokens]
            coords = group_coords(sae, model, toks, hook_name, layer, group_id)
            hit = torch.isin(toks, want)
            hit[:, 0] = False
            for row, col in zip(*hit.nonzero(as_tuple=True)):
                label, category = vocab[int(toks[row, col])]
                if counts.get(label, 0) >= max_per_label:
                    continue
                vector = coords[row, col].numpy()
                if float(np.linalg.norm(vector)) <= min_norm:
                    continue
                vectors.append(vector)
                labels.append(label)
                categories.append(category)
                counts[label] = counts.get(label, 0) + 1
            texts = []
            if seen >= max_docs:
                break
    if not vectors:
        raise SystemExit("No temporal activations collected; check --group-id.")
    print(f"collected {len(vectors)} vectors over {len(set(labels))} labels", flush=True)
    return np.stack(vectors), np.array(labels), np.array(categories)


def classify(token: str) -> str | None:
    t = token.strip().strip(",.;:!?").lower()
    if not t:
        return None
    for day in DAYS:
        if t == day.lower() or t in {a.lower() for a in DAY_ALIASES[day]}:
            return "Day"
    for month in MONTHS:
        if t == month.lower() or t in {a.lower() for a in MONTH_ALIASES[month]}:
            return "Month"
    if t.capitalize() in SEASONS:
        return "Season"
    if YEAR_RE.match(t):
        return "Year"
    if t.isdigit():
        return "Number"
    return None


def plot_profiles(sae, model, hook_name, layer, group_id, out_path):
    """Panel (a): where the group fires inside each prompt."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(PROMPTS), 1, figsize=(7.2, 6.4))
    for ax, prompt in zip(axes, PROMPTS):
        toks = model.to_tokens([prompt])
        coords = group_coords(sae, model, toks, hook_name, layer, group_id)[0]
        norms = np.linalg.norm(coords.numpy(), axis=-1)
        pieces = [model.tokenizer.decode([t]) for t in toks[0].tolist()]
        ax.plot(range(len(norms)), norms, color="0.3", lw=1.2)
        for i, piece in enumerate(pieces):
            category = classify(piece)
            if category is None:
                continue
            ax.scatter([i], [norms[i]], s=26, color=CATEGORY_COLOR[category], zorder=3)
            ax.annotate(piece.strip(), (i, norms[i]), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7,
                        color=CATEGORY_COLOR[category])
        ax.set_title(prompt, fontsize=9)
        ax.set_xlabel("Token position", fontsize=8)
        ax.set_ylabel("Group norm", fontsize=8)
        ax.grid(alpha=0.25, ls=":")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
               for k, c in CATEGORY_COLOR.items()]
    fig.suptitle(f"Temporal Token Activation Profiles (Group {group_id})", y=0.99)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


SUBSPACE_COLOR = {"Day": "#4C78A8", "Month": "#F28E2B",
                  "Season": "#59A14F", "Year": "#9C755F"}
VIEW_ELEV, VIEW_AZIM = 30.0, 321.0


def plot_subspace(vectors, categories, out_path, elev=VIEW_ELEV, azim=VIEW_AZIM):
    """Panel (b): PCA of the collected coordinates, coloured by category.

    Plotted as (PC3, PC2, -PC1) under a fixed camera, which is the orientation the
    day cluster separates cleanly in.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(vectors)
    pca = PCA(n_components=3, random_state=0)
    proj = pca.fit_transform(X)
    coords = proj[:, [2, 1, 0]].copy()
    coords[:, 2] *= -1.0

    fig = plt.figure(figsize=(7.8, 6.1))
    ax = fig.add_subplot(111, projection="3d")
    for category in ("Day", "Month", "Season", "Year"):
        mask = categories == category
        if not mask.any():
            continue
        color = SUBSPACE_COLOR[category]
        label = {"Day": "Day Of Week"}.get(category, category)
        ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2], s=16, alpha=0.35,
                   color=color, edgecolors="none", label=label)
        mean = coords[mask].mean(axis=0)
        ax.scatter([mean[0]], [mean[1]], [mean[2]], s=120, color=color,
                   edgecolors="white", linewidths=0.8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("PC3", labelpad=6)
    ax.set_ylabel("PC2", labelpad=6)
    ax.set_zlabel("PC1", labelpad=6)
    ax.set_title("Temporal categories")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.94)
    fig.savefig(out_path)
    plt.close(fig)
    return {"explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_]}


def circular_projection(centroids, month_index, ridge=1e-6):
    """Ridge least squares sending month m to (cos(2*pi*m/12), sin(2*pi*m/12))."""
    angles = 2.0 * math.pi * (np.asarray(month_index, dtype=float) / 12.0)
    Y = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    Xc = centroids - centroids.mean(axis=0, keepdims=True)
    W = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(Xc.shape[1]), Xc.T @ Y)
    return Xc @ W


def plot_season_order(vectors, labels, categories, out_path):
    """Panel (c): do the four seasons come out in calendar order?"""
    import matplotlib.pyplot as plt

    names, centroids, index = [], [], []
    for m, month in enumerate(MONTHS, start=1):
        mask = (categories == "Month") & (labels == month)
        if not mask.any():
            continue
        names.append(month)
        centroids.append(vectors[mask].mean(axis=0))
        index.append(m)
    if len(centroids) < 4:
        raise SystemExit("Fewer than four months collected; cannot fit the ring.")

    coords = circular_projection(np.stack(centroids), index)
    season_xy = {}
    for season in SEASONS:
        rows = [i for i, n in enumerate(names) if SEASON_OF_MONTH[n] == season]
        if rows:
            season_xy[season] = coords[rows].mean(axis=0)

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for i, name in enumerate(names):
        ax.scatter(*coords[i], s=18, color=CATEGORY_COLOR["Month"], alpha=0.55)
        ax.annotate(name[:3].lower(), coords[i], textcoords="offset points",
                    xytext=(4, 3), fontsize=7, color="0.4")
    ring = [season_xy[s] for s in SEASONS if s in season_xy]
    if len(ring) > 2:
        loop = np.stack(ring + [ring[0]])
        ax.plot(loop[:, 0], loop[:, 1], color="0.35", lw=1.0, zorder=1)
    for season, xy in season_xy.items():
        ax.scatter(*xy, s=90, edgecolor="0.2", zorder=3,
                   color={"Winter": "#1f77b4", "Spring": "#2ca02c",
                          "Summer": "#ff7f0e", "Autumn": "#8c564b"}[season], label=season)
    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 2")
    ax.set_title("Season ordering from month clusters")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {s: [float(v) for v in xy] for s, xy in season_xy.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--group-id", type=int, default=1473)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--corpus-docs", type=int, default=2000)
    ap.add_argument("--corpus-max-tokens", type=int, default=128)
    ap.add_argument("--max-samples-per-label", type=int, default=40)
    ap.add_argument("--min-group-norm", type=float, default=0.1)
    ap.add_argument("--year-min", type=int, default=1980)
    ap.add_argument("--year-max", type=int, default=2024)
    ap.add_argument("--elev", type=float, default=VIEW_ELEV)
    ap.add_argument("--azim", type=float, default=VIEW_AZIM)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    register_topk_sasa_classes()
    device = resolve_device(args.device)
    sae = SAE.load_from_disk(args.sae_dir, device=device).eval()
    hook_name = sae.cfg.metadata.hook_name
    layer = int(re.search(r"blocks\.(\d+)\.", hook_name).group(1))
    model = HookedTransformer.from_pretrained(
        sae.cfg.metadata.model_name, device=device, center_writing_weights=False).eval()

    out_dir = Path(args.output_dir or REPO_ROOT / "results" / f"temporal_group{args.group_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_profiles(sae, model, hook_name, layer, args.group_id, out_dir / "fig3a_profiles.pdf")
    vectors, labels, categories = collect(
        sae, model, hook_name, layer, args.group_id,
        max_docs=args.corpus_docs, max_tokens=args.corpus_max_tokens,
        max_per_label=args.max_samples_per_label, min_norm=args.min_group_norm,
        year_min=args.year_min, year_max=args.year_max)
    pca_info = plot_subspace(vectors, categories, out_dir / "fig3b_subspace.pdf",
                             elev=args.elev, azim=args.azim)
    seasons = plot_season_order(vectors, labels, categories,
                                out_dir / "fig3c_season_order.pdf")

    (out_dir / "summary.json").write_text(json.dumps(
        {"group_id": args.group_id, "sae_dir": args.sae_dir, "hook_name": hook_name,
         "n_vectors": int(len(vectors)), "n_labels": int(len(set(labels))),
         **pca_info, "season_coords": seasons}, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sae_lens import SAE
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

import paths
from synthetic.manifolds import MANIFOLDS, manifold
from synthetic.manifolds import OUT as FITS
from synthetic.subspaces import effective_rank

OUT = paths.RESULTS / "figures" / "synthetic_manifolds"
TITLE = {"relu_sae": "ReLU SAE", "topk_sae": "TopK SAE", "batchtopk_sae": "BatchTopK SAE", "sasa": "SASA"}
ROW_TITLE = {"circle": "Circle $S^1$", "sphere": "Sphere $S^2$", "helix": "Helix"}
CYCLIC = ("circle", "sphere")  # concept value is an angle


@torch.no_grad()
def load(name: str, sae_name: str) -> dict:
    points, _, value = manifold(name)
    sae = SAE.load_from_disk(str(FITS / name / sae_name), device="cpu")
    points = torch.as_tensor(points)
    codes = sae.encode(points)
    record = {"value": value}
    if sae_name == "sasa":
        record["activity"] = codes.view(len(points), sae.n_groups, sae.group_rank).norm(dim=-1).numpy()
        record["unit"] = record["activity"].argmax(1)
        group = int(np.bincount(record["unit"]).argmax())
        columns = slice(group * sae.group_rank, (group + 1) * sae.group_rank)
        record["rank"] = effective_rank(torch.linalg.svdvals(sae.W_enc[:, columns] @ sae.W_dec[columns]))
    else:
        record["activity"] = codes.numpy()
        directions = sae.W_dec / sae.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        alignment = torch.where((codes > 0).any(0)[None, :], points @ directions.T, torch.tensor(-np.inf))
        record["unit"] = alignment.argmax(1).numpy()
    return record


def save(fig, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT / filename), bbox_inches="tight", dpi=200)
    plt.close(fig)


def scatter3d(ax, P, colors, cmap) -> None:
    floor = P[:, 2].min() - 0.2 * (np.ptp(P[:, 2]) + 1e-6)
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), floor), c="0.75", s=3, alpha=0.2, edgecolors="none")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=colors, cmap=cmap, s=7, alpha=0.9, edgecolors="none")
    ax.set_box_aspect((1, 1, 0.85))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-62)


def splitting_figure() -> None:
    columns = [("Concept value", None, "k")] + [(title, sae_name, "#0b7a3b" if sae_name == "sasa" else "#8a2b2b")
                                                for sae_name, title in TITLE.items()]
    fig = plt.figure(figsize=(2.35 * len(columns), 2.5 * len(MANIFOLDS)))
    for row, name in enumerate(MANIFOLDS):
        points, _, value = manifold(name)
        P = PCA(n_components=3).fit(points).transform(points)
        for col, (title, sae_name, title_color) in enumerate(columns):
            ax = fig.add_subplot(len(MANIFOLDS), len(columns), row * len(columns) + col + 1, projection="3d")
            if sae_name is None:
                scatter3d(ax, P, value, "twilight" if name in CYCLIC else "viridis")
            else:
                record = load(name, sae_name)
                units, unit_index = np.unique(record["unit"], return_inverse=True)
                cmap = plt.get_cmap("tab20" if len(units) > 10 else "tab10")
                scatter3d(ax, P, unit_index % cmap.N, cmap)
                if sae_name == "sasa":
                    label = f"{len(units)} group{'s' if len(units) > 1 else ''} (rank={record['rank']:.0f})"
                else:
                    label = f"{len(units)} atoms"
                ax.text2D(0.5, 0.0, label, fontsize=8, ha="center", transform=ax.transAxes, color="0.3")
            if row == 0:
                ax.set_title(title, fontsize=10.5, color=title_color,
                             fontweight="bold" if sae_name == "sasa" else "normal")
            if col == 0:
                ax.text2D(-0.14, 0.5, ROW_TITLE[name], rotation=90, fontsize=11, transform=ax.transAxes, va="center")
    fig.tight_layout()
    save(fig, "splitting.pdf")


def tuning_panel(ax, record: dict, n_bins: int = 80, max_curves: int = 40) -> None:
    value, activity = record["value"], record["activity"]
    edges = np.linspace(value.min(), value.max(), n_bins + 1)
    bins = np.clip(np.digitize(value, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins).clip(1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    strength = activity.max(0)
    firing = np.where(strength > 0.05 * strength.max())[0]
    curves = {j: gaussian_filter1d(np.bincount(bins, weights=activity[:, j], minlength=n_bins) / counts, 1.5)
              for j in firing}
    order = sorted(firing, key=lambda j: centers[curves[j].argmax()])
    if len(order) > max_curves:
        order = [order[i] for i in np.linspace(0, len(order) - 1, max_curves).round().astype(int)]
    cmap = plt.get_cmap("magma")
    for position, j in enumerate(order):
        curve = curves[j] / curves[j].max() if curves[j].max() > 1e-9 else curves[j]
        color = cmap(0.08 + 0.82 * position / max(len(order) - 1, 1))
        ax.plot(centers, curve, color=color, lw=1.1, alpha=0.9)
        ax.fill_between(centers, 0, curve, color=color, alpha=0.12)
    ax.set_yticks([])
    sns.despine(ax=ax, left=True)


def tuning_figures() -> None:
    for name in ("circle", "helix"):
        fig, axes = plt.subplots(1, len(TITLE), figsize=(2.9 * len(TITLE), 2.5), squeeze=False)
        for ax, (sae_name, title) in zip(axes[0], TITLE.items()):
            tuning_panel(ax, load(name, sae_name))
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("concept parameter")
        axes[0, 0].set_ylabel("activation")
        fig.tight_layout()
        save(fig, f"tuning_{name}.pdf")


def line_covering_number(coords: np.ndarray, eps: float) -> int:
    directions = coords / np.linalg.norm(coords, axis=1, keepdims=True).clip(1e-9)
    min_cosine = np.sqrt(max(1 - eps**2, 0.0))
    covered = np.zeros(len(directions), bool)
    n = 0
    while not covered.all():
        line = directions[np.argmax(~covered)]
        covered |= np.abs(directions @ line) >= min_cosine
        n += 1
    return n


def scaling_figure() -> None:
    sns.set_style("whitegrid", {"grid.color": "#e6e6e6"})
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4))
    eps = np.logspace(np.log10(0.4), np.log10(0.03), 12)
    cmap = plt.get_cmap("viridis")
    for i, name in enumerate(MANIFOLDS):
        coords = manifold(name)[1]
        span_dim = coords.shape[1]
        counts = [line_covering_number(coords, e) for e in eps]
        color = cmap(0.15 + 0.7 * i / max(len(MANIFOLDS) - 1, 1))
        ax.loglog(1 / eps, counts, "o-", color=color, ms=4, label=f"{ROW_TITLE[name]} ($d_i={span_dim}$)")
        reference = (1 / eps) ** (span_dim - 1)
        ax.loglog(1 / eps, reference / reference[0] * counts[0], "--", color=color, lw=1.0, alpha=0.7)
    ax.set_xlabel("$1/\\varepsilon$")
    ax.set_ylabel("# decoder directions $N$")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "scaling.pdf")


def main():
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    splitting_figure()
    tuning_figures()
    scaling_figure()


if __name__ == "__main__":
    main()

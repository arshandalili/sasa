import json
import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paths

OUT = paths.RESULTS / "figures" / "matched_reconstruction_absorption.pdf"
CONFIG, HOOK, COLUMNS = "gpt2_l7", "blocks.7.hook_resid_pre", 24576
NAME_PREFIX = {"TopK": "topk", "BatchTopK": "batchtopk", "Matryoshka": "matryoshka", "SASA": "sasa_r6"}
COLOR = {"TopK": "#333333", "BatchTopK": "#0072B2", "Matryoshka": "#D55E00", "SASA": "#009E73"}
L0S = (60, 120, 240)
MARKER = {30: "o", 60: "^", 120: "s", 240: "D"}
PANELS = (("First letters", "Mean absorption (%)", "letters", "mean_absorption"),  # title, y label, evaluation, measure
          ("Concept families", "Mean absorption (%)", "families", "mean_absorption"),
          ("", "Full absorption (%)", "letters", "full_absorption"),
          ("", "Full absorption (%)", "families", "full_absorption"))


def ce_score(sae_name: str) -> float:
    path = paths.RESULTS / "core" / CONFIG / f"core_{sae_name}_{HOOK}_custom_sae_eval_results.json"
    return 100 * json.loads(path.read_text())["eval_result_metrics"]["model_performance_preservation"]["ce_loss_score"]


def absorption(evaluation: str, sae_name: str, measure: str) -> tuple[float, float]:
    seeds = (paths.RESULTS / "absorption" / evaluation / CONFIG / sae_name).glob("seed*.json")
    values = [100 * json.loads(path.read_text())[measure] for path in seeds]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "STIXGeneral",
                         "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.7), sharex=True, sharey="row")
    for panel, (ax, (title, ylabel, evaluation, measure)) in enumerate(zip(axes.flat, PANELS)):
        for method, prefix in NAME_PREFIX.items():
            sae_names = [f"{prefix}_l0{l0}_w{COLUMNS}" for l0 in L0S]
            ce = [ce_score(sae_name) for sae_name in sae_names]
            values = [absorption(evaluation, sae_name, measure) for sae_name in sae_names]
            ax.plot(ce, [mean for mean, _ in values], "-", color=COLOR[method], lw=1.6, zorder=2)
            for l0, x, (mean, sd) in zip(L0S, ce, values):
                ax.errorbar(x, mean, yerr=sd, fmt=MARKER[l0], color=COLOR[method], ms=4.5, lw=0.9, capsize=1.8,
                            zorder=3)
        ax.set_title(title)
        if panel >= 2:
            ax.set_xlabel("CE score (%)")
        if panel % 2 == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for row in axes:
        row[0].set_ylim(bottom=0)
    lines = [Line2D([], [], color=COLOR[method], lw=1.8) for method in NAME_PREFIX]
    markers = [Line2D([], [], color="#444444", marker=MARKER[l0], ls="", ms=4.5) for l0 in L0S]
    marker_labels = [f"$\\ell_0={l0}$" for l0 in L0S]
    handles, labels = [], []
    for i, method in enumerate(NAME_PREFIX):
        handles.append(lines[i])
        labels.append(method)
        if i < len(markers):
            handles.append(markers[i])
            labels.append(marker_labels[i])
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8.5, handlelength=2.0,
               columnspacing=1.6, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)


if __name__ == "__main__":
    main()

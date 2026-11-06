import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import paths

RESULTS = paths.RESULTS / "real_manifolds"
OUT = paths.RESULTS / "figures" / "real_manifolds"
INK, MUTED, GRID = "#222222", "#666666", "#e6e6e6"
TOP_HEIGHT, BOTTOM_HEIGHT = 2.75, 1.40  # inches
BOTTOM = BOTTOM_HEIGHT / (TOP_HEIGHT + BOTTOM_HEIGHT)  # figure fraction of the bottom row
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", plt.get_cmap("viridis_r")(np.linspace(0.06, 0.94, 64)))
TEMPERATURE = LinearSegmentedColormap.from_list("temp", ["#2b5fb4", "#9bb7de", "#d9d9d9", "#e8a07f", "#b2302a"])
RING_COLORS = plt.get_cmap("tab20")(np.r_[0, 2, 4, 6, 8, 10, 12, 16, 18, 1, 3, 5, 7, 9, 11, 13, 17, 19])
OTHER_COLORS = ("#9e9e9e", "#c4c4c4")
MONTH_LETTERS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def cyclic(fraction):
    hue = np.deg2rad(25 + 360 * np.asarray(fraction, dtype=float))
    a, b = 0.14 * np.cos(hue), 0.14 * np.sin(hue)
    l_ = (0.62 + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (0.62 - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (0.62 - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = np.stack([4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
                    -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
                    -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_], -1)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1)


def top_rect(x, y, w, h):
    return [x, BOTTOM + y * (1 - BOTTOM), w, h * (1 - BOTTOM)]


def bottom_rect(x, y, w, h):
    return [x, y * BOTTOM * 0.985, w, h * BOTTOM * 0.985]


def manifold_panel(ax, Y, colors, title, elev, azim, size, zoom, ring=None):
    Y = Y / np.abs(Y).max()
    lo, hi = Y.min(0), Y.max(0)
    span = (hi - lo).max()
    mid = (hi + lo) / 2
    x0, x1 = mid[0] - 0.56 * span, mid[0] + 0.56 * span
    y0, y1 = mid[1] - 0.56 * span, mid[1] + 0.56 * span
    floor = lo[2] - 0.38 * span
    ax.computed_zorder = False
    ax.set_facecolor((1, 1, 1, 0))
    ax.add_collection3d(Poly3DCollection([[(x0, y0, floor), (x1, y0, floor), (x1, y1, floor), (x0, y1, floor)]],
                                         facecolor="#efefef", edgecolor="none", zorder=0))
    ax.scatter(Y[:, 0], Y[:, 1], np.full(len(Y), floor), s=size * 0.6, c="#8f8f8f", alpha=0.45, edgecolors="none",
               depthshade=False, rasterized=True, zorder=1)
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    order = np.argsort(Y @ np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]))  # far points first
    ax.scatter(Y[order, 0], Y[order, 1], Y[order, 2], s=size, c=np.asarray(colors)[order], alpha=0.95,
               edgecolors="none", depthshade=False, rasterized=True, zorder=2)
    if ring is not None:
        values, n = ring
        C = np.stack([Y[values == i].mean(0) for i in range(n)] + [Y[values == 0].mean(0)])
        ax.plot(C[:, 0], C[:, 1], C[:, 2], color="#555555", lw=0.6, alpha=0.8, zorder=3)
        ax.plot(C[:, 0], C[:, 1], np.full(len(C), floor), color="#9a9a9a", lw=0.5, alpha=0.8, zorder=1.5)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(floor, floor + 1.2 * span)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.text2D(0.5, 0.0, title, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK)


def key_strip(fig, ax, colors, left, right, width=0.42, height=0.017, gap=0.052):
    box = ax.get_position()
    center = box.x0 + box.width / 2
    key = fig.add_axes([center - width * box.width / 2, box.y0 - gap - height, width * box.width, height])
    key.imshow(np.asarray(colors)[None, :, :3], aspect="auto", extent=(0, 1, 0, 1), interpolation="nearest")
    key.set_axis_off()
    key.text(-0.05, 0.5, left, transform=key.transAxes, ha="right", va="center", fontsize=5.5, color=MUTED)
    key.text(1.05, 0.5, right, transform=key.transAxes, ha="left", va="center", fontsize=5.5, color=MUTED)


def light_spines(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9a9a")


def steering_panel(ax, curves, colors, xticks, xlabels, title, ylabel=False, period=None):
    t, change = np.array(curves["t"]), np.array(curves["change"])
    if period:
        tail = t >= period - 0.5
        t = np.concatenate([t[tail] - period, t[~tail]])
        change = np.concatenate([change[tail], change[~tail]])
    for j in range(len(curves["words"])):
        ax.plot(t, change[:, j], color=colors[j], lw=0.85, solid_capstyle="round", zorder=3)
    ax.axhline(0, color="#bdbdbd", lw=0.5, zorder=1)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_xlim(t[0], t[-1])
    ax.tick_params(axis="both", labelsize=6, pad=1.5, colors=MUTED)
    light_spines(ax)
    ax.grid(axis="y", color=GRID, lw=0.4, zorder=0)
    lo, hi = change.min(), change.max()
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.12 * (hi - lo))
    ax.text(0.0, 1.06, title, transform=ax.transAxes, fontsize=7, color=INK, ha="left", va="bottom")
    if ylabel:
        ax.set_ylabel(r"$\Delta$ token probability", fontsize=6.5, color=INK, labelpad=2)
    return t, change


def share_bars(ax, bars):
    for x, (_, slices) in enumerate(bars):
        stacked = 0.0
        for share, colour, in_ring in slices:
            ax.bar(x, share, bottom=stacked, width=0.6, color=colour, edgecolor="white" if in_ring else "none",
                   linewidth=0.3, zorder=2)
            stacked += share
        ring_total = sum(share for share, _, in_ring in slices if in_ring)
        ax.text(x + 0.33, ring_total / 2, f"{100 * ring_total:.0f}%", fontsize=6, color=INK, ha="left", va="center")
    ax.set_xlim(-0.55, len(bars) - 0.2)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([label for label, _ in bars], fontsize=6.3, color=INK, linespacing=1.15)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "50%", "100%"], fontsize=6)
    ax.tick_params(axis="both", length=1.5, pad=1.5, colors=MUTED)
    ax.tick_params(axis="x", length=0, pad=2.5)
    light_spines(ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
    ax.set_ylabel("share of month activation", fontsize=6.3, color=INK, labelpad=2)


def group_panel(ax, z, month, group):
    centroids = np.stack([z[month == m].mean(0) for m in range(12)])
    mean = centroids.mean(0)
    _, _, Vt = np.linalg.svd(centroids - mean, full_matrices=False)
    P, Pc = (z - mean) @ Vt[:2].T, (centroids - mean) @ Vt[:2].T
    shown = np.random.default_rng(0).choice(len(P), size=720, replace=False)
    ax.scatter(P[shown, 0], P[shown, 1], c=cyclic(month[shown] / 12), s=1.0, alpha=0.5, edgecolors="none",
               rasterized=True, zorder=1)
    loop = np.vstack([Pc, Pc[:1]])
    ax.plot(loop[:, 0], loop[:, 1], color="#4a4a4a", lw=0.6, zorder=2)
    ax.scatter(Pc[:, 0], Pc[:, 1], c=cyclic(np.arange(12) / 12), s=14, edgecolors="white", linewidths=0.5, zorder=3)
    lo, hi = np.percentile(P[shown], [1, 99], axis=0)
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_axis_off()
    ax.text(0.0, 1.04, f"Inside group {group}", transform=ax.transAxes, fontsize=7, color=INK, ha="left", va="bottom")


def main():
    results = json.loads((RESULTS / "mistral_l8.json").read_text())
    concepts, steering, dictionaries = results["concepts"], results["steering"], results["dictionaries"]
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "xtick.major.size": 2, "ytick.major.size": 2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(5.5, TOP_HEIGHT + BOTTOM_HEIGHT))
    keys = {"height": 0.017 * (1 - BOTTOM), "gap": 0.052 * (1 - BOTTOM)}

    # top left: the day and month rings, and the year, temperature and age curves
    for name, title, n, x, size, first, last in (("days", "Days of the week", 7, 0.0, 6, "Mon", "Sun"),
                                                 ("months", "Months", 12, 0.30, 3.5, "Jan", "Dec")):
        Y, values = np.array(concepts[name]["points"]), np.array(concepts[name]["values"])
        ax = fig.add_axes(top_rect(x, 0.53, 0.31, 0.47), projection="3d")
        manifold_panel(ax, Y, cyclic(values / n), title, elev=40, azim=-62, size=size, zoom=1.18, ring=(values, n))
        key_strip(fig, ax, cyclic(np.arange(n) / n), first, last, **keys)
    for i, (name, cmap, title, first, last) in enumerate((("years", SEQUENTIAL, "Years", "1800", "1998"),
                                                          ("temperature", TEMPERATURE, "Temperature (°F)", "−30",
                                                           "119"),
                                                          ("age", SEQUENTIAL, "Age (years)", "1", "99"))):
        Y, values = np.array(concepts[name]["points"]), np.array(concepts[name]["values"], dtype=float)
        ax = fig.add_axes(top_rect(0.205 * i, 0.105, 0.215, 0.37), projection="3d")
        manifold_panel(ax, Y, cmap((values - values.min()) / np.ptp(values)), title, elev=22, azim=-60, size=3.0,
                       zoom=1.2)
        key_strip(fig, ax, cmap(np.linspace(0, 1, 64)), first, last, **keys)

    # top right: the answer words' probabilities while steering around the day and month rings and along age
    steering_panel(fig.add_axes(top_rect(0.695, 0.775, 0.29, 0.17)), steering["days"], cyclic(np.arange(7) / 7),
                   np.arange(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "Steering along the days ring",
                   period=7)
    steering_panel(fig.add_axes(top_rect(0.695, 0.455, 0.29, 0.17)), steering["months"], cyclic(np.arange(12) / 12),
                   np.arange(12), MONTH_LETTERS, "Steering along the months ring", ylabel=True, period=12)
    ax = fig.add_axes(top_rect(0.695, 0.135, 0.29, 0.17))
    words = steering["age"]["words"]
    shade = {"child": 0.08, "teenager": 0.38, "young": 0.6, "senior": 0.92}
    t, change = steering_panel(ax, steering["age"], [SEQUENTIAL(shade[word]) for word in words], [5, 25, 50, 75, 95],
                               ["5", "25", "50", "75", "95"], "Steering along age (years)")
    labels = {"child": (6, 0.006, "left"), "teenager": (6, 0.0055, "left"), "young": (6, -0.0075, "left"),
              "senior": (94, 0.0075, "right")}  # word: (age, vertical offset, alignment) of its label
    for j, word in enumerate(words):
        age, offset, align = labels[word]
        k = int(np.abs(t - age).argmin())
        ax.text(t[k], change[k, j] + offset, word, fontsize=5.5, color=INK, ha=align, va="center")

    dash = {"transform": fig.transFigure, "color": "#a6a6a6", "lw": 0.6, "ls": (0, (2.5, 2.5))}
    fig.add_artist(plt.Line2D([0.632, 0.632], [BOTTOM + 0.05 * (1 - BOTTOM), BOTTOM + 0.97 * (1 - BOTTOM)], **dash))
    fig.add_artist(plt.Line2D([0.01, 0.99], [BOTTOM, BOTTOM], **dash))

    # bottom left: the month ring with every point coloured by the unit that places it, for the SAE and for SASA
    colors = {"sae": {u: RING_COLORS[i] for i, u in enumerate(dictionaries["sae"]["ring_units"])},
              "sasa": {g: plt.get_cmap("tab10")(i) for i, g in enumerate(dictionaries["sasa"]["ring_units"])}}
    Y = np.array(concepts["months"]["points"])
    for j, (d, title) in enumerate((("sae", f"{len(colors['sae'])} SAE latents"),
                                    ("sasa", f"{len(colors['sasa'])} SASA groups"))):
        ax = fig.add_axes(bottom_rect(0.01 + 0.17 * j, 0.17, 0.165, 0.80), projection="3d")
        manifold_panel(ax, Y, np.array([colors[d][u] for u in dictionaries[d]["ring_owner"]]), title, elev=40,
                       azim=-62, size=2.4, zoom=1.2)

    # bottom middle: each dictionary's month units, the ring's units first in their colours, the rest in two greys
    bars = []
    for d, name, unit_word in (("sae", "Standard SAE", "latents"), ("sasa", "SASA", "groups")):
        share = dict(zip(dictionaries[d]["month_units"], dictionaries[d]["shares"]))
        ring = dictionaries[d]["ring_units"]
        rest = sorted((u for u in share if u not in ring), key=lambda u: -share[u])
        bars.append((f"{name}\n{len(share)} {unit_word}", [(share[u], colors[d][u], True) for u in ring]
                     + [(share[u], OTHER_COLORS[i % 2], False) for i, u in enumerate(rest)]))
    bar_ax = fig.add_axes(bottom_rect(0.465, 0.27, 0.23, 0.62))
    share_bars(bar_ax, bars)

    # bottom right: inside the SASA group with the largest share, joined to its bar slice
    group = results["group"]["index"]
    group_ax = fig.add_axes(bottom_rect(0.78, 0.12, 0.2, 0.69))
    group_panel(group_ax, np.array(results["group"]["codes"]), np.array(results["group"]["month"]), group)
    height = fig.get_figheight()
    key_strip(fig, group_ax, cyclic(np.arange(12) / 12), "Jan", "Dec", width=0.42, height=0.047 / height,
              gap=0.025 / height)
    fig.add_artist(ConnectionPatch(xyA=(1.31, 0.16), coordsA="data", axesA=bar_ax, xyB=(-0.04, 0.40),
                                   coordsB="axes fraction", axesB=group_ax, arrowstyle="-|>", mutation_scale=6, lw=0.8,
                                   color=colors["sasa"][group], connectionstyle="arc3,rad=0.25", shrinkA=0.5,
                                   shrinkB=0.5))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "real_manifolds.pdf", dpi=400)


if __name__ == "__main__":
    main()

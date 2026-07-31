"""How much of a feature's internal structure lives in the units that stand for it."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.gpt2_dicts import GRID, Dict2, load
from paths import PAPER_GPT2_SASA, SCRATCH

HOOK = "blocks.7.hook_resid_pre"
CTX, BATCH = 256, 16
CAP_PER_VALUE, MIN_PER_VALUE, MIN_VALUES = 120, 20, 3
N_BG = 20000
TAU_FEATURE, PREC_FLOOR = 0.50, 0.20
COLS = (6, 12, 24, 48)
SEED = 0

STATES = SCRATCH / "gpt2_feature_states.pt"
BACKGROUND = SCRATCH / "gpt2_background.pt"

TERMS = {
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                "Sunday"],
    "month": ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"],
    "country": ["France", "Germany", "Italy", "Spain", "Japan", "China", "India",
                "Brazil", "Canada", "Mexico", "Egypt", "Kenya", "Norway", "Greece",
                "Turkey", "Vietnam"],
    "uscity": ["Boston", "Chicago", "Denver", "Seattle", "Miami", "Atlanta",
               "Dallas", "Phoenix", "Portland", "Detroit", "Houston", "Philadelphia"],
    "number": ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
               "Nine", "Ten", "Eleven", "Twelve"],
    "color": ["Red", "Blue", "Green", "Yellow", "Black", "White", "Brown",
              "Orange", "Purple", "Pink", "Gray", "Silver"],
    "direction": ["North", "South", "East", "West"],
    "season": ["Spring", "Summer", "Autumn", "Winter"],
    "language": ["English", "French", "German", "Spanish", "Italian", "Russian",
                 "Chinese", "Japanese", "Arabic", "Hindi", "Korean", "Dutch"],
    "animal": ["Dog", "Cat", "Horse", "Cow", "Sheep", "Bird", "Fish", "Lion",
               "Tiger", "Bear", "Wolf", "Fox"],
}


def _gpt2():
    import os

    os.environ.pop("HF_HUB_OFFLINE", None)
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2", device="cuda")


def collect_states(max_docs=10000):
    """Layer-7 states at every occurrence of a feature value in natural text."""
    if STATES.exists():
        return torch.load(STATES, weights_only=False)["store"]
    from datasets import load_dataset

    model = _gpt2()
    tok = model.tokenizer
    tmap = {}
    for concept, terms in TERMS.items():
        value = 0
        for t in terms:
            # TransformerLens sets add_bos_token=True, which would make every term two
            # tokens and silently match nothing.
            ids = [e[0] for v in (" " + t, " " + t.lower())
                   for e in [tok.encode(v, add_special_tokens=False)] if len(e) == 1]
            if not ids:
                continue
            for i in set(ids):
                tmap.setdefault(i, []).append((concept, value))
            value += 1

    texts = load_dataset("NeelNanda/pile-10k", split="train")["text"][:max_docs]
    want = torch.tensor(sorted(tmap), device="cuda")
    store, counts = {}, {}
    for s in range(0, len(texts), BATCH):
        toks = model.to_tokens(texts[s:s + BATCH], truncate=True)[:, :CTX]
        _, cache = model.run_with_cache(toks, names_filter=HOOK, stop_at_layer=8)
        hit = torch.isin(toks, want)
        hit[:, 0] = False
        if hit.any():
            rows, cols = hit.nonzero(as_tuple=True)
            vecs = cache[HOOK][rows, cols].float().cpu()
            for j, t in enumerate(toks[rows, cols].tolist()):
                for concept, value in tmap[t]:
                    if counts.get((concept, value), 0) >= CAP_PER_VALUE:
                        continue
                    store.setdefault((concept, value), []).append(vecs[j])
                    counts[(concept, value)] = counts.get((concept, value), 0) + 1
        del cache
        if (s // BATCH) % 100 == 0:
            filled = sum(1 for v in counts.values() if v >= CAP_PER_VALUE)
            print(f"  doc {s}/{len(texts)}  filled={filled}", flush=True)

    STATES.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"store": store}, STATES)
    del model
    torch.cuda.empty_cache()
    return store


def collect_background(n_tokens=40000):
    """A plain Pile sample, used only to price a unit's precision."""
    if BACKGROUND.exists():
        return torch.load(BACKGROUND, weights_only=False)
    from datasets import load_dataset

    model = _gpt2()
    texts = load_dataset("NeelNanda/pile-10k", split="train")["text"]
    out, n = [], 0
    for s in range(0, 4000, BATCH):
        toks = model.to_tokens(texts[s:s + BATCH], truncate=True)[:, :128]
        _, cache = model.run_with_cache(toks, names_filter=HOOK, stop_at_layer=8)
        out.append(cache[HOOK].reshape(-1, 768).float().cpu())
        n += out[-1].shape[0]
        del cache
        if n >= n_tokens:
            break
    H = torch.cat(out)[:n_tokens]
    BACKGROUND.parent.mkdir(parents=True, exist_ok=True)
    torch.save(H, BACKGROUND)
    del model
    torch.cuda.empty_cache()
    return H


def features():
    store = collect_states()
    out = {}
    for concept in sorted({c for c, _ in store}):
        X, V = [], []
        for (c, value), vs in store.items():
            if c != concept or len(vs) < MIN_PER_VALUE:
                continue
            X.append(torch.stack(vs))
            V += [value] * len(vs)
        if len(set(V)) < MIN_VALUES:
            continue
        out[concept] = (torch.cat(X).float(), np.array(V))
    return out, collect_background()[:N_BG].float()


def unit_norms(A, rank):
    """(N, n_units) activation of the object the gate opens."""
    return A.view(A.shape[0], -1, rank).norm(dim=-1) if rank > 1 else A.abs()


def fire_stats(F, y):
    """Precision, recall and F1 of every unit's firing against a binary label."""
    y = torch.as_tensor(y).float().cuda()
    npos = y.sum().clamp_min(1)
    tp = (F.float() * y[:, None]).sum(0)
    fired = F.float().sum(0).clamp_min(1e-8)
    prec = tp / fired
    rec = tp / npos
    return prec, rec, 2 * prec * rec / (prec + rec).clamp_min(1e-8)


def rank_units(U, Y):
    """Multi-class Fisher criterion per unit; the ordinary per-atom statistic at r = 1."""
    N, G, r = U.shape
    present = [c for c in range(int(Y.max()) + 1) if (Y == c).sum() >= 3]
    mu = U.mean(0, keepdim=True)
    sb = torch.zeros(G, device=U.device)
    sw = torch.zeros(G, device=U.device)
    for c in present:
        m = (Y == c).to(U.device)
        Uc = U[m]
        mc = Uc.mean(0, keepdim=True)
        sb += float(m.sum()) * (mc - mu).pow(2).sum((0, 2))
        sw += (Uc - mc).pow(2).sum((0, 2))
    return sb / sw.clamp_min(1e-8)


def split_idx(n, seed=SEED):
    idx = np.random.RandomState(seed).permutation(n)
    return idx[: int(0.7 * n)], idx[int(0.7 * n):]


def value_acc(Fea, V, tr, te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    if Fea.shape[1] == 0 or len(set(V[tr].tolist())) < 2:
        return None
    s = Fea[tr].std(0) + 1e-6
    m = LogisticRegression(max_iter=2000, C=1.0).fit(Fea[tr] / s, V[tr])
    return float(accuracy_score(V[te], m.predict(Fea[te] / s)))


def analyse(tag, d, feats, bg):
    rank = d.rank
    Fbg = (unit_norms(d.coeffs(bg), rank) > 0).cuda()
    rows = []
    for name, (X, V) in feats.items():
        A = d.coeffs(X)
        Fp = (unit_norms(A, rank) > 0).cuda()
        F = torch.cat([Fp, Fbg])
        y = np.concatenate([np.ones(len(Fp)), np.zeros(len(Fbg))])
        prec, rec, f1 = fire_stats(F, y)

        sel = ((rec >= TAU_FEATURE) & (prec >= PREC_FLOOR)).nonzero().flatten()
        sel = sel[f1[sel].argsort(descending=True)]
        tr, te = split_idx(len(V))
        Vi = np.unique(V, return_inverse=True)[1]
        base = float(np.bincount(Vi[te]).max() / len(te))

        row = {"feature": name, "n_values": int(Vi.max() + 1), "n_tokens": int(len(V)),
               "n_feature_units": int(sel.numel()), "majority_baseline": round(base, 4),
               "feature_only": {}, "unrestricted": {}}

        blocks = A.view(A.shape[0], -1, rank).cuda()
        order_any = rank_units(blocks[tr], torch.as_tensor(Vi[tr])
                               ).nan_to_num(-1e9).argsort(descending=True).cpu()
        del blocks

        for c in COLS:
            k = max(1, c // rank)
            for label, src in (("feature_only", sel[:k]), ("unrestricted", order_any[:k])):
                if src.numel() == 0:
                    row[label][str(k * rank)] = None
                    continue
                cols = (src[:, None].cpu() * rank + torch.arange(rank)[None, :]).reshape(-1)
                acc = value_acc(A[:, cols].numpy(), Vi, tr, te)
                row[label][str(k * rank)] = None if acc is None else round(acc, 4)
        rows.append(row)
        del A, Fp, F
        torch.cuda.empty_cache()
    del Fbg
    torch.cuda.empty_cache()

    # best_any takes whichever ranking did better for that arm on that feature, so no arm
    # is held back by a ranking rule that happens not to suit it.
    for r in rows:
        r["best_any"] = {k: max([x for x in (r["feature_only"].get(k),
                                             r["unrestricted"].get(k)) if x is not None],
                                default=None)
                         for k in r["feature_only"]}

    out = {"rank": rank, "l0_cols": d.l0_cols, "features": rows}
    for label in ("feature_only", "unrestricted", "best_any"):
        for c in COLS:
            k = str(max(1, c // rank) * rank)
            v = [r[label].get(k) for r in rows if r[label].get(k) is not None]
            out[f"median_{label}_{k}"] = round(float(np.median(v)), 4) if v else None
    out["median_n_feature_units"] = float(np.median([r["n_feature_units"] for r in rows]))
    out["median_majority_baseline"] = round(
        float(np.median([r["majority_baseline"] for r in rows])), 4)

    k6 = str(max(1, 6 // rank) * rank)
    print(f"{tag:22s} r={rank:2d}  value accuracy from FEATURE units at {k6}/24/48 cols: "
          f"{out[f'median_feature_only_{k6}']}/{out.get('median_feature_only_24')}/"
          f"{out.get('median_feature_only_48')}   best-any at {k6}: "
          f"{out[f'median_best_any_{k6}']}   (majority {out['median_majority_baseline']}, "
          f"{out['median_n_feature_units']:.0f} feature units)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    feats, bg = features()
    print(f"{len(feats)} features: {', '.join(feats)}", flush=True)

    p = Path(args.out or REPO / "results" / "within_feature.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(p.read_text()) if p.exists() else {}

    cells = [(f"{a}:{l}", None) for a, l in GRID] + [("sasa_paper:60", str(PAPER_GPT2_SASA))]
    if args.cells:
        cells = [c for c in cells if c[0] in args.cells]

    for key, path in cells:
        if key in out:
            continue
        d = Dict2(path) if path else load(*key.split(":")[:1], int(key.split(":")[1]))
        out[key] = analyse(key, d, feats, bg)
        p.write_text(json.dumps(out, indent=1))
        del d
        torch.cuda.empty_cache()
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()

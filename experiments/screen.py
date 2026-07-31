"""Appendix F.2.1-style group screening, re-run on the freshly trained Mistral SASA.

Selection rule, fixed before results are seen: take the highest synthetic-AUC group,
require corpus AUC >= 0.80 on a held-out Pile slice.
"""

import argparse
import json
import random
import re

import torch

from experiments.common import encode, load_lm, load_sasa

CORPUS_AUC_FLOOR = 0.80

WEEKDAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH = ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"]
SEASON = ["spring", "summer", "autumn", "winter"]
YEAR = [str(y) for y in range(1980, 2025, 2)]
GEO = ["France", "Germany", "Japan", "Brazil", "Canada", "Egypt", "India", "Italy",
       "Kenya", "Mexico", "Norway", "Poland", "Spain", "Sweden", "Turkey", "Chile",
       "Paris", "London", "Berlin", "Tokyo", "Chicago", "Toronto", "Madrid", "Rome",
       "Boston", "Seattle", "Dublin", "Vienna", "Moscow", "Cairo", "Sydney", "Lisbon",
       "Europe", "Africa", "Asia", "Australia"]
OBJECT = ["lamp", "chair", "bottle", "hammer", "basket", "mirror", "pillow", "kettle",
          "ladder", "blanket", "candle", "bucket", "camera", "clock", "cushion",
          "drawer", "envelope", "fork", "guitar", "helmet", "jacket", "knife",
          "napkin", "pencil", "plate", "radio", "ribbon", "shovel", "spoon", "towel",
          "umbrella", "vase", "wallet", "whistle", "bicycle", "curtain"]

TEMPLATES = {
    "temporal": [
        "The event happened in {term}.", "The meeting is on {term}.",
        "She will return in {term}.", "They met every {term}.",
        "He works best in the {term}.", "The deadline is {term}.",
        "We arrived {term}.", "The festival is held in {term}.",
        "The schedule changed by {term}.", "She called me on {term}.",
        "The report is due in {term}.", "The appointment is {term}.",
        "He studies each {term}.", "We meet once a {term}.",
        "The contract ends in {term}.", "The anniversary is in {term}.",
    ],
    "geography": [
        "I traveled to {term}.", "She moved to {term} for work.",
        "He lives in {term} now.", "The city of {term} is famous.",
        "The region around {term} is beautiful.", "We drove across {term}.",
        "The capital of {term} is well known.", "The border of {term} is disputed.",
        "Flights to {term} were delayed.", "A map of {term} was published.",
        "The river in {term} floods in spring.", "The cuisine of {term} is popular.",
        "{term} appears on the map.", "The mountains in {term} are high.",
        "The coastline of {term} is long.", "The embassy in {term} reopened.",
    ],
    "negative": [
        "I bought a {term} at the store.", "She found a {term} in the garden.",
        "He repaired the {term} at home.", "We cooked a {term} for dinner.",
        "The {term} was on the table.", "A {term} fell from the shelf.",
        "He cleaned the {term} yesterday.", "She carried a {term} to work.",
        "The {term} needs a new battery.", "They painted the {term} blue.",
    ],
}

CONCEPTS = {
    "weekday": ("temporal", WEEKDAY),
    "month": ("temporal", MONTH),
    "season": ("temporal", SEASON),
    "year": ("temporal", YEAR),
    "temporal": ("temporal", WEEKDAY + MONTH + SEASON + YEAR),
    "geo": ("geography", GEO),
}


def build(family, terms, cap=800, seed=42):
    combos = [(t, w) for t in TEMPLATES[family] for w in terms]
    random.Random(seed).shuffle(combos)
    return [(t.replace("{term}", w), t.index("{term}"), len(w)) for t, w in combos[:cap]]


@torch.no_grad()
def group_norms(model, sae, items, layer, batch=16):
    """Group norms at the substituted-term positions only. -> (n_tokens, n_groups)"""
    out = []
    for i in range(0, len(items), batch):
        for text, cstart, clen in items[i:i + batch]:
            enc = model.tokenizer(text, return_offsets_mapping=True)
            spans = [j for j, (a, b) in enumerate(enc["offset_mapping"])
                     if b > cstart and a < cstart + clen]
            toks = torch.tensor([enc["input_ids"]], device=model.cfg.device)
            _, cache = model.run_with_cache(
                toks, names_filter=f"blocks.{layer}.hook_resid_post", stop_at_layer=layer + 1
            )
            h = cache[f"blocks.{layer}.hook_resid_post"][0]
            a, _ = encode(sae, h)
            out.append(a.norm(dim=-1)[spans].cpu())
    return torch.cat(out)


def auc(scores, y):
    """Mann-Whitney AUC per column, exact under ties. scores (N,G), y (N,) bool."""
    pos = scores[y].T.contiguous().cuda()
    neg = scores[~y].T.contiguous().cuda().sort(dim=1).values
    lt = torch.searchsorted(neg, pos, right=False).float()
    le = torch.searchsorted(neg, pos, right=True).float()
    return (((lt + le) / 2).mean(1) / neg.shape[1]).cpu()


def cohen_d(scores, y):
    p, n = scores[y], scores[~y]
    pooled = ((p.var(0) + n.var(0)) / 2).sqrt().clamp(min=1e-8)
    return (p.mean(0) - n.mean(0)) / pooled


def precision_at_q99(scores, y):
    thr = scores.kthvalue(max(1, int(0.99 * scores.shape[0])), dim=0).values
    fired = scores >= thr
    return (fired & y.unsqueeze(1)).sum(0).float() / fired.sum(0).clamp(min=1).float()


def l1_lr_weights(scores, y):
    """l1-logistic-regression weights over all group norms (paper's sanity check)."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.05, max_iter=2000)
    lr.fit(scores.numpy(), y.numpy())
    w = lr.coef_[0]
    return w / max(abs(w).max(), 1e-8)


@torch.no_grad()
def corpus_scores(model, sae, layer, concept_terms, keep, n_docs=800, ctx=128, min_pos=200):
    """Token-level group norms over a Pile slice, labelled by lexicon string match."""
    from datasets import load_dataset

    lex = {t.lower() for t in concept_terms}
    year = re.compile(r"^(19|20)\d{2}$")
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    rows, labels, npos = [], [], 0
    for k, doc in enumerate(ds):
        if k >= n_docs or npos >= min_pos:
            break
        toks = model.to_tokens(doc["text"])[:, :ctx]
        if toks.shape[1] < 4:
            continue
        _, cache = model.run_with_cache(
            toks, names_filter=f"blocks.{layer}.hook_resid_post", stop_at_layer=layer + 1
        )
        a, _ = encode(sae, cache[f"blocks.{layer}.hook_resid_post"][0])
        gn = a.norm(dim=-1)[1:][:, keep].cpu()  # drop BOS: never seen in training
        strs = [model.tokenizer.decode([t]).strip().lower().strip(".,;:!?'\"")
                for t in toks[0, 1:].tolist()]
        y = torch.tensor([s in lex or bool(year.match(s)) for s in strs])
        rows.append(gn)
        labels.append(y)
        npos += int(y.sum())
    return torch.cat(rows), torch.cat(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--concepts", default="weekday,month,temporal,geo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    sae = load_sasa(args.ckpt)
    model = load_lm()
    print(f"G={sae.n_groups} r={sae.group_rank} s={sae.k_groups} layer={args.layer}", flush=True)

    neg_items = build("negative", OBJECT)
    neg = group_norms(model, sae, neg_items, args.layer)
    print(f"negative term tokens: {neg.shape[0]}", flush=True)

    report = {"ckpt": args.ckpt, "layer": args.layer, "n_neg": int(neg.shape[0]),
              "rule": f"argmax synthetic AUC, corpus AUC >= {CORPUS_AUC_FLOOR}", "concepts": {}}

    for name in args.concepts.split(","):
        family, terms = CONCEPTS[name]
        pos = group_norms(model, sae, build(family, terms), args.layer)
        scores = torch.cat([pos, neg])
        y = torch.zeros(scores.shape[0], dtype=torch.bool)
        y[: pos.shape[0]] = True

        a = auc(scores, y)
        d = cohen_d(scores, y)
        p = precision_at_q99(scores, y)
        top = a.topk(args.top).indices
        w = l1_lr_weights(scores, y)

        gsel = int(top[0])
        cs, cy = corpus_scores(model, sae, args.layer, terms, top)
        ca = auc(cs, cy)  # positional over `top`

        report["concepts"][name] = {
            "n_pos": int(pos.shape[0]),
            "selected": gsel,
            "auc": float(a[gsel]), "cohen_d": float(d[gsel]),
            "prec_q99": float(p[gsel]), "l1_lr": float(w[gsel]),
            "corpus_auc": float(ca[0]), "corpus_n_pos": int(cy.sum()),
            "passes": bool(ca[0] >= CORPUS_AUC_FLOOR),
            "top": [{"g": int(g), "auc": float(a[g]), "d": float(d[g]),
                     "corpus_auc": float(ca[i])} for i, g in enumerate(top)],
        }
        r = report["concepts"][name]
        print(f"[{name}] pos={r['n_pos']} -> group {gsel}: AUC={r['auc']:.3f} "
              f"d={r['cohen_d']:.2f} p@q99={r['prec_q99']:.2f} l1LR={r['l1_lr']:.2f} "
              f"corpusAUC={r['corpus_auc']:.3f} pass={r['passes']}", flush=True)
        print("   runners-up: " + ", ".join(
            f"g{t['g']}:{t['auc']:.3f}/{t['corpus_auc']:.3f}" for t in r["top"][1:6]), flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

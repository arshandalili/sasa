"""Numbers for the Q5 ablation table."""

import argparse
import json
from collections import defaultdict

import torch

from experiments.common import contrib, dec, encode, load_lm, load_sasa
from experiments.concept_causal import (ECHO, LAYER, SEQ, TERMS, class_p, edit_hook,
                                        find_pos, single_token_ids, states)
from experiments.screen import OBJECT

N_DRAW = 5


def proj_fn(sae, g):
    q, _ = torch.linalg.qr(dec(sae)[g].T.float())

    def f(h):
        mu = h.mean(-1, keepdim=True)
        c = h - mu
        std = c.std(-1, keepdim=True)
        xn = c / (std + 1e-5)
        return (xn - (xn @ q) @ q.T) * (std + 1e-5) + mu
    return f


def contrib_fn(sae, g):
    def f(h):
        a, std = encode(sae, h)
        return h - contrib(sae, a, std, g)
    return f


def rand_dir_fn(sae, g, seed):
    def f(h):
        a, std = encode(sae, h)
        n = contrib(sae, a, std, g).norm(dim=-1, keepdim=True)
        gen = torch.Generator(device=h.device).manual_seed(seed)
        r = torch.randn(h.shape, generator=gen, device=h.device, dtype=h.dtype)
        return h - r / r.norm(dim=-1, keepdim=True) * n
    return f


def active_groups(sae, h, g_own, k, seed=0):
    """Other groups that also fire at these tokens, sampled once for the whole task.

    Prefers groups active at every position; falls back to a lower firing rate when
    the sparsity level leaves no such group."""
    a, _ = encode(sae, h)
    on = (a.norm(dim=-1) > 0).float().mean(0)
    for thr in (0.5, 0.25, 0.1):
        cand = [c for c in (on >= thr).nonzero().flatten().tolist() if c != g_own]
        if cand:
            gen = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(cand), generator=gen)[:k].tolist()
            return [cand[i] for i in idx], float(thr)
    return [], 0.0


def noun_assignment(nouns, n_slot, draw):
    """Distinct nouns per slot, distinct across draws where the list allows."""
    return [nouns[(draw * n_slot + j) % len(nouns)] for j in range(n_slot)]


@torch.no_grad()
def run_sentence(model, sae, concept, g, cls_ids, terms, nouns, g_runner=None):
    """One-sentence recall: 'The deadline is Friday. The deadline is __'."""
    tids, _ = single_token_ids(model, terms)
    n = len(terms)
    out = defaultdict(list)

    packs = []
    for fr in ECHO[concept]:
        texts = [fr.replace("{t}", t) for t in terms]
        toks = model.to_tokens(texts)
        pos = torch.tensor([find_pos(model, toks[i:i + 1], tids[i]) for i in range(n)],
                           device="cuda")
        _, h = states(model, texts, pos)
        packs.append((fr, texts, toks, pos, h))
    ctrl, thr = active_groups(sae, torch.cat([p[-1] for p in packs]), g, N_DRAW)
    print(f"  control groups {ctrl} (firing rate >= {thr})", flush=True)

    for fr, texts, toks, pos, h in packs:
        out["none"].append(class_p(model, toks, cls_ids))
        out["contrib"].append(class_p(model, toks, cls_ids, pos, contrib_fn(sae, g)))
        out["subspace"].append(class_p(model, toks, cls_ids, pos, proj_fn(sae, g)))
        out["zero_state"].append(
            class_p(model, toks, cls_ids, pos, lambda x: torch.zeros_like(x)))
        if g_runner is not None:
            out["runner_up"].append(
                class_p(model, toks, cls_ids, pos, proj_fn(sae, g_runner)))
        for j, go in enumerate(ctrl):
            out[f"active_group/{j}"].append(
                class_p(model, toks, cls_ids, pos, proj_fn(sae, go)))
        for s in range(N_DRAW):
            out[f"rand_dir/{s}"].append(
                class_p(model, toks, cls_ids, pos, rand_dir_fn(sae, g, s)))
            words = noun_assignment(nouns, n, s)
            _, hn = states(model, [fr.replace("{t}", w) for w in words], pos)
            out[f"noun_state/{s}"].append(
                class_p(model, toks, cls_ids, pos, lambda x, v=hn: v.to(x.dtype)))
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def run_list(model, sae, concept, g, cls_ids, terms, nouns, g_runner=None):
    """List continuation: 'Monday Tuesday Wednesday __'."""
    tids, _ = single_token_ids(model, terms)
    nid, _ = single_token_ids(model, nouns)
    n = len(terms)
    out = defaultdict(list)

    def masked(toks, mask, fn):
        def mh(resid, hook):
            o = resid.clone()
            o[mask] = fn(o[mask].float()).to(o.dtype)
            return o
        lg = model.run_with_hooks(toks, fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", mh)])
        return lg[:, -1].float().softmax(-1)[:, cls_ids].sum(-1)

    packs = []
    for fr in SEQ[concept]:
        texts = [fr.format(a=terms[i], b=terms[(i + 1) % n], c=terms[(i + 2) % n])
                 for i in range(n)]
        toks = model.to_tokens(texts)
        mask = torch.isin(toks, tids)
        _, cache = model.run_with_cache(
            toks, names_filter=f"blocks.{LAYER}.hook_resid_post", stop_at_layer=LAYER + 1)
        h = cache[f"blocks.{LAYER}.hook_resid_post"][mask].float()
        packs.append((fr, toks, mask, h))
    ctrl, thr = active_groups(sae, torch.cat([p[-1] for p in packs]), g, N_DRAW)
    print(f"  control groups {ctrl} (firing rate >= {thr})", flush=True)

    for fr, toks, mask, h in packs:
        out["none"].append(class_p(model, toks, cls_ids))
        out["contrib"].append(masked(toks, mask, contrib_fn(sae, g)))
        out["subspace"].append(masked(toks, mask, proj_fn(sae, g)))
        out["zero_state"].append(masked(toks, mask, lambda x: torch.zeros_like(x)))
        if g_runner is not None:
            out["runner_up"].append(masked(toks, mask, proj_fn(sae, g_runner)))
        for j, go in enumerate(ctrl):
            out[f"active_group/{j}"].append(masked(toks, mask, proj_fn(sae, go)))
        for s in range(N_DRAW):
            out[f"rand_dir/{s}"].append(masked(toks, mask, rand_dir_fn(sae, g, s)))
            w = noun_assignment(nouns, 3 * n, s)
            ntexts = [fr.format(a=w[3 * i], b=w[3 * i + 1], c=w[3 * i + 2]) for i in range(n)]
            ntoks = model.to_tokens(ntexts)
            nmask = torch.isin(ntoks, nid)
            assert ntoks.shape == toks.shape and bool((nmask == mask).all()), "misaligned"
            _, nc = model.run_with_cache(
                ntoks, names_filter=f"blocks.{LAYER}.hook_resid_post",
                stop_at_layer=LAYER + 1)
            src = nc[f"blocks.{LAYER}.hook_resid_post"]

            def swap(resid, hook, s=src, m=mask):
                o = resid.clone()
                o[m] = s[m].to(o.dtype)
                return o
            lg = model.run_with_hooks(
                toks, fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", swap)])
            out[f"noun_state/{s}"].append(
                lg[:, -1].float().softmax(-1)[:, cls_ids].sum(-1))
    return {k: torch.cat(v) for k, v in out.items()}


def summarise(raw):
    """Collapse the per-draw keys, keeping mean, spread and per-prompt drop counts."""
    base = raw["none"]
    fam = defaultdict(list)
    for k, v in raw.items():
        fam[k.split("/")[0]].append(v)
    out = {"n_prompts": int(base.numel())}
    for name, vs in fam.items():
        assert all(v.numel() == base.numel() for v in vs), f"{name}: ragged draws"
        per_draw = torch.stack([v.mean() for v in vs])
        allv = torch.cat(vs)
        drops = torch.cat([(base - v) > 0 for v in vs])
        out[name] = {
            "mean": float(allv.mean()),
            "draws": [round(float(x), 4) for x in per_draw],
            "draw_min": float(per_draw.min()), "draw_max": float(per_draw.max()),
            "prompt_sd": float(allv.std()),
            "frac_prompts_down": float(drops.float().mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--g-weekday", type=int, required=True)
    ap.add_argument("--g-month", type=int, required=True)
    ap.add_argument("--runner-weekday", type=int, required=True,
                    help="second-best group for the concept in the screening")
    ap.add_argument("--runner-month", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    sae = load_sasa(args.ckpt)
    model = load_lm()
    _, nouns = single_token_ids(model, OBJECT)
    print(f"single-token nouns: {len(nouns)}", flush=True)

    res = {"ckpt": args.ckpt, "n_draw": N_DRAW,
           "groups": {"weekday": args.g_weekday, "month": args.g_month},
           "runners": {"weekday": args.runner_weekday, "month": args.runner_month}}
    for concept, g, gr in (("weekday", args.g_weekday, args.runner_weekday),
                           ("month", args.g_month, args.runner_month)):
        cls_ids, _ = single_token_ids(model, TERMS[concept])
        for task, fn in (("list", run_list), ("sentence", run_sentence)):
            s = summarise(fn(model, sae, concept, g, cls_ids, TERMS[concept], nouns, gr))
            res[f"{concept}/{task}"] = s
            print(f"\n[{concept}/{task}] n={s['n_prompts']}", flush=True)
            for k in ("none", "contrib", "subspace", "runner_up", "active_group",
                      "rand_dir", "noun_state", "zero_state"):
                if k not in s:
                    continue
                d = s[k]
                print(f"  {k:13s} {100*d['mean']:5.1f}%  draws[{100*d['draw_min']:.1f}"
                      f"-{100*d['draw_max']:.1f}]  sd={100*d['prompt_sd']:.1f}"
                      f"  down={d['frac_prompts_down']:.2f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

"""Concept-level causal tests for SASA groups, on SASA's own terms."""

import argparse
import json
import math

import torch

from experiments.common import contrib, dec, encode, load_lm, load_sasa
from experiments.screen import MONTH, OBJECT, WEEKDAY

LAYER = 19

ECHO = {
    "weekday": ["The meeting is on {t}. The meeting is on",
                "She called me on {t}. She called me on",
                "The deadline is {t}. The deadline is",
                "They travel every {t}. They travel every",
                "The class meets on {t}. The class meets on",
                "He arrived last {t}. He arrived last",
                "Payday falls on {t}. Payday falls on",
                "The show airs every {t}. The show airs every"],
    "month": ["The event happened in {t}. The event happened in",
              "The contract ends in {t}. The contract ends in",
              "She will return in {t}. She will return in",
              "The festival is held in {t}. The festival is held in",
              "The report is due in {t}. The report is due in",
              "The season starts in {t}. The season starts in"],
}
SEQ = {"weekday": ["{a} {b} {c}", "Days: {a} {b} {c}", "In order: {a} {b} {c}"],
       "month": ["{a} {b} {c}", "Months: {a} {b} {c}", "In order: {a} {b} {c}"]}
JUDGE = {"weekday": "day of the week", "month": "month of the year"}
STEER_CTX = ["The meeting is on", "The event happened in", "She will return in",
             "The appointment is", "It reminded her of"]
TERMS = {"weekday": WEEKDAY, "month": MONTH}


def single_token_ids(model, words):
    out, keep = [], []
    for w in words:
        t = model.tokenizer(" " + w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            out.append(t[0])
            keep.append(w)
    return torch.tensor(out, device="cuda"), keep


def find_pos(model, toks, tid):
    hit = (toks == tid).nonzero()
    return int(hit[0, 1])


@torch.no_grad()
def states(model, texts, positions):
    toks = model.to_tokens(texts)
    _, cache = model.run_with_cache(
        toks, names_filter=f"blocks.{LAYER}.hook_resid_post", stop_at_layer=LAYER + 1)
    h = cache[f"blocks.{LAYER}.hook_resid_post"][torch.arange(len(texts)), positions]
    return toks, h.float()


def edit_hook(pos, fn):
    def h(resid, hook):
        o = resid.clone()
        idx = torch.arange(o.shape[0], device=o.device)
        o[idx, pos] = fn(o[idx, pos].float()).to(o.dtype)
        return o
    return (f"blocks.{LAYER}.hook_resid_post", h)


@torch.no_grad()
def class_p(model, toks, cls_ids, pos=None, fn=None, read_pos=None):
    hooks = [edit_hook(pos, fn)] if fn is not None else []
    lg = model.run_with_hooks(toks, fwd_hooks=hooks).float()
    if read_pos is None:
        lg = lg[:, -1]
    else:
        lg = lg[torch.arange(lg.shape[0]), read_pos]
    return lg.softmax(-1)[:, cls_ids].sum(-1)


def ab_conditions(sae, g_own, g_other, seed):
    """name -> h-editor. Projection = remove full subspace; contribution = remove
    only what the group wrote."""
    def proj(g):
        q, _ = torch.linalg.qr(dec(sae)[g].T.float())
        def f(h):
            mu = h.mean(-1, keepdim=True)
            c = h - mu
            std = c.std(-1, keepdim=True)
            xn = c / (std + 1e-5)
            return (xn - (xn @ q) @ q.T) * (std + 1e-5) + mu
        return f

    def contrib_abl(g):
        def f(h):
            a, std = encode(sae, h)
            return h - contrib(sae, a, std, g)
        return f

    def randvec(h):
        a, std = encode(sae, h)
        n = contrib(sae, a, std, g_own).norm(dim=-1, keepdim=True)
        gen = torch.Generator(device=h.device).manual_seed(seed)
        r = torch.randn(h.shape, generator=gen, device=h.device, dtype=h.dtype)
        return h - r / r.norm(dim=-1, keepdim=True) * n

    g_rand = (g_own + 1234) % sae.n_groups
    return {
        "ablate_own_contrib": contrib_abl(g_own),
        "ablate_own_subspace": proj(g_own),
        "ablate_other_concept_subspace": proj(g_other),
        "ablate_random_group_subspace": proj(g_rand),
        "remove_random_vector_matched": randvec,
    }


def logodds(p):
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


@torch.no_grad()
def battery_echo(model, sae, concept, g_own, g_other, cls_ids, terms, nouns, seed):
    conds = ab_conditions(sae, g_own, g_other, seed)
    rows = {k: [] for k in ["baseline", "ceiling_noun_state", *conds]}
    for fr in ECHO[concept]:
        texts = [fr.replace("{t}", t) for t in terms]
        tids, _ = single_token_ids(model, terms)
        toks = model.to_tokens(texts)
        pos = torch.tensor([find_pos(model, toks[i:i + 1], tids[i]) for i in range(len(terms))],
                           device="cuda")
        _, h_noun = states(model, [fr.replace("{t}", nouns[i % len(nouns)])
                                   for i in range(len(terms))], pos)
        rows["baseline"] += class_p(model, toks, cls_ids).tolist()
        rows["ceiling_noun_state"] += class_p(
            model, toks, cls_ids, pos, lambda h: h_noun.to(h.dtype)).tolist()
        for name, fn in conds.items():
            rows[name] += class_p(model, toks, cls_ids, pos, fn).tolist()
    out = {k: sum(v) / len(v) for k, v in rows.items()}
    base, ceil = out["baseline"], out["ceiling_noun_state"]
    out["frac_of_ceiling"] = {k: (base - out[k]) / max(base - ceil, 1e-6)
                              for k in conds}
    return out


@torch.no_grad()
def battery_seq(model, sae, concept, g_own, g_other, cls_ids, terms, seed):
    conds = ab_conditions(sae, g_own, g_other, seed)
    tids, _ = single_token_ids(model, terms)
    n = len(terms)
    rows = {k: [] for k in ["baseline", *conds]}
    for fr in SEQ[concept]:
        texts = [fr.format(a=terms[i], b=terms[(i + 1) % n], c=terms[(i + 2) % n])
                 for i in range(n)]
        toks = model.to_tokens(texts)
        mask = torch.isin(toks, tids)
        for name, fn in [("baseline", None), *conds.items()]:
            if fn is None:
                p = class_p(model, toks, cls_ids)
            else:
                def mh(resid, hook, f=fn):
                    o = resid.clone()
                    o[mask] = f(o[mask].float()).to(o.dtype)
                    return o
                p = model.run_with_hooks(
                    toks, fwd_hooks=[(f"blocks.{LAYER}.hook_resid_post", mh)]
                )[:, -1].float().softmax(-1)[:, cls_ids].sum(-1)
            rows[name] += p.tolist()
    return {k: sum(v) / len(v) for k, v in rows.items()}


@torch.no_grad()
def battery_judge(model, sae, concept, g_own, g_other, terms, nouns, seed):
    conds = ab_conditions(sae, g_own, g_other, seed)
    yes = model.tokenizer(" Yes", add_special_tokens=False)["input_ids"][0]
    no = model.tokenizer(" No", add_special_tokens=False)["input_ids"][0]
    cat = JUDGE[concept]
    tmpl = (f"Q: Is lamp a {cat}? A: No\n"
            f"Q: Is chair a {cat}? A: No\n"
            f"Q: Is {{x}} a {cat}? A:")
    out = {}
    for tag, words in (("pos", terms), ("neg", nouns[2:10])):
        tids, kept = single_token_ids(model, words)
        texts = [tmpl.format(x=w) for w in kept]
        toks = model.to_tokens(texts)
        pos = torch.tensor([int((toks[i] == tids[i]).nonzero()[-1]) for i in range(len(kept))],
                           device="cuda")
        res = {}
        lg = model(toks)[:, -1].float()
        res["baseline"] = float((lg[:, yes] - lg[:, no]).mean())
        for name, fn in conds.items():
            lg = model.run_with_hooks(toks, fwd_hooks=[edit_hook(pos, fn)])[:, -1].float()
            res[name] = float((lg[:, yes] - lg[:, no]).mean())
        out[tag] = res
    return out


@torch.no_grad()
def battery_steer(model, sae, concept, cls_own, cls_other, proto_own, proto_other,
                  nouns, seed):
    """Sufficiency: inject a group prototype at a NOUN in the concept slot of the
    echo frames; does class mass at the end recover from the noun floor toward the
    concept baseline?"""
    alphas = [0.5, 1, 2, 4]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rvec = torch.randn(proto_own.shape, generator=gen, device="cuda")
    rvec = rvec / rvec.norm() * proto_own.norm()
    rows = {}
    for fr in ECHO[concept]:
        texts = [fr.replace("{t}", nouns[i]) for i in range(8)]
        tids, _ = single_token_ids(model, nouns[:8])
        toks = model.to_tokens(texts)
        pos = torch.tensor([find_pos(model, toks[i:i + 1], tids[i]) for i in range(8)],
                           device="cuda")
        rows.setdefault("noun_floor", []).extend(class_p(model, toks, cls_own).tolist())
        rows.setdefault("noun_floor_other", []).extend(class_p(model, toks, cls_other).tolist())
        for a in alphas:
            for tag, v in (("own_proto", proto_own), ("other_group_proto", proto_other),
                           ("random_vector", rvec)):
                f = lambda h, vv=v, aa=a: h + aa * vv.to(h.dtype)
                rows.setdefault(f"{tag}_a{a}", []).extend(
                    class_p(model, toks, cls_own, pos, f).tolist())
                rows.setdefault(f"{tag}_a{a}_other", []).extend(
                    class_p(model, toks, cls_other, pos, f).tolist())
    return {k: sum(v) / len(v) for k, v in rows.items()}


@torch.no_grad()
def battery_restore(model, sae, concept, cls_ids, terms, nouns, proto_own, proto_other,
                    seed):
    """Wipe the concept site (noun-state replacement), then add back only a group
    prototype. Recovery of the baseline-vs-wipe gap = sufficiency at this site."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    rvec = torch.randn(proto_own.shape, generator=gen, device="cuda")
    rvec = rvec / rvec.norm() * proto_own.norm()
    rows = {}
    for fr in ECHO[concept]:
        texts = [fr.replace("{t}", t) for t in terms]
        tids, _ = single_token_ids(model, terms)
        toks = model.to_tokens(texts)
        pos = torch.tensor([find_pos(model, toks[i:i + 1], tids[i]) for i in range(len(terms))],
                           device="cuda")
        _, h_noun = states(model, [fr.replace("{t}", nouns[i % len(nouns)])
                                   for i in range(len(terms))], pos)
        conds = {
            "baseline": None,
            "wiped": lambda h: h_noun.to(h.dtype),
            "wiped_plus_own_proto": lambda h: (h_noun + proto_own).to(h.dtype),
            "wiped_plus_other_proto": lambda h: (h_noun + proto_other).to(h.dtype),
            "wiped_plus_random_vec": lambda h: (h_noun + rvec).to(h.dtype),
        }
        for name, fn in conds.items():
            rows.setdefault(name, []).extend(class_p(model, toks, cls_ids, pos, fn).tolist())
    out = {k: sum(v) / len(v) for k, v in rows.items()}
    gap = max(out["baseline"] - out["wiped"], 1e-6)
    out["recovery"] = {k: (out[k] - out["wiped"]) / gap
                       for k in ("wiped_plus_own_proto", "wiped_plus_other_proto",
                                 "wiped_plus_random_vec")}
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--g-weekday", type=int, required=True)
    ap.add_argument("--g-month", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    sae = load_sasa(args.ckpt)
    model = load_lm()
    cls = {}
    for c in ("weekday", "month"):
        cls[c], kept = single_token_ids(model, TERMS[c])
        assert len(kept) == len(TERMS[c])
    _, nouns = single_token_ids(model, OBJECT)

    protos = {}
    for c, g in (("weekday", args.g_weekday), ("month", args.g_month)):
        fr = ECHO[c][0]
        tids, _ = single_token_ids(model, TERMS[c])
        toks = model.to_tokens([fr.replace("{t}", t) for t in TERMS[c]])
        pos = torch.tensor([find_pos(model, toks[i:i + 1], tids[i])
                            for i in range(len(TERMS[c]))], device="cuda")
        _, h = states(model, [fr.replace("{t}", t) for t in TERMS[c]], pos)
        a, std = encode(sae, h)
        protos[c] = contrib(sae, a, std, g).mean(0)
        print(f"[{c}] proto norm {float(protos[c].norm()):.2f} "
              f"fire {float((a[:, g].norm(dim=-1) > 0).float().mean()):.2f}", flush=True)

    out = {"ckpt": args.ckpt, "groups": {"weekday": args.g_weekday, "month": args.g_month}}
    pair = {"weekday": ("month", args.g_weekday, args.g_month),
            "month": ("weekday", args.g_month, args.g_weekday)}
    for c, (oc, g_own, g_other) in pair.items():
        e = battery_echo(model, sae, c, g_own, g_other, cls[c], TERMS[c], nouns, args.seed)
        out[f"echo_{c}"] = e
        print(f"[echo {c}] base={e['baseline']:.3f} ceiling={e['ceiling_noun_state']:.3f}")
        for k, v in e["frac_of_ceiling"].items():
            print(f"   {k:34s} p={e[k]:.3f}  frac_of_ceiling={v:+.2f}")
        j = battery_judge(model, sae, c, g_own, g_other, TERMS[c], nouns, args.seed)
        out[f"judge_{c}"] = j
        print(f"[judge {c}] pos base={j['pos']['baseline']:+.2f} "
              f"own_sub={j['pos']['ablate_own_subspace']:+.2f} "
              f"other={j['pos']['ablate_other_concept_subspace']:+.2f} "
              f"randg={j['pos']['ablate_random_group_subspace']:+.2f} | "
              f"neg base={j['neg']['baseline']:+.2f}", flush=True)
        rst = battery_restore(model, sae, c, cls[c], TERMS[c], nouns, protos[c],
                              protos[oc], args.seed)
        out[f"restore_{c}"] = rst
        print(f"[restore {c}] base={rst['baseline']:.3f} wiped={rst['wiped']:.3f} "
              f"+own={rst['wiped_plus_own_proto']:.3f} "
              f"(recovery {rst['recovery']['wiped_plus_own_proto']:+.2f}) "
              f"+other={rst['wiped_plus_other_proto']:.3f} "
              f"({rst['recovery']['wiped_plus_other_proto']:+.2f}) "
              f"+rand={rst['wiped_plus_random_vec']:.3f} "
              f"({rst['recovery']['wiped_plus_random_vec']:+.2f})", flush=True)
        s = battery_steer(model, sae, c, cls[c], cls[oc], protos[c], protos[oc],
                          nouns, args.seed)
        out[f"steer_{c}"] = s
        print(f"[steer {c}] noun_floor own={s['noun_floor']:.3f} "
              f"other={s['noun_floor_other']:.3f} (concept baseline={e['baseline']:.3f})")
        for a in (0.5, 1, 2, 4):
            print(f"   a={a}: own_proto own={s[f'own_proto_a{a}']:.3f} "
                  f"other={s[f'own_proto_a{a}_other']:.4f} | "
                  f"othergrp own={s[f'other_group_proto_a{a}']:.4f} "
                  f"other={s[f'other_group_proto_a{a}_other']:.3f} | "
                  f"rand own={s[f'random_vector_a{a}']:.4f}", flush=True)

    for c, (oc, g_own, g_other) in pair.items():
        sq = battery_seq(model, sae, c, g_own, g_other, cls[c], TERMS[c], args.seed)
        out[f"seq_{c}"] = sq
        print(f"[seq {c}] " + " ".join(f"{k}={v:.3f}" for k, v in sq.items()), flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

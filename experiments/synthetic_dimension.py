"""tXJ7 Q1: vary intrinsic feature dimension d_i with every other factor fixed."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from common import configure_runtime_environment, ensure_workspace_directories
from common import PATHS

configure_runtime_environment()
ensure_workspace_directories()

from sae_lens.registry import get_sae_training_class  # noqa: E402
from sae_lens.saes.sae import SAEMetadata, TrainStepInput  # noqa: E402

from sasa.model import TopKSASA, TopKSASAConfig  # noqa: E402


REPORT_DIR = PATHS.reports_root / "txj7_q1_splitting"
DIMS = (1, 2, 3, 4, 6, 8, 12, 16)
EPS_GRID = (0.5, 0.3)
K_GRID = (1, 2, 4, 8)


@dataclass(frozen=True)
class WorldSpec:
    d_in: int = 768
    per_dim: int = 32
    n_active: int = 4
    energy: float = 1.0
    noise_frac: float = 0.01
    energy_mode: str = "total"

    @property
    def dims(self) -> tuple[int, ...]:
        return tuple(d for d in DIMS for _ in range(self.per_dim))

    @property
    def n_features(self) -> int:
        return len(self.dims)


class World:
    def __init__(self, spec: WorldSpec, seed: int, device: str):
        self.spec = spec
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.true_dims = torch.tensor(spec.dims, dtype=torch.long)
        self.bases = []
        for d_i in self.true_dims.tolist():
            q, _ = torch.linalg.qr(torch.randn(spec.d_in, d_i, generator=g))
            self.bases.append(q.to(device))
        # 'total'  : E||V_i z_i||^2 = energy for all d_i (equal energy; per-direction SNR falls with d_i)
        # 'per_dim': per-coordinate variance fixed (equal SNR; total energy grows with d_i)
        if spec.energy_mode == "total":
            sig = [(spec.energy / d) ** 0.5 for d in self.true_dims.tolist()]
        elif spec.energy_mode == "per_dim":
            sig = [spec.energy**0.5 for _ in self.true_dims.tolist()]
        else:
            raise ValueError(spec.energy_mode)
        self.sigma = torch.tensor(sig, device=device)

    def coherence(self, max_pairs: int = 20000, seed: int = 0) -> dict:
        n = len(self.bases)
        rng = np.random.default_rng(seed)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(pairs) > max_pairs:
            pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]
        vals = []
        for i, j in pairs:
            vals.append(float(torch.linalg.svdvals(self.bases[i].T @ self.bases[j]).max()))
        v = np.array(vals)
        return {"mu_max": float(v.max()), "mu_mean": float(v.mean()),
                "mu_p99": float(np.percentile(v, 99)), "n_pairs": len(pairs)}

    def sample(self, batch: int, gen: torch.Generator):
        s = self.spec
        n = s.n_features
        x = torch.zeros(batch, s.d_in, device=self.device)
        idx = torch.argsort(torch.rand(batch, n, device=self.device, generator=gen), dim=1)[:, : s.n_active]
        active = torch.zeros(batch, n, dtype=torch.bool, device=self.device)
        active.scatter_(1, idx, True)
        for k in range(n):
            rows = active[:, k].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            d_k = int(self.true_dims[k])
            z = torch.randn(rows.numel(), d_k, device=self.device, generator=gen) * self.sigma[k]
            x[rows] += z @ self.bases[k].T
        sig = x.pow(2).sum(-1).mean().sqrt()
        x = x + (s.noise_frac**0.5) * sig / (s.d_in**0.5) * torch.randn(
            x.shape, device=self.device, generator=gen)
        return x, active


class SignedTopK(torch.nn.Module):
    """Top-k by magnitude, keeping sign. Same gate as TopK, no rectification."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        _, idx = torch.topk(x.abs(), k=self.k, dim=-1, sorted=False)
        out = torch.zeros_like(x)
        return out.scatter_(-1, idx, x.gather(-1, idx))


def build_sae(arch, *, d_in, width, l0, group_rank, device, coeff):
    common = dict(d_in=d_in, d_sae=width, metadata=SAEMetadata(), device=device,
                  normalize_activations="none", apply_b_dec_to_input=True)
    if arch == "sasa":
        cfg = TopKSASAConfig(**common, n_groups=width // group_rank, group_rank=group_rank,
                             k_groups=max(l0 // group_rank, 1), nuclear_coefficient=0.0)
        return TopKSASA(cfg).to(device), {"aux": 1.0}
    C, Cfg = get_sae_training_class("topk" if arch == "topk_signed" else arch)
    if arch in ("topk", "batchtopk", "topk_signed"):
        sae = C(Cfg(**common, k=l0)).to(device)
        if arch == "topk_signed":
            sae.activation_fn = SignedTopK(l0)
        return sae, {"aux_loss": 1.0}
    if arch in ("standard", "gated"):
        return C(Cfg(**common, l1_coefficient=coeff)).to(device), {"l1": coeff}
    if arch == "jumprelu":
        return C(Cfg(**common, l0_coefficient=coeff)).to(device), {"l0": coeff}
    raise ValueError(arch)


def train(sae, coeffs, world, *, steps, batch, lr, seed, device):
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed + 5150)
    for step in range(steps):
        x, _ = world.sample(batch, gen)
        out = sae.training_forward_pass(
            TrainStepInput(sae_in=x, coefficients=coeffs, dead_neuron_mask=None, n_training_steps=step))
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
    return sae


@torch.no_grad()
def _omp_residual(pts, A, k):
    """Batched OMP: residual norm of each point against its best <=k-subset of atoms A."""
    if A.shape[0] == 0:
        return pts.norm(dim=-1)
    k = min(k, A.shape[0])
    n = pts.shape[0]
    R = pts.clone()
    chosen = torch.zeros(n, A.shape[0], dtype=torch.bool, device=pts.device)
    for _ in range(k):
        corr = (R @ A.T).abs().masked_fill(chosen, -1.0)
        pick = corr.argmax(dim=1)
        chosen[torch.arange(n, device=pts.device), pick] = True
        idx = chosen.float().argsort(dim=1, descending=True, stable=True)
        sel = idx[:, : int(chosen.sum(1).max())]
        S = A[sel]                                    # (n, k_cur, d)
        S = S * chosen.gather(1, sel).unsqueeze(-1).float()
        sol = torch.linalg.lstsq(S.transpose(1, 2), pts.unsqueeze(-1)).solution
        R = pts - (S.transpose(1, 2) @ sol).squeeze(-1)
    return R.norm(dim=-1)


@torch.no_grad()
def covering_numbers(W_alive, basis, eps_grid, k_grid, n_pts=96, cap=32, top_m=192, gen=None):
    """Realized L_i^(k)(eps) of Definition 2: min atoms so every unit h in V_i is within
    eps of the span of SOME <=k-subset. Greedy outer loop; OMP inner (upper-bounds dist,
    so this upper-bounds L). Splitting at budget k iff L > k."""
    d_i = basis.shape[1]
    h = torch.randn(n_pts, d_i, device=basis.device, generator=gen)
    pts = (h / h.norm(dim=-1, keepdim=True)) @ basis.T

    rel = (W_alive @ basis).pow(2).sum(-1)
    cand = W_alive[rel.argsort(descending=True)[:top_m]]

    out = {}
    for k in k_grid:
        for eps in eps_grid:
            sel = []
            uncov = torch.ones(n_pts, dtype=torch.bool, device=pts.device)
            n = 0
            while uncov.any() and n < cap:
                R = pts[uncov] - 0.0
                if sel:
                    A = cand[torch.tensor(sel, device=pts.device)]
                    r = _omp_residual(pts[uncov], A, k)
                    scale = (r / pts[uncov].norm(dim=-1)).unsqueeze(-1)
                    R = pts[uncov] * scale
                gain = (R @ cand.T).abs().pow(2).sum(0)
                gain[torch.tensor(sel, dtype=torch.long, device=pts.device)] = -1.0 if sel else gain[0] * 0 - 1.0
                sel.append(int(gain.argmax()))
                n += 1
                A = cand[torch.tensor(sel, device=pts.device)]
                uncov = _omp_residual(pts, A, k) > eps
            out[(k, eps)] = n if not uncov.any() else cap
    return out


@torch.no_grad()
def measure(sae, world, *, arch, n_eval, device, seed, alive_thresh=1e-6, mass=0.9, cover_per_dim=6):
    """N_i = atoms carrying `mass` of the energy the SAE puts into V_i when i is active."""
    spec = world.spec
    n_feat = spec.n_features
    d_sae = sae.cfg.d_sae
    gen = torch.Generator(device=device).manual_seed(seed + 271)

    fired = torch.zeros(d_sae, device=device)
    A2 = torch.zeros(n_feat, d_sae, device=device)     # E[a_j^2 | i active]
    cnt = torch.zeros(n_feat, device=device)
    recon_num = np.zeros(n_feat)
    recon_den = np.zeros(n_feat)
    l0s, done = [], 0
    while done < n_eval:
        b = min(4096, n_eval - done)
        x, active = world.sample(b, gen)
        acts = sae.encode(x)
        nz = acts.abs() > alive_thresh
        fired += nz.float().sum(0)
        l0s.append(nz.sum(-1).float().mean().item())
        xr = sae.decode(acts)
        A2 += active.float().T @ acts.pow(2)
        cnt += active.float().sum(0)
        for k in range(n_feat):
            rows = active[:, k].nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            P = world.bases[k]
            tgt, got = x[rows] @ P, xr[rows] @ P
            recon_num[k] += float((tgt - got).pow(2).sum())
            recon_den[k] += float(tgt.pow(2).sum())
        done += b

    alive = fired > 0
    Wr = sae.W_dec.detach().float()
    Wn = Wr / Wr.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    feat_recon = 1.0 - recon_num / np.maximum(recon_den, 1e-12)

    # energy atom j places inside V_k, when k is active
    proj = torch.zeros(d_sae, n_feat, device=device)
    for k in range(n_feat):
        proj[:, k] = (Wr @ world.bases[k]).pow(2).sum(-1)
    E = (A2 / cnt.clamp(min=1).unsqueeze(1)) * proj.T
    E = E * alive.float().unsqueeze(0)

    Ns = np.zeros(n_feat)
    for k in range(n_feat):
        e = E[k]
        tot = float(e.sum())
        if tot <= 0:
            Ns[k] = np.nan
            continue
        s, _ = torch.sort(e, descending=True)
        Ns[k] = int(torch.searchsorted(torch.cumsum(s, 0), mass * tot).item() + 1)

    # ---- top-k slot competition -------------------------------------------
    # An atom "belongs" to feature k if >=50% of its decoder norm sits in V_k.
    share = proj / Wr.pow(2).sum(-1, keepdim=True).clamp(min=1e-12)   # [d_sae, n_feat]
    owner = share.argmax(1)
    owned = (share.gather(1, owner.unsqueeze(1)).squeeze(1) >= 0.5) & alive
    M = torch.zeros(n_feat, d_sae, device=device)
    M[owner[owned], owned.nonzero(as_tuple=True)[0]] = 1.0
    own_count = M.sum(1)

    kcut = max(1, int(round(float(np.mean(l0s)))))
    gen2 = torch.Generator(device=device).manual_seed(seed + 733)
    slots_won = torch.zeros(n_feat, device=device)
    own_pre = torch.zeros(n_feat, device=device)
    cnt2 = torch.zeros(n_feat, device=device)
    cut_acc, done2 = [], 0
    n_slot = min(n_eval, 40960)
    while done2 < n_slot:
        b = min(4096, n_slot - done2)
        x, active = world.sample(b, gen2)
        pre = sae.encode_with_hidden_pre(x)[1] if hasattr(sae, "encode_with_hidden_pre") \
            else sae.encode(x)
        p = pre.abs()
        cut = torch.topk(p, k=min(kcut, p.shape[-1]), dim=-1).values[:, -1]   # [b]
        cut_acc.append(float(cut.mean()))
        af = active.float()
        slots_won += (af * ((p >= cut.unsqueeze(1)).float() @ M.T)).sum(0)
        own_pre += (af * (p @ M.T)).sum(0)
        cnt2 += af.sum(0)
        done2 += b
    c2 = cnt2.clamp(min=1)
    slots_won = (slots_won / c2).cpu().numpy()
    own_pre = (own_pre / c2 / own_count.clamp(min=1)).cpu().numpy()

    gpts = torch.Generator(device=device).manual_seed(seed + 99)
    W_alive = Wn[alive]
    td = world.true_dims.numpy()
    # subsample features for the covering measurement (it is the expensive part)
    probe = np.concatenate([np.where(td == d)[0][:cover_per_dim] for d in sorted(set(td.tolist()))])
    L = {f"k{k}_eps{e}": {} for k in K_GRID for e in EPS_GRID}
    for k in probe.tolist():
        cn = covering_numbers(W_alive, world.bases[k], EPS_GRID, K_GRID, gen=gpts)
        for (kk, ee), v in cn.items():
            L[f"k{kk}_eps{ee}"][str(k)] = int(v)
    out = {
        "arch": arch,
        "l0_realized": float(np.mean(l0s)),
        "frac_atoms_alive": float(alive.float().mean()),
        "n_atoms_alive": int(alive.sum()),
        "true_dim": td.tolist(),
        "feature_recon": feat_recon.tolist(),
        "atoms_per_feature": Ns.tolist(),
        "own_atoms": own_count.cpu().numpy().tolist(),
        "slots_won": slots_won.tolist(),
        "own_pre_level": own_pre.tolist(),
        "topk_cut_level": float(np.mean(cut_acc)),
        "splitting_ratio": (Ns / td).tolist(),
        "covering": L,
        "covering_probe_idx": probe.tolist(),
    }

    if arch == "sasa":
        n_groups, r = sae.n_groups, sae.group_rank
        Wg = sae.W_dec.detach().float().view(n_groups, r, -1)
        gcap = torch.zeros(n_groups, n_feat, device=device)
        ranks = torch.zeros(n_groups)
        Bs = []
        for gi in range(n_groups):
            S = torch.linalg.svdvals(Wg[gi])
            keep = int((S > S.max() * 1e-6).sum().clamp(min=1))
            _, _, Vh = torch.linalg.svd(Wg[gi], full_matrices=False)
            B = Vh[:keep]
            Bs.append(B)
            csum = torch.cumsum(S.pow(2), 0)
            ranks[gi] = int(torch.searchsorted(csum, 0.9 * float(S.pow(2).sum())).item() + 1)
            for k in range(n_feat):
                gcap[gi, k] = (B @ world.bases[k]).pow(2).sum() / world.true_dims[k]
        rows, cols = linear_sum_assignment(-gcap.cpu().numpy())
        gr = np.zeros(n_feat)
        gc = np.zeros(n_feat)
        for a, b in zip(rows, cols):
            gr[b] = float(ranks[a])
            gc[b] = float(gcap[a, b])

        # min groups whose UNION spans 90% of the feature subspace (greedy over
        # the best-capturing candidates); the old (gcap >= 0.5).sum(0) could not
        # exceed 1 by construction, since one rank-r group caps at r/d_i.
        gpf = np.full(n_feat, np.nan)
        gcov = np.zeros(n_feat)
        for k in range(n_feat):
            Vk = world.bases[k]
            order = torch.argsort(gcap[:, k], descending=True)[:24].tolist()
            Q = None
            for j, gi in enumerate(order, 1):
                Q = Bs[gi] if Q is None else torch.cat([Q, Bs[gi]], 0)
                Qo = torch.linalg.qr(Q.T.contiguous())[0].T
                cap = float((Qo @ Vk).pow(2).sum() / world.true_dims[k])
                gcov[k] = cap
                if cap >= 0.9:
                    gpf[k] = j
                    break
        out["groups_per_feature"] = gpf.tolist()
        out["groups_union_capture"] = gcov.tolist()
        out["groups_dominant"] = (gcap >= 0.5).sum(0).cpu().numpy().tolist()
        out["matched_group_rank"] = gr.tolist()
        out["matched_group_capture"] = gc.tolist()
    return out


def group_by_dim(rec, key, gate=None):
    td = np.array(rec["true_dim"])
    v = np.array(rec[key] if key in rec else rec["covering"][key], dtype=float)
    m = np.ones(len(td), bool) if gate is None else gate
    return {int(d): v[(td == d) & m] for d in sorted(set(td.tolist()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["standard", "topk", "batchtopk", "jumprelu", "gated", "sasa"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--l0", type=int, default=64)
    ap.add_argument("--group-rank", type=int, default=16)
    ap.add_argument("--d-in", type=int, default=768)
    ap.add_argument("--per-dim", type=int, default=32)
    ap.add_argument("--n-active", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--coeff", type=float, default=0.03)
    ap.add_argument("--coeff-map", type=str, default="standard=0.03,gated=0.06,jumprelu=1.0")
    ap.add_argument("--n-eval", type=int, default=60000)
    ap.add_argument("--recon-gate", type=float, default=0.5)
    ap.add_argument("--energy-mode", type=str, default="total", choices=["total", "per_dim"])
    ap.add_argument("--cover-per-dim", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda:6")
    ap.add_argument("--tag", type=str, default="main")
    args = ap.parse_args()

    torch.set_grad_enabled(True)
    out_dir = REPORT_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = WorldSpec(d_in=args.d_in, per_dim=args.per_dim, n_active=args.n_active, energy_mode=args.energy_mode)
    ratio = sum(spec.dims) / spec.d_in
    print(f"world: d_in={spec.d_in} m={spec.n_features} sum(d_i)={sum(spec.dims)} "
          f"(={ratio:.2f}x d_in -> superposition) s={spec.n_active} width={args.width} l0={args.l0} "
          f"energy={spec.energy_mode}")

    coh = World(spec, seed=0, device=args.device).coherence()
    lim = 1 / (2 * spec.n_active - 1)
    print(f"coherence over {coh['n_pairs']} pairs: mu_max={coh['mu_max']:.3f} "
          f"mu_p99={coh['mu_p99']:.3f} mu_mean={coh['mu_mean']:.3f}   "
          f"[Prop 3.5 wants mu<{lim:.3f}; 2*sqrt(max d_i/d_in)={2*(max(DIMS)/spec.d_in)**0.5:.3f}]")
    print()

    results = []
    for seed in args.seeds:
        world = World(spec, seed=seed, device=args.device)
        for arch in args.archs:
            torch.manual_seed(seed)
            cmap = dict(kv.split("=") for kv in args.coeff_map.split(",") if kv)
            coeff = float(cmap.get(arch, args.coeff))
            sae, co = build_sae(arch, d_in=spec.d_in, width=args.width, l0=args.l0,
                                group_rank=args.group_rank, device=args.device, coeff=coeff)
            sae = train(sae, co, world, steps=args.steps, batch=args.batch, lr=args.lr,
                        seed=seed, device=args.device)
            rec = measure(sae, world, arch=arch, n_eval=args.n_eval, device=args.device, seed=seed,
                          cover_per_dim=args.cover_per_dim)
            rec["seed"] = seed
            results.append(rec)
            gate = np.array(rec["feature_recon"]) >= args.recon_gate
            g = group_by_dim(rec, "atoms_per_feature", gate)
            line = "  ".join(f"d={d}:{(g[d].mean() if len(g[d]) else float('nan')):4.1f}" for d in sorted(g))
            print(f"seed={seed} {arch:<10} l0={rec['l0_realized']:>6.1f} alive={rec['frac_atoms_alive']*100:5.1f}% "
                  f"gated={int(gate.sum())}/{len(gate)} | atoms/feature  {line}", flush=True)

    (out_dir / "results.json").write_text(json.dumps(
        {"spec": asdict(spec), "args": vars(args), "coherence": coh, "results": results}, indent=1))
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

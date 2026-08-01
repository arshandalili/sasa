"""Synthetic planted-dimension recovery: does the nuclear penalty find the true rank?"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment

from common import configure_runtime_environment, ensure_workspace_directories
from common import PATHS

configure_runtime_environment()
ensure_workspace_directories()

from sae_lens.saes.sae import SAEMetadata, TrainStepInput  # noqa: E402

from sasa.model import TopKSASA, TopKSASAConfig  # noqa: E402

from analysis.rank_analysis import spectral_metrics_from_values  # noqa: E402

REPORT_DIR = PATHS.reports_root / "synthetic_rank_recovery"


# --------------------------------------------------------------------------- #
# ground-truth generative model
# --------------------------------------------------------------------------- #
# Measured on GPT-2 blocks.7.hook_resid_pre over OpenWebText: E||x||^2 = 80652.6.
# The synthetic stream is rescaled to this power so that nuclear_coefficient means
GPT2_L7_RESID_POWER: float = 80652.6


@dataclass(frozen=True)
class WorldSpec:
    d_in: int = 768
    n_concepts: int = 256
    max_true_dim: int = 8          # d_k drawn balanced from 1..max_true_dim
    n_active: int = 8              # concepts active per sample
    noise_frac: float = 0.01       # isotropic noise as a fraction of signal power
    act_scale_lo: float = 0.5      # per-concept magnitude spread (realism)
    act_scale_hi: float = 2.0
    target_power: float = GPT2_L7_RESID_POWER  # E||x||^2, matched to GPT-2 layer 7


class PlantedWorld:
    """K concepts, concept k spanning a known d_k-dimensional subspace of R^d_in."""

    def __init__(self, spec: WorldSpec, seed: int, device: str):
        self.spec = spec
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed)

        # balanced dimensions so every d_k is equally represented in the correlation
        reps = int(np.ceil(spec.n_concepts / spec.max_true_dim))
        dims = np.tile(np.arange(1, spec.max_true_dim + 1), reps)[: spec.n_concepts]
        self.true_dims = torch.tensor(
            dims[torch.randperm(spec.n_concepts, generator=g).numpy()], dtype=torch.long
        )

        # random orthonormal basis per concept (concepts are NOT mutually orthogonal:
        # sum(d_k) >> d_in, so this is a superposed regime like a real residual stream)
        self.bases: list[torch.Tensor] = []
        for k in range(spec.n_concepts):
            a = torch.randn(spec.d_in, int(self.true_dims[k]), generator=g)
            q, _ = torch.linalg.qr(a)
            self.bases.append(q.to(device))

        self.scales = (
            torch.empty(spec.n_concepts).uniform_(spec.act_scale_lo, spec.act_scale_hi, generator=g)
        ).to(device)

        # calibrate once so E||x||^2 == spec.target_power (see GPT2_L7_RESID_POWER)
        self.power_scale = 1.0
        cal = torch.Generator(device=device).manual_seed(seed + 777)
        raw = self._raw_sample(8192, cal)
        self.power_scale = float(
            (spec.target_power / raw.pow(2).sum(dim=-1).mean().clamp(min=1e-12)).sqrt()
        )

    def sample(self, batch: int, gen: torch.Generator) -> torch.Tensor:
        return self._raw_sample(batch, gen) * self.power_scale

    def _raw_sample(self, batch: int, gen: torch.Generator) -> torch.Tensor:
        s = self.spec
        x = torch.zeros(batch, s.d_in, device=self.device)
        # which concepts fire in each row
        idx = torch.argsort(
            torch.rand(batch, s.n_concepts, device=self.device, generator=gen), dim=1
        )[:, : s.n_active]
        for k in range(s.n_concepts):
            rows = (idx == k).any(dim=1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            d_k = int(self.true_dims[k])
            # ISOTROPIC in the concept's own basis -> d_k is unambiguous
            c = torch.randn(rows.numel(), d_k, device=self.device, generator=gen)
            x[rows] += self.scales[k] * (c @ self.bases[k].T)
        sig = x.pow(2).sum(dim=-1).mean().sqrt()
        x = x + (s.noise_frac ** 0.5) * sig / (s.d_in ** 0.5) * torch.randn(
            x.shape, device=self.device, generator=gen
        )
        return x


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_one(
    world: PlantedWorld,
    *,
    group_rank: int,
    nuclear_coefficient: float,
    steps: int,
    batch: int,
    lr: float,
    seed: int,
    device: str,
) -> tuple[TopKSASA, dict]:
    s = world.spec
    torch.manual_seed(seed)
    cfg = TopKSASAConfig(
        d_in=s.d_in,
        d_sae=s.n_concepts * group_rank,
        n_groups=s.n_concepts,
        group_rank=group_rank,
        k_groups=s.n_active,
        nuclear_coefficient=float(nuclear_coefficient),
        metadata=SAEMetadata(),
        device=device,
        normalize_activations="none",
        apply_b_dec_to_input=True,
    )
    sae = TopKSASA(cfg).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed + 9999)

    hist = []
    for step in range(steps):
        x = world.sample(batch, gen)
        out = sae.training_forward_pass(
            TrainStepInput(
                sae_in=x, coefficients={"aux": 1.0}, dead_neuron_mask=None, n_training_steps=step
            )
        )
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            with torch.no_grad():
                ev = 1.0 - (
                    (out.sae_out - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()
                ).item()
            hist.append({"step": step, "loss": out.loss.item(), "explained_variance": ev})
    return sae, {"history": hist, "explained_variance": hist[-1]["explained_variance"]}


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure(sae: TopKSASA, world: PlantedWorld, *, n_eval: int, device: str, seed: int) -> dict:
    s = world.spec
    r = sae.group_rank
    W = sae.W_dec.detach().float().view(s.n_concepts, r, s.d_in)

    # ---- learned rank per group (identical statistic to rank_analysis.py) ----
    dec_rank, spans = [], []
    for g in range(s.n_concepts):
        U, S, Vh = torch.linalg.svd(W[g], full_matrices=False)
        dec_rank.append(spectral_metrics_from_values(S.pow(2).cpu().numpy())["rank90"])
        keep = (S > S.max() * 1e-6).sum().clamp(min=1)
        spans.append(Vh[:keep])                       # orthonormal basis of the row space
    dec_rank = np.array(dec_rank)

    # ---- coefficient-covariance rank, the proxy used on GPT-2 ----
    gen = torch.Generator(device=device).manual_seed(seed + 4242)
    tot = torch.zeros(s.n_concepts, r, device=device)
    sec = torch.zeros(s.n_concepts, r, r, device=device)
    cnt = torch.zeros(s.n_concepts, device=device)
    done = 0
    while done < n_eval:
        b = min(4096, n_eval - done)
        acts = sae.encode(world.sample(b, gen)).view(-1, s.n_concepts, r)
        on = acts.norm(dim=-1) > 1e-6
        masked = acts * on.unsqueeze(-1)
        tot += masked.sum(0)
        sec += torch.einsum("tgr,tgs->grs", masked, masked)
        cnt += on.sum(0)
        done += b
    act_rank = np.zeros(s.n_concepts)
    for g in range(s.n_concepts):
        if cnt[g] < 2:
            continue
        mu = tot[g] / cnt[g]
        cov = (sec[g] / cnt[g] - torch.outer(mu, mu)).cpu().numpy()
        act_rank[g] = spectral_metrics_from_values(
            np.clip(np.linalg.eigvalsh(cov), 0, None)
        )["rank90"]

    # ---- match groups to planted concepts by subspace capture ----
    # capture(g,k) = fraction of concept k's subspace lying inside group g's span
    cap = np.zeros((s.n_concepts, s.n_concepts))
    for g in range(s.n_concepts):
        V = spans[g]                                   # (q, d_in)
        for k in range(s.n_concepts):
            Uk = world.bases[k]                        # (d_in, d_k)
            cap[g, k] = float((V @ Uk).pow(2).sum() / Uk.shape[1])
    rows, cols = linear_sum_assignment(-cap)
    matched_cap = cap[rows, cols]
    true_d = world.true_dims.numpy()[cols]
    learned = dec_rank[rows]
    learned_act = act_rank[rows]
    alive = cnt.cpu().numpy()[rows] >= 2

    def corr(a, b):
        if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan"), float("nan")
        return stats.pearsonr(a, b)

    ok = alive & (matched_cap >= 0.5)                  # interpretable subset
    pear, pp = corr(learned[ok], true_d[ok])
    spear, sp = (
        stats.spearmanr(learned[ok], true_d[ok]) if ok.sum() >= 5 and np.std(learned[ok]) > 0
        else (float("nan"), float("nan"))
    )
    apear, app = corr(learned_act[ok], true_d[ok])

    return {
        "n_groups": int(s.n_concepts),
        "mean_capture": float(matched_cap.mean()),
        "median_capture": float(np.median(matched_cap)),
        "frac_well_matched": float((matched_cap >= 0.5).mean()),
        "frac_alive": float(alive.mean()),
        "n_interpretable": int(ok.sum()),
        "mean_learned_rank": float(learned.mean()),
        "sd_learned_rank": float(learned.std()),
        "frac_at_max_rank": float((learned >= r - 0.5).mean()),
        "n_distinct_ranks": int(len(np.unique(learned))),
        "mae_rank": float(np.abs(learned[ok] - true_d[ok]).mean()) if ok.sum() else float("nan"),
        "bias_rank": float((learned[ok] - true_d[ok]).mean()) if ok.sum() else float("nan"),
        "exact_frac": float((learned[ok] == true_d[ok]).mean()) if ok.sum() else float("nan"),
        "pearson_rank_vs_true": float(pear),
        "pearson_p": float(pp),
        "spearman_rank_vs_true": float(spear),
        "spearman_p": float(sp),
        "actrank_pearson_vs_true": float(apear),
        "actrank_pearson_p": float(app),
        "mean_true_dim": float(true_d[ok].mean()) if ok.sum() else float("nan"),
        "_per_group": {
            "true_dim": true_d.tolist(),
            "learned_rank": learned.tolist(),
            "activation_rank": learned_act.tolist(),
            "capture": matched_cap.tolist(),
            "interpretable": ok.tolist(),
        },
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0, 3.0, 10.0, 30.0, 100.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--group-rank", type=int, default=10, help="r_max; > max true dim")
    ap.add_argument("--max-true-dim", type=int, default=8)
    ap.add_argument("--n-concepts", type=int, default=256)
    ap.add_argument("--n-active", type=int, default=8)
    ap.add_argument("--d-in", type=int, default=768)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-eval", type=int, default=200000)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--tag", type=str, default="main")
    args = ap.parse_args()

    torch.set_grad_enabled(True)
    out_dir = REPORT_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = WorldSpec(
        d_in=args.d_in,
        n_concepts=args.n_concepts,
        max_true_dim=args.max_true_dim,
        n_active=args.n_active,
    )
    print(f"world: {asdict(spec)}")
    print(f"group_rank(r_max)={args.group_rank}  steps={args.steps}  device={args.device}\n")

    results = []
    for seed in args.seeds:
        world = PlantedWorld(spec, seed=seed, device=args.device)
        for lam in args.lambdas:
            sae, tr = train_one(
                world,
                group_rank=args.group_rank,
                nuclear_coefficient=lam,
                steps=args.steps,
                batch=args.batch,
                lr=args.lr,
                seed=seed,
                device=args.device,
            )
            m = measure(sae, world, n_eval=args.n_eval, device=args.device, seed=seed)
            m.update(
                lam=lam, seed=seed, group_rank=args.group_rank,
                explained_variance=tr["explained_variance"],
            )
            results.append(m)
            print(
                f"lam={lam:>6.1f} seed={seed} | EV {tr['explained_variance']:.4f} "
                f"| capture {m['mean_capture']:.3f} ({m['frac_well_matched']*100:.0f}% >=0.5) "
                f"| rank {m['mean_learned_rank']:.2f}+-{m['sd_learned_rank']:.2f} "
                f"({m['n_distinct_ranks']} distinct, {m['frac_at_max_rank']*100:.0f}% at max) "
                f"| MAE {m['mae_rank']:.2f} r={m['pearson_rank_vs_true']:.3f} "
                f"| act-rank r={m['actrank_pearson_vs_true']:.3f}",
                flush=True,
            )

    (out_dir / "results.json").write_text(json.dumps(results, indent=1))

    print("\n" + "=" * 108)
    print("SUMMARY  (mean +/- sd over seeds)")
    print("=" * 108)
    print(f"{'lambda':>8} {'EV':>8} {'capture':>9} {'mean rank':>12} {'#distinct':>10} "
          f"{'MAE':>10} {'r(rank,d_k)':>16} {'r(actrank,d_k)':>16}")
    print("-" * 108)
    for lam in args.lambdas:
        rs = [r for r in results if r["lam"] == lam]
        if not rs:
            continue
        f = lambda k: np.array([r[k] for r in rs], dtype=float)
        print(
            f"{lam:>8.1f} {f('explained_variance').mean():>8.4f} {f('mean_capture').mean():>9.3f} "
            f"{f('mean_learned_rank').mean():>6.2f}+-{f('mean_learned_rank').std():<5.2f} "
            f"{f('n_distinct_ranks').mean():>10.1f} "
            f"{f('mae_rank').mean():>5.2f}+-{f('mae_rank').std():<4.2f} "
            f"{f('pearson_rank_vs_true').mean():>10.3f}+-{f('pearson_rank_vs_true').std():<5.3f} "
            f"{f('actrank_pearson_vs_true').mean():>10.3f}+-{f('actrank_pearson_vs_true').std():<5.3f}"
        )
    print(f"\nwrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()

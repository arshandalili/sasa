import json

import numpy as np
import torch
from sae_lens import TopKTrainingSAE, TopKTrainingSAEConfig

import paths
from sasa.model import SASA, SASAConfig
from synthetic.data import SyntheticActivations, random_subspaces
from synthetic.subspaces import captured_fraction, group_spans
from synthetic.training import fit

OUT = paths.RESULTS / "synthetic" / "feature_dimension"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_IN = 768
FEATURE_DIMS = (1, 2, 3, 4, 6, 8, 12, 16)
FEATURES_PER_DIM = 32
ACTIVE_FEATURES = 4
NOISE_FRACTION = 0.01
WIDTH = 4096
DICTIONARIES = {"topk_l64": 64, "topk_l32": 32, "sasa_r16": 16, "sasa_r8": 8}  # l0 of TopK, group rank of SASA
SASA_ACTIVE_GROUPS = 4
STEPS, BATCH, LR = 6000, 2048, 1e-3
N_EVAL, EVAL_BATCH = 60000, 4096
SEEDS = (0, 1, 2)
TRAINING_SEED_OFFSET, EVALUATION_SEED_OFFSET = 5150, 271
ENERGY_SHARE = 0.9
MAX_GROUPS = 24
SUMMARY_DIMS = (1, 4, 8, 12, 16)


def activations(seed: int) -> SyntheticActivations:
    generator = torch.Generator().manual_seed(seed)
    dims = [d for d in FEATURE_DIMS for _ in range(FEATURES_PER_DIM)]
    scales = torch.tensor([(1.0 / d) ** 0.5 for d in dims])
    return SyntheticActivations(random_subspaces(dims, D_IN, generator), scales, ACTIVE_FEATURES, NOISE_FRACTION,
                                DEVICE)


def build(name: str):
    if name.startswith("topk"):
        return TopKTrainingSAE(TopKTrainingSAEConfig(d_in=D_IN, d_sae=WIDTH, k=DICTIONARIES[name], device=DEVICE))
    return SASA(SASAConfig(d_in=D_IN, d_sae=WIDTH, n_groups=WIDTH // DICTIONARIES[name], group_rank=DICTIONARIES[name],
                           k_groups=SASA_ACTIVE_GROUPS, gate="topk", nuclear_coefficient=0.0, encoder_norm_renorm=True,
                           device=DEVICE))


@torch.no_grad()
def evaluate(sae, data: SyntheticActivations, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(DEVICE).manual_seed(seed + EVALUATION_SEED_OFFSET)
    n_features = len(data.bases)
    # what each column decodes to; not W_dec, since the TopK training SAE divides a code by its decoder column's norm
    directions =sae.decode(torch.eye(WIDTH, device=DEVICE)) - sae.decode(torch.zeros(1, WIDTH, device=DEVICE))
    residual_energy = torch.zeros(n_features, device=DEVICE)
    input_energy = torch.zeros(n_features, device=DEVICE)
    squared_codes, times_active = torch.zeros(n_features, WIDTH, device=DEVICE), torch.zeros(n_features, device=DEVICE)
    for start in range(0, N_EVAL, EVAL_BATCH):
        x, active = data.sample(min(EVAL_BATCH, N_EVAL - start), generator)
        codes = sae.encode(x)
        residual = x - sae.decode(codes)
        for i, basis in enumerate(data.bases):
            rows = active[:, i]
            residual_energy[i] += (residual[rows] @ basis).pow(2).sum()
            input_energy[i] += (x[rows] @ basis).pow(2).sum()
        squared_codes += active.float().T @ codes.pow(2)
        times_active += active.float().sum(0)
    in_subspace = torch.stack([(directions @ basis).pow(2).sum(-1) for basis in data.bases])
    energy = squared_codes / times_active.clamp(min=1)[:, None] * in_subspace
    return 1 - residual_energy / input_energy, energy


def atoms_per_feature(energy: torch.Tensor) -> list[float]:
    counts = []
    for feature_energy in energy:
        total = float(feature_energy.sum())
        cumulative = torch.cumsum(torch.sort(feature_energy, descending=True).values, 0)
        counts.append(int(torch.searchsorted(cumulative, ENERGY_SHARE * total).item() + 1) if total > 0 else np.nan)
    return counts


@torch.no_grad()
def groups_per_feature(sae: SASA, data: SyntheticActivations) -> list[float]:
    spans = group_spans(sae.W_dec, sae.cfg.n_groups)
    counts = []
    for basis in data.bases:
        capture = torch.tensor([captured_fraction(span, basis) for span in spans])
        best = torch.argsort(capture, descending=True)[:MAX_GROUPS].tolist()
        counts.append(np.nan)
        for n in range(1, len(best) + 1):
            union = torch.linalg.qr(torch.cat([spans[g] for g in best[:n]]).T)[0].T
            if captured_fraction(union, basis) >= ENERGY_SHARE:
                counts[-1] = n
                break
    return counts


def by_dimension(runs: list[dict], key: str) -> dict:
    means = {}
    for d in SUMMARY_DIMS:
        seed_means = []
        for run in runs:
            values = np.array(run[key], dtype=float)[np.array(run["feature_dim"]) == d]
            values = values[np.isfinite(values)]
            if values.size:
                seed_means.append(values.mean())
        means[str(d)] = float(np.mean(seed_means)) if seed_means else float("nan")
    return means


def run(name: str) -> dict:
    runs = []
    for seed in SEEDS:
        data = activations(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(DEVICE).manual_seed(seed + TRAINING_SEED_OFFSET)
        sae = fit(build(name), lambda: data.sample(BATCH, generator)[0], STEPS, BATCH, LR)
        explained_variance, energy = evaluate(sae, data, seed)
        units = atoms_per_feature(energy) if name.startswith("topk") else groups_per_feature(sae, data)
        runs.append({"seed": seed, "feature_dim": data.dims, "units": units,
                     "explained_variance": explained_variance.tolist()})
    return {"units_per_feature": by_dimension(runs, "units"),
            "explained_variance": by_dimension(runs, "explained_variance"), "runs": runs}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in DICTIONARIES:
        (OUT / f"{name}.json").write_text(json.dumps(run(name)))


if __name__ == "__main__":
    main()

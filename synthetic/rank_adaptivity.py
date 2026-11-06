import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import paths
from sasa.model import SASA, SASAConfig
from synthetic.data import SyntheticActivations, random_subspaces
from synthetic.subspaces import captured_fraction, effective_rank, group_spans
from synthetic.training import fit

OUT = paths.RESULTS / "synthetic" / "rank_adaptivity.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_IN = 768
N_FEATURES = 128
MAX_DIM = 8  # feature dimensions 1-8, balanced
ACTIVE_FEATURES = 8
NOISE_FRACTION = 0.01
SCALE_RANGE = (0.5, 2.0)
GPT2_MEAN_SQUARED_NORM = 80652.6  # GPT-2 small at blocks.7.hook_resid_pre on OpenWebText; the data are rescaled to it
GROUP_WIDTH = 10
NUCLEAR_COEFFICIENTS = (0.0, 300.0)
STEPS, BATCH, LR = 15000, 4096, 1e-3
N_EVAL, EVAL_BATCH = 200000, 4096
SEEDS = (0, 1, 2)
CALIBRATION_SEED_OFFSET, TRAINING_SEED_OFFSET, EVALUATION_SEED_OFFSET = 777, 9999, 4242


def activations(seed: int) -> SyntheticActivations:
    generator = torch.Generator().manual_seed(seed)
    dims = np.resize(np.arange(1, MAX_DIM + 1), N_FEATURES)[torch.randperm(N_FEATURES, generator=generator).numpy()]
    bases = random_subspaces(dims.tolist(), D_IN, generator)
    scales = torch.empty(N_FEATURES).uniform_(*SCALE_RANGE, generator=generator)
    data = SyntheticActivations(bases, scales, ACTIVE_FEATURES, NOISE_FRACTION, DEVICE)
    data.rescale(GPT2_MEAN_SQUARED_NORM, torch.Generator(DEVICE).manual_seed(seed + CALIBRATION_SEED_OFFSET))
    return data


def train(data: SyntheticActivations, nuclear_coefficient: float, seed: int) -> SASA:
    torch.manual_seed(seed)
    config = SASAConfig(d_in=D_IN, d_sae=N_FEATURES * GROUP_WIDTH, n_groups=N_FEATURES, group_rank=GROUP_WIDTH,
                        k_groups=ACTIVE_FEATURES, gate="topk", nuclear_coefficient=nuclear_coefficient,
                        encoder_norm_renorm=True, device=DEVICE)
    generator = torch.Generator(DEVICE).manual_seed(seed + TRAINING_SEED_OFFSET)
    return fit(SASA(config), lambda: data.sample(BATCH, generator)[0], STEPS, BATCH, LR)


@torch.no_grad()
def evaluate(sae: SASA, data: SyntheticActivations, seed: int) -> tuple[np.ndarray, float]:
    generator = torch.Generator(DEVICE).manual_seed(seed + EVALUATION_SEED_OFFSET)
    open_count = torch.zeros(N_FEATURES, device=DEVICE)
    error = variance = 0.0
    for start in range(0, N_EVAL, EVAL_BATCH):
        x, _ = data.sample(min(EVAL_BATCH, N_EVAL - start), generator)
        codes = sae.encode(x)
        open_count += (codes.view(-1, N_FEATURES, GROUP_WIDTH).norm(dim=-1) > 1e-6).sum(0)
        error += float((sae.decode(codes) - x).pow(2).sum())
        variance += float((x - x.mean(0)).pow(2).sum())
    return open_count.cpu().numpy(), 1.0 - error / variance


@torch.no_grad()
def measure(sae: SASA, data: SyntheticActivations, seed: int) -> dict:
    # every feature is matched to one group; a group is counted if it is open on at least two inputs and captures at
    # least half of its feature's subspace
    decoders = sae.W_dec.view(N_FEATURES, GROUP_WIDTH, D_IN)  # group k: D_k^T
    encoders = sae.W_enc.view(D_IN, N_FEATURES, GROUP_WIDTH).permute(1, 0, 2)  # group k: E_k^T
    ranks = [effective_rank(torch.linalg.svdvals(encoder @ decoder)) for encoder, decoder in zip(encoders, decoders)]
    spans = group_spans(sae.W_dec, N_FEATURES)
    capture = np.array([[captured_fraction(span, basis) for basis in data.bases] for span in spans])
    groups, features = linear_sum_assignment(-capture)
    open_count, explained_variance = evaluate(sae, data, seed)
    counted = (open_count[groups] >= 2) & (capture[groups, features] >= 0.5)
    return {"explained_variance": explained_variance, "feature_dim": [data.dims[i] for i in features],
            "effective_rank": [ranks[k] for k in groups], "counted": counted.tolist()}


def summarize(runs: list[dict]) -> dict:
    # per coefficient: the effective rank of the counted groups by feature dimension, and the explained variance, each
    # averaged within a seed and then over seeds
    summary = {}
    for coefficient in NUCLEAR_COEFFICIENTS:
        selected = [run for run in runs if run["nuclear_coefficient"] == coefficient]
        ranks = [np.array(run["effective_rank"])[run["counted"]] for run in selected]
        dims = [np.array(run["feature_dim"])[run["counted"]] for run in selected]
        rank_by_dim = {}
        for d in range(1, MAX_DIM + 1):
            seed_means = [rank[dim == d].mean() for rank, dim in zip(ranks, dims) if (dim == d).any()]
            if seed_means:
                rank_by_dim[str(d)] = float(np.mean(seed_means))
        summary[f"{coefficient:g}"] = {"effective_rank": rank_by_dim, "explained_variance": float(
            np.mean([run["explained_variance"] for run in selected]))}
    return summary


def main():
    runs = []
    for seed in SEEDS:
        data = activations(seed)
        for coefficient in NUCLEAR_COEFFICIENTS:
            sae = train(data, coefficient, seed)
            runs.append({"seed": seed, "nuclear_coefficient": coefficient, **measure(sae, data, seed)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summarize(runs), "runs": runs}))


if __name__ == "__main__":
    main()

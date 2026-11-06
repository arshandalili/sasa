import numpy as np
import torch
from sae_lens import BatchTopKTrainingSAEConfig, StandardTrainingSAEConfig, TopKTrainingSAEConfig
from sae_lens.registry import get_sae_training_class

import paths
import sasa  # noqa: F401  registers SASA with sae_lens
from sasa.model import SASAConfig
from synthetic.training import fit

OUT = paths.RESULTS / "synthetic" / "manifolds"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MANIFOLDS = ("circle", "sphere", "helix")
N_POINTS, AMBIENT_DIM, NOISE = 4000, 64, 0.05
WIDTH = 256
GROUP_RANK, ACTIVE_GROUPS = 4, 1
STEPS, BATCH, LR = 5000, 512, 1e-3
SEED = 0


def manifold(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # points in R^AMBIENT_DIM, their coordinates in the manifold's own span, and their concept value: the angle on the
    # circle and sphere, the height on the helix
    rng = np.random.default_rng(SEED)
    if name == "circle":
        value = rng.uniform(0, 2 * np.pi, N_POINTS)
        coords = np.stack([np.cos(value), np.sin(value)], 1)
    elif name == "sphere":
        coords = rng.standard_normal((N_POINTS, 3))
        coords /= np.linalg.norm(coords, axis=1, keepdims=True)
        value = np.arctan2(coords[:, 1], coords[:, 0])
    else:  # helix: three turns of height 2
        value = rng.uniform(0, 1, N_POINTS)
        angle = 2 * np.pi * 3.0 * value
        coords = np.stack([np.cos(angle), np.sin(angle), 2.0 * (value - 0.5)], 1)
    rng = np.random.default_rng(SEED + 1)
    basis, _ = np.linalg.qr(rng.standard_normal((AMBIENT_DIM, coords.shape[1])))
    points = coords @ basis.T + (NOISE / np.sqrt(AMBIENT_DIM)) * rng.standard_normal((N_POINTS, AMBIENT_DIM))
    return points.astype(np.float32), coords.astype(np.float32), value.astype(np.float32)


def dictionaries() -> dict:
    # the l1 and nuclear-norm coefficients are the strongest on {1, 3} x 10^n keeping explained variance >= 0.99
    common = dict(d_in=AMBIENT_DIM, d_sae=WIDTH, device=DEVICE)
    return {"relu_sae": StandardTrainingSAEConfig(**common, l1_coefficient=0.03),
            "topk_sae": TopKTrainingSAEConfig(**common, k=GROUP_RANK * ACTIVE_GROUPS),
            "batchtopk_sae": BatchTopKTrainingSAEConfig(**common, k=GROUP_RANK * ACTIVE_GROUPS),
            "sasa": SASAConfig(**common, n_groups=WIDTH // GROUP_RANK, group_rank=GROUP_RANK, k_groups=ACTIVE_GROUPS,
                               gate="topk", nuclear_coefficient=3.0)}


def start_on_points(sae, points: torch.Tensor, generator: torch.Generator) -> None:
    rows = points[torch.randint(0, len(points), (WIDTH,), device=points.device, generator=generator)]
    rows = rows / rows.norm(dim=1, keepdim=True)
    with torch.no_grad():
        sae.W_dec.copy_(rows)
        sae.W_enc.copy_(rows.T)


def main():
    for name in MANIFOLDS:
        x = torch.as_tensor(manifold(name)[0], device=DEVICE)
        for sae_name, config in dictionaries().items():
            torch.manual_seed(SEED)
            sae = get_sae_training_class(config.architecture())[0](config)
            generator = torch.Generator(DEVICE).manual_seed(SEED)
            start_on_points(sae, x, generator)
            fit(sae, lambda: x[torch.randint(0, len(x), (BATCH,), device=DEVICE, generator=generator)], STEPS, BATCH,
                LR)
            sae.save_inference_model(OUT / name / sae_name)


if __name__ == "__main__":
    main()

import json

import numpy as np
import torch

import paths
from configs import load_config
from eval.causal import (DATE_ONLY_MONTHS, DAYS, MONTHS, LanguageModel, SASAGroups, inference_threshold,
                         natural_windows, pile_states, pile_texts, window_states)
from eval.dictionaries import ENGELS, load_sae

SASA = "mistral_l8/sasa_r8_k10"
OUT = paths.RESULTS / "real_manifolds"
TIMES = ["the crack of dawn", "early morning", "late morning", "noon", "afternoon", "evening", "night", "midnight"]
MONTH_CARRIERS = ["It happened in {w}", "The meeting was moved to {w}", "She was born in {w}",
                  "The report is due in {w}", "They got married in {w}", "Sales peaked in {w}", "He resigned in {w}",
                  "The festival is held in {w}", "Construction began in {w}", "The results were published in {w}",
                  "We travelled there in {w}", "The season starts in {w}"]
AGES = np.arange(1, 100)
READOUTS = {  # steered concept: (question after the prompt, answer words whose probabilities are read)
    "days": (". What day is it today? It is", DAYS),
    "months": (". Which month was it? It was", MONTHS),
    "age": (" Is this person a child, a teenager, an adult or a senior? They are a",
            ["child", "teenager", "young", "senior"]),
}
AGE_GRID, AGE_BANDWIDTH = np.arange(2, 99, 1.0), 3.0
PILE_DOCUMENTS = 200
MONTH_FIRING, PILE_FIRING = 0.2, 0.01


def concept_sets() -> dict:
    return {
        "days": [(f"It's {t} on {d}", i, j) for i, d in enumerate(DAYS) for j, t in enumerate(TIMES)],
        "months": [(c.format(w=m), i, j) for i, m in enumerate(MONTHS) for j, c in enumerate(MONTH_CARRIERS)],
        "years": [(f"The date is {y}", y, None) for y in range(1800, 1999)],
        "temperature": [(f"Today it's {f} degrees Fahrenheit outside", f, None) for f in range(-30, 120)],
        "age": [(f"They are {a} years old.", int(a), None) for a in AGES],
    }


def centre_by_carrier(M: torch.Tensor, carriers: np.ndarray) -> torch.Tensor:
    M = M.clone()
    for carrier in np.unique(carriers):
        rows = torch.from_numpy(carriers == carrier)
        M[rows] = M[rows] - M[rows].mean(0, keepdim=True)
    return M


def principal_directions(states: torch.Tensor, carriers):
    norm = states.norm(dim=1)
    keep = (norm <= norm.mean() + 3 * norm.std()).numpy()
    X = states[keep] if carriers is None else centre_by_carrier(states[keep], carriers[keep])
    return keep, X, torch.linalg.svd(X - X.mean(0), full_matrices=False)[2][:3].T


def concept_coordinates(sets: dict, states: dict) -> dict:
    coordinates = {}
    for name, prompts in sets.items():
        values = np.array([value for _, value, _ in prompts])
        carriers = None if prompts[0][2] is None else np.array([carrier for _, _, carrier in prompts])
        keep, X, directions = principal_directions(states[name], carriers)
        coordinates[name] = {"points": ((X - X.mean(0)) @ directions).tolist(), "values": values[keep].tolist()}
    return coordinates


def ring_path(states: torch.Tensor, values: np.ndarray, n: int):
    centroids = torch.stack([states[torch.from_numpy(values == i)].mean(0) for i in range(n)])
    t = np.arange(0, n, 0.1)
    lower = np.floor(t).astype(int)
    fraction = torch.tensor(t - lower, dtype=torch.float32).unsqueeze(1)
    return t, (1 - fraction) * centroids[lower] + fraction * centroids[(lower + 1) % n], centroids[values]


def age_path(states: torch.Tensor, ages: np.ndarray):
    grid = torch.tensor(AGE_GRID, dtype=torch.float32)
    weights = torch.exp(-0.5 * ((grid[:, None] - torch.tensor(ages, dtype=torch.float32)[None]) / AGE_BANDWIDTH) ** 2)
    path = (weights / weights.sum(1, keepdim=True)) @ states
    return AGE_GRID, path, torch.stack([path[int((grid - float(a)).abs().argmin())] for a in ages])


def steer(lm: LanguageModel, prompts: list[str], own: torch.Tensor, path: torch.Tensor, readout: tuple) -> np.ndarray:
    question, words = readout
    answers = [lm.first_id(word) for word in words]
    changes = []
    for prompt, own_point in zip(prompts, own):
        ids, position = lm.site(prompt, question)
        deltas = torch.cat([torch.zeros_like(own_point)[None], path - own_point])
        probabilities = lm.edited_log_probs(ids, position, deltas, answers).exp()
        changes.append(probabilities[1:] - probabilities[0])
    return torch.stack(changes).mean(0).numpy()


def steering_curves(lm: LanguageModel, sets: dict, states: dict) -> dict:
    curves = {}
    for name in ("days", "months"):
        values = np.array([value for _, value, _ in sets[name]])
        t, path, own = ring_path(states[name], values, len(READOUTS[name][1]))
        curves[name] = (t, steer(lm, [prompt for prompt, _, _ in sets[name]], own, path, READOUTS[name]))
    prompts = [f"They are {a} years old." for a in AGES]
    t, path, own = age_path(lm.last_token_states(prompts), AGES)
    inside = (AGES >= AGE_GRID[0]) & (AGES <= AGE_GRID[-1])
    curves["age"] = (t, steer(lm, [p for p, kept in zip(prompts, inside) if kept], own[torch.from_numpy(inside)],
                              path, READOUTS["age"]))
    return {name: {"t": t.tolist(), "words": READOUTS[name][1], "change": change.tolist()}
            for name, (t, change) in curves.items()}


def natural_month_names(lm: LanguageModel, texts: list[str]):
    names = {lm.first_id(word): word for word in DAYS + MONTHS}
    windows = [w for w in natural_windows(lm, texts, names, set(DATE_ONLY_MONTHS)) if names[w[-1]] in MONTHS]
    return window_states(lm, windows), np.array([MONTHS.index(names[w[-1]]) for w in windows])


def sasa_codes(sasa: SASAGroups, states: torch.Tensor) -> torch.Tensor:
    codes, _, scale = sasa.codes(states.to(sasa.device))
    return (codes * scale.unsqueeze(-1)).cpu()


def sae_activations(sae, states: torch.Tensor) -> torch.Tensor:
    codes = sae.encode(states.to(sae.W_enc.device))
    return (codes / sae.input_scale).cpu()


def ring_owners(writes: torch.Tensor, active: torch.Tensor, position: torch.Tensor, carriers: np.ndarray):
    centred = centre_by_carrier(writes.reshape(len(writes), -1), carriers).view_as(writes)
    return torch.einsum("nuk,nk->nu", centred, position).masked_fill(~active, -float("inf")).argmax(1)


def month_ring(sasa: SASAGroups, sae, states: torch.Tensor, carriers: np.ndarray) -> dict:
    keep, X, directions = principal_directions(states, carriers)
    position = X @ directions
    position = position / position.norm(dim=1, keepdim=True).clamp_min(1e-9)
    codes = sasa_codes(sasa, states[keep])
    opened = codes.norm(dim=-1) > 0
    groups = opened.any(0).nonzero().flatten()
    writes = torch.einsum("nur,urk->nuk", codes[:, groups], sasa.decoder_blocks[groups].cpu() @ directions)
    sasa_owner = groups[ring_owners(writes, opened[:, groups], position, carriers[keep])]
    activation = sae_activations(sae, states[keep])
    latents = (activation > 0).any(0).nonzero().flatten()
    writes = activation[:, latents, None] * ((sae.W_dec[latents] * sae.decoder_norms[latents, None]).cpu() @ directions)
    sae_owner = latents[ring_owners(writes, activation[:, latents] > 0, position, carriers[keep])]
    return {"sae": sae_owner.numpy(), "sasa": sasa_owner.numpy()}


def pile_rates(sasa: SASAGroups, sae, corpus: torch.Tensor, chunk: int = 4096) -> dict:
    opened = torch.zeros(sasa.decoder_blocks.shape[0], device=sasa.device)
    fired = torch.zeros(sae.W_enc.shape[1], device=sasa.device)
    for start in range(0, len(corpus), chunk):
        h = corpus[start:start + chunk].to(sasa.device).float()
        opened += (sasa.pre_activations(h)[0].norm(dim=-1) > sasa.threshold).sum(0)
        fired += (sae.encode(h) > 0).sum(0)
    return {"sasa": (opened / len(corpus)).cpu().numpy(), "sae": (fired / len(corpus)).cpu().numpy()}


def month_units(activity: np.ndarray, month: np.ndarray, pile_rate: np.ndarray) -> dict:
    firing = np.stack([(activity[month == m] > 0).mean(0) for m in range(len(MONTHS))])
    units = np.flatnonzero((firing.max(0) >= MONTH_FIRING) & (pile_rate < PILE_FIRING))
    mass = activity[:, units].sum(0)
    return dict(zip(units.tolist(), (mass / mass.sum()).tolist()))


def month_feature(sasa: SASAGroups, sae, owner: dict, natural: torch.Tensor, month: np.ndarray,
                  corpus: torch.Tensor) -> dict:
    rates = pile_rates(sasa, sae, corpus)
    codes = sasa_codes(sasa, natural)
    activity = {"sae": sae_activations(sae, natural).numpy(), "sasa": codes.norm(dim=-1).numpy()}
    shares = {d: month_units(activity[d], month, rates[d]) for d in ("sae", "sasa")}
    sae_ring = np.unique(owner["sae"])
    preferred = np.stack([activity["sae"][month == m][:, sae_ring].mean(0) for m in range(len(MONTHS))]).argmax(0)
    ring = {"sae": sae_ring[np.argsort(preferred)].tolist(),
            "sasa": sorted(np.unique(owner["sasa"]).tolist(), key=lambda g: -shares["sasa"][g])}
    assert all(set(ring[d]) <= set(shares[d]) for d in ring), "every ring unit is a month unit"
    group = ring["sasa"][0]
    return {"dictionaries": {d: {"ring_owner": owner[d].tolist(), "ring_units": ring[d], "month_units": list(shares[d]),
                                 "shares": list(shares[d].values())} for d in ("sae", "sasa")},
            "group": {"index": group, "codes": codes[:, group].tolist(), "month": month.tolist()}}


def main() -> None:
    torch.set_grad_enabled(False)
    config = load_config(SASA.split("/")[0])
    lm = LanguageModel(config["model_name"], layer=int(config["hook_name"].split(".")[1]))
    sets = concept_sets()
    states = {name: lm.last_token_states([prompt for prompt, _, _ in prompts]) for name, prompts in sets.items()}
    result = {"concepts": concept_coordinates(sets, states), "steering": steering_curves(lm, sets, states)}

    texts = pile_texts()
    corpus = pile_states(lm, texts[:PILE_DOCUMENTS], config["context_size"])
    natural, month = natural_month_names(lm, texts)

    sasa = SASAGroups(load_sae(SASA))
    sasa.threshold = inference_threshold(sasa, corpus)
    sae = load_sae(ENGELS)
    carriers = np.array([carrier for _, _, carrier in sets["months"]])
    owner = month_ring(sasa, sae, states["months"], carriers)
    result.update(month_feature(sasa, sae, owner, natural, month, corpus))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mistral_l8.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()

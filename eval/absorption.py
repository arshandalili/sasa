import argparse
import gc
import gzip
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sae_bench.evals.absorption.probing import train_multi_probe
from sae_bench.evals.absorption.prompting import create_icl_prompt, first_letter_formatter
from sae_bench.evals.absorption.vocab import get_alpha_tokens
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from transformer_lens import HookedTransformer

import paths
from configs import load_config
from eval.dictionaries import EngelsReLU, load_sae
from eval.gate import gate_threshold
from eval.member_prediction import family_state_files

LETTERS = "abcdefghijklmnopqrstuvwxyz"
# SAEBench's first-letter benchmark (AbsorptionEvalConfig, feature_absorption_calculator.py, k_sparse_probing.py)
PROMPT_TEMPLATE = "{word} has the first letter:"
WORD_POSITION = -6
SILENT = 1e-8
ABSORBER_COSINE, FULL_ABSORBER_COSINE = 0.1, 0.025
ABSORBED_SHARE = 0.4
N_ABSORBERS = 3
F1_GAIN = 0.03
MAX_MAIN_UNITS = 10
MIN_PROBE_F1, MIN_ELIGIBLE = 0.6, 20
SPARSE_PROBE_L1 = 0.01
N_BACKGROUND = 20000
PROBE_SET_BATCH, PROMPT_BATCH, PROBE_EVAL_BATCH, CODE_BATCH = 64, 32, 24, 4096


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_dir(evaluation: str, config_name: str) -> Path:
    return paths.RESULTS / "absorption" / evaluation / config_name / "prep"


def probe_dir(config: dict) -> Path:
    layer = int(config["hook_name"].split(".")[1])
    return paths.DATA / "absorption" / "probes" / config["model_name"].split("/")[-1] / f"layer_{layer}"


@torch.no_grad()
def build_probe_data(model, config: dict, directory: Path) -> None:
    hook = config["hook_name"]
    arrays = {}
    for split in ("train", "test"):
        frame = pd.read_csv(directory / f"{split}_df.csv", keep_default_na=False, na_values=[""])
        prompts = frame["prompt"].tolist()
        activations = np.empty((len(prompts), model.cfg.d_model), dtype=np.float32)
        for start in range(0, len(prompts), PROBE_SET_BATCH):
            cache = model.run_with_cache(prompts[start:start + PROBE_SET_BATCH], names_filter=[hook])[1]
            activations[start:start + PROBE_SET_BATCH] = cache[hook][:, WORD_POSITION, :].float().cpu().numpy()
        arrays[f"X_{split}"], arrays[f"y_{split}"] = activations, frame["answer_class"].to_numpy(dtype=np.int64)
    np.savez_compressed(directory / "data.npz", **arrays)


def load_probe_split(tokenizer, directory: Path, split: str) -> tuple[torch.Tensor, list]:
    data = np.load(directory / "data.npz")
    frame = pd.read_csv(directory / f"{split}_df.csv", keep_default_na=False, na_values=[""])
    activations = torch.from_numpy(data[f"X_{split}"]).to("cpu", dtype=torch.bfloat16)
    rows, words = [], []
    for row, letter in enumerate(data[f"y_{split}"].tolist()):
        token = frame.iloc[row]["token"]
        if not isinstance(token, str) or re.match(r"[\d<>]", token):  # digits and byte tokens such as <0x6A>
            continue
        words.append((tokenizer.convert_tokens_to_string([token]), letter))
        rows.append(row)
    return activations[rows].clone(), words


def letter_prompts(model, config_name: str, seed: int, positives: dict) -> tuple[list, list, list]:
    words = [word for letter in LETTERS for word in positives[letter]]
    letters = [index for index, letter in enumerate(LETTERS) for _ in positives[letter]]
    # SAEBench draws the in-context examples from a set whose order changes between processes, so they are shipped
    shipped_file = paths.DATA / "absorption" / "prompts" / f"{config_name}.json.gz"
    shipped = json.loads(gzip.open(shipped_file).read()) if shipped_file.exists() else {"prompts": {}}
    if str(seed) in shipped["prompts"]:
        assert shipped["word"] == words and shipped["letter"] == letters, "shipped prompts do not match this probe set"
        return words, letters, shipped["prompts"][str(seed)]
    seed_everything(seed)
    vocabulary, formatter = sorted(get_alpha_tokens(model.tokenizer)), first_letter_formatter()
    prompts = [create_icl_prompt(word, examples=vocabulary, base_template=PROMPT_TEMPLATE, answer_formatter=formatter,
                                 example_separator="\n", max_icl_examples=10, shuffle_examples=True,
                                 prepend_separator_to_first_example=True).base for word in words]
    return words, letters, prompts


def same_length_mask(letters: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    # the word sits at WORD_POSITION only in the prompts of the most common length of its letter
    keep = torch.zeros(len(lengths), dtype=torch.bool)
    for letter in range(len(LETTERS)):
        rows = (letters == letter).nonzero().flatten().tolist()
        if rows:
            most_common = Counter(lengths[i] for i in rows).most_common(1)[0][0]
            for i in rows:
                keep[i] = lengths[i] == most_common
    return keep


@torch.no_grad()
def word_activations(model, prompts: list[str], lengths: list[int], hook: str, dtype) -> torch.Tensor:
    activations = torch.zeros(len(prompts), model.cfg.d_model, dtype=dtype)
    rows_by_length = {}
    for row, length in enumerate(lengths):
        rows_by_length.setdefault(length, []).append(row)
    for length, rows in rows_by_length.items():
        for start in range(0, len(rows), PROMPT_BATCH):
            batch = rows[start:start + PROMPT_BATCH]
            tokens = model.to_tokens([prompts[i] for i in batch])
            assert tokens.shape[1] == length
            cache = model.run_with_cache(tokens, names_filter=[hook])[1]
            activations[batch] = cache[hook][:, WORD_POSITION, :].to("cpu", dtype)
    return activations


def prep_letters(config_name: str, seeds: list[int]) -> None:
    config = load_config(config_name)
    out = prep_dir("letters", config_name)
    out.mkdir(parents=True, exist_ok=True)
    probes = probe_dir(config)
    llm_dtype = getattr(torch, config["dtype"])
    model = HookedTransformer.from_pretrained_no_processing(config["model_name"], device="cuda", dtype=llm_dtype)
    if not (probes / "data.npz").exists():
        build_probe_data(model, config, probes)
    probe = torch.load(probes / "probe.pth", map_location="cuda", weights_only=False).to(dtype=torch.bfloat16)
    train_x, train_words = load_probe_split(model.tokenizer, probes, "train")
    test_x, test_words = load_probe_split(model.tokenizer, probes, "test")
    with torch.no_grad():
        scores = torch.cat([probe(test_x[i:i + PROBE_EVAL_BATCH].cuda()).float().cpu()
                            for i in range(0, len(test_x), PROBE_EVAL_BATCH)])
    test_letters = np.array([letter for _, letter in test_words])
    test_tokens = [word for word, _ in test_words]
    f1, positives = {}, {}
    for index, letter in enumerate(LETTERS):
        is_letter, predicted = test_letters == index, scores[:, index].numpy() > 0.0
        f1[letter] = float(metrics.f1_score(is_letter, predicted))
        positives[letter] = [test_tokens[i] for i in np.where(is_letter & predicted)[0]]
    eligible = [letter for letter in LETTERS if f1[letter] > MIN_PROBE_F1]
    assert len(eligible) >= MIN_ELIGIBLE, "SAEBench would refuse this model and layer"
    torch.save({"concepts": list(LETTERS), "n_classes": len(LETTERS), "train_x": train_x,
                "train_y": torch.tensor([letter for _, letter in train_words]), "test_x": test_x,
                "test_y": torch.tensor(test_letters), "probe_weights": probe.weights.detach().float().cpu(),
                "eligible": eligible}, out / "probe_data.pt")
    for seed in seeds:
        word_file = out / f"words_seed{seed}.pt"
        if word_file.exists():
            continue
        _, letter_indices, prompts = letter_prompts(model, config_name, seed, positives)
        lengths = [len(model.to_tokens(prompt)[0]) for prompt in prompts]
        letters = torch.tensor(letter_indices)
        scored = same_length_mask(letters, lengths)
        if config["model_name"] == "gpt2":
            assert bool(scored.all()), "GPT-2 prompts must all have one length"
        activations = word_activations(model, prompts, lengths, config["hook_name"], llm_dtype)
        torch.save({"activations": activations, "concept": letters, "scored": scored}, word_file)


def half_split(labels: list[str]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    rows_by_label = {}
    for row, label in enumerate(labels):
        rows_by_label.setdefault(label, []).append(row)
    train = torch.zeros(len(labels), dtype=torch.bool)
    for rows in rows_by_label.values():
        chosen = torch.randperm(len(rows), generator=generator)[: len(rows) // 2]
        train[torch.tensor(rows)[chosen]] = True
    return train


def prep_families(config_name: str, seeds: list[int]) -> None:
    out = prep_dir("families", config_name)
    out.mkdir(parents=True, exist_ok=True)
    states_file, background_file = family_state_files(config_name)
    collected = torch.load(states_file, map_location="cpu", weights_only=False)
    states, members = collected["states"], collected["members"]
    background = torch.load(background_file, map_location="cpu", weights_only=False)[:N_BACKGROUND].float()
    concepts = list(members)
    activations, classes, labels = [], [], []
    for index, concept in enumerate(concepts):
        for member_index, member in enumerate(members[concept]):
            occurrences = states.get((concept, member_index), [])
            if len(occurrences):
                activations.append(torch.stack(list(occurrences)).float())
                classes += [index] * len(occurrences)
                labels += [f"{concept}:{member}"] * len(occurrences)
    activations = torch.cat(activations + [background])
    classes = torch.tensor(classes + [len(concepts)] * len(background))
    labels = labels + ["background"] * len(background)
    train = half_split(labels)
    test = ~train
    seed_everything(0)
    targets = torch.nn.functional.one_hot(classes[train], num_classes=len(concepts) + 1)[:, : len(concepts)]
    probe = train_multi_probe(activations[train].cuda(), targets, num_probes=len(concepts), batch_size=64,
                              num_epochs=50, lr=1e-2, weight_decay=1e-4, show_progress=False,
                              device=torch.device("cuda"))
    with torch.no_grad():
        scores = probe(activations[test].cuda()).float().cpu()
    test_classes = classes[test].numpy()
    f1, positives = {}, {}
    for index, concept in enumerate(concepts):
        is_concept, predicted = test_classes == index, scores[:, index].numpy() > 0.0
        f1[concept] = float(metrics.f1_score(is_concept, predicted))
        positives[concept] = np.where(is_concept & predicted)[0].tolist()
    eligible = [concept for concept in concepts if f1[concept] > MIN_PROBE_F1]
    test_x = activations[test]
    torch.save({"concepts": concepts, "n_classes": len(concepts) + 1, "train_x": activations[train],
                "train_y": classes[train], "test_x": test_x, "test_y": classes[test],
                "probe_weights": probe.weights.detach().float().cpu(), "eligible": eligible}, out / "probe_data.pt")
    rows = [i for concept in concepts for i in positives[concept]]
    words = {"activations": test_x[rows],
             "concept": torch.tensor([index for index, concept in enumerate(concepts) for _ in positives[concept]]),
             "scored": torch.ones(len(rows), dtype=torch.bool)}
    for seed in seeds:
        torch.save(words, out / f"words_seed{seed}.pt")


class Dictionary:
    # codes in activation units (times the decoder-row norm, and the layer-norm scale of SASA's input) on unit-norm
    # decoder rows, read in units of unit_size consecutive columns: a latent, or a SASA group for concept families
    def __init__(self, sae, evaluation: str, threshold: float | None = None):
        self.sae, self.threshold = sae, threshold
        self.engels = isinstance(sae, EngelsReLU)
        self.unit_size = getattr(sae, "group_rank", 1) if evaluation == "families" else 1
        decoder = sae.W_dec.detach().float()
        self.n_columns, self.d_in = decoder.shape
        self.decoder_norms = decoder.norm(dim=-1)
        self.directions = decoder / self.decoder_norms[:, None]
        self.layer_norm = getattr(sae.cfg, "normalize_activations", None) == "layer_norm"
        self.n_units = self.n_columns // self.unit_size
        unit_directions = self.directions.reshape(self.n_units, self.unit_size, self.d_in)
        self.unit_gram = unit_directions @ unit_directions.transpose(-1, -2)
        if self.unit_size > 1:
            # a unit's output in orthonormal coordinates, V_u^T y_u with V_u the right singular vectors of its decoder
            right_singular = torch.linalg.svd(decoder.reshape(self.n_units, self.unit_size, self.d_in).double(),
                                              full_matrices=False)[2]
            self.output_map = (unit_directions.double() @ right_singular.transpose(-1, -2)).transpose(-1, -2).float()
        self.signs = None

    @torch.no_grad()
    def codes(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.threshold is not None:  # BatchTopK-trained SASA: the groups above the inference threshold
            pre_acts = self.sae.process_sae_in(x) @ self.sae.W_enc + self.sae.b_enc
            groups = pre_acts.view(-1, self.sae.n_groups, self.sae.group_rank)
            codes = (groups * (groups.norm(dim=-1, keepdim=True) > self.threshold)).reshape(x.shape[0], -1)
        else:
            codes = self.sae.encode(x).float()
        if self.engels:
            return codes * (self.sae.decoder_norms.float()[None, :] / self.sae.input_scale.reshape(-1, 1).float())
        scale = self.decoder_norms[None, :]
        if self.layer_norm:
            scale = scale * self.sae.ln_std.reshape(-1, 1).float()
        return codes * scale

    def units(self, columns: torch.Tensor) -> torch.Tensor:
        return columns.reshape(columns.shape[0], self.n_units, self.unit_size)

    def orient(self, train_codes: torch.Tensor) -> None:
        mean = torch.zeros(self.n_units, self.unit_size, device=self.output_map.device)
        for start in range(0, train_codes.shape[0], CODE_BATCH):
            mean += self.units(train_codes[start:start + CODE_BATCH].cuda()).sum(0)
        mean /= train_codes.shape[0]
        self.signs = torch.where(torch.einsum("uij,uj->ui", self.output_map, mean) < 0, -1.0, 1.0)

    def unit_coordinates(self, codes: torch.Tensor) -> torch.Tensor:
        out = torch.empty(codes.shape[0], self.n_columns, dtype=torch.float32)
        for start in range(0, codes.shape[0], CODE_BATCH):
            batch = codes[start:start + CODE_BATCH].cuda()
            coordinates = torch.einsum("uij,nuj->nui", self.output_map, self.units(batch)) * self.signs[None]
            out[start:start + CODE_BATCH] = coordinates.reshape(batch.shape[0], -1).cpu()
        return out


def encode_all(dictionary: Dictionary, activations: torch.Tensor) -> torch.Tensor:
    out = torch.empty(len(activations), dictionary.n_columns, dtype=torch.float32)
    for start in range(0, len(activations), CODE_BATCH):
        out[start:start + CODE_BATCH] = dictionary.codes(activations[start:start + CODE_BATCH].cuda()).cpu()
    return out


def sparsity_penalty(unit_size: int, n_probes: int):
    def penalty(probe, _activations, _labels):
        if unit_size == 1:
            return SPARSE_PROBE_L1 * probe.weights.abs().sum(dim=-1).mean()
        return SPARSE_PROBE_L1 * probe.weights.reshape(n_probes, -1, unit_size).norm(dim=-1).sum(dim=-1).mean()
    return penalty


def train_sparse_probe(codes, labels, unit_size: int, seed: int, n_probes: int, n_classes: int) -> torch.Tensor:
    seed_everything(seed)
    targets = torch.nn.functional.one_hot(labels, num_classes=n_classes)[:, :n_probes]
    probe = train_multi_probe(codes, targets, num_probes=n_probes, batch_size=4096, num_epochs=50, lr=0.01, end_lr=1e-5,
                              weight_decay=1e-6, show_progress=False, verbose=False, device=torch.device("cuda"),
                              extra_loss_fn=sparsity_penalty(unit_size, n_probes), map_acts=None,
                              probe_dim=codes.shape[1])
    return probe.weights.detach().float()


def select_main_units(train_codes, train_labels, test_codes, test_labels, probe_weights, unit_size: int,
                      eligible: list, concepts) -> dict:
    n_units = probe_weights.shape[1] // unit_size
    selection = {}
    for concept in eligible:
        index = list(concepts).index(concept)
        weights = probe_weights[index]
        if unit_size == 1:
            unit_scores = weights
        else:
            positive_sum = train_codes[(train_labels == index).to(train_codes.device)].sum(0).to(weights.device)
            alignment = (weights * positive_sum).reshape(n_units, unit_size).sum(-1)
            unit_scores = weights.reshape(n_units, unit_size).norm(dim=-1) * torch.sign(alignment)
        train_target = (train_labels == index).numpy().astype(np.int64)
        test_target = (test_labels == index).numpy()
        f1_by_k, units_by_k = [], []
        for k in range(1, MAX_MAIN_UNITS + 1):
            units = unit_scores.topk(k).indices
            columns = (units[:, None] * unit_size + torch.arange(unit_size, device=units.device)[None]).flatten()
            columns = columns.to(train_codes.device)
            regression = LogisticRegression(max_iter=500, class_weight="balanced")
            regression.fit(train_codes[:, columns].float().cpu().numpy(), train_target)
            scores = (test_codes[:, columns.to(test_codes.device)].float().cpu().numpy() @ regression.coef_[0]
                      + regression.intercept_[0])
            f1_by_k.append(float(metrics.f1_score(test_target, scores > 0.0)))
            units_by_k.append(units.cpu().tolist())
        best_f1, main_units = -100.0, []
        for f1, units in zip(f1_by_k, units_by_k):
            if not f1 > best_f1 + F1_GAIN:
                break
            best_f1, main_units = f1, units
        selection[concept] = main_units
    return selection


def word_absorption(contributions, cosines, silent, projection, main_units) -> tuple[torch.Tensor, torch.Tensor]:
    n_words, n_units = contributions.shape
    is_main = torch.zeros(n_units, dtype=torch.bool, device=contributions.device)
    is_main[main_units] = True
    main = contributions[:, is_main].sum(1)
    candidates = (~is_main)[None, :] & (cosines >= ABSORBER_COSINE) & (contributions > 0)
    top = torch.where(candidates, contributions, torch.full_like(contributions, -float("inf")))
    top = top.topk(N_ABSORBERS, dim=1).values
    absorbed = torch.where(torch.isfinite(top), top, torch.zeros_like(top)).sum(1)
    no_absorption = (main >= projection) | (absorbed / projection < ABSORBED_SHARE)
    main_not_positive = ~no_absorption & (main <= 0.0)
    partial = ~no_absorption & ~main_not_positive
    fraction = torch.zeros(n_words, dtype=torch.float64, device=contributions.device)
    fraction[main_not_positive] = 1.0
    capped = torch.minimum(absorbed, projection - main)
    fraction[partial] = (capped / (capped + main)).clamp(0.0, 1.0)[partial].double()
    strongest = contributions.argmax(1, keepdim=True)
    strongest_contribution = contributions.gather(1, strongest).squeeze(1)
    strongest_cosine = cosines.gather(1, strongest).squeeze(1)
    full = (silent[:, is_main].all(1) & (strongest_cosine >= FULL_ABSORBER_COSINE) & (projection >= 0)
            & (strongest_contribution / projection >= ABSORBED_SHARE))
    return fraction, full


def latent_absorption(dictionary: Dictionary, activations, projection, direction, main_units) -> tuple:
    codes = dictionary.codes(activations)
    cosines = dictionary.directions @ direction
    contributions = codes * cosines[None, :]
    return word_absorption(contributions, cosines[None, :].expand_as(contributions), codes < SILENT, projection,
                           main_units)


def unit_absorption(dictionary: Dictionary, activations, projection, direction, main_units) -> tuple:
    # a group contributes its output y_u = sum_j c_j d_j projected on the probe, with the cosine of y_u
    codes = dictionary.codes(activations)
    cosines = dictionary.directions @ direction
    unit_codes = dictionary.units(codes)
    contributions = dictionary.units(codes * cosines[None, :]).sum(-1)
    output_norms = torch.einsum("nur,urs,nus->nu", unit_codes, dictionary.unit_gram, unit_codes).clamp(min=0).sqrt()
    unit_cosines = torch.where(output_norms > 0, contributions / output_norms.clamp(min=1e-30),
                               torch.zeros_like(contributions))
    return word_absorption(contributions, unit_cosines, output_norms < SILENT, projection, main_units)


def score_dictionary(evaluation: str, name: str, dictionary: Dictionary, probe_data: dict, word_files: dict,
                     seeds: list[int], out: Path) -> None:
    concepts, eligible = probe_data["concepts"], probe_data["eligible"]
    n_probes, n_classes = len(concepts), probe_data["n_classes"]
    probe_weights = probe_data["probe_weights"]
    probe_directions = (probe_weights / probe_weights.norm(dim=-1, keepdim=True)).cuda()
    train_codes = encode_all(dictionary, probe_data["train_x"])
    test_codes = encode_all(dictionary, probe_data["test_x"])
    train_labels, test_labels = probe_data["train_y"], probe_data["test_y"]
    if dictionary.unit_size > 1:
        dictionary.orient(train_codes)
        train_codes, test_codes = dictionary.unit_coordinates(train_codes), dictionary.unit_coordinates(test_codes)
    absorption = latent_absorption if evaluation == "letters" else unit_absorption
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        probe = train_sparse_probe(train_codes, train_labels, dictionary.unit_size, seed, n_probes, n_classes)
        selection = select_main_units(train_codes, train_labels, test_codes, test_labels, probe, dictionary.unit_size,
                                      eligible, concepts)
        words = word_files[seed]
        mean, full = [], []
        for concept in eligible:
            index = concepts.index(concept)
            rows = ((words["concept"] == index) & words["scored"]).nonzero().flatten()
            activations = words["activations"][rows].cuda().float()
            direction = probe_directions[index]
            fraction, fully_absorbed = absorption(dictionary, activations, activations @ direction, direction,
                                                  selection[concept])
            mean.append(float(fraction.sum()) / max(len(rows), 1))
            full.append(float(fully_absorbed.sum()) / max(len(rows), 1))
        record = {"sae": name, "seed": seed, "evaluation": evaluation, "mean_absorption": float(np.mean(mean)),
                  "full_absorption": float(np.mean(full))}
        (out / f"seed{seed}.json").write_text(json.dumps(record, indent=1))


def score(evaluation: str, names: list[str], seeds: list[int]) -> None:
    config_name = names[0].split("/")[0]
    assert all(name.split("/")[0] == config_name for name in names), "SAEs of one config per run"
    prep = prep_dir(evaluation, config_name)
    probe_data = torch.load(prep / "probe_data.pt", weights_only=False)
    word_files = {seed: torch.load(prep / f"words_seed{seed}.pt", weights_only=False) for seed in seeds}
    for name in names:
        out = paths.RESULTS / "absorption" / evaluation / name
        todo = [seed for seed in seeds if not (out / f"seed{seed}.json").exists()]
        if not todo:
            continue
        torch.cuda.empty_cache()
        sae = load_sae(name)
        threshold = None
        if getattr(sae.cfg, "gate", None) == "batchtopk":
            threshold = gate_threshold(name)
            if threshold is None:
                raise SystemExit(f"{name}: calibrate the inference threshold first (python -m eval.gate {name})")
        dictionary = Dictionary(sae, evaluation, threshold)
        score_dictionary(evaluation, name, dictionary, probe_data, word_files, todo, out)
        del sae, dictionary
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    def seed_list(text):
        return [int(seed) for seed in text.split(",")]

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prep", help="probe and word activations of a config, shared by its SAEs")
    prep.add_argument("evaluation", choices=("letters", "families"))
    prep.add_argument("config", help="gpt2_l7 or mistral_l8 (first letters), gpt2_l7 (concept families)")
    prep.add_argument("--seeds", type=seed_list, default=[0, 1, 2])
    scoring = commands.add_parser("score", help="absorption of SAEs, one result file per seed")
    scoring.add_argument("evaluation", choices=("letters", "families"))
    scoring.add_argument("saes", nargs="+", help="<config>/<sae> ...")
    scoring.add_argument("--seeds", type=seed_list, default=[0, 1, 2])
    args = parser.parse_args()
    if args.command == "prep":
        (prep_letters if args.evaluation == "letters" else prep_families)(args.config, args.seeds)
    else:
        score(args.evaluation, args.saes, args.seeds)


if __name__ == "__main__":
    main()

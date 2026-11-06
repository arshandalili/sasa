import argparse
import json

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import paths
from configs import load_config
from eval.dictionaries import load_sae

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
          "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CONCEPTS = {"month": MONTHS, "day": DAYS}
OTHER_CONCEPT = {"month": "day", "day": "month"}
TEMPLATES = {  # "{t}" is the concept word; the text after it repeats the context, so the next word is a concept word
    "month": ["The event happened in {t}. The event happened in", "The contract ends in {t}. The contract ends in",
              "She will return in {t}. She will return in", "The festival is held in {t}. The festival is held in",
              "The report is due in {t}. The report is due in", "The season starts in {t}. The season starts in",
              "The meeting was moved to {t}. The meeting was moved to", "Our lease ends in {t}. Our lease ends in",
              "The store closed in {t}. The store closed in", "The trial began in {t}. The trial began in",
              "The book came out in {t}. The book came out in",
              "The treaty was signed in {t}. The treaty was signed in",
              "The project was approved in {t}. The project was approved in",
              "We adopted a cat in {t}. We adopted a cat in",
              "The bridge was completed in {t}. The bridge was completed in",
              "Her contract expires in {t}. Her contract expires in"],
    "day": ["The meeting is on {t}. The meeting is on", "She called me on {t}. She called me on",
            "The deadline is {t}. The deadline is", "They travel every {t}. They travel every",
            "The class meets on {t}. The class meets on", "He arrived last {t}. He arrived last",
            "Payday falls on {t}. Payday falls on", "The show airs every {t}. The show airs every",
            "The market opens on {t}. The market opens on", "We play tennis on {t}. We play tennis on",
            "The package arrived on {t}. The package arrived on", "Her shift starts on {t}. Her shift starts on",
            "The museum is closed on {t}. The museum is closed on", "Rehearsal is every {t}. Rehearsal is every",
            "The trash is collected on {t}. The trash is collected on", "Our next game is on {t}. Our next game is on"],
}
NOUNS = ["lamp", "chair", "bottle", "hammer", "basket", "mirror", "pillow", "kettle", "ladder", "blanket", "candle",
         "bucket", "camera", "clock", "drawer", "envelope", "fork", "guitar", "helmet", "jacket", "knife", "napkin",
         "pencil", "plate", "radio", "ribbon", "shovel", "spoon", "towel", "umbrella", "vase", "wallet", "whistle"]
DATE_WORDS = {"▁in", "▁of", "▁early", "▁late", "▁mid", "▁since", "▁until", "▁by", "▁from", "▁last", "▁next", "▁this",
              "▁during", "▁through", "▁on", "▁between", "▁before", "▁after"}
DATE_ONLY_MONTHS = ("May", "March")  # kept as positives only in a date context
N_PROTOTYPE_TEMPLATES = 4  # templates that estimate the steering vectors; the others evaluate them
NOUNS_PER_TEMPLATE = 8
N_CONTROL_DRAWS = 5
STEERING_SCALES = [0.5, 1, 2, 4]
WINDOW, PER_NAME, PER_DOCUMENT = 64, 150, 3
N_BACKGROUND = 20000
PILE_DOCUMENTS, MIN_DOCUMENT_TOKENS, THRESHOLD_BATCH = 400, 32, 4096
LAYER_NORM_EPS = 1e-5


class _StopForward(Exception):
    pass


class LanguageModel:
    def __init__(self, model_name: str, layer: int, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
        self.block = self.model.model.layers[layer]

    def ids(self, text: str) -> list[int]:
        return self.tokenizer(text)["input_ids"]  # starts with BOS

    def site(self, prefix: str, suffix: str) -> tuple[list[int], int]:
        prefix_ids, ids = self.ids(prefix), self.ids(prefix + suffix)
        assert ids[:len(prefix_ids)] == prefix_ids, (prefix, suffix)
        return ids, len(prefix_ids) - 1

    def first_id(self, word: str) -> int:
        base, full = self.ids("x"), self.ids("x " + word)
        assert full[:len(base)] == base
        return full[len(base)]

    @torch.no_grad()
    def block_input(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        captured = {}

        def capture(_module, args, kwargs):
            captured["h"] = (args[0] if args else kwargs["hidden_states"]).float()
            raise _StopForward

        handle = self.block.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            self.model.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device))
        except _StopForward:
            pass
        finally:
            handle.remove()
        return captured["h"]

    @torch.no_grad()
    def last_token_states(self, texts: list[str], batch: int = 64) -> torch.Tensor:
        out = []
        for start in range(0, len(texts), batch):
            sequences = [self.ids(text) for text in texts[start:start + batch]]
            length = max(map(len, sequences))
            ids = torch.zeros(len(sequences), length, dtype=torch.long)
            mask = torch.zeros(len(sequences), length, dtype=torch.long)
            for row, sequence in enumerate(sequences):
                ids[row, :len(sequence)], mask[row, :len(sequence)] = torch.tensor(sequence), 1
            hidden = self.block_input(ids, mask)
            out.append(hidden[torch.arange(len(sequences)), torch.tensor([len(s) - 1 for s in sequences])].cpu())
        return torch.cat(out)

    @torch.no_grad()
    def all_states(self, ids: list[int]) -> torch.Tensor:
        sequence = torch.tensor([ids])
        return self.block_input(sequence, torch.ones_like(sequence))[0].cpu()

    @torch.no_grad()
    def edited_log_probs(self, ids: list[int], position: int, deltas: torch.Tensor, targets: list[int],
                         batch: int = 128) -> torch.Tensor:
        sequence = torch.tensor([ids], device=self.device)
        targets = torch.tensor(targets, device=self.device)
        out = []
        for start in range(0, len(deltas), batch):
            chunk = deltas[start:start + batch].to(self.device, torch.bfloat16)

            def add(_module, args, kwargs):
                hidden = (args[0] if args else kwargs["hidden_states"]).clone()
                hidden[:, position] += chunk
                return ((hidden,) + tuple(args[1:]), kwargs) if args else (args, dict(kwargs, hidden_states=hidden))

            handle = self.block.register_forward_pre_hook(add, with_kwargs=True)
            try:
                logits = self.model(input_ids=sequence.expand(len(chunk), -1)).logits
            finally:
                handle.remove()
            out.append(logits[:, -1].float().log_softmax(-1)[:, targets].cpu())
        return torch.cat(out)

    def concept_mass(self, text: str, suffix: str, deltas: torch.Tensor, id_sets: list[list[int]]) -> list[np.ndarray]:
        ids, position = self.site(text, suffix)
        probs = self.edited_log_probs(ids, position, deltas, [i for id_set in id_sets for i in id_set]).exp()
        masses, start = [], 0
        for id_set in id_sets:
            masses.append(probs[:, start:start + len(id_set)].sum(1).numpy())
            start += len(id_set)
        return masses


class SASAGroups:
    def __init__(self, sae, device: str = "cuda"):
        self.device = device
        self.W_enc, self.b_enc, self.W_dec = (getattr(sae, k).detach().float() for k in ("W_enc", "b_enc", "W_dec"))
        self.rank, self.k_groups = sae.group_rank, sae.k_groups
        self.decoder_blocks = self.W_dec.view(sae.n_groups, self.rank, -1)  # (K, r, d)
        self.d_model = self.decoder_blocks.shape[-1]
        self.threshold = None

    def pre_activations(self, h: torch.Tensor):
        mean = h.mean(-1, keepdim=True)
        centered = h - mean
        scale = centered.std(-1, keepdim=True)
        pre_acts = (centered / (scale + LAYER_NORM_EPS)) @ self.W_enc + self.b_enc
        return pre_acts.view(len(h), -1, self.rank), mean, scale

    def norms(self, h: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        return torch.cat([self.pre_activations(h[i:i + chunk].to(self.device))[0].norm(dim=-1).cpu()
                          for i in range(0, len(h), chunk)])

    def codes(self, h: torch.Tensor):
        pre_acts, mean, scale = self.pre_activations(h)
        return pre_acts * (pre_acts.norm(dim=-1, keepdim=True) > self.threshold), mean, scale

    def contribution(self, h: torch.Tensor, group: int) -> torch.Tensor:
        codes, _, scale = self.codes(h.to(self.device))
        return ((codes[:, group] @ self.decoder_blocks[group]) * scale).cpu()

    def basis(self, groups: list[int]) -> torch.Tensor:
        return torch.linalg.qr(self.decoder_blocks[groups].reshape(-1, self.d_model).T)[0]

    def project_out(self, h: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        h = h.to(self.device)
        mean = h.mean(-1, keepdim=True)
        centered = h - mean
        scale = centered.std(-1, keepdim=True)
        normalized = centered / (scale + LAYER_NORM_EPS)
        return ((normalized - (normalized @ basis) @ basis.T) * (scale + LAYER_NORM_EPS) + mean).cpu()


def template_mean(values) -> float:
    return round(float(np.mean(values)), 4)


def random_basis(d: int, dimension: int, seed: int, device: str) -> torch.Tensor:
    return torch.linalg.qr(torch.randn(d, dimension, generator=torch.Generator().manual_seed(seed)))[0].to(device)


def random_unit(d: int, seed: int) -> torch.Tensor:
    vector = torch.randn(1, d, generator=torch.Generator().manual_seed(seed))
    return vector / vector.norm()


def pile_texts() -> list[str]:
    return load_dataset("NeelNanda/pile-10k", split="train")["text"]


def pile_states(lm: LanguageModel, texts: list[str], context: int) -> torch.Tensor:
    states = []
    for ids in map(lm.ids, texts[:PILE_DOCUMENTS]):
        if len(ids) >= MIN_DOCUMENT_TOKENS:
            states.append(lm.all_states(ids[:context])[1:].half())
    return torch.cat(states)


def inference_threshold(groups: SASAGroups, corpus: torch.Tensor) -> float:
    cutoffs = []
    for start in range(0, len(corpus) - THRESHOLD_BATCH + 1, THRESHOLD_BATCH):
        norms = groups.pre_activations(corpus[start:start + THRESHOLD_BATCH].to(groups.device).float())[0].norm(dim=-1)
        cutoffs.append(norms.flatten().topk(groups.k_groups * THRESHOLD_BATCH).values[-1])
    return float(torch.stack(cutoffs).mean())


def date_context(tokens: list[str], position: int) -> bool:
    previous = tokens[position - 1]
    following = tokens[position + 1] if position + 1 < len(tokens) else ""
    after_following = tokens[position + 2] if position + 2 < len(tokens) else ""
    return (previous in DATE_WORDS or previous.strip("▁").isdigit() or following.isdigit()
            or (following == "▁" and after_following.isdigit()))


def natural_windows(lm: LanguageModel, texts: list[str], name_of_id: dict, date_only: set) -> list[list[int]]:
    counts, windows = {name: 0 for name in name_of_id.values()}, []
    for text in texts:
        ids = lm.tokenizer(text, add_special_tokens=False)["input_ids"][:4096]
        tokens = lm.tokenizer.convert_ids_to_tokens(ids) if date_only else None
        taken = 0
        for position in range(WINDOW - 1, len(ids) - 1):
            name = name_of_id.get(ids[position])
            if name is None or counts[name] >= PER_NAME or taken >= PER_DOCUMENT:
                continue
            if name in date_only and not date_context(tokens, position):
                continue
            windows.append([lm.tokenizer.bos_token_id] + ids[position - WINDOW + 1:position + 1])
            counts[name] += 1
            taken += 1
        if min(counts.values()) >= PER_NAME:
            break
    return windows


def window_states(lm: LanguageModel, windows: list[list[int]], batch: int = 64) -> torch.Tensor:
    states = []
    for start in range(0, len(windows), batch):
        chunk = windows[start:start + batch]
        mask = torch.ones(len(chunk), WINDOW + 1, dtype=torch.long)
        states.append(lm.block_input(torch.tensor(chunk), mask)[:, -1].cpu())
    return torch.cat(states)


def screen(lm: LanguageModel, groups: SASAGroups, texts: list[str], background: torch.Tensor) -> dict:
    month_ids = {lm.first_id(name): index for index, name in enumerate(MONTHS)}
    day_ids = {lm.first_id(name): index for index, name in enumerate(DAYS)}
    date_only = {MONTHS.index(name) for name in DATE_ONLY_MONTHS}
    positives = {"month": window_states(lm, natural_windows(lm, texts, month_ids, date_only)).half().float(),
                 "day": window_states(lm, natural_windows(lm, texts, day_ids, set()))}
    background_norms = groups.norms(background)
    ranking = {}
    for concept in CONCEPTS:
        positive_norms = groups.norms(positives[concept])
        n_positive, n_background = len(positive_norms), len(background_norms)
        ranks = torch.cat([positive_norms, background_norms]).argsort(0).argsort(0).float() + 1
        auroc = (ranks[:n_positive].sum(0) - n_positive * (n_positive + 1) / 2) / (n_positive * n_background)
        group, runner_up = auroc.argsort(descending=True)[:2].tolist()
        ranking[concept] = {"group": group, "runner_up": runner_up,
                            "auroc": [round(float(auroc[g]), 4) for g in (group, runner_up)]}
    return ranking


def ablation(lm: LanguageModel, groups: SASAGroups, concept: str, ranking: dict, word_ids: dict,
             nouns: list[str]) -> dict:
    group, runner_up = ranking[concept]["group"], ranking[concept]["runner_up"]
    other_group = ranking[OTHER_CONCEPT[concept]]["group"]
    templates = TEMPLATES[concept]
    prompts = [(index, template.split("{t}")[0] + word, template.split("{t}")[1])
               for index, template in enumerate(templates) for word in CONCEPTS[concept]]
    states = lm.last_token_states([text for _, text, _ in prompts])
    codes, _, _ = groups.codes(states.to(groups.device))
    firing = (codes.norm(dim=-1) > 0).float().mean(0).cpu()
    firing[[group, runner_up, other_group]] = -1
    candidates = torch.nonzero(firing >= 0.5).flatten().tolist()
    order = torch.randperm(len(candidates), generator=torch.Generator().manual_seed(0))[:N_CONTROL_DRAWS].tolist()
    co_active = [candidates[i] for i in order]
    subspaces = {"group_subspace": groups.basis([group]), "runner_up_subspace": groups.basis([runner_up]),
                 "other_concept_group_subspace": groups.basis([other_group])}
    subspaces.update({f"active_group_subspace/{draw}": groups.basis([g]) for draw, g in enumerate(co_active)})
    subspaces.update({f"random_8d_subspace/{draw}": random_basis(groups.d_model, groups.rank, draw, groups.device)
                      for draw in range(N_CONTROL_DRAWS)})
    contributions = groups.contribution(states, group)
    mass = {}  # intervention -> template -> masses of its prompts (and control draws)
    for row, (template, text, suffix) in enumerate(prompts):
        h = states[row:row + 1]
        edits = {"none": torch.zeros_like(h), "group_write": -contributions[row:row + 1]}
        edits.update({name: groups.project_out(h, basis) - h for name, basis in subspaces.items()})
        removed_norm = float(edits["group_subspace"].norm())
        for draw in range(N_CONTROL_DRAWS):
            edits[f"random_direction_subspace_norm/{draw}"] = -random_unit(groups.d_model, draw) * removed_norm
            write_norm = float(contributions[row].norm())
            edits[f"random_direction_write_norm/{draw}"] = -random_unit(groups.d_model, draw) * write_norm
            noun = nouns[(draw * len(prompts) + row) % len(nouns)]
            edits[f"noun_state/{draw}"] = lm.last_token_states([text.rsplit(" ", 1)[0] + " " + noun]) - h
        edits["zero_state"] = -h
        (masses,) = lm.concept_mass(text, suffix, torch.cat(list(edits.values())), [word_ids[concept]])
        for name, value in zip(edits, masses):
            mass.setdefault(name.split("/")[0], {}).setdefault(template, []).append(float(value))
    per_template = {name: np.array([np.mean(values[t]) for t in range(len(templates))])
                    for name, values in mass.items()}
    unedited, patched = per_template["none"], per_template["noun_state"]
    effect = {name: (unedited - values) / (unedited - patched) for name, values in per_template.items()
              if name not in ("none", "noun_state")}
    controls = ("runner_up_subspace", "other_concept_group_subspace", "active_group_subspace", "random_8d_subspace",
                "random_direction_subspace_norm")
    return {"group": group, "runner_up": runner_up, "other_concept_group": other_group, "active_groups": co_active,
            "probability": {name: template_mean(values) for name, values in per_template.items()},
            "normalized_effect": {name: template_mean(values) for name, values in effect.items()},
            "effect_difference": {f"group_subspace - {name}":
                                  template_mean(effect["group_subspace"] - effect[name]) for name in controls}}


def steering_vectors(lm: LanguageModel, groups: SASAGroups, ranking: dict) -> dict:
    vectors = {}
    for concept in CONCEPTS:
        states = lm.last_token_states([TEMPLATES[concept][t].split("{t}")[0] + word
                                       for t in range(N_PROTOTYPE_TEMPLATES) for word in CONCEPTS[concept]])
        vectors[concept] = {"own": groups.contribution(states, ranking[concept]["group"]).mean(0),
                            "runner_up": groups.contribution(states, ranking[concept]["runner_up"]).mean(0)}
    return vectors


def steering(lm: LanguageModel, groups: SASAGroups, concept: str, vectors: dict, word_ids: dict,
             nouns: list[str]) -> dict:
    other = OTHER_CONCEPT[concept]
    norm = vectors[concept]["own"].norm()
    controls = {"group_prototype": vectors[concept]["own"],
                "runner_up_prototype": vectors[concept]["runner_up"] / vectors[concept]["runner_up"].norm() * norm,
                "other_concept_prototype": vectors[other]["own"] / vectors[other]["own"].norm() * norm}
    controls.update({f"random_vector/{draw}": random_unit(groups.d_model, 100 + draw)[0] * norm
                     for draw in range(N_CONTROL_DRAWS)})
    names = ["none|0"] + [f"{name}|{scale}" for name in controls for scale in STEERING_SCALES]
    deltas = torch.stack([torch.zeros(groups.d_model)] + [scale * vector for vector in controls.values()
                                                            for scale in STEERING_SCALES])
    mass = {}  # "vector|scale" -> template -> (own, other) masses
    evaluation_templates = range(N_PROTOTYPE_TEMPLATES, len(TEMPLATES[concept]))
    for t in evaluation_templates:
        prefix, suffix = TEMPLATES[concept][t].split("{t}")
        for q in range(NOUNS_PER_TEMPLATE):
            noun = nouns[(t * NOUNS_PER_TEMPLATE + q) % len(nouns)]
            own, other_mass = lm.concept_mass(prefix + noun, suffix, deltas, [word_ids[concept], word_ids[other]])
            for name, own_value, other_value in zip(names, own, other_mass):
                key = name.split("/")[0].split("|")[0] + "|" + name.split("|")[1]  # draws of a control share a key
                mass.setdefault(key, {}).setdefault(t, []).append((float(own_value), float(other_value)))
    return {key: {"own": template_mean([np.mean([m[0] for m in values[t]]) for t in evaluation_templates]),
                  "other": template_mean([np.mean([m[1] for m in values[t]]) for t in evaluation_templates])}
            for key, values in mass.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sae", help="mistral_l8/sasa_r8_k10")
    name = parser.parse_args().sae
    torch.set_grad_enabled(False)
    config = load_config(name.split("/")[0])
    lm = LanguageModel(config["model_name"], layer=int(config["hook_name"].split(".")[1]))
    groups = SASAGroups(load_sae(name))
    word_ids = {concept: [lm.first_id(word) for word in words] for concept, words in CONCEPTS.items()}
    for concept, words in CONCEPTS.items():
        assert all(lm.tokenizer.convert_ids_to_tokens(i) == "▁" + w for i, w in zip(word_ids[concept], words)), concept
    nouns = [noun for noun in NOUNS if lm.tokenizer.convert_ids_to_tokens(lm.first_id(noun)) == "▁" + noun
             and len(lm.ids("x " + noun)) == len(lm.ids("x")) + 1]
    texts = pile_texts()
    corpus = pile_states(lm, texts, config["context_size"])
    groups.threshold = inference_threshold(groups, corpus)
    ranking = screen(lm, groups, texts, corpus[:N_BACKGROUND].float())
    result = {"sae": name, "screen": ranking,
              "ablation": {concept: ablation(lm, groups, concept, ranking, word_ids, nouns) for concept in CONCEPTS}}
    vectors = steering_vectors(lm, groups, ranking)
    result["steering"] = {concept: steering(lm, groups, concept, vectors, word_ids, nouns)
                          for concept in CONCEPTS}
    out = paths.RESULTS / "causal" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "causal.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformer_lens import HookedTransformer

import paths
from configs import load_config
from eval.dictionaries import load_sae

CONCEPT_FAMILIES = {
    "weekday": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "month": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"],
    "country": ["France", "Germany", "Italy", "Spain", "Japan", "China", "India", "Brazil", "Canada", "Mexico", "Egypt",
                "Kenya", "Norway", "Greece", "Turkey", "Vietnam"],
    "uscity": ["Boston", "Chicago", "Denver", "Seattle", "Miami", "Atlanta", "Dallas", "Phoenix", "Portland", "Detroit",
               "Houston", "Philadelphia"],
    "number": ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"],
    "color": ["Red", "Blue", "Green", "Yellow", "Black", "White", "Brown", "Orange", "Purple", "Pink", "Gray",
              "Silver"],
    "direction": ["North", "South", "East", "West"],
    "season": ["Spring", "Summer", "Autumn", "Winter"],
    "language": ["English", "French", "German", "Spanish", "Italian", "Russian", "Chinese", "Japanese", "Arabic",
                 "Hindi", "Korean", "Dutch"],
    "animal": ["Dog", "Cat", "Horse", "Cow", "Sheep", "Bird", "Fish", "Lion", "Tiger", "Bear", "Wolf", "Fox"],
}
CONTEXT, TEXTS_PER_BATCH, MAX_OCCURRENCES = 256, 16, 120
BACKGROUND_TOKENS, BACKGROUND_CONTEXT = 40000, 128
MIN_OCCURRENCES, MIN_MEMBERS = 20, 3
N_BACKGROUND = 20000
MIN_RECALL, MIN_PRECISION = 0.5, 0.2
COLUMN_BUDGETS = (6, 24)
TRAIN_FRACTION, SPLIT_SEEDS = 0.7, (0, 1, 2)
ENCODE_BATCH = 512


def family_state_files(config_name: str) -> tuple[Path, Path]:
    return paths.DATA / config_name / "family_states.pt", paths.DATA / config_name / "background.pt"


def single_token_ids(tokenizer, member: str) -> list[int]:
    ids = []
    for form in (" " + member, " " + member.lower()):
        tokens = tokenizer.encode(form, add_special_tokens=False)
        if len(tokens) == 1:
            ids.append(tokens[0])
    return ids


@torch.no_grad()
def collect(config_name: str) -> None:
    config = load_config(config_name)
    hook = config["hook_name"]
    stop_at_layer = int(hook.split(".")[1]) + 1
    model = HookedTransformer.from_pretrained_no_processing(config["model_name"], device="cuda")
    members_of_token, members = {}, {}  # token id -> [(family, member index)], family -> its single-token members
    for concept, terms in CONCEPT_FAMILIES.items():
        member_index = 0
        for term in terms:
            ids = single_token_ids(model.tokenizer, term)
            if not ids:
                continue
            for token in set(ids):
                members_of_token.setdefault(token, []).append((concept, member_index))
            members.setdefault(concept, []).append(term)
            member_index += 1
    texts = load_dataset("NeelNanda/pile-10k", split="train")["text"][:10000]
    wanted = torch.tensor(sorted(members_of_token), device="cuda")
    states = {}
    for start in range(0, len(texts), TEXTS_PER_BATCH):
        tokens = model.to_tokens(texts[start:start + TEXTS_PER_BATCH], truncate=True)[:, :CONTEXT]
        cache = model.run_with_cache(tokens, names_filter=hook, stop_at_layer=stop_at_layer)[1]
        hits = torch.isin(tokens, wanted)
        hits[:, 0] = False
        if not hits.any():
            continue
        rows, positions = hits.nonzero(as_tuple=True)
        vectors = cache[hook][rows, positions].float().cpu()
        for vector, token in zip(vectors, tokens[rows, positions].tolist()):
            for member in members_of_token[token]:
                if len(states.get(member, [])) < MAX_OCCURRENCES:
                    states.setdefault(member, []).append(vector)
    states_file, background_file = family_state_files(config_name)
    states_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": states, "members": members}, states_file)
    background, n_tokens = [], 0
    for start in range(0, 4000, TEXTS_PER_BATCH):
        tokens = model.to_tokens(texts[start:start + TEXTS_PER_BATCH], truncate=True)[:, :BACKGROUND_CONTEXT]
        cache = model.run_with_cache(tokens, names_filter=hook, stop_at_layer=stop_at_layer)[1]
        background.append(cache[hook].reshape(-1, config["d_in"]).float().cpu())
        n_tokens += background[-1].shape[0]
        if n_tokens >= BACKGROUND_TOKENS:
            break
    torch.save(torch.cat(background)[:BACKGROUND_TOKENS], background_file)


def load_families(config_name: str) -> tuple[dict, torch.Tensor]:
    states_file, background_file = family_state_files(config_name)
    states = torch.load(states_file, weights_only=False)["states"]
    families = {}
    for concept in sorted({concept for concept, _ in states}):
        activations, members = [], []
        for (owner, member), occurrences in states.items():
            if owner != concept or len(occurrences) < MIN_OCCURRENCES:
                continue
            activations.append(torch.stack(occurrences))
            members += [member] * len(occurrences)
        if len(set(members)) >= MIN_MEMBERS:
            families[concept] = (torch.cat(activations).float(), np.array(members))
    background = torch.load(background_file, weights_only=False)[:N_BACKGROUND].float()
    return families, background


@torch.no_grad()
def encode(sae, activations: torch.Tensor) -> torch.Tensor:
    out = []
    for start in range(0, activations.shape[0], ENCODE_BATCH):
        codes = sae.encode(activations[start:start + ENCODE_BATCH].cuda().float())
        if getattr(sae, "group_rank", 1) > 1 and sae.cfg.normalize_activations == "layer_norm":
            codes = codes * sae.ln_std  # SASA codes in activation units
        out.append(codes.cpu())
    return torch.cat(out)


def unit_activity(codes: torch.Tensor, rank: int) -> torch.Tensor:
    return codes.view(codes.shape[0], -1, rank).norm(dim=-1) if rank > 1 else codes.abs()


def detection_scores(fires: torch.Tensor, is_member: np.ndarray):
    is_member = torch.as_tensor(is_member).float().cuda()
    true_positives = (fires.float() * is_member[:, None]).sum(0)
    precision = true_positives / fires.float().sum(0).clamp_min(1e-8)
    recall = true_positives / is_member.sum().clamp_min(1)
    return precision, recall, 2 * precision * recall / (precision + recall).clamp_min(1e-8)


def member_accuracy(features: np.ndarray, members: np.ndarray, train, test) -> float | None:
    if features.shape[1] == 0 or len(set(members[train].tolist())) < 2:
        return None
    scale = features[train].std(0) + 1e-6
    regression = LogisticRegression(max_iter=2000, C=1.0).fit(features[train] / scale, members[train])
    return float(accuracy_score(members[test], regression.predict(features[test] / scale)))


def evaluate(sae, families: dict, background: torch.Tensor) -> dict:
    if getattr(sae.cfg, "gate", None) == "batchtopk":
        raise SystemExit("member prediction gates SASA per token; this dictionary has the batch gate")
    rank = getattr(sae, "group_rank", 1)
    background_fires = (unit_activity(encode(sae, background), rank) > 0).cuda()
    unit_counts = [max(1, budget // rank) for budget in COLUMN_BUDGETS]
    accuracy = {n_units: [[] for _ in SPLIT_SEEDS] for n_units in unit_counts}
    for activations, member_of_occurrence in families.values():
        codes = encode(sae, activations)
        fires = torch.cat([(unit_activity(codes, rank) > 0).cuda(), background_fires])
        is_member = np.concatenate([np.ones(len(codes)), np.zeros(len(background_fires))])
        precision, recall, f1 = detection_scores(fires, is_member)
        units = ((recall >= MIN_RECALL) & (precision >= MIN_PRECISION)).nonzero().flatten()
        units = units[f1[units].argsort(descending=True)]
        members = np.unique(member_of_occurrence, return_inverse=True)[1]
        for split, seed in enumerate(SPLIT_SEEDS):
            order = np.random.RandomState(seed).permutation(len(members))
            n_train = int(TRAIN_FRACTION * len(members))
            train, test = order[:n_train], order[n_train:]
            for n_units in unit_counts:
                chosen = units[:n_units]
                if chosen.numel():
                    columns = (chosen[:, None].cpu() * rank + torch.arange(rank)[None, :]).reshape(-1)
                    value = member_accuracy(codes[:, columns].numpy(), members, train, test)
                    if value is not None:
                        accuracy[n_units][split].append(round(value, 4))
    return {str(n_units * rank): [round(float(np.mean(values)), 4) if values else None for values in per_split]
            for n_units, per_split in accuracy.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collecting = commands.add_parser("collect", help="occurrence states and background tokens of a config (LLM)")
    collecting.add_argument("config")
    scoring = commands.add_parser("score", help="member prediction of SAEs, SASA_RESULTS/member_prediction/")
    scoring.add_argument("saes", nargs="+", help="<config>/<sae> ...")
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.config)
        return
    config_name = args.saes[0].split("/")[0]
    assert all(name.split("/")[0] == config_name for name in args.saes), "SAEs of one config per run"
    out = paths.RESULTS / "member_prediction" / f"{config_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    families, background = load_families(config_name)
    for name in args.saes:
        sae_name = name.split("/")[1]
        if sae_name not in results:
            results[sae_name] = evaluate(load_sae(name), families, background)
            out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

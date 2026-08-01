"""First-letter feature absorption, ported from SAEBench and computed locally."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import statistics
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval import _common as adapter_mod  # noqa: E402
from paths import CHECKPOINTS, RESULTS  # noqa: E402

LETTERS = "abcdefghijklmnopqrstuvwxyz"
PROBES_DIR = REPO_ROOT / "artifacts" / "absorption" / "probes"
REPORT_DIR = RESULTS / "absorption"

PROMPT_TEMPLATE = "{word} has the first letter:"
PROMPT_TOKEN_POS = -6
ICL_EXAMPLES = 10

MAX_K = 10
F1_JUMP_THRESHOLD = 0.03
MIN_GT_PROBE_F1 = 0.6
MIN_FEATS_FOR_EVAL = 20
L1_DECAY = 0.01
L1_PROBE_EPOCHS = 50
L1_PROBE_BATCH = 4096

TOPK_FEATS = 10
FULL_ABSORPTION_COS_THRESHOLD = 0.025
ABSORPTION_FRACTION_COS_THRESHOLD = 0.1
PROBE_PROJECTION_PROPORTION_THRESHOLD = 0.4
MAX_ABSORBING_LATENTS = 3
EPS = 1e-8


def patch_adapter_encode_to_raw_scale() -> None:
    if getattr(adapter_mod.SAEBenchAdapter, "_raw_scale_patched", False):
        return
    original_encode = adapter_mod.SAEBenchAdapter.encode

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = original_encode(self, x)
        if str(getattr(self.sae.cfg, "normalize_activations", "none")) != "layer_norm":
            return z
        ln_std = getattr(self.sae, "ln_std", None)
        if ln_std is None:
            raise RuntimeError("layer_norm SAE did not expose ln_std after encode.")
        rows = int(torch.tensor(z.shape[:-1]).prod().item())
        if ln_std.numel() != rows:
            raise RuntimeError(
                f"ln_std has {ln_std.numel()} entries but encode returned {rows} rows."
            )
        scale = ln_std.reshape(*z.shape[:-1], 1).to(device=z.device, dtype=z.dtype)
        return z * scale

    adapter_mod.SAEBenchAdapter.encode = encode
    adapter_mod.SAEBenchAdapter._raw_scale_patched = True


def load_adapter(sae_dir: Path, device: str):
    from sae_lens import SAE, register_sae_class

    from sasa import TopKSASAInference, TopKSASAInferenceConfig

    try:
        register_sae_class("topk_sasa", TopKSASAInference, TopKSASAInferenceConfig)
    except ValueError:
        pass

    try:
        sae = SAE.load_from_disk(str(sae_dir), device=device)
    except KeyError:
        # BatchTopK/Matryoshka save as jumprelu; convert if given a training checkpoint.
        from sae_lens.saes.sae import TrainingSAE

        training_sae = TrainingSAE.load_from_disk(str(sae_dir), device=device)
        converted = Path(tempfile.mkdtemp(prefix="sasa_infer_"))
        training_sae.save_inference_model(str(converted))
        del training_sae
        sae = SAE.load_from_disk(str(converted), device=device)

    adapter_mod._canonicalize_model_name_for_saebench(sae)

    n_groups = getattr(sae, "n_groups", None) or getattr(sae.cfg, "n_groups", None)
    group_rank = getattr(sae, "group_rank", None) or getattr(sae.cfg, "group_rank", None)
    return adapter_mod.SAEBenchAdapter(
        sae,
        match_token_norm=False,
        n_groups=int(n_groups) if n_groups is not None else None,
        group_rank=int(group_rank) if group_rank is not None else None,
    )


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_outputs: int = 1):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    @property
    def weights(self) -> torch.Tensor:
        return self.fc.weight


class _ProbeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        return LinearProbe if name == "LinearProbe" else super().find_class(module, name)


def load_ground_truth_probe(probe_dir: Path, device: str) -> LinearProbe:
    """probe.pth is a pickled SAEBench LinearProbe; rebind it onto the local class."""
    shim = types.ModuleType("absorption_probe_pickle")
    shim.Unpickler, shim.load = _ProbeUnpickler, pickle.load
    probe = torch.load(
        probe_dir / "probe.pth", map_location=device, pickle_module=shim, weights_only=False
    )
    return probe.float().eval()


def load_split(probe_dir: Path, split: str, tokenizer, device: str):
    data = np.load(probe_dir / "data.npz")
    df = pd.read_csv(probe_dir / f"{split}_df.csv", keep_default_na=False, na_values=[""])
    tokens = df["token"].tolist()
    keep = [
        i for i, t in enumerate(tokens)
        if isinstance(t, str) and not re.match(r"[\d<>]", t)
    ]
    acts = torch.from_numpy(data[f"X_{split}"][keep]).to(device=device, dtype=torch.float32)
    words = [tokenizer.convert_tokens_to_string([tokens[i]]) for i in keep]
    return acts, words, data[f"y_{split}"][keep]


@torch.no_grad()
def encode_all(sae, acts: torch.Tensor, batch_size: int = L1_PROBE_BATCH) -> torch.Tensor:
    return torch.cat([sae.encode(batch) for batch in acts.split(batch_size)])


def train_l1_probe(latents: torch.Tensor, labels: np.ndarray) -> torch.Tensor:
    """26-way one-vs-rest logistic probe on SAE latents, L1-penalised, for feature selection."""
    device = latents.device
    y = nn.functional.one_hot(torch.from_numpy(labels), len(LETTERS)).to(device, torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(y) - y.sum(0)) / y.sum(0))
    probe = LinearProbe(latents.shape[-1], len(LETTERS)).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=0.01, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(1e-5 / 0.01) ** (1 / L1_PROBE_EPOCHS))
    for _ in range(L1_PROBE_EPOCHS):
        for idx in torch.randperm(len(latents), device=device).split(L1_PROBE_BATCH):
            opt.zero_grad()
            loss = loss_fn(probe(latents[idx]), y[idx])
            loss = loss + L1_DECAY * probe.weights.abs().sum(dim=-1).mean()
            loss.backward()
            opt.step()
        sched.step()
    return probe.weights.detach()


def train_k_sparse_probes(l1_weights: torch.Tensor, latents: np.ndarray, labels: np.ndarray):
    probes = {}
    for letter_i in range(len(LETTERS)):
        feats = l1_weights[letter_i].topk(MAX_K).indices.cpu().numpy()
        y = (labels == letter_i).astype(np.int64)
        for k in range(1, MAX_K + 1):
            ids = feats[:k]
            fit = LogisticRegression(max_iter=500, class_weight="balanced").fit(latents[:, ids], y)
            probes[letter_i, k] = (ids, fit.coef_[0].astype(np.float32), float(fit.intercept_[0]))
    return probes


def letter_metrics(probes, latents: np.ndarray, labels: np.ndarray, probe_scores: np.ndarray):
    """Ground-truth probe F1 plus the split latents, i.e. the k-sparse probe's feature set at
    the largest k that still bought an F1 jump."""
    rows = []
    for letter_i, letter in enumerate(LETTERS):
        y = labels == letter_i
        split_feats, best = [], -100.0
        for k in range(1, MAX_K + 1):
            ids, weight, bias = probes[letter_i, k]
            f1 = f1_score(y, latents[:, ids] @ weight + bias > 0, zero_division=0)
            if f1 <= best + F1_JUMP_THRESHOLD:
                break
            best, split_feats = f1, ids.tolist()
        rows.append({
            "letter": letter,
            "f1_probe": f1_score(y, probe_scores[:, letter_i] > 0, zero_division=0),
            "split_feats": split_feats,
        })
    return rows


def get_alpha_tokens(tokenizer) -> list[str]:
    alpha = set(LETTERS + LETTERS.upper())
    words = []
    for token in tokenizer.vocab:
        word = tokenizer.convert_tokens_to_string([token])
        body = word[1:] if word.startswith(" ") else word
        if body and all(char in alpha for char in body):
            words.append(word)
    return words


def icl_prompt(word: str, vocab: list[str]) -> str:
    while True:
        examples = random.sample(vocab, ICL_EXAMPLES)
        if word not in examples:
            break
    shots = [PROMPT_TEMPLATE.format(word=w) + " " + w.strip()[0].upper() for w in examples]
    return "\n" + "\n".join(shots) + "\n" + PROMPT_TEMPLATE.format(word=word)


def absorption_fraction(proj: torch.Tensor, act_proj: float, cos: torch.Tensor,
                        main_feats: list[int]) -> float:
    main_proj = proj[main_feats].sum().item()
    candidates = torch.ones_like(proj, dtype=torch.bool)
    candidates[main_feats] = False
    candidates &= cos >= ABSORPTION_FRACTION_COS_THRESHOLD
    candidates &= proj > 0
    pool = proj[candidates]
    absorber_proj = pool.topk(min(MAX_ABSORBING_LATENTS, pool.numel())).values.sum().item()
    if (main_proj >= act_proj
            or absorber_proj / act_proj < PROBE_PROJECTION_PROPORTION_THRESHOLD):
        return 0.0
    if main_proj <= 0.0:
        return 1.0
    absorbed = min(absorber_proj, act_proj - main_proj)
    return float(np.clip(absorbed / (absorbed + main_proj), 0.0, 1.0))


def is_full_absorption(proj: torch.Tensor, act_proj: float, cos: torch.Tensor,
                       latents: torch.Tensor, main_feats: list[int]) -> bool:
    if (latents[main_feats] >= EPS).any().item():
        return False
    top = proj.topk(TOPK_FEATS).indices[0].item()
    if cos[top].item() < FULL_ABSORPTION_COS_THRESHOLD or act_proj < 0:
        return False
    return proj[top].item() / act_proj >= PROBE_PROJECTION_PROPORTION_THRESHOLD


@torch.no_grad()
def score_letter(model, sae, hook: str, direction: torch.Tensor, main_feats: list[int],
                 words: list[str], vocab: list[str], batch_size: int):
    direction = (direction / direction.norm()).to(sae.device)
    cos = nn.functional.cosine_similarity(direction, sae.W_dec, dim=-1).float().cpu()
    prompts = [icl_prompt(word, vocab) for word in words]
    if len({len(ids) for ids in model.tokenizer(prompts)["input_ids"]}) > 1:
        raise ValueError("all prompts for a letter must have the same token length")

    fractions, full = [], []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        acts = model.run_with_cache(batch, names_filter=[hook])[1][hook][:, PROMPT_TOKEN_POS, :]
        latents = sae.encode(acts).float().cpu()
        proj = latents * cos
        act_proj = (acts.float() @ direction.float()).cpu()
        for i in range(len(batch)):
            fractions.append(absorption_fraction(proj[i], act_proj[i].item(), cos, main_feats))
            full.append(is_full_absorption(proj[i], act_proj[i].item(), cos, latents[i], main_feats))
    return fractions, full


def load_gpt2(device: str):
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Hand TransformerLens a built HF model so it never forwards the deprecated torch_dtype.
    return HookedTransformer.from_pretrained_no_processing(
        "gpt2",
        hf_model=AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32),
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
        device=device,
        dtype=torch.float32,
    )


def load_sae(sae_dir: Path, device: str):
    with warnings.catch_warnings():
        # SAE-Lens advises loading the LLM with cfg.model_from_pretrained_kwargs; load_gpt2 does.
        warnings.filterwarnings(
            "ignore", r"\s*This SAE has non-empty model_from_pretrained_kwargs", UserWarning
        )
        return load_adapter(sae_dir, device)


def run(sae_dir: Path, device: str, batch_size: int, seed: int, raw_scale: bool) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if raw_scale:
        patch_adapter_encode_to_raw_scale()
    sae = load_sae(sae_dir, device)
    # Read activations at the hook the SAE was trained on; SAEBench hardcodes hook_resid_post.
    hook = sae.cfg.hook_name
    probe_dir = PROBES_DIR / "gpt2" / f"layer_{int(hook.split('.')[1])}"

    model = load_gpt2(device)
    probe = load_ground_truth_probe(probe_dir, device)
    train_acts, _, train_labels = load_split(probe_dir, "train", model.tokenizer, device)
    test_acts, test_words, test_labels = load_split(probe_dir, "test", model.tokenizer, device)

    train_latents = encode_all(sae, train_acts)
    l1_weights = train_l1_probe(train_latents, train_labels)
    k_probes = train_k_sparse_probes(l1_weights, train_latents.cpu().numpy(), train_labels)
    del train_acts, train_latents

    with torch.no_grad():
        test_latents = encode_all(sae, test_acts).cpu().numpy()
        probe_scores = probe(test_acts).cpu().numpy()
    rows = letter_metrics(k_probes, test_latents, test_labels, probe_scores)
    del test_acts, test_latents

    usable = [row for row in rows if row["f1_probe"] > MIN_GT_PROBE_F1]
    if len(usable) < MIN_FEATS_FOR_EVAL:
        raise SystemExit(
            f"only {len(usable)} letters reach ground-truth probe F1 > {MIN_GT_PROBE_F1}; "
            f"the eval needs {MIN_FEATS_FOR_EVAL}. Check artifacts/absorption/probes."
        )

    vocab = get_alpha_tokens(model.tokenizer)
    fraction_scores, full_scores, split_counts, details = [], [], [], []
    for row in usable:
        letter_i = LETTERS.index(row["letter"])
        # the probe's true positives: words the ground-truth probe fires on
        hits = (test_labels == letter_i) & (probe_scores[:, letter_i] > 0)
        words = [word for word, hit in zip(test_words, hits) if hit]
        fractions, full = score_letter(
            model, sae, hook, probe.weights[letter_i], row["split_feats"],
            words, vocab, batch_size,
        )
        fraction_scores.append(sum(fractions) / len(words))
        full_scores.append(sum(full) / len(words))
        split_counts.append(len(row["split_feats"]))
        details.append({
            "letter": row["letter"],
            "f1_probe": row["f1_probe"],
            "mean_absorption_fraction": fraction_scores[-1],
            "full_absorption_rate": full_scores[-1],
            "num_probe_true_positives": len(words),
            "num_split_features": split_counts[-1],
        })

    return {
        "mean_absorption_fraction_score": statistics.mean(fraction_scores),
        "mean_full_absorption_score": statistics.mean(full_scores),
        "mean_num_split_features": statistics.mean(split_counts),
        "std_dev_absorption_fraction_score": statistics.stdev(fraction_scores),
        "letters": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae", required=True,
                    help="Checkpoint directory name (under SASA_CHECKPOINTS) or an absolute path.")
    ap.add_argument("--label", required=True, help="Output label; must be unique per run.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-raw-scale", action="store_true",
                    help="Skip the layer-norm rescaling, i.e. score layer_norm SAEs in their "
                         "own units rather than raw activation units.")
    args = ap.parse_args()

    adapter_mod._set_hf_cache_defaults(REPO_ROOT)
    sae_dir = Path(args.sae) if Path(args.sae).is_absolute() else CHECKPOINTS / args.sae
    result = run(sae_dir, args.device, args.batch_size, args.seed, not args.no_raw_scale)
    result = {"label": args.label, "sae": str(sae_dir), "seed": args.seed, **result}

    out_dir = REPORT_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "letters"}, indent=2))


if __name__ == "__main__":
    main()

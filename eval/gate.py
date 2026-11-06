import argparse
import hashlib
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

import paths
from configs import load_config
from eval.dictionaries import load_sae

N_BATCHES, BATCH_TOKENS, SEQUENCES_PER_FORWARD = 24, 4096, 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def gate_threshold(name: str) -> float | None:
    # the threshold calibrated by main(), else the shipped one (data/gate/) if the checkpoint holds the released weights
    calibrated = paths.RESULTS / "gate" / f"{name}.json"
    if calibrated.exists():
        return json.loads(calibrated.read_text())["threshold"]
    config_name, sae_name = name.split("/")
    shipped_file = paths.DATA / "gate" / f"{config_name}.json"
    shipped = json.loads(shipped_file.read_text()).get(sae_name) if shipped_file.exists() else None
    if shipped is None or sha256(paths.CHECKPOINTS / name / "sae_weights.safetensors") != shipped["sha256"]:
        return None
    return shipped["threshold"]


def pile_sequences(model, context: int) -> torch.Tensor:
    sequences = []
    for text in load_dataset("NeelNanda/pile-10k", split="train")["text"]:
        if len(sequences) * context >= N_BATCHES * BATCH_TOKENS:
            break
        tokens = model.to_tokens(text, prepend_bos=True)[0, :context]
        if tokens.numel() == context:
            sequences.append(tokens)
    return torch.stack(sequences)


@torch.no_grad()
def calibrate(sae, config: dict) -> float:
    model = HookedTransformer.from_pretrained_no_processing(config["model_name"], device="cuda", dtype=torch.bfloat16)
    sequences = pile_sequences(model, config["context_size"])
    hook = config["hook_name"]
    norms = []
    for start in range(0, len(sequences), SEQUENCES_PER_FORWARD):
        hidden = model.run_with_cache(sequences[start:start + SEQUENCES_PER_FORWARD], names_filter=[hook])[1][hook]
        sae_in = sae.process_sae_in(hidden.reshape(-1, sae.cfg.d_in).float())
        pre_acts = sae_in @ sae.W_enc + sae.b_enc
        norms.append(pre_acts.view(-1, sae.n_groups, sae.group_rank).norm(dim=-1).cpu())
    norms = torch.cat(norms)
    norms = norms[torch.randperm(len(norms), generator=torch.Generator().manual_seed(0))]  # batches mix documents
    n_open = sae.k_groups * BATCH_TOKENS
    cutoffs = [norms[b * BATCH_TOKENS:(b + 1) * BATCH_TOKENS].reshape(-1).topk(n_open).values[-1].item()
               for b in range(len(norms) // BATCH_TOKENS)]
    return sum(cutoffs) / len(cutoffs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sae", help="a SASA trained with the BatchTopK gate: mistral_l8/sasa_r8_k10")
    name = parser.parse_args().sae
    sae = load_sae(name)
    if getattr(sae.cfg, "gate", None) != "batchtopk":
        raise SystemExit(f"{name} is not a SASA trained with the BatchTopK gate")
    out = paths.RESULTS / "gate" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sae": name, "threshold": calibrate(sae, load_config(name.split("/")[0]))}, indent=1))


if __name__ == "__main__":
    main()

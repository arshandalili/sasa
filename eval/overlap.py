import argparse
import json

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

import paths
from configs import load_config
from eval.dictionaries import load_sae

N_TOKENS, BATCH_PROMPTS = 2_000_000, 32
TOP_GROUPS = 500
ENERGY = 0.90


def token_batches(dataset: str, context_size: int) -> tuple[torch.Tensor, ...]:
    windows = []
    for document in load_dataset(dataset, split="train", streaming=True):
        ids = document["input_ids"]
        windows += [ids[start:start + context_size] for start in range(0, len(ids) - context_size + 1, context_size)]
        if len(windows) * context_size >= N_TOKENS:
            break
    return torch.tensor(windows[:N_TOKENS // context_size]).split(BATCH_PROMPTS)


def code_moments(sae, model, hook: str, batches) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # per group: the number of tokens on which it is open, and the sums of its code and of the code's outer product
    count = torch.zeros(sae.n_groups, dtype=torch.float64, device="cuda")
    first = torch.zeros(sae.n_groups, sae.group_rank, dtype=torch.float64, device="cuda")
    second = torch.zeros(sae.n_groups, sae.group_rank, sae.group_rank, dtype=torch.float64, device="cuda")
    for batch in batches:
        hidden = model.run_with_cache(batch.cuda(), names_filter=[hook])[1][hook]
        codes = sae.encode(hidden).reshape(-1, sae.n_groups, sae.group_rank).double()  # zero for a closed group
        count += (codes.norm(dim=-1) > 0).sum(0)
        first += codes.sum(0)
        second += torch.einsum("tkr,tks->krs", codes, codes)
    return count, first, second


def subspaces(decoder_rows, count, first, second) -> torch.Tensor:
    # An open group's output D_k a_k has covariance D_k C_k D_k^T, with C_k the covariance of its code, so its principal
    # directions are the right singular vectors of C_k^{1/2} D_k^T. The group's subspace is spanned by the leading
    # directions that explain ENERGY of the variance; the bases are zero-padded to the group width, [groups, d, r].
    mean = first / count[:, None]
    covariance = second / count[:, None, None] - mean[:, :, None] * mean[:, None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    root = eigenvectors @ torch.diag_embed(eigenvalues.clamp(min=0).sqrt()) @ eigenvectors.transpose(1, 2)
    _, singular_values, directions = torch.linalg.svd(root @ decoder_rows, full_matrices=False)
    explained = singular_values.square().cumsum(-1) / singular_values.square().sum(-1, keepdim=True)
    ranks = (explained < ENERGY).sum(-1) + 1
    kept = torch.arange(directions.shape[1], device="cuda") < ranks[:, None]
    return (directions * kept[..., None]).transpose(1, 2).float()


def pair_overlaps(bases: torch.Tensor) -> np.ndarray:
    # the overlap of two subspaces is the largest cosine between their directions, the top singular value of B_i^T B_j
    largest = torch.linalg.svdvals(torch.einsum("idr,jds->ijrs", bases, bases))[..., 0]
    i, j = torch.triu_indices(len(bases), len(bases), offset=1, device="cuda")
    return largest[i, j].double().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sae", help="a GPT-2 SASA, e.g. gpt2_l7/sasa_r6_l060_w12288")
    name = parser.parse_args().sae
    torch.set_grad_enabled(False)
    config = load_config(name.split("/")[0])
    sae = load_sae(name)
    model = HookedTransformer.from_pretrained_no_processing(config["model_name"], device="cuda").eval()
    batches = token_batches(config["dataset_path"], config["context_size"])
    count, first, second = code_moments(sae, model, config["hook_name"], batches)
    top = count.topk(TOP_GROUPS).indices  # the groups open on the most tokens
    decoder_rows = sae.W_dec.double().view(sae.n_groups, sae.group_rank, -1)  # D_k^T
    overlaps = pair_overlaps(subspaces(decoder_rows[top], count[top], first[top], second[top]))
    result = {"sae": name, "median_overlap": float(np.median(overlaps)),
              "share_above_0.5": float((overlaps > 0.5).mean())}
    out = paths.RESULTS / "overlap" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlap.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

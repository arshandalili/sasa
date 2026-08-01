# SASA: Subspace-Aware Sparse Autoencoders

A standard SAE ties every latent to a single decoder direction, which assumes each feature
is one-dimensional. Many features in an LLM are not: day of the week, month, year and
colour are carried by low-dimensional *subspaces*, often with circular structure. The
mismatch is not benign. Reconstructing a feature of intrinsic dimension `d_i` to error
`eps` out of single directions takes a number of atoms exponential in `d_i`, and the
l1-regularized objective descends toward exactly that tiling, so a trained dictionary
fragments one coherent feature across many near-collinear latents. That is feature
splitting: reading the feature means aggregating a cluster of latents rather than
inspecting one unit.

SASA makes the decoder *subspace* the unit of representation. Each latent gets a rank-`r`
decoder block instead of a vector, gating is Top-`s` over the group norms `||p_k(h)||_2`
so a whole block switches on at once, and a trace-norm penalty on each block lets its
effective rank adapt to the feature it holds. A model is `(K, r, s)`: `K` groups of rank
`r`, `s` active per token, so the dictionary has `m = Kr` columns and `l0 = sr`. Once
`r >= d_i` a single group is the global minimizer of the SASA objective, unique up to
block index and orthogonal rotation, which turns feature recovery into principal subspace
estimation and makes the sample complexity polynomial in `d_i` instead of exponential.

![Vector-based SAEs split a multi-dimensional feature across many near-collinear atoms;
SASA captures it as one subspace.](assets/figure1.png)

Three ground-truth manifolds -- a circle (`d_i=2`), a sphere (`d_i=3`) and a helix
(`d_i=3`) -- embedded in `d=64` and fit by six dictionaries of width 256. Every
vector-based SAE spreads a manifold over tens to hundreds of atoms; SASA covers it with
one group of effective rank `d_i`. `experiments/synthetic_dimension.py` reproduces this.

## Setup

```bash
git lfs install
git lfs pull

uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt

export SASA_SCRATCH=/path/to/activations
export SASA_CHECKPOINTS=$PWD/checkpoints
export SASA_RESULTS=$PWD/results
export SASA_CACHE_ROOT=$PWD/.cache
```

`git lfs pull` fetches the two large files kept out of ordinary git history: the probe
activation cache used by the absorption eval, and the Mistral-7B checkpoint used by the
causal intervention.

Run everything as a module from this directory.
`checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10` is the GPT-2 layer-7 model behind Tables 1-3
(`K=2048, r=6, s=10`, so `m=12288`, `l0=60`), so the eval commands below work without
training anything first.

## Cache

Cache the activation stream once, every model then reads the same tokens in the same order.

```bash
python -m scripts.cache_activations \
  --model-name gpt2 --hook-name blocks.7.hook_resid_pre --d-in 768 \
  --dataset apollo-research/Skylion007-openwebtext-tokenizer-gpt2 \
  --context-size 128 --training-tokens 300000000 \
  --output-path $SASA_SCRATCH/gpt2_l7_300m
```

## Train

The paper's GPT-2 model:

```bash
python -m scripts.train_sasa \
  --model-name gpt2 --hook-name blocks.7.hook_resid_pre --d-in 768 \
  --n-groups 2048 --group-rank 6 --k-groups 10 \
  --dataset apollo-research/Skylion007-openwebtext-tokenizer-gpt2 --tokenized \
  --training-tokens 300000000 --context-size 128 --save-dir $SASA_CHECKPOINTS
```

`scripts/train_sasa.py` takes any TransformerLens model.

The matched-budget sweep. Every arm shares width (12,288 columns), hook, token stream,
token budget, optimizer, schedule and seed; architecture and sparsity vary.
These are the sixteen models behind the absorption and explained-variance tables:

```bash
for arch in topk batchtopk matryoshka topk_sasa; do
  for l0 in 30 60 120 240; do
    python -m scripts.train_matched_budget --arch $arch --l0 $l0 --use-cached-activations
  done
done
```

The rank x nuclear-coefficient ablation, all at width 12,288 and `l0=60`:

```bash
for r in 3 6 12; do for nuc in 0 10 100; do
  python -m scripts.train_matched_budget --arch topk_sasa --l0 60 \
    --group-rank $r --nuclear-coefficient $nuc --use-cached-activations
done; done
```

`--arch relu|jumprelu|gated` trains the l1-family baselines (their `l0` follows from the
sparsity coefficient, so `--l0` is ignored). `--nuclear-coefficient` is `lambda_dim`.
`metrics/mean_stable_rank` should sit strictly between 1 and `group_rank`.

## Evaluate

Reconstruction and faithfulness (KL, CE, explained variance, l0), against the pretrained
GPT-2 SAE `gpt2-small-res-jb` at the same hook:

```bash
python -m eval.run_core \
  --topk-sasa-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --hook blocks.7.hook_resid_pre
```

First-letter feature absorption, scored by `eval/absorption.py`. The ground-truth probes
and their cached activations ship in `artifacts/absorption/probes/`, so nothing is trained
here; `scripts/rebuild_probe_activations.py` regenerates the cache from the shipped CSVs
if you ever need to.

```bash
python -m eval.absorption --sae topk_sasa_gpt2_l7_r6_nuc100_l060 --label sasa_l060
```

Activations are read at the SAE's own hook. The score is not scale-invariant, so SASA
latents are rescaled by the per-token `ln_std` back to raw activation units and every
architecture is scored on one scale; `--no-raw-scale` turns that off.

The matched-budget sweep, mean absorption fraction, lower is better:

| `l0` | SASA | TopK | BatchTopK | Matryoshka |
| --- | --- | --- | --- | --- |
| 30 | **0.046** | 0.092 | 0.093 | 0.130 |
| 60 | **0.068** | 0.097 | 0.191 | 0.140 |
| 120 | **0.022** | 0.131 | 0.339 | 0.403 |
| 240 | **0.096** | 0.178 | 0.262 | 0.394 |

The metric carries about +-0.03 absolute run to run, so read the ordering rather than
individual cells.

The temporal subspace of Figure 3. Group 1473 spans years; the three panels are its
explained-variance spectrum, the 3D PCA of the year directions, and the circular fit:

```bash
python -m analysis.temporal_subspace \
  --sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 --group-id 1473
```

AutoInterp:

```bash
python -m eval.run_autointerp --sae-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --label sasa_paper --api-key-file openai_key.txt
```

For SASA the unit this evaluates is the *group*, whose activation is the group norm
`||p_k(h)||_2`. A group is charged all `r` of its columns wherever a column budget is
compared.

Evaluate the final saved model, not a mid-training snapshot: SAE-Lens folds the decoder row
norms into the encoder on the final save, and the scalar baselines are norm-inflated
without it.

## Synthetic experiments

Both generate their own data, so they need no LLM, no cached activations and no checkpoint.

Feature splitting against intrinsic dimension. Plants features of known `d_i` with every
other factor held fixed and counts how many atoms each architecture spends per feature.
This is the experiment behind the figure above:

```bash
python -m experiments.synthetic_dimension --tag main
```

Rank recovery. Plants subspaces of known dimension and sweeps the nuclear coefficient to
check the penalty drives each group's effective rank to the true `d_i`:

```bash
python -m experiments.synthetic_rank --tag main
```

The defaults are the paper's settings (8k and 20k steps, three seeds, six architectures or
six coefficients) and take a few hours each; `--steps`, `--seeds` and `--archs`/`--lambdas`
cut that down. Both write to `results/reports/<name>/<tag>/results.json`.

## Causal intervention

Whether a SASA group *is* the concept rather than merely correlating with it, tested by
intervening on the group and reading the effect off the model's own predictions. This runs
on Mistral-7B v0.1 at `blocks.19.hook_resid_post`.

`checkpoints/topk_sasa_mistral7b_l19_n8192_r4_k15` is the model behind these numbers
(`K=8192, r=4, s=15`, so `m=32768`, `l0=60`). It is 1.1 GB and ships through Git LFS, so
fetch it once after cloning:

```bash
git lfs install
git lfs pull
```

`screen.py` ranks groups per concept by synthetic AUC, keeping those above 0.80 corpus AUC
on a held-out Pile slice; the top group per concept is `--g-*` and the second-best is the
`--runner-*` control:

```bash
CKPT=checkpoints/topk_sasa_mistral7b_l19_n8192_r4_k15

python -m experiments.screen --ckpt $CKPT --out results/screen.json
python -m experiments.concept_causal --ckpt $CKPT \
  --g-weekday 5596 --g-month 4502 --out results/concept_causal.json
python -m experiments.ablation_table --ckpt $CKPT \
  --g-weekday 5596 --g-month 4502 --runner-weekday 7444 --runner-month 3372 \
  --out results/ablation_table.json
```

Those indices are the screening result for this checkpoint: weekday and month both reach
synthetic AUC 1.000, at corpus AUC 1.000 and 0.859. Group indices are not stable across
training runs, so re-run `screen.py` against any other checkpoint. To train one:

```bash
python -m scripts.train_sasa \
  --model-name mistralai/Mistral-7B-v0.1 --hook-name blocks.19.hook_resid_post --d-in 4096 \
  --n-groups 8192 --group-rank 4 --k-groups 15 --save-dir $SASA_CHECKPOINTS
```

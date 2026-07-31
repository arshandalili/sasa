# SASA: Subspace-Aware Sparse Autoencoders

SASA replaces each latent's single decoder direction with a rank-`r` decoder block, gates
on the block's norm (Top-`s` over group norms), and regularizes the block with a
trace-norm penalty. A model is `(K, r, s)`: `K` groups of rank `r`, `s` active per token,
so the dictionary has `m = Kr` columns and `l0 = sr`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Paths come from `paths.py` and are all environment variables:

```bash
export SASA_SCRATCH=/fast/disk/sasa      # activation caches (large)
export SASA_CHECKPOINTS=$PWD/checkpoints
export SASA_RESULTS=$PWD/results
export SASA_CACHE_ROOT=$PWD/.cache       # HuggingFace, wandb
```

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

`scripts/train_sasa.py` takes any TransformerLens model; pass `--model-dtype bfloat16` and
`--exclude-special-tokens` for larger ones.

The matched-budget sweep. Every arm shares width (12,288 columns), hook, token stream,
token budget, optimizer, schedule and seed; only architecture and sparsity vary. These are
the sixteen models behind the rebuttal's absorption and explained-variance tables:

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
sparsity coefficient, so `--l0` is ignored). `--nuclear-coefficient` is `lambda_dim`;
`metrics/mean_stable_rank` should sit strictly between 1 and `group_rank`.

## Evaluate

Reconstruction and faithfulness (KL, CE, explained variance, l0), against the pretrained
GPT-2 SAE `gpt2-small-res-jb` at the same hook:

```bash
python -m eval.run_core \
  --topk-sasa-dir checkpoints/topk_sasa_gpt2_l7_n2048_r6_k10 \
  --hook blocks.7.hook_resid_pre
```

First-letter feature absorption. The ground-truth probes this eval scores against are
shipped in `artifacts/absorption/probes/`, but their activations are 134 MB, so rebuild
those once (one GPT-2 pass, a couple of minutes) before the first run:

```bash
python -m scripts.rebuild_probe_activations
python -m eval.run_absorption --sae topk_sasa_gpt2_l7_n2048_r6_k10 --label sasa_paper
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

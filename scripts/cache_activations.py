import argparse
import os
from pathlib import Path

import torch

cache_root = Path(os.environ.get("SASA_CACHE_ROOT", Path.home() / ".cache" / "sasa"))
cache_root.mkdir(parents=True, exist_ok=True)
tmp_root = cache_root / "tmp"
tmp_root.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(cache_root / "hf")
os.environ["HF_DATASETS_CACHE"] = str(cache_root / "hf_datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hf_hub")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TMPDIR"] = str(tmp_root)
for d in ["hf", "hf_datasets", "hf_hub"]:
    (cache_root / d).mkdir(parents=True, exist_ok=True)

import datasets.config

from sae_lens import CacheActivationsRunner, CacheActivationsRunnerConfig
from sasa.compat import patch_rope_theta

datasets.config.TORCHVISION_AVAILABLE = False
patch_rope_theta()


def consolidate(output_path: str):
    """Merge the shards written by all ranks into one loadable dataset."""
    out = Path(output_path)
    ds = CacheActivationsRunner._consolidate_shards(
        out / ".tmp_shards", out, copy_files=False
    )
    print(f"Consolidated {len(ds)} sequences into: {output_path}")
    return ds


def cache_activations(
    model_name: str,
    hook_name: str,
    d_in: int,
    dataset_path: str,
    context_size: int,
    training_tokens: int,
    model_batch_size: int,
    act_dtype: str,
    output_path: str,
    model_dtype: str | None,
    autocast_lm: bool,
    device: str | None,
    seed: int,
    shuffle: bool,
    buffer_size_gb: float,
    rank: int = 0,
    world_size: int = 1,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_from_pretrained_kwargs = {"center_writing_weights": False}
    if model_dtype is not None:
        model_from_pretrained_kwargs["dtype"] = model_dtype

    cfg = CacheActivationsRunnerConfig(
        dataset_path=dataset_path,
        model_name=model_name,
        model_batch_size=model_batch_size,
        hook_name=hook_name,
        d_in=d_in,
        training_tokens=training_tokens // world_size,
        context_size=context_size,
        new_cached_activations_path=output_path,
        shuffle=shuffle,
        seed=seed,
        dtype=act_dtype,
        device=device,
        buffer_size_gb=buffer_size_gb,
        model_from_pretrained_kwargs=model_from_pretrained_kwargs,
        autocast_lm=autocast_lm,
        prepend_bos=True,
        streaming=True,
    )

    if world_size == 1:
        runner = CacheActivationsRunner(cfg)
        print(runner)
        dataset = runner.run()
        print(f"Cached {len(dataset)} sequences to: {output_path}")
        return

    # Multi-GPU: each rank consumes a disjoint set of input shards and writes
    # interleaved shard indices into a shared .tmp_shards, so a single
    # consolidate pass afterwards yields the same layout as a 1-GPU run.
    from datasets import load_dataset
    from datasets.distributed import split_dataset_by_node
    from tqdm.auto import tqdm

    ds = load_dataset(dataset_path, split="train", streaming=True)
    if ds.num_shards % world_size != 0:
        raise ValueError(
            f"dataset has {ds.num_shards} shards, not divisible by world_size "
            f"{world_size}; ranks would re-read the same data"
        )
    ds = split_dataset_by_node(ds, rank=rank, world_size=world_size)

    runner = CacheActivationsRunner(cfg, override_dataset=ds)
    print(runner)

    tmp = Path(output_path) / ".tmp_shards"
    tmp.mkdir(parents=True, exist_ok=True)
    store = runner.activations_store
    with torch.no_grad():
        for i in tqdm(range(cfg.n_buffers), desc=f"rank{rank}"):
            try:
                buffers = [
                    store.get_raw_llm_batch() for _ in range(cfg.n_batches_in_buffer)
                ]
            except StopIteration:
                print(f"rank{rank}: dataset exhausted at buffer {i}")
                break
            acts = torch.cat([b[0] for b in buffers], dim=0)
            token_ids = (
                torch.cat([b[1] for b in buffers]) if buffers[0][1] is not None else None
            )
            shard = runner._create_shard((acts, token_ids))
            shard.save_to_disk(
                f"{tmp}/shard_{i * world_size + rank:05d}", num_shards=1
            )
            del buffers, acts, token_ids, shard
    print(f"rank{rank}: done, wrote shards to {tmp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--hook-name", type=str)
    parser.add_argument("--d-in", type=int)
    parser.add_argument("--dataset", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--training-tokens", type=int, default=500_000_000)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument(
        "--act-dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Stored dtype; arrow has no bfloat16. float16 halves disk but can "
        "overflow on outlier activations of large models.",
    )
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--model-dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="LLM dtype; use bfloat16 for 8B+ models.",
    )
    parser.add_argument("--autocast-lm", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--buffer-size-gb", type=float, default=2.0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help="Merge .tmp_shards written by all ranks, then exit.",
    )
    args = parser.parse_args()

    if args.consolidate_only:
        consolidate(args.output_path)
        raise SystemExit(0)

    missing = [n for n in ("model_name", "hook_name", "d_in") if getattr(args, n) is None]
    if missing:
        parser.error(f"required unless --consolidate-only: {missing}")

    cache_activations(
        model_name=args.model_name,
        hook_name=args.hook_name,
        d_in=args.d_in,
        dataset_path=args.dataset,
        context_size=args.context_size,
        training_tokens=args.training_tokens,
        model_batch_size=args.model_batch_size,
        act_dtype=args.act_dtype,
        output_path=args.output_path,
        model_dtype=args.model_dtype,
        autocast_lm=args.autocast_lm,
        device=args.device,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        buffer_size_gb=args.buffer_size_gb,
        rank=args.rank,
        world_size=args.world_size,
    )

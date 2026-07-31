"""Filesystem roots, all overridable by environment variable.

    SASA_CACHE_ROOT   HuggingFace / wandb / temp caches
    SASA_SCRATCH      cached LLM activations (large; put this on fast local disk)
    SASA_CHECKPOINTS  trained SAE directories (cfg.json + sae_weights.safetensors)
    SASA_RESULTS      JSON/CSV/figure outputs
    SASA_PROBE_DATA   SAEBench absorption probe artifacts for GPT-2 layer 7
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _root(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


CACHE_ROOT = _root("SASA_CACHE_ROOT", Path.home() / ".cache" / "sasa")
SCRATCH = _root("SASA_SCRATCH", REPO / "scratch")
CHECKPOINTS = _root("SASA_CHECKPOINTS", REPO / "checkpoints")
RESULTS = _root("SASA_RESULTS", REPO / "results")
PROBE_DATA = _root("SASA_PROBE_DATA", SCRATCH / "absorption_probes")

# GPT-2 layer 7, K=2048, r=6, s=10.
PAPER_GPT2_SASA = CHECKPOINTS / "topk_sasa_gpt2_l7_n2048_r6_k10"

GPT2_ACT_CACHE = SCRATCH / "gpt2_l7_probe_acts"

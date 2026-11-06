import argparse
import os
import sys

import sae_bench.evals.core.main as saebench_core
import sae_bench.sae_bench_utils.general_utils as saebench_utils
import torch

import paths
from configs import load_config
from eval.dictionaries import load_sae

N_RECONSTRUCTION_BATCHES, N_SPARSITY_BATCHES, BATCH_PROMPTS = 10, 50, 16


def use_gpt2_model_name(sae) -> None:
    # SAEBench loads the LLM that the SAE names; Bloom's GPT-2 SAE calls it gpt2-small
    name = getattr(sae.cfg, "model_name", None) or getattr(getattr(sae.cfg, "metadata", None), "model_name", None)
    if isinstance(name, str) and name.startswith("gpt2"):
        sae.cfg.model_name = "gpt2"
        if getattr(sae.cfg, "metadata", None) is not None:
            sae.cfg.metadata.model_name = "gpt2"


class SAEBenchConfig:
    # sae.cfg with its sae_lens metadata (hook_name, model_name, ...) readable as attributes, as SAEBench reads it
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, name):
        if name in ("hook_head_index", "hook_head_indices"):
            return None
        metadata = getattr(self._cfg, "metadata", None)
        for source in (self._cfg, metadata):
            value = getattr(source, name, None)
            if value is not None:
                return value
        if name == "hook_layer" and isinstance(getattr(metadata, "hook_name", None), str):
            return int(metadata.hook_name.split(".")[1])
        raise AttributeError(name)


class SAEBenchAdapter:
    # SAEBench reads unit-norm decoder rows; decode() applies the row norms to the codes instead
    def __init__(self, sae):
        self.sae, self.cfg = sae, SAEBenchConfig(sae.cfg)
        self._read_weights()

    @torch.no_grad()
    def _read_weights(self) -> None:
        decoder = self.sae.W_dec.detach()
        self.row_norms = decoder.norm(dim=1).clamp_min(1e-8)
        self.W_dec = (decoder / self.row_norms[:, None]).contiguous()
        self.W_enc, self.b_dec, self.device = self.sae.W_enc, self.sae.b_dec, self.sae.device
        self.dtype = getattr(self.sae, "dtype", None) or self.sae.W_dec.dtype

    def to(self, *args, **kwargs):
        self.sae = self.sae.to(*args, **kwargs)
        self._read_weights()
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.sae.device or x.dtype != self.dtype:
            x = x.to(self.sae.device, dtype=self.dtype)
        return self.sae.encode(x)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        shape = codes.shape
        flat = codes.reshape(-1, shape[-1])
        out = ((flat * self.row_norms[None, :].to(flat.dtype)) @ self.W_dec.to(flat.dtype)
               + self.sae.b_dec.to(flat.dtype))
        out = self.sae.hook_sae_recons(out.view(*shape[:-1], out.shape[-1]))
        # the normalisation statistics are per token, [N, 1]
        return self.sae.run_time_activation_norm_fn_out(out.reshape(-1, out.shape[-1])).view(*shape[:-1], out.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("saes", nargs="+", help="<config>/<sae> ...")
    names = parser.parse_args().saes
    saebench_utils.setup_environment()
    for name in names:
        config_name, sae_name = name.split("/")
        config = load_config(config_name)
        out = paths.RESULTS / "core" / config_name
        out.mkdir(parents=True, exist_ok=True)
        sae = load_sae(name)
        use_gpt2_model_name(sae)
        # one call per SAE: within a call SAEBench reuses the LLM loaded with the first SAE's settings
        saebench_core.multiple_evals(
            selected_saes=[(f"core_{sae_name}_{config['hook_name']}", SAEBenchAdapter(sae))],
            n_eval_reconstruction_batches=N_RECONSTRUCTION_BATCHES,
            n_eval_sparsity_variance_batches=N_SPARSITY_BATCHES, eval_batch_size_prompts=BATCH_PROMPTS,
            # SAEBench's result schema requires the per-latent statistics; without them it saves nothing
            compute_featurewise_density_statistics=True, compute_featurewise_weight_based_metrics=True,
            exclude_special_tokens_from_reconstruction=True, dataset=config["dataset_path"],
            context_size=config["context_size"], output_folder=str(out), verbose=True, dtype="float32", device="cuda",
            force_rerun=True)
        del sae


if __name__ == "__main__":
    main()
    # Arrow threads can still be reading the streamed dataset through Python, which aborts or hangs the
    # interpreter's shutdown; the results are written, so end the process without it
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

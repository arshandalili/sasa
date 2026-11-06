import argparse
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from sae_lens import SAE
from safetensors.torch import load_file, save_file

import paths
import sasa  # noqa: F401  registers SASA with sae_lens

BLOOM = "gpt2_l7/jb_relu_w24576"  # Bloom (2024), from sae_lens
ENGELS = "mistral_l8/engels_relu_w65536"  # Engels et al. (2025), converted from their release by main()
ENGELS_ENCODE_CHUNK = 32  # rows per matmul; the chunk size changes the rounding of the codes


def load_sae(name: str, device: str = "cuda"):
    if name == BLOOM:
        return SAE.from_pretrained("gpt2-small-res-jb", "blocks.7.hook_resid_pre", device=device).eval()
    if name == ENGELS:
        return EngelsReLU(paths.CHECKPOINTS / name, device)
    return SAE.load_from_disk(str(paths.CHECKPOINTS / name), device=device).eval()


class EngelsReLU(nn.Module):
    # rescales each input to norm sqrt(d) before encoding; encode() keeps the scale of every token
    def __init__(self, path: Path, device: str = "cuda"):
        super().__init__()
        weights = load_file(str(Path(path) / "sae_weights.safetensors"), device=device)
        self.W_enc, self.b_enc, self.b_dec = (nn.Parameter(weights[k].float()) for k in ("W_enc", "b_enc", "b_dec"))
        norms = weights["W_dec"].float().norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec = nn.Parameter(weights["W_dec"].float() / norms)
        self.decoder_norms = norms.squeeze(1)
        self.input_scale = None
        self.cfg = types.SimpleNamespace(normalize_activations="none")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.W_enc.dtype)
        self.input_scale = (x.shape[-1] ** 0.5) / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scaled = x * self.input_scale
        return torch.cat([torch.relu(scaled[i:i + ENGELS_ENCODE_CHUNK] @ self.W_enc + self.b_enc)
                          for i in range(0, len(scaled), ENGELS_ENCODE_CHUNK)])


class _ReleaseConfig:  # stands in for the class of Engels et al.'s code that their pickle names
    pass


def convert_engels(release_file: str, out_dir: Path) -> None:
    saes, config = types.ModuleType("saes"), types.ModuleType("saes.config")
    config.LanguageModelSAERunnerConfig = _ReleaseConfig
    saes.config = config
    sys.modules.setdefault("saes", saes)
    sys.modules.setdefault("saes.config", config)
    state = torch.load(release_file, map_location="cpu", weights_only=False)["state_dict"]
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = {k: state[k].contiguous() for k in ("W_enc", "W_dec", "b_enc", "b_dec")}
    save_file(weights, str(out_dir / "sae_weights.safetensors"))
    cfg = {"architecture": "engels_relu", "d_in": state["W_enc"].shape[0], "d_sae": state["W_enc"].shape[1],
           "hook_name": "blocks.8.hook_resid_pre", "model_name": "mistralai/Mistral-7B-v0.1"}
    (out_dir / "cfg.json").write_text(json.dumps(cfg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_file", help="Mistral-7B-v0.1_blocks.8.hook_resid_pre_65536_final.pt")
    convert_engels(parser.parse_args().release_file, paths.CHECKPOINTS / ENGELS)


if __name__ == "__main__":
    main()

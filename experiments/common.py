"""Loading and exact group-contribution machinery for the SASA causal experiments."""

import torch
from sae_lens import SAE, register_sae_class

from sasa.compat import patch_rope_theta
from sasa.model import (
    TopKSASAInference,
    TopKSASAInferenceConfig,
    _per_sample_topk_mask,
)

register_sae_class("topk_sasa", TopKSASAInference, TopKSASAInferenceConfig)

LN_EPS = 1e-5  # must equal sae_lens run_time_activation_ln_in's eps


def load_sasa(path, device="cuda"):
    return SAE.load_from_disk(path, device=device).eval()


def load_lm(name="mistralai/Mistral-7B-v0.1", device="cuda"):
    patch_rope_theta()
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        name, device=device, dtype=torch.bfloat16, center_writing_weights=False
    ).eval()


def dec(sae):
    """W_dec as (n_groups, r, d_in)."""
    return sae.W_dec.view(sae.n_groups, sae.group_rank, -1)


@torch.no_grad()
def encode(sae, h):
    """Raw h (..., d) -> gated group coords (..., G, r) and per-token ln std (..., 1)."""
    h = h.to(sae.W_enc.dtype)
    c = h - h.mean(-1, keepdim=True)
    std = c.std(-1, keepdim=True)
    xn = c / (std + LN_EPS)
    if sae.cfg.apply_b_dec_to_input:
        xn = xn - sae.b_dec
    pre = xn @ sae.W_enc + sae.b_enc
    flat = pre.reshape(-1, sae.n_groups, sae.group_rank)
    mask = _per_sample_topk_mask(flat.norm(dim=-1), sae.k_groups)
    a = flat * mask.unsqueeze(-1)
    return a.reshape(*pre.shape[:-1], sae.n_groups, sae.group_rank), std


def contrib(sae, a, std, g):
    """Raw-space vector that group g writes into the residual stream."""
    return (a[..., g, :] @ dec(sae)[g]) * std


def project_out(sae, h, g):
    """Raw-space h with its component in col(D_g) removed (subspace ablation)."""
    h = h.to(sae.W_dec.dtype)
    mu = h.mean(-1, keepdim=True)
    c = h - mu
    std = c.std(-1, keepdim=True)
    xn = c / (std + LN_EPS)
    q, _ = torch.linalg.qr(dec(sae)[g].T)  # (d, r) orthonormal basis of the group span
    return (xn - (xn @ q) @ q.T) * (std + LN_EPS) + mu


@torch.no_grad()
def recon(sae, h):
    """Full SASA reconstruction, rebuilt from the same pieces the interventions use."""
    a, std = encode(sae, h)
    mu = h.to(sae.W_dec.dtype).mean(-1, keepdim=True)
    flat = a.reshape(*a.shape[:-2], -1)
    return (flat @ sae.W_dec + sae.b_dec) * std + mu


def selftest(sae, n=64, seed=0):
    """Check that our decomposition reproduces the model's own forward pass."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = (torch.randn(n, sae.cfg.d_in, generator=g) * 14.0).to(sae.W_dec.device)
    ref = sae(h).float()
    ours = recon(sae, h).float()
    rel = (ours - ref).norm() / ref.norm()

    a, std = encode(sae, h)
    mu = h.to(a.dtype).mean(-1, keepdim=True)
    active = (a.norm(dim=-1) > 0)[0].nonzero().flatten().tolist()
    parts = sum(contrib(sae, a[:1], std[:1], gi) for gi in active)
    whole = parts + sae.b_dec * std[:1] + mu[:1]
    rel_parts = (whole - ref[:1]).norm() / ref[:1].norm()
    return rel.item(), rel_parts.item(), len(active)

"""Matched-budget GPT-2 layer-7 dictionaries."""

import json
from pathlib import Path

import torch

from paths import CHECKPOINTS

SWEEP = CHECKPOINTS
LN_EPS = 1e-5

L0_VALUES = (30, 60, 120, 240)
PAPER_RANK, PAPER_NUC = 6, 100

# Written by scripts/train_matched_budget.py.
GRID = {
    **{(arch, l0): f"{arch}_gpt2_l7_l0{l0}"
       for arch in ("topk", "batchtopk", "matryoshka") for l0 in L0_VALUES},
    **{("sasa", l0): f"topk_sasa_gpt2_l7_r{PAPER_RANK}_nuc{PAPER_NUC}_l0{l0}"
       for l0 in L0_VALUES},
}


class Dict2:
    """One dictionary, exposed in columns."""

    def __init__(self, path, device="cuda"):
        from safetensors.torch import load_file
        self.path = str(path)
        self.cfg = json.loads((Path(path) / "cfg.json").read_text())
        w = load_file(str(Path(path) / "sae_weights.safetensors"))
        self.arch = self.cfg["architecture"]
        self.W_enc = w["W_enc"].to(device).float()
        self.b_enc = w["b_enc"].to(device).float()
        self.b_dec = w["b_dec"].to(device).float()
        W_dec = w["W_dec"].to(device).float()
        if "threshold" in w:
            self.thr = w["threshold"].to(device).float()
        elif "topk_threshold" in w:
            self.thr = float(w["topk_threshold"])
        else:
            self.thr = None
        self.rescale = bool(self.cfg.get("rescale_acts_by_decoder_norm"))
        self.ln = self.cfg.get("normalize_activations") == "layer_norm"
        self.apply_b_dec = bool(self.cfg.get("apply_b_dec_to_input"))
        self.rank = int(self.cfg.get("group_rank") or 1)
        self.k = self.cfg.get("k")
        self.k_groups = self.cfg.get("k_groups")
        self.dec_norm = W_dec.norm(dim=-1).clamp_min(1e-8)
        self.W = (W_dec / self.dec_norm[:, None]) if self.rescale else W_dec

        if self.rank > 1:
            self.l0_cols = int(self.k_groups * self.rank)
        elif self.k is not None:
            self.l0_cols = int(self.k)
        else:
            self.l0_cols = None

    @torch.no_grad()
    def _pre(self, h):
        x = h
        if self.ln:
            x = x - x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)
            x = x / (std + LN_EPS)
        else:
            std = torch.ones(h.shape[0], 1, device=h.device)
        if self.apply_b_dec:
            x = x - self.b_dec
        pre = x @ self.W_enc + self.b_enc
        if self.rescale:
            pre = pre * self.dec_norm
        return pre, std

    @torch.no_grad()
    def coeffs(self, H, chunk=512):
        """(N, ncols) coefficients c with contribution_j = c_j * W[j] in RAW space."""
        out = []
        for i in range(0, H.shape[0], chunk):
            h = H[i:i + chunk].cuda().float()
            pre, std = self._pre(h)
            if self.rank > 1:
                g = pre.reshape(pre.shape[0], -1, self.rank)
                nrm = g.norm(dim=-1)
                idx = nrm.topk(self.k_groups, dim=-1).indices
                mask = torch.zeros_like(nrm).scatter_(1, idx, 1.0)
                a = (g * mask.unsqueeze(-1)).reshape(pre.shape[0], -1)
                a = a * std                       # decoded through the layer-norm scale
            elif self.thr is not None:            # batchtopk / matryoshka at inference
                a = pre * (pre > self.thr)
            else:                                  # per-token topk
                v, idx = pre.topk(int(self.k), dim=-1)
                a = torch.zeros_like(pre).scatter_(1, idx, v.clamp_min(0))
            out.append(a.cpu())
        return torch.cat(out)

    @torch.no_grad()
    def recon(self, H, chunk=512):
        out = []
        for i in range(0, H.shape[0], chunk):
            h = H[i:i + chunk].cuda().float()
            c = self.coeffs(h).cuda()
            r = c @ self.W + self.b_dec * (h.std(-1, keepdim=True) if self.ln else 1.0)
            if self.ln:
                r = r + h.mean(-1, keepdim=True)
            out.append(r.cpu())
        return torch.cat(out)


def load(arch, l0, device="cuda"):
    return Dict2(SWEEP / GRID[(arch, l0)], device)


def selfcheck(H):
    """Measured l0 (columns) and FVE for every cell -- the convention check."""
    rows = []
    for (arch, l0) in sorted(GRID, key=lambda x: (x[0], x[1])):
        d = load(arch, l0)
        c = d.coeffs(H)
        meas_l0 = float((c != 0).float().sum(-1).mean())
        r = d.recon(H)
        fve = float(1 - (H - r).pow(2).sum() / (H - H.mean(0)).pow(2).sum())
        rows.append((arch, l0, d.l0_cols, meas_l0, fve))
        cfg_l0 = f"{d.l0_cols:4d}" if d.l0_cols is not None else "thr "
        print(f"  {arch:11s} l0={l0:4d} l0_cfg={cfg_l0}  measured l0={meas_l0:7.1f}  FVE={fve:6.3f}",
              flush=True)
        del d, c, r
        torch.cuda.empty_cache()
    return rows

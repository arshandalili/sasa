"""Shared SAEBench plumbing: the SAE adapter and the core-eval summary writers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from sae_lens import SAE


def _set_hf_cache_defaults(workspace_root: Path) -> None:
    cache_root = workspace_root / ".cache"
    # Set explicitly, not setdefault: clusters often pre-point these at a quota-limited $HOME.
    os.environ["HF_HOME"] = str(cache_root / "hf")
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "hf_datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hf_hub")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _canonicalize_model_name_for_saebench(sae: SAE) -> None:
    """
    SAE-Lens metadata sometimes says "gpt2-small"; SAEBench wants "gpt2".
    """
    model_name = getattr(sae.cfg, "model_name", None)
    if isinstance(model_name, str):
        # Canonicalize common GPT-2 aliases
        if model_name in {"gpt2-small", "openai/gpt2", "gpt2"} or model_name.startswith(
            "gpt2"
        ):
            sae.cfg.model_name = "gpt2"
            meta = getattr(sae.cfg, "metadata", None)
            if meta is not None:
                # SAELens metadata types have changed across versions; try a few strategies.
                try:
                    setattr(meta, "model_name", "gpt2")
                except Exception:
                    pass
                try:
                    meta["model_name"] = "gpt2"  # type: ignore[index]
                except Exception:
                    pass


class SAEBenchAdapter:
    """Wraps a SAE-Lens SAE so SAEBench sees a unit-norm decoder and group-aware stats."""

    def __init__(
        self,
        sae: SAE,
        *,
        match_token_norm: bool,
        n_groups: int | None = None,
        group_rank: int | None = None,
        group_sizes: list[int] | None = None,
        use_group_norm: bool = False,
    ) -> None:
        self.sae = sae
        self.match_token_norm = match_token_norm
        self.use_group_norm = use_group_norm

        self.n_groups = n_groups
        self.group_rank = group_rank
        self.group_sizes = [int(value) for value in group_sizes] if group_sizes is not None else None
        
        if self.use_group_norm and self.group_sizes is None and (n_groups is None or group_rank is None):
            raise ValueError(
                "use_group_norm=True requires either uniform n_groups/group_rank metadata or explicit group_sizes."
            )
        if self.group_sizes is not None:
            self.n_groups = len(self.group_sizes)

        # (z * row_norms) @ (W_dec / row_norms) == z @ W_dec, so SAEBench sees a
        # unit-norm decoder without the SAE being mutated.
        self._wdec_row_norms: torch.Tensor | None = None  # [d_sae]
        self._wdec_unit: torch.Tensor | None = None  # [d_sae, d_in]
        
        self._wdec_group_centroids: torch.Tensor | None = None # [n_groups, d_in]
        
        self._sync_decoder_cache()

        self._last_input_norms: torch.Tensor | None = None  # [tokens, 1] on sae.device

        self._tokens_seen = 0
        self._sum_true_l1_abs = 0.0
        self._sum_l0_atoms = 0.0

        self._sum_l0_groups = 0.0
        self._sum_l1_group_norms = 0.0
        self._group_token_counts: torch.Tensor | None = None  # [n_groups]

    # --- SAE-Lens passthrough attributes SAEBench expects ---
    @property
    def cfg(self):
        # SAE-Lens keeps hook_layer/model_name in metadata; SAEBench wants them as attributes.
        adapter_self = self
        
        class CfgProxy:
            def __init__(self, original_cfg, n_groups, use_group_norm):
                self._original_cfg = original_cfg
                self._n_groups = n_groups
                self._use_group_norm = use_group_norm
                
            def __getattr__(self, name):
                # Override d_sae when using group norms
                if name == "d_sae" and self._use_group_norm:
                    return self._n_groups

                if name in {"hook_head_index", "hook_head_indices"}:
                    return None
                    
                try:
                    val = getattr(self._original_cfg, name)
                    if val is not None:
                        return val
                except AttributeError:
                    pass
                    
                meta = getattr(self._original_cfg, "metadata", None)
                if meta is not None:
                    if hasattr(meta, name):
                        val = getattr(meta, name, None)
                        if val is not None:
                            return val
                    if hasattr(meta, "__getitem__"):
                        try:
                            val = meta[name]
                            if val is not None:
                                return val
                        except (KeyError, TypeError):
                            pass
                    meta_dict = dict(meta) if hasattr(meta, "__iter__") else {}
                    if name in meta_dict and meta_dict[name] is not None:
                        return meta_dict[name]
                    
                    if name == "hook_layer":
                        hook_name = None
                        try:
                            hook_name = getattr(meta, "hook_name", None) or meta_dict.get("hook_name")
                        except:
                            pass
                        if hook_name:
                            try:
                                return int(hook_name.split(".")[1])
                            except (IndexError, ValueError):
                                pass
                
                raise AttributeError(f"Config has no attribute '{name}'")
                
        return CfgProxy(self.sae.cfg, self.n_groups, self.use_group_norm)

    @property
    def W_enc(self):
        return self.sae.W_enc

    @property
    def W_dec(self):
        if self.use_group_norm:
            return self._wdec_group_centroids
        return self._wdec_unit if self._wdec_unit is not None else self.sae.W_dec

    @property
    def b_dec(self):
        return self.sae.b_dec

    @property
    def device(self):
        return self.sae.device

    @property
    def dtype(self):
        """
        SAEBench expects `sae.dtype` to exist (SAE Lens provides it).
        Provide it here so evals like absorption can do `.to(sae.device, dtype=sae.dtype)`.
        """
        dt = getattr(self.sae, "dtype", None)
        if dt is not None:
            return dt
        return self.sae.W_dec.dtype

    def to(self, *args, **kwargs):
        self.sae = self.sae.to(*args, **kwargs)
        self._sync_decoder_cache()
        return self

    def _sync_decoder_cache(self) -> None:
        with torch.no_grad():
            W = self.sae.W_dec.detach()
            norms = W.norm(dim=1).clamp_min(1e-8)  # [d_sae]
            self._wdec_row_norms = norms
            self._wdec_unit = (W / norms[:, None]).contiguous()
            
            if self.use_group_norm and self.n_groups is not None:
                if self.group_sizes is not None:
                    splits = list(torch.split(W, self.group_sizes, dim=0))
                    centroids = torch.stack([split.mean(dim=0) for split in splits], dim=0)
                    centroid_norms = centroids.norm(dim=1, keepdim=True)
                    tiny = centroid_norms.squeeze(-1) < 1e-8
                    if tiny.any():
                        fallback_rows = torch.stack([split[0] for split in splits], dim=0)
                        fallback_rows = fallback_rows / fallback_rows.norm(dim=1, keepdim=True).clamp_min(1e-8)
                        centroids = torch.where(tiny[:, None], fallback_rows, centroids)
                else:
                    W_groups = W.view(self.n_groups, self.group_rank, -1)
                    centroids = W_groups.mean(dim=1) # [n_groups, d_in]
                centroid_norms = centroids.norm(dim=1, keepdim=True).clamp_min(1e-8)
                self._wdec_group_centroids = (centroids / centroid_norms).contiguous()

    # --- Core API ---
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.sae.device or x.dtype != self.dtype:
            x = x.to(self.sae.device, dtype=self.dtype)
        x_flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            self._last_input_norms = x_flat.norm(dim=-1, keepdim=True).to(self.sae.device)

        z = self.sae.encode(x)

        z_flat = z.reshape(-1, z.shape[-1])
        
        if self.use_group_norm and self.n_groups is not None:
            if self.group_sizes is not None:
                z_norms = torch.stack(
                    [chunk.norm(dim=-1) for chunk in torch.split(z_flat, self.group_sizes, dim=-1)],
                    dim=-1,
                )
            else:
                if self.group_rank is None or z_flat.shape[-1] != self.n_groups * self.group_rank:
                    raise ValueError("Shape mismatch in encode")
                z_groups_view = z_flat.view(-1, self.n_groups, self.group_rank)
                z_norms = z_groups_view.norm(dim=-1) # [tokens, n_groups]
            
            with torch.no_grad():
                self._tokens_seen += int(z_flat.shape[0])
                group_active = (z_norms > 0).to(z_norms.dtype)
                self._sum_l0_groups += float(group_active.sum(dim=-1).sum().item())
            
            return z_norms.view(*z.shape[:-1], self.n_groups)

        with torch.no_grad():
            self._tokens_seen += int(z_flat.shape[0])
            self._sum_true_l1_abs += float(z_flat.abs().sum(dim=-1).sum().item())
            self._sum_l0_atoms += float((z_flat != 0).sum(dim=-1).float().sum().item())

            if self.n_groups is not None:
                if self.group_sizes is not None:
                    group_norms = torch.stack(
                        [chunk.norm(dim=-1) for chunk in torch.split(z_flat, self.group_sizes, dim=-1)],
                        dim=-1,
                    )
                elif self.group_rank is not None:
                    if z_flat.shape[-1] != self.n_groups * self.group_rank:
                        raise ValueError(
                            f"Expected d_sae={self.n_groups*self.group_rank} for group metrics, got {z_flat.shape[-1]}"
                        )
                    z_groups = z_flat.view(-1, self.n_groups, self.group_rank)
                    group_norms = z_groups.norm(dim=-1)  # [tokens, n_groups]
                else:
                    raise ValueError("Missing group metadata for group statistics.")
                group_active = (group_norms > 0).to(group_norms.dtype)
                self._sum_l0_groups += float(group_active.sum(dim=-1).sum().item())
                self._sum_l1_group_norms += float((group_norms * group_active).sum(dim=-1).sum().item())
                if self._group_token_counts is None:
                    self._group_token_counts = torch.zeros(self.n_groups, dtype=torch.float64)
                self._group_token_counts += group_active.sum(dim=0).double().cpu()

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Unit-normalized W_dec with the exact original reconstruction: folding the norms
        # into the weights would change thresholding behavior.
        z_shape = z.shape
        z_flat = z.reshape(-1, z.shape[-1])

        if self.use_group_norm:
             if self._wdec_group_centroids is None:
                 raise RuntimeError("Group centroids not initialized")
             
             centroids = self._wdec_group_centroids.to(device=z_flat.device, dtype=z_flat.dtype)
             out_flat = z_flat @ centroids + self.sae.b_dec.to(device=z_flat.device, dtype=z_flat.dtype)
        else:
            if self._wdec_row_norms is None or self._wdec_unit is None:
                out_flat = z_flat @ self.sae.W_dec + self.sae.b_dec
            else:
                row_norms = self._wdec_row_norms.to(device=z_flat.device, dtype=z_flat.dtype)
                W_unit = self._wdec_unit.to(device=z_flat.device, dtype=z_flat.dtype)
                out_flat = (z_flat * row_norms[None, :]) @ W_unit + self.sae.b_dec.to(
                    device=z_flat.device, dtype=z_flat.dtype
                )

        out = out_flat.view(*z_shape[:-1], out_flat.shape[-1])

        # Preserve SAE-Lens decode hook semantics if present (e.g., hook points).
        hook = getattr(self.sae, "hook_sae_recons", None)
        if hook is not None and callable(hook):
            try:
                out = hook(out)
            except Exception:
                pass

        # CRITICAL: Apply layer norm de-normalization if the SAE uses it.
        # The SAE's encode() stores ln_mu/ln_std; we must use them to rescale output.
        run_time_norm_out = getattr(self.sae, "run_time_activation_norm_fn_out", None)
        if run_time_norm_out is not None and callable(run_time_norm_out):
            original_shape = out.shape
            # encode() may leave ln_mu/ln_std shaped like its own input (2D or 3D).
            # Flatten them to match the flattened `out` or the multiply broadcasts wrong.
            ln_mu = getattr(self.sae, "ln_mu", None)
            ln_std = getattr(self.sae, "ln_std", None)
            saved = None
            if ln_mu is not None and ln_std is not None and ln_mu.dim() > 2:
                saved = (ln_mu, ln_std)
                self.sae.ln_mu = ln_mu.reshape(-1, ln_mu.shape[-1])
                self.sae.ln_std = ln_std.reshape(-1, ln_std.shape[-1])
            try:
                out = run_time_norm_out(out.reshape(-1, out.shape[-1])).view(*original_shape)
            finally:
                if saved is not None:
                    self.sae.ln_mu, self.sae.ln_std = saved

        if not self.match_token_norm or self._last_input_norms is None:
            return out

        # Rescale reconstructed activations token-wise to match norms of input activations.
        out_flat = out.reshape(-1, out.shape[-1])
        target_norms = self._last_input_norms
        if target_norms.shape[0] != out_flat.shape[0]:
            # If shapes don't align, skip scaling rather than silently doing something wrong.
            return out

        with torch.no_grad():
            eps = 1e-8
            out_norms = out_flat.norm(dim=-1, keepdim=True).clamp_min(eps)
            scale = target_norms / out_norms
            out_flat = out_flat * scale
        return out_flat.view_as(out)

    # --- Stats export ---
    def get_stats(self) -> dict[str, float]:
        if self._tokens_seen == 0:
            return {}
        return {
            "true_l1_abs_atoms": self._sum_true_l1_abs / self._tokens_seen,
            "l0_atoms": self._sum_l0_atoms / self._tokens_seen,
        }

    def get_group_stats(self) -> dict[str, float]:
        if self._tokens_seen == 0 or self.n_groups is None or self.group_rank is None:
            return {}
        out: dict[str, float] = {
            "l0_groups": self._sum_l0_groups / self._tokens_seen,
            "l1_groups_norm": self._sum_l1_group_norms / self._tokens_seen,
        }
        if self._group_token_counts is not None:
            freqs = (self._group_token_counts / float(self._tokens_seen)).numpy()
            out["frac_groups_alive"] = float((freqs > 0).mean())
            out["freq_groups_over_1_percent"] = float((freqs > 0.01).mean())
            out["freq_groups_over_10_percent"] = float((freqs > 0.1).mean())
        return out


def _flatten_metrics(prefix: str, obj) -> list[tuple[str, float]]:
    """
    Flatten nested SAEBench metric dicts into (metric_path, value) rows.
    """
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(_flatten_metrics(key, v))
        return out
    if isinstance(obj, (int, float)) and obj is not None:
        return [(prefix, float(obj))]
    return out


def _write_core_summary(output_dir: Path) -> None:
    """
    Read SAEBench core JSON outputs from output_dir and write a compact CSV/JSON summary.
    """
    result_files = sorted(output_dir.glob("*_eval_results.json"))
    if not result_files:
        print(f"No SAEBench result files found in {output_dir}")
        return

    # Load extra metrics (true L1, group-level sparsity) if present.
    extra_path = output_dir / "extra_metrics.json"
    extra: dict[str, object] | None = None
    if extra_path.exists():
        try:
            extra = json.loads(extra_path.read_text())
        except Exception:
            extra = None

    rows: list[dict[str, object]] = []
    for p in result_files:
        data = json.loads(p.read_text())
        sae_name = data.get("sae_name", p.stem)
        metrics = data.get("eval_result_metrics", {})
        for metric_path, value in _flatten_metrics("", metrics):
            # IMPORTANT: SAEBench core's 'sparsity.l1' is a signed sum (not abs),
            # which can be negative for SAEs with signed activations.
            if metric_path == "sparsity.l1":
                metric_path = "sparsity.l1_signed_sum"
            rows.append({"sae": sae_name, "metric": metric_path, "value": value})

        # Inject corrected metrics into the main summary so users don't misread SAEBench's l1.
        if extra is not None:
            is_control = str(p.name).startswith("control_")
            is_sasa = not is_control
            if is_control and isinstance(extra.get("control"), dict):
                ctrl = extra["control"]
                if "true_l1_abs_atoms" in ctrl:
                    rows.append(
                        {
                            "sae": sae_name,
                            "metric": "sparsity.true_l1_abs_atoms",
                            "value": float(ctrl["true_l1_abs_atoms"]),
                        }
                    )
                if "l0_atoms" in ctrl:
                    rows.append(
                        {
                            "sae": sae_name,
                            "metric": "sparsity.l0_atoms_measured",
                            "value": float(ctrl["l0_atoms"]),
                        }
                    )
            if is_sasa and isinstance(extra.get("sasa"), dict):
                man = extra["sasa"]
                if "true_l1_abs_atoms" in man:
                    rows.append(
                        {
                            "sae": sae_name,
                            "metric": "sparsity.true_l1_abs_atoms",
                            "value": float(man["true_l1_abs_atoms"]),
                        }
                    )
                if "l0_atoms" in man:
                    rows.append(
                        {
                            "sae": sae_name,
                            "metric": "sparsity.l0_atoms_measured",
                            "value": float(man["l0_atoms"]),
                        }
                    )
                # Group-level metrics are already prefixed with "group_sparsity." in extra_metrics.json
                for k, v in man.items():
                    if isinstance(k, str) and k.startswith("group_sparsity.") and isinstance(
                        v, (int, float)
                    ):
                        rows.append({"sae": sae_name, "metric": k, "value": float(v)})

    # Long-form CSV (easy to diff/plot)
    csv_path = output_dir / "core_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sae", "metric", "value"])
        w.writeheader()
        w.writerows(rows)

    # Also dump to JSON
    json_path = output_dir / "core_summary.json"
    json_path.write_text(json.dumps(rows, indent=2))

    print(f"Wrote summary: {csv_path}")


def _write_extra_metrics(
    output_dir: Path, *, control_adapter: SAEBenchAdapter, sasa_adapter: SAEBenchAdapter
) -> None:
    """Write SAEBench-independent metrics:
      - true L1 (abs) at atom level (SAEBench core's 'l1' is a signed sum)
      - group-level sparsity for SASA (groups-as-features)
    """
    extra = {
        "control": control_adapter.get_stats(),
        "sasa": {
            **sasa_adapter.get_stats(),
            **{f"group_sparsity.{k}": v for k, v in sasa_adapter.get_group_stats().items()},
        },
        "note": {
            "saebench_core_l1_is_signed_sum": True,
            "group_definition": "A SASA group is active if any of its atoms is non-zero in encode(x).",
            "sasa_decode_scaling": sasa_adapter.match_token_norm,
        },
    }
    p = output_dir / "extra_metrics.json"
    p.write_text(json.dumps(extra, indent=2))


def _postprocess_saebench_core_jsons(
    output_dir: Path, *, control_adapter: SAEBenchAdapter, sasa_adapter: SAEBenchAdapter
) -> None:
    """Modify SAEBench Core JSON files in-place so downstream analysis doesn't misread metrics."""
    extra = {
        "control": control_adapter.get_stats(),
        "sasa": {
            **sasa_adapter.get_stats(),
            **{f"group_sparsity.{k}": v for k, v in sasa_adapter.get_group_stats().items()},
        },
    }

    for p in sorted(output_dir.glob("*_eval_results.json")):
        data = json.loads(p.read_text())
        metrics = data.get("eval_result_metrics", {})
        sparsity = metrics.get("sparsity", {})
        if isinstance(sparsity, dict) and "l1" in sparsity and "l1_signed_sum" not in sparsity:
            sparsity["l1_signed_sum"] = sparsity.pop("l1")
            metrics["sparsity"] = sparsity
            data["eval_result_metrics"] = metrics

        if p.name.startswith("control_"):
            ctrl = extra["control"]
            if isinstance(ctrl, dict):
                metrics.setdefault("sparsity", {})
                metrics["sparsity"]["true_l1_abs_atoms"] = float(
                    ctrl.get("true_l1_abs_atoms", 0.0)
                )
                metrics["sparsity"]["l0_atoms_measured"] = float(ctrl.get("l0_atoms", 0.0))
        else:
            man = extra["sasa"]
            if isinstance(man, dict):
                metrics.setdefault("sparsity", {})
                metrics["sparsity"]["true_l1_abs_atoms"] = float(
                    man.get("true_l1_abs_atoms", 0.0)
                )
                metrics["sparsity"]["l0_atoms_measured"] = float(man.get("l0_atoms", 0.0))
                metrics.setdefault("group_sparsity", {})
                for k, v in man.items():
                    if isinstance(k, str) and k.startswith("group_sparsity.") and isinstance(
                        v, (int, float)
                    ):
                        metrics["group_sparsity"][k.split(".", 1)[1]] = float(v)

        data["eval_result_metrics"] = metrics
        p.write_text(json.dumps(data, indent=2))

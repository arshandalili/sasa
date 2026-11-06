from dataclasses import dataclass

import torch
import torch.nn as nn
from sae_lens.saes.sae import SAE, SAEConfig, TrainingSAE, TrainingSAEConfig, TrainStepInput, TrainStepOutput

GATES = ("topk", "batchtopk")
NUCLEAR_TARGETS = ("composition", "decoder")
EIGENVALUE_FLOOR = 1e-6


def topk_group_mask(group_norms: torch.Tensor, k_groups: int) -> torch.Tensor:
    top = group_norms.topk(k_groups, dim=-1, sorted=False).indices
    return torch.zeros_like(group_norms).scatter_(1, top, 1.0)


def batch_topk_group_mask(group_norms: torch.Tensor, k_groups: int) -> torch.Tensor:
    flat = group_norms.reshape(-1)
    n_open = k_groups * group_norms.shape[0]
    if n_open >= flat.numel():
        return torch.ones_like(group_norms)
    mask = torch.zeros_like(flat)
    mask[flat.topk(n_open, sorted=True).indices] = 1.0  # by index: ties at the cut-off must not open extra groups
    return mask.view_as(group_norms)


def _eigh(matrices: torch.Tensor):
    try:
        return torch.linalg.eigh(matrices)
    except torch.linalg.LinAlgError:  # the batched GPU solver can fail on degenerate matrices
        eigenvalues, eigenvectors = torch.linalg.eigh(matrices.cpu().double())
        return eigenvalues.to(matrices.device, matrices.dtype), eigenvectors.to(matrices.device, matrices.dtype)


def decoder_nuclear_norm(decoder_rows: torch.Tensor) -> torch.Tensor:
    gram = decoder_rows @ decoder_rows.transpose(-1, -2)
    return torch.linalg.eigvalsh(gram).clamp(min=0).sqrt().sum(dim=-1)


def _sqrt_and_inverse_sqrt(gram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eigenvalues, eigenvectors = _eigh(gram.double())
    eigenvalues = eigenvalues.clamp(min=EIGENVALUE_FLOOR)
    transposed = eigenvectors.transpose(-1, -2)
    return ((eigenvectors * eigenvalues.sqrt().unsqueeze(-2)) @ transposed,
            (eigenvectors * eigenvalues.rsqrt().unsqueeze(-2)) @ transposed)


def composition_nuclear_norm(encoder_rows: torch.Tensor, decoder_rows: torch.Tensor) -> torch.Tensor:
    # ||D_k E_k||_* from the (K, r, d) rows of E_k and D_k^T. With G_E = E_k E_k^T and G_D = D_k^T D_k, D_k E_k has
    # the singular values of S = G_E^{1/2} G_D^{1/2}, and ||S||_* = tr(C S) with the polar factor C of S. Writing
    # S = G_E^{-1/2} (E_k E_k^T) (D_k^T D_k) G_D^{-1/2} with C, G^{-1/2} and one copy of each factor fixed gives
    # the gradient.
    with torch.no_grad():
        fixed_encoder, fixed_decoder = encoder_rows.detach(), decoder_rows.detach()
        encoder_sqrt, encoder_inv_sqrt = _sqrt_and_inverse_sqrt(fixed_encoder @ fixed_encoder.transpose(-1, -2))
        decoder_sqrt, decoder_inv_sqrt = _sqrt_and_inverse_sqrt(fixed_decoder @ fixed_decoder.transpose(-1, -2))
        s = encoder_sqrt @ decoder_sqrt
        eigenvalues, v = _eigh(s.transpose(-1, -2) @ s)
        inv_abs_s = (v * eigenvalues.clamp(min=EIGENVALUE_FLOOR).rsqrt().unsqueeze(-2)) @ v.transpose(-1, -2)
        dtype = encoder_rows.dtype
        polar = (s @ inv_abs_s).transpose(-1, -2).to(dtype)
        encoder_inv_sqrt, decoder_inv_sqrt = encoder_inv_sqrt.to(dtype), decoder_inv_sqrt.to(dtype)
    encoder_factor = encoder_inv_sqrt @ (fixed_encoder @ encoder_rows.transpose(-1, -2))
    decoder_factor = (decoder_rows @ fixed_decoder.transpose(-1, -2)) @ decoder_inv_sqrt
    return torch.diagonal(polar @ encoder_factor @ decoder_factor, dim1=-2, dim2=-1).sum(-1)


@dataclass
class SASAInferenceConfig(SAEConfig):
    n_groups: int = 2048
    group_rank: int = 6
    k_groups: int = 10
    gate: str = "batchtopk"

    @classmethod
    def architecture(cls) -> str:
        return "batchtopk_manifold_sae"


@dataclass
class TopKSASAInferenceConfig(SASAInferenceConfig):
    gate: str = "topk"

    @classmethod
    def architecture(cls) -> str:
        return "topk_sasa"


@dataclass
class SASAConfig(TrainingSAEConfig):
    n_groups: int = 2048  # K
    group_rank: int = 6  # r
    k_groups: int = 10  # s
    gate: str = "batchtopk"
    k_aux: int = 512
    aux_coefficient: float = 1.0
    nuclear_coefficient: float = 0.5  # lambda_dim
    nuclear_target: str = "composition"  # ||D_k E_k||_*, or "decoder": ||D_k||_*
    encoder_norm_renorm: bool = False

    @classmethod
    def architecture(cls) -> str:
        return "batchtopk_manifold_sae"

    def __post_init__(self):
        super().__post_init__()
        if self.gate not in GATES or self.nuclear_target not in NUCLEAR_TARGETS:
            raise ValueError(f"gate must be one of {GATES} and nuclear_target one of {NUCLEAR_TARGETS}")
        self.d_sae = self.n_groups * self.group_rank

    def get_inference_config_class(self) -> type[SAEConfig]:
        return TopKSASAInferenceConfig if self.gate == "topk" else SASAInferenceConfig


@dataclass
class TopKSASAConfig(SASAConfig):
    gate: str = "topk"

    @classmethod
    def architecture(cls) -> str:
        return "topk_sasa"


class _GroupCodes:
    @property
    def n_groups(self) -> int:
        return self.cfg.n_groups

    @property
    def group_rank(self) -> int:
        return self.cfg.group_rank

    @property
    def k_groups(self) -> int:
        return self.cfg.k_groups

    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.b_enc = nn.Parameter(torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device))

    def gated_codes(self, sae_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_acts = sae_in @ self.W_enc + self.b_enc
        groups = pre_acts.view(-1, self.n_groups, self.group_rank)
        gate = topk_group_mask if self.cfg.gate == "topk" else batch_topk_group_mask
        return pre_acts, groups * gate(groups.norm(dim=-1), self.k_groups).unsqueeze(-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, codes = self.gated_codes(self.process_sae_in(x.reshape(-1, x.shape[-1])))
        return self.hook_sae_acts_post(codes.reshape(*x.shape[:-1], -1))

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        sae_out = self.hook_sae_recons(feature_acts @ self.W_dec + self.b_dec)
        # the layer-norm statistics of encode() are per token, [N, 1]
        return self.run_time_activation_norm_fn_out(sae_out.reshape(-1, sae_out.shape[-1])).view(sae_out.shape)


class SASA(_GroupCodes, TrainingSAE[SASAConfig]):
    cfg: SASAConfig
    b_enc: nn.Parameter

    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_acts, codes = self.gated_codes(self.process_sae_in(x))
        return codes.view(-1, self.cfg.d_sae), pre_acts

    def training_forward_pass(self, step_input: TrainStepInput) -> TrainStepOutput:
        self._normalize_weights()
        x = step_input.sae_in
        pre_acts, codes = self.gated_codes(self.process_sae_in(x))
        feature_acts = codes.view(-1, self.cfg.d_sae)
        reconstruction = self.decode(feature_acts)
        mse_loss = ((reconstruction - x) ** 2).sum(dim=-1).mean()
        aux_losses = self.calculate_aux_loss(step_input, feature_acts, pre_acts, reconstruction)
        losses = {"mse_loss": mse_loss, "nuclear_loss": self._nuclear_loss(), **aux_losses}
        return TrainStepOutput(sae_in=x, sae_out=reconstruction, feature_acts=feature_acts, hidden_pre=pre_acts,
                               loss=sum(losses.values()), losses=losses)

    @torch.no_grad()
    def _normalize_weights(self) -> None:
        self.W_dec.data = self.W_dec.data / self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        if self.cfg.encoder_norm_renorm:
            self.W_enc.data = self.W_enc.data / self.W_enc.norm(dim=0, keepdim=True).clamp(min=1e-8)

    def _nuclear_loss(self) -> torch.Tensor:
        if self.cfg.nuclear_coefficient == 0:
            return self.W_dec.new_zeros(())
        decoder_rows = self.W_dec.view(self.n_groups, self.group_rank, -1)  # rows of D_k^T
        if self.cfg.nuclear_target == "decoder":
            norms = decoder_nuclear_norm(decoder_rows)
        else:
            encoder_rows = self.W_enc.view(-1, self.n_groups, self.group_rank).permute(1, 2, 0)
            norms = composition_nuclear_norm(encoder_rows, decoder_rows)
        return self.cfg.nuclear_coefficient * norms.mean()

    def calculate_aux_loss(self, step_input: TrainStepInput, feature_acts: torch.Tensor, hidden_pre: torch.Tensor,
                           sae_out: torch.Tensor) -> dict[str, torch.Tensor]:
        dead = step_input.dead_neuron_mask
        k_aux = 0 if dead is None else min(self.cfg.k_aux, int(dead.sum()))
        if k_aux == 0:
            return {"aux_loss": sae_out.new_tensor(0.0)}
        residual = (step_input.sae_in - sae_out).detach()
        dead_pre_acts = torch.where(dead.view(1, -1), hidden_pre, torch.full_like(hidden_pre, -torch.inf))
        top = dead_pre_acts.topk(k_aux, dim=-1).indices
        aux_acts = torch.zeros_like(hidden_pre).scatter_(1, top, dead_pre_acts.gather(1, top))
        aux_mse = ((self.decode(aux_acts) - residual) ** 2).sum(dim=-1).mean()
        return {"aux_loss": self.cfg.aux_coefficient * aux_mse}

    def get_coefficients(self) -> dict[str, float]:
        return {"aux": self.cfg.aux_coefficient}


class SASAInference(_GroupCodes, SAE[SASAInferenceConfig]):
    cfg: SASAInferenceConfig
    b_enc: nn.Parameter

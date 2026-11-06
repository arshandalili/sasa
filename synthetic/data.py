import torch


def random_subspaces(dims: list[int], d: int, generator: torch.Generator) -> list[torch.Tensor]:
    return [torch.linalg.qr(torch.randn(d, k, generator=generator))[0] for k in dims]


class SyntheticActivations:
    # each activation sums n_active random features, feature i a Gaussian vector in the subspace spanned by bases[i]
    # times scales[i], plus isotropic noise holding noise_fraction of the signal's mean squared norm
    def __init__(self, bases: list[torch.Tensor], scales: torch.Tensor, n_active: int, noise_fraction: float,
                 device: str):
        self.bases = [basis.to(device) for basis in bases]
        self.dims = [basis.shape[1] for basis in bases]
        self.scales = scales.to(device)
        self.n_active, self.noise_fraction, self.device = n_active, noise_fraction, device
        self.d = self.bases[0].shape[0]
        self.gain = 1.0

    def sample(self, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        order = torch.argsort(torch.rand(n, len(self.bases), device=self.device, generator=generator), dim=1)
        active = torch.zeros(n, len(self.bases), dtype=torch.bool, device=self.device)
        active.scatter_(1, order[:, :self.n_active], True)
        x = torch.zeros(n, self.d, device=self.device)
        for k, basis in enumerate(self.bases):
            rows = active[:, k].nonzero(as_tuple=True)[0]
            if rows.numel():
                coefficients = torch.randn(rows.numel(), self.dims[k], device=self.device, generator=generator)
                x[rows] += self.scales[k] * (coefficients @ basis.T)
        signal_rms = x.pow(2).sum(dim=-1).mean().sqrt()
        noise = torch.randn(x.shape, device=self.device, generator=generator)
        return (x + self.noise_fraction**0.5 * signal_rms / self.d**0.5 * noise) * self.gain, active

    def rescale(self, mean_squared_norm: float, generator: torch.Generator, n: int = 8192) -> None:
        x, _ = self.sample(n, generator)
        self.gain = float((mean_squared_norm / x.pow(2).sum(dim=-1).mean().clamp(min=1e-12)).sqrt())

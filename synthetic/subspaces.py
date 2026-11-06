import numpy as np
import torch


def group_spans(decoder_rows: torch.Tensor, n_groups: int) -> list[torch.Tensor]:
    spans = []
    for block in decoder_rows.view(n_groups, -1, decoder_rows.shape[-1]):
        _, singular_values, right_vectors = torch.linalg.svd(block, full_matrices=False)
        spans.append(right_vectors[: int((singular_values > singular_values.max() * 1e-6).sum().clamp(min=1))])
    return spans


def captured_fraction(span: torch.Tensor, basis: torch.Tensor) -> float:
    return float((span @ basis).pow(2).sum() / basis.shape[1])


def effective_rank(singular_values: torch.Tensor) -> float:
    squared = singular_values.pow(2).cpu().numpy().astype(np.float64)
    squared = squared[squared > 0]
    return float(np.searchsorted(np.cumsum(np.sort(squared)[::-1]), 0.9 * squared.sum()) + 1)

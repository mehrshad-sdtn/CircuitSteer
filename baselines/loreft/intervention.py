"""The LoReFT intervention module.

Reference: Wu, Arora, Wang, Geiger, Jurafsky, Manning and Potts,
"ReFT: Representation Finetuning for Language Models", NeurIPS 2024
(arXiv:2404.03592); official implementation `pyreft`
(stanfordnlp/pyreft, pyreft/interventions.py).

    LoReFT(h) = h + R^T ( W h + b - R h )

R is [d, r] with orthonormal columns, kept orthonormal by torch's
orthogonal parametrisation rather than by a penalty, exactly as the
reference does. W and b are a learned linear map into the same r
dimensions. The edit therefore only ever moves h inside the r-dimensional
subspace R spans: outside it, R^T(...) contributes nothing.

Kept free of sae_lens and transformer_lens so it can be tested without a
model.
"""
from __future__ import annotations

import torch


class LoReFT(torch.nn.Module):
    """Low-rank subspace edit with an orthonormal rotation."""

    def __init__(self, d_model: int, rank: int = 4, dtype=torch.float32):
        super().__init__()
        if rank <= 0 or rank > d_model:
            raise ValueError("rank must be in [1, d_model]")
        self.d_model = d_model
        self.rank = rank

        rotation = torch.empty(d_model, rank, dtype=dtype)
        torch.nn.init.orthogonal_(rotation)
        self.rotation = torch.nn.Parameter(rotation)
        self.source = torch.nn.Linear(d_model, rank, dtype=dtype)

    @property
    def basis(self) -> torch.Tensor:
        """R as [d, r] with orthonormal columns, re-derived every call.

        The reference wraps the parameter in
        `torch.nn.utils.parametrizations.orthogonal`. On torch 2.7 that
        map returns an all-zero matrix as soon as an optimiser step moves
        the underlying parameter off the manifold - the parameter stays
        healthy, the exposed weight silently becomes zero, and the
        intervention turns into a no-op that still trains and still
        generates. A QR factorisation imposes the same constraint,
        differentiably, and actually holds; `test_rotation_stays_
        orthonormal_after_optimisation` is what pins the difference.
        """
        return torch.linalg.qr(self.rotation)[0]

    def delta(self, hidden: torch.Tensor) -> torch.Tensor:
        """The edit R^T(Wh + b - Rh), before any scaling."""
        original = hidden.dtype
        hidden = hidden.to(self.source.weight.dtype)
        basis = self.basis                          # [d, r]
        projected = hidden @ basis                  # R h
        target = self.source(hidden)                # W h + b
        return ((target - projected) @ basis.T).to(original)

    def forward(self, hidden: torch.Tensor, scale: float = 1.0):
        return hidden + scale * self.delta(hidden)

    def trainable_parameters(self):
        return list(self.parameters())

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

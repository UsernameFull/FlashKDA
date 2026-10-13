"""Layout helpers for the AscendC KDA path.

Every K1/K2 kernel addresses the sequence in chunk-major order: the leading
index ``c`` runs over ``(batch, head, chunk)`` as
``c = (b * H + h) * NT + chunk``, matching the ``beta`` packing in
``kda_ascendc_v1.api``.  ``pack_tokens`` converts the public
``[B, T, H, D]`` layout into that form so callers can allocate matching
output buffers.
"""
from __future__ import annotations

import torch

CHUNK = 16


def pack_tokens(x: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """``[B, T, H, D]`` -> ``[B*H*(T//chunk), chunk, D]`` (contiguous)."""
    if x.dim() != 4:
        raise ValueError("pack_tokens expects a 4-D [B, T, H, D] tensor")
    b, t, h, d = x.shape
    if t % chunk:
        raise ValueError(f"T={t} is not a multiple of chunk={chunk}")
    nt = t // chunk
    return x.reshape(b, nt, chunk, h, d).permute(0, 3, 1, 2, 4).contiguous().reshape(b * h * nt, chunk, d)


def unpack_tokens(x: torch.Tensor, b: int, t: int, h: int, chunk: int = CHUNK) -> torch.Tensor:
    """Inverse of :func:`pack_tokens`: ``[c, chunk, D]`` -> ``[B, T, H, D]``."""
    d = x.shape[-1]
    nt = t // chunk
    return x.reshape(b, h, nt, chunk, d).permute(0, 2, 3, 1, 4).contiguous().reshape(b, t, h, d)

"""Pure-PyTorch reference for the FlashKDA forward formulation."""

from __future__ import annotations

import torch


LOG2E = 1.4426950408889634


def _forward_substitute(lower: torch.Tensor) -> torch.Tensor:
    """Return ``(I + lower)^-1`` for a strictly lower-triangular matrix."""
    size = lower.shape[-1]
    inverse = torch.eye(size, dtype=lower.dtype, device=lower.device)
    for row in range(1, size):
        for col in range(row):
            inverse[..., row, :] -= lower[..., row, col] * inverse[..., col, :]
    return inverse


def _chunk_cumsum(gate: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch, steps, heads, dim = gate.shape
    chunks = steps // chunk_size
    gated = gate.view(batch, chunks, chunk_size, heads, dim)
    return gated.cumsum(dim=2).view(batch, steps, heads, dim)


def flashkda_reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    chunk_size: int = 16,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the chunked KDA forward in FlashKDA's state-first layout.

    ``q/k/v`` and ``g`` are ``[B, T, H, D]``, while ``beta`` is pre-activation
    ``[B, T, H]``. The returned state is fp32 ``[B, H, V, K]``.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have identical [B, T, H, D] shapes")
    if g.shape != q.shape or g.dtype != torch.float32:
        raise ValueError("g must be FP32 [B, T, H, D]")
    batch, steps, heads, dim = q.shape
    if steps % chunk_size:
        raise ValueError(f"T={steps} must be divisible by chunk_size={chunk_size}")
    if beta.shape != (batch, steps, heads) or beta.dtype != torch.float32:
        raise ValueError("beta must be FP32 [B, T, H]")
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        raise ValueError("A_log must be FP32 [H]")
    if dt_bias.shape != (heads, dim) or dt_bias.dtype != torch.float32:
        raise ValueError("dt_bias must be FP32 [H, D]")
    if initial_state is not None and initial_state.shape != (batch, heads, dim, dim):
        raise ValueError("initial_state must be [B, H, D, D]")

    solve_dtype = torch.float32 if q.device.type == "npu" else torch.float64
    q_normalized = (q.float() / torch.sqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
    k_normalized = (k.float() / torch.sqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
    gate_input = g + dt_bias.view(1, 1, heads, dim)
    gate = lower_bound * torch.sigmoid(torch.exp(A_log).view(1, 1, heads, 1) * gate_input) * LOG2E
    gate_cumsum = _chunk_cumsum(gate, chunk_size)
    beta_sigmoid = torch.sigmoid(beta)

    state = (torch.zeros(batch, heads, dim, dim, dtype=torch.float32, device=q.device)
             if initial_state is None else initial_state.float().clone())
    output = torch.empty_like(q, dtype=torch.float32)

    for batch_index in range(batch):
        for head_index in range(heads):
            state_bh = state[batch_index, head_index]
            for chunk_index in range(steps // chunk_size):
                start = chunk_index * chunk_size
                stop = start + chunk_size
                q_chunk = q_normalized[batch_index, start:stop, head_index].float()
                k_chunk = k_normalized[batch_index, start:stop, head_index].float()
                v_chunk = v[batch_index, start:stop, head_index].float()
                g_chunk = gate_cumsum[batch_index, start:stop, head_index]
                beta_chunk = beta_sigmoid[batch_index, start:stop, head_index].float()
                g_middle = g_chunk[chunk_size // 2]
                g_recentred = g_chunk - g_middle

                q_decayed = q_chunk * torch.exp2(g_recentred)
                k_recentred = k_chunk * torch.exp2(g_recentred)
                k_inverse = k_chunk * torch.exp2(-g_recentred)
                aqk = torch.tril(q_decayed @ k_inverse.T * scale)
                akk = torch.tril(k_recentred @ k_inverse.T * beta_chunk[:, None], diagonal=-1)
                inverse = _forward_substitute(akk.to(solve_dtype)).float()

                k_weighted = k_chunk * beta_chunk[:, None] * torch.exp2(g_chunk)
                v_weighted = v_chunk * beta_chunk[:, None]
                w = inverse @ k_weighted
                u = inverse @ v_weighted

                g_last = g_chunk[-1]
                v_new = u - w @ state_bh.T
                output[batch_index, start:stop, head_index] = (
                    (q_chunk * torch.exp2(g_chunk)) @ state_bh.T * scale + aqk @ v_new
                )
                k_update = k_chunk * torch.exp2(g_last[None, :] - g_chunk)
                state_bh = state_bh * torch.exp2(g_last)[None, :] + v_new.T @ k_update
            state[batch_index, head_index] = state_bh

    return output, state

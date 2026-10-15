"""Pure-PyTorch reference for the FlashKDA forward formulation."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


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

STATE_IO = [(True, True), (True, False), (False, True), (False, False)]
STATE_IDS = ["in+out", "in_only", "out_only", "no_state"]


def make_inputs(batch, steps, heads, device="cpu", dim=128, gate_dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(42)
    shape = (batch, steps, heads, dim)
    q, k = [F.normalize(torch.randn(shape, generator=generator), dim=-1).bfloat16()
            for _ in range(2)]
    v = torch.randn(shape, generator=generator).bfloat16()
    # Mild decay supplies stable ordinary cases alongside the extreme-gate tests.
    g = (torch.randn(shape, generator=generator) * 0.5 - 2).to(gate_dtype)
    beta = torch.randn(shape[:-1], generator=generator).bfloat16()
    a_log = torch.rand(heads, generator=generator)
    bias = torch.rand(heads, dim, generator=generator) * 0.25
    return tuple(x.to(device) for x in (q, k, v, g, beta, a_log, bias))


def make_state(sequences, heads, dim, dtype, device):
    generator = torch.Generator().manual_seed(123)
    return (torch.randn(sequences, heads, dim, dim, generator=generator) * 0.1).to(
        device=device, dtype=dtype)


def recurrent_reference(inputs, initial_state=None, seq_lens=None, lower_bound=-5.0):
    """Token recurrence in public [sequence, head, value, key] state layout."""
    q, k, v, g, beta, a_log, bias = [x.detach().cpu().double() for x in inputs]
    batch, steps, heads, dim = q.shape
    q, k = [F.normalize(x, dim=-1, eps=1e-6) for x in (q, k)]
    decay = torch.exp(lower_bound * torch.sigmoid(
        a_log.exp()[None, None, :, None] * (g + bias)))
    beta = beta.sigmoid()
    lengths = [steps] * batch if seq_lens is None else seq_lens
    state = (torch.zeros(len(lengths), heads, dim, dim, dtype=torch.float64)
             if initial_state is None else initial_state.detach().cpu().double().clone())
    output = torch.empty_like(v).reshape(-1, heads, dim)
    q, k, v, decay = [x.reshape(-1, heads, dim) for x in (q, k, v, decay)]
    beta = beta.reshape(-1, heads)
    start = 0
    for sequence, length in enumerate(lengths):
        s = state[sequence]
        for token in range(start, start + length):
            s = s * decay[token].unsqueeze(-2)
            residual = (v[token] - (s * k[token].unsqueeze(-2)).sum(-1)) * beta[token, :, None]
            s = s + residual.unsqueeze(-1) * k[token].unsqueeze(-2)
            output[token] = (s * q[token].unsqueeze(-2)).sum(-1) / math.sqrt(dim)
        state[sequence] = s
        start += length
    return output.reshape(batch, steps, heads, dim).float(), state.float()


def assert_close(actual, expected):
    torch.testing.assert_close(actual.detach().cpu().float(), expected.float(),
                               rtol=5e-3, atol=5e-3)


def assert_relative_rms(actual, expected):
    """CUDA gate-sweep criterion: relative RMS error below 0.5%."""
    actual, expected = actual.detach().cpu().float(), expected.float()
    assert torch.isfinite(actual).all(), "nonfinite output/state"
    ratio = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    assert ratio.item() < 5e-3, f"relative RMS error {ratio.item():.6g} >= 0.005"

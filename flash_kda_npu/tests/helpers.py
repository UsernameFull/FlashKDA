"""Deterministic inputs and an independent CPU fp64 recurrent oracle."""
import math

import torch
from torch.nn import functional as F

STATE_IO = [(True, True), (True, False), (False, True), (False, False)]
STATE_IDS = ["in+out", "in_only", "out_only", "no_state"]


def make_inputs(batch, steps, heads, device="cpu", dim=128, gate_dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(42)
    shape = (batch, steps, heads, dim)
    q, k = [F.normalize(torch.randn(shape, generator=generator), dim=-1).bfloat16()
            for _ in range(2)]
    v = torch.randn(shape, generator=generator).bfloat16()
    # Mild decay avoids overflow in the fallback's uncentred chunk exponentials.
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

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.nn import functional as F

CHUNK = 64
LOWER_BOUND_DEFAULT = -5.0
ATOL = RTOL = 5e-3


def is_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except (ImportError, AttributeError):
        return False


def _normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2.0, dim=-1, eps=1e-6).to(x.dtype)


def _activate_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    gate = g.float() + dt_bias.unsqueeze(0)
    gate_scale = torch.special.exp2(A_log.float() * math.log2(math.e))
    gate = gate_scale.view(1, -1, 1) * gate
    gate = torch.sigmoid(gate)
    return (lower_bound * math.log2(math.e)) * gate


def _state_dtype(
    initial_state: Optional[torch.Tensor],
    final_state: Optional[torch.Tensor],
) -> torch.dtype:
    if (initial_state is not None and initial_state.dtype == torch.float32) or (
        final_state is not None and final_state.dtype == torch.float32
    ):
        return torch.float32
    return torch.bfloat16


def _pad_chunk(value: torch.Tensor, length: int, chunk: int) -> torch.Tensor:
    if length == chunk:
        return value
    shape = list(value.shape)
    shape[0] = chunk - length
    return torch.cat([value, value.new_zeros(shape)], dim=0)


def _torch_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: Optional[torch.Tensor],
    final_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
) -> None:
    """Pure PyTorch implementation used for unsupported shapes and hosts."""

    batch_size, total_length, heads, dim = q.shape
    q = _normalize_last_dim(q.contiguous())
    k = _normalize_last_dim(k.contiguous())
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    beta = beta.contiguous()
    gate = _activate_gate(g.contiguous(), A_log, dt_bias, lower_bound)

    q = q.reshape(-1, heads, dim)
    k = k.reshape(-1, heads, dim)
    v = v.reshape(-1, heads, dim)
    beta = beta.reshape(-1, heads)
    gate = gate.reshape(-1, heads, dim)
    output = torch.empty_like(v)

    if cu_seqlens is None:
        cu_seqlens_cpu = torch.arange(0, batch_size * total_length + 1, total_length)
        sequences = batch_size
    else:
        cu_seqlens_cpu = cu_seqlens.detach().cpu()
        sequences = cu_seqlens_cpu.numel() - 1

    state_work = torch.zeros(
        (sequences, heads, dim, dim),
        dtype=_state_dtype(initial_state, final_state),
        device=q.device,
    )
    if initial_state is not None:
        state_work.copy_(initial_state.reshape(state_work.shape))

    beta = torch.sigmoid(beta.float()).to(torch.bfloat16)
    fallback_chunk = CHUNK

    for sequence in range(sequences):
        start = int(cu_seqlens_cpu[sequence])
        end = int(cu_seqlens_cpu[sequence + 1])
        for chunk_start in range(start, end, fallback_chunk):
            chunk_end = min(chunk_start + fallback_chunk, end)
            length = chunk_end - chunk_start
            qc = q[chunk_start:chunk_end]
            kc = k[chunk_start:chunk_end]
            vc = v[chunk_start:chunk_end]
            gc = gate[chunk_start:chunk_end]
            bc = beta[chunk_start:chunk_end]
            qc = _pad_chunk(qc, length, fallback_chunk)
            kc = _pad_chunk(kc, length, fallback_chunk)
            vc = _pad_chunk(vc, length, fallback_chunk)
            gc = _pad_chunk(gc, length, fallback_chunk)
            bc = _pad_chunk(bc, length, fallback_chunk)

            qd = qc.permute(1, 0, 2)
            kd = kc.permute(1, 0, 2)
            vd = vc.permute(1, 0, 2)
            bc = bc.permute(1, 0).unsqueeze(-1)
            gate_cumsum = gc.permute(1, 0, 2).cumsum(dim=1)
            gate_total = gate_cumsum[:, -1:, :]

            k_decayed = kd * torch.special.exp2(gate_cumsum).to(torch.bfloat16)
            q_decayed = (
                qd
                * torch.special.exp2(gate_cumsum).to(torch.bfloat16)
                * torch.tensor(scale, dtype=torch.bfloat16, device=q.device)
            )
            k_inverse = kd * torch.special.exp2(-gate_cumsum).to(torch.bfloat16)
            k_restored = k_inverse * torch.special.exp2(gate_total).to(torch.bfloat16)

            lower = k_decayed @ k_inverse.transpose(-1, -2)
            attention = q_decayed @ k_inverse.transpose(-1, -2)
            lower = lower * torch.tril(
                torch.ones_like(lower, dtype=torch.float32), diagonal=-1
            ).to(torch.bfloat16) * bc
            attention = torch.tril(attention)
            inverse = torch.inverse(torch.eye(fallback_chunk, device=q.device) + lower)

            state = state_work[sequence].float()
            value = (
                vd.float() - k_decayed.float() @ state.transpose(-1, -2)
            ) * bc.float()
            value = inverse.float() @ value
            result = q_decayed.float() @ state.transpose(-1, -2) + attention.float() @ value
            delta = k_restored.float().transpose(-1, -2) @ value
            decay = torch.special.exp2(gate_total.float()).squeeze(1).unsqueeze(-1)
            state_new = (delta.float() + state.transpose(-1, -2) * decay).to(
                state_work.dtype
            )
            state_work[sequence] = state_new
            output[chunk_start:chunk_end] = result[:, :length].transpose(0, 1)

    out.copy_(output.reshape(out.shape))
    if final_state is not None:
        final_state.copy_(state_work.reshape(final_state.shape))


def _supports_native_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    final_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
) -> bool:
    if q.device.type != "npu":
        return False
    if q.ndim != 4 or q.shape[-1] != 128 or v.shape[-1] != 128:
        return False
    if cu_seqlens is not None or q.shape[1] % CHUNK:
        return False
    if q.shape != k.shape or q.shape != v.shape or q.shape != g.shape:
        return False
    if beta.shape != q.shape[:3]:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False
    if v.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        return False
    if not out.is_contiguous():
        return False
    if g.dtype not in (torch.bfloat16, torch.float32):
        return False
    if A_log.numel() != q.shape[2]:
        return False
    if tuple(dt_bias.shape) != (q.shape[2], q.shape[-1]):
        return False
    if initial_state is not None:
        expected = (q.shape[0], q.shape[2], q.shape[-1], q.shape[-1])
        if tuple(initial_state.shape) != expected:
            return False
    if final_state is not None:
        expected = (q.shape[0], q.shape[2], q.shape[-1], q.shape[-1])
        if tuple(final_state.shape) != expected:
            return False
        if final_state.dtype != torch.float32:
            return False
    return True


def _run_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: Optional[torch.Tensor],
    final_state: Optional[torch.Tensor],
) -> None:
    from .ascendc import kda_bt16_fwd_ascendc

    state_output = (
        final_state.contiguous() if final_state is not None else None
    )
    output, state = kda_bt16_fwd_ascendc(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous().to(torch.float32) if g.dtype != torch.float32 else g.contiguous(),
        beta.contiguous().to(torch.float32),
        scale=float(scale),
        initial_state=None if initial_state is None else initial_state.contiguous().to(torch.float32),
        output_final_state=final_state is not None,
        A_log=A_log.contiguous().to(torch.float32),
        bias=dt_bias.contiguous().to(torch.float32),
        lower_bound=float(lower_bound),
        out=out,
        final_state=state_output,
    )
    out.copy_(output)
    if (final_state is not None and state is not None
            and state.data_ptr() != final_state.data_ptr()):
        final_state.copy_(state)


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = LOWER_BOUND_DEFAULT,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> None:
    """Run FlashKDA forward on Ascend NPU, with a PyTorch fallback."""
    if is_available() and _supports_native_npu(
        q, k, v, g, beta, out, A_log, dt_bias,
        initial_state, final_state, cu_seqlens,
    ):
        try:
            _run_ascendc(
                q, k, v, g, beta, float(scale), out, A_log, dt_bias,
                float(lower_bound), initial_state, final_state,
            )
            return
        except ImportError:
            pass

    _torch_fallback(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        out,
        A_log,
        dt_bias,
        float(lower_bound),
        initial_state,
        final_state,
        cu_seqlens,
    )

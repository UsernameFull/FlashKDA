from __future__ import annotations

from typing import Optional

import torch

CHUNK = 64
LOWER_BOUND_DEFAULT = -5.0


def is_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except (ImportError, AttributeError):
        return False


def _check_public_inputs(q, k, v, g, beta, out, A_log, dt_bias,
                         initial_state, final_state, cu_seqlens):
    """Match the CUDA public tensor contract before any conversion or launch."""
    tensors = dict(q=q, k=k, v=v, g=g, beta=beta, out=out,
                   A_log=A_log, dt_bias=dt_bias)
    if initial_state is not None:
        tensors["initial_state"] = initial_state
    if final_state is not None:
        tensors["final_state"] = final_state
    if cu_seqlens is not None:
        tensors["cu_seqlens"] = cu_seqlens
    for name, tensor in tensors.items():
        if tensor.device.type != "npu":
            raise RuntimeError(f"{name} must be an NPU tensor")
        if tensor.device != q.device:
            raise RuntimeError("all inputs must be on q's NPU device")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    for name in ("q", "k", "v", "g", "beta", "out"):
        if tensors[name].dtype != torch.bfloat16:
            raise RuntimeError(f"{name} must be bfloat16")
    for name in ("A_log", "dt_bias"):
        if tensors[name].dtype != torch.float32:
            raise RuntimeError(f"{name} must be float32")
    for name in ("initial_state", "final_state"):
        if name in tensors and tensors[name].dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError(f"{name} must be bfloat16 or float32")
    if (initial_state is not None and final_state is not None
            and initial_state.dtype != final_state.dtype):
        raise RuntimeError("initial_state and final_state must have the same dtype")

    if q.ndim != 4:
        raise RuntimeError("q must be [B, T, H, D]")
    batch, steps, heads, dim = q.shape
    for name in ("k", "v", "g", "out"):
        if tensors[name].shape != q.shape:
            raise RuntimeError(f"{name} must match q shape")
    if beta.shape != (batch, steps, heads):
        raise RuntimeError("beta must be [B, T, H]")
    if A_log.shape != (heads,):
        raise RuntimeError("A_log must be [H]")
    if dt_bias.shape != (heads, dim):
        raise RuntimeError("dt_bias must be [H, D]")
    if dim != 128:
        raise RuntimeError("currently only supports D == 128")
    # These are current AscendC capability limits, not reasons to switch backend.
    if cu_seqlens is not None:
        raise NotImplementedError("NPU AscendC does not support cu_seqlens yet")
    if batch < 1 or heads < 1 or steps < CHUNK or steps % CHUNK:
        raise RuntimeError("NPU AscendC requires B,H > 0, T >= 64 and T % 64 == 0")
    for name in ("initial_state", "final_state"):
        if name in tensors and tensors[name].shape != (batch, heads, dim, dim):
            raise RuntimeError(f"{name} must be [B, H, D, D]")


def _run_ascendc(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                 initial_state, final_state):
    from .ascendc import kda_bt16_fwd_ascendc

    # AscendC consumes FP32 gate parameters and initial state internally. Public
    # dtypes have already been validated against CUDA's contract.
    kda_bt16_fwd_ascendc(
        q, k, v, g.float(), beta.float(), scale=float(scale),
        initial_state=None if initial_state is None else initial_state.float(),
        output_final_state=final_state is not None,
        A_log=A_log, bias=dt_bias, lower_bound=float(lower_bound),
        out=out, final_state=final_state,
    )


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
    """Run native AscendC with CUDA's public dtypes and no automatic fallback.

    Writes ``out`` and optional ``final_state`` in place and returns ``None``.
    Currently requires fixed-length NPU inputs with T >= 64 and T % 64 == 0.
    Input, import, compilation and launch failures propagate to the caller.
    """
    _check_public_inputs(q, k, v, g, beta, out, A_log, dt_bias,
                         initial_state, final_state, cu_seqlens)
    _run_ascendc(q, k, v, g, beta, float(scale), out, A_log, dt_bias,
                 float(lower_bound), initial_state, final_state)

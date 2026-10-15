from __future__ import annotations

from typing import Optional

import torch

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
    sequences = batch
    if cu_seqlens is not None:
        if batch != 1:
            raise RuntimeError("B must be 1 when cu_seqlens is provided")
        if cu_seqlens.dtype != torch.int64:
            raise RuntimeError("cu_seqlens must be int64")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise RuntimeError("cu_seqlens must be 1D with at least 2 elements")
        sequences = cu_seqlens.numel() - 1
    if batch < 1 or heads < 1 or (steps < 1 and cu_seqlens is None):
        raise RuntimeError("NPU AscendC requires B,H > 0 and T > 0 for fixed-length inputs")
    for name in ("initial_state", "final_state"):
        if name in tensors and tensors[name].shape != (sequences, heads, dim, dim):
            raise RuntimeError(f"{name} must be [N, H, D, D]")


def _sequence_offsets(cu_seqlens, steps):
    # Only sequence metadata visits the host. All tensor packing and KDA math
    # stay on NPU. Validate all boundaries before writing caller buffers.
    offsets = cu_seqlens.detach().cpu().tolist()
    if offsets[0] != 0 or offsets[-1] != steps:
        raise RuntimeError("cu_seqlens must start at 0 and end at T")
    if any(a > b for a, b in zip(offsets, offsets[1:])):
        raise RuntimeError("cu_seqlens must be nondecreasing")
    return offsets


def _run_ascendc(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                 initial_state, final_state, cu_seqlens):
    from . import ascendc

    # AscendC consumes FP32 gate parameters and initial state internally. Public
    # dtypes have already been validated against CUDA's contract.
    if cu_seqlens is None and q.shape[1] % ascendc.CHUNK == 0:
        ascendc.kda_bt16_fwd_ascendc(
            q, k, v, g.float(), beta.float(), scale=float(scale),
            initial_state=None if initial_state is None else initial_state.float(),
            output_final_state=final_state is not None,
            A_log=A_log, bias=dt_bias, lower_bound=float(lower_bound),
            out=out, final_state=final_state,
        )
        return

    batch, steps, heads, dim = q.shape
    offsets = (list(range(0, (batch + 1) * steps, steps)) if cu_seqlens is None
               else _sequence_offsets(cu_seqlens, steps))
    lengths = [b - a for a, b in zip(offsets, offsets[1:])]
    uniform = lengths[0]
    if uniform > 0 and uniform % ascendc.CHUNK == 0 and all(n == uniform for n in lengths):
        # Packed equal-length sequences are already a contiguous native batch.
        shape = (len(lengths), uniform, heads, dim)
        ascendc.kda_bt16_fwd_ascendc(
            q.view(shape), k.view(shape), v.view(shape), g.view(shape).float(),
            beta.view(shape[:3]).float(), scale=float(scale), out=out.view(shape),
            A_log=A_log, bias=dt_bias, lower_bound=float(lower_bound),
            initial_state=None if initial_state is None else initial_state.float(),
            output_final_state=final_state is not None, final_state=final_state)
        return
    groups = {}
    for sequence, (start, end) in enumerate(zip(offsets, offsets[1:])):
        length = end - start
        padded = (length + ascendc.CHUNK - 1) // ascendc.CHUNK * ascendc.CHUNK
        groups.setdefault(padded, []).append((sequence, start, length))

    sources = [x.view(-1, heads, dim) for x in (q, k, v, g)]
    beta_flat = beta.view(-1, heads)
    out_flat = out.view(-1, heads, dim)
    profiles = []
    # One native batch per rounded length keeps padding below one chunk per
    # nonempty sequence, even for severely imbalanced sequence lengths.
    for padded, sequences in groups.items():
        if padded == 0:
            if final_state is not None:
                for sequence, _, _ in sequences:
                    if initial_state is None:
                        final_state[sequence].zero_()
                    else:
                        final_state[sequence].copy_(initial_state[sequence])
            continue
        count = len(sequences)
        shape = (count, padded, heads, dim)
        packed = [torch.zeros(shape, dtype=x.dtype, device=q.device) for x in sources]
        packed_beta = torch.zeros((count, padded, heads), dtype=beta.dtype, device=q.device)
        for row, (_, start, length) in enumerate(sequences):
            for target, source in zip(packed, sources):
                target[row, :length].copy_(source[start:start + length])
            packed_beta[row, :length].copy_(beta_flat[start:start + length])
        indices = torch.tensor([s[0] for s in sequences], dtype=torch.int64, device=q.device)
        lengths = torch.tensor([s[2] for s in sequences], dtype=torch.int32, device=q.device)
        state_in = (None if initial_state is None else
                    initial_state.index_select(0, indices).float())
        state_out = (None if final_state is None else
                     torch.empty((count, heads, dim, dim), dtype=final_state.dtype, device=q.device))
        result, state = ascendc.kda_bt16_fwd_ascendc(
            *packed[:3], packed[3].float(), packed_beta.float(), scale=float(scale),
            A_log=A_log, bias=dt_bias, lower_bound=float(lower_bound),
            initial_state=state_in, output_final_state=final_state is not None,
            final_state=state_out, valid_lengths=lengths)
        profiles.append(ascendc.get_last_profile())
        for row, (_, start, length) in enumerate(sequences):
            out_flat[start:start + length].copy_(result[row, :length])
        if final_state is not None:
            final_state.index_copy_(0, indices, state)
    ascendc._merge_profiles(profiles, sum(b - a for a, b in zip(offsets, offsets[1:])),
                            sum(padded * len(rows) for padded, rows in groups.items()))


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
    Supports fixed and packed variable-length NPU inputs, including partial chunks.
    Input, import, compilation and launch failures propagate to the caller.
    """
    _check_public_inputs(q, k, v, g, beta, out, A_log, dt_bias,
                         initial_state, final_state, cu_seqlens)
    _run_ascendc(q, k, v, g, beta, float(scale), out, A_log, dt_bias,
                 float(lower_bound), initial_state, final_state, cu_seqlens)

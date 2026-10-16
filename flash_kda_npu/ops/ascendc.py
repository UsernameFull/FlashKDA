"""AscendC (RTC-compiled) KDA v1 forward."""
from __future__ import annotations

import math
import heapq
import struct
import os
import time
from functools import lru_cache, wraps
from pathlib import Path

import torch
import torch_npu

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT.parent
EXT = PACKAGE_ROOT / "lib"
import sys
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))
from flash_kda_npu_launcher import launch_argsarray_engine, rtc_compile, runtime_key

# Chunk size of the whole pipeline; has to match the kernels' KDA_CHUNK
# define (see _defines).  Production builds with 64.
CHUNK = int(os.environ.get("KDA_CHUNK", "64"))
# Only these chunk sizes are validated; anything else is refused in
# kda_bt16_fwd_ascendc before the RTC compile.
SUPPORTED_CHUNKS = frozenset({16, 32, 64})
# Chunks one wide-solve block handles, capped by the 192 KB UB: four live
# [NC, CHUNK, CHUNK] tiles plus the Brcb expansion.  At CHUNK = 64 the
# two-level path shrinks the tiles to [NC, M, M] (M = 32), lifting the cap
# from 4 to 12.
SOLVE_WIDE_NCHUNK = (int(os.environ.get("KDA_SOLVE_WIDE_NCHUNK", "0"))
                     or (32 if CHUNK <= 16 else (16 if CHUNK <= 32 else 12)))
# Two-level solve (R1): the wide kernel runs the row recursion on M = CHUNK/SB
# sub-blocks and the assemble kernel forms the coupling block on the Cube, so
# the vector work drops with SB^2.  SB = 2 needs M >= 16 for the assemble
# kernel's 16-row fractal copies, so only the C = 64 build gets it.
SOLVE_WIDE_SUBB = 2 if CHUNK >= 64 else 1
# Chunks one wide block solves (the wide kernel's tile runs over sub-blocks x
# chunks, see its header): SB sub-blocks of every chunk share one tile.
SOLVE_WIDE_NCH = SOLVE_WIDE_NCHUNK // SOLVE_WIDE_SUBB
# Chunks one assemble block forms the coupling block of.  The KDA_ASM_LOADS=0
# queue form issues a whole pass's loads before its arithmetic, so its queue
# has to be exactly this deep; the other modes have no such constraint.
ASM_NCHUNK = int(os.environ.get("KDA_ASM_NCHUNK", "0")) or 4
# The wide half is AIV-only and the assemble/Cube half AIC-only, so the chunk
# range is cut into this many slices and the two halves run on separate
# streams behind a per-slice event.  0 keeps a single stream.
SOLVE_OVERLAP = int(os.environ.get("KDA_SOLVE_OVERLAP", "24"))
# Slice count the default whole-AIV-wave sizing targets
# (KDA_SOLVE_SLICE_CHUNKS overrides).
SOLVE_SLICE_TARGET = int(os.environ.get("KDA_SOLVE_SLICE_TARGET", "20"))
# Heads per block in the persistent K2 loop, and therefore the KDA_MAXH the
# kernel is compiled with (they must agree: the kernel's head map only walks
# MAXH heads per block, so a smaller define silently drops heads).
# The fp32 state is 32 KB of UB per head and stays resident, so all the
# staging has to fit in the rest of the 192 KB.  The staging is aliased
# across stage 2/4 (fwd_kernel2_persistent.cpp carries the budget), so 4
# heads fit at every chunk size; a UB overrun is a kernel-side aivec error,
# not a host-side allocation failure.  Heads per block also sets the block
# count and hence the wave count: the K2 picker searches whole-wave shapes
# and takes the smallest waves x heads (nh = 1 loses the depth-one
# protocol's engine overlap, so it is only kept for single-wave grids).
# The env can lower or raise MAXH within the cap.
PERSIST_MAXH = max(1, min(4 if CHUNK <= 64 else 2,
                          int(os.environ.get("KDA_PERSIST_LOOP_MAXH", "0"))
                          or (4 if CHUNK <= 64 else 2)))
# Solve-Cube chunks per block: its L0C holds one slot per (pass, chunk),
# so 4 fits up to C = 32 and 2 at C = 64 (the whole 128 KB either way).
WU_NCHUNK = int(os.environ.get("KDA_WU_NCHUNK", "0")) or (4 if CHUNK <= 32 else 2)
SOLVE_FUSE = SOLVE_WIDE_SUBB > 1 and os.environ.get("KDA_SOLVE_FUSE", "1") == "1"
D = 128
BV = 64
NV = 2
_COMPILED: set[tuple[int, int]] = set()
_LAST_PROFILE: dict[str, object] = {}
# Triangular 0/1 masks for the intra-chunk Gram kernel, built once per device.
_GRAM_MASKS: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
# Identity tile read by the K1 solve kernel, built once per (device, width).
_EYE_TILES: dict[tuple, torch.Tensor] = {}
_LAUNCH_COUNTS: dict[str, int] = {}
_LAUNCH_BLOCKS: dict[str, int] = {}


def _tri_eye(device: torch.device, n: int | None = None) -> torch.Tensor:
    """Identity tile for the K1 solve: A_inv starts from it.

    The wide kernel reads it as one dense n x n block, so the two-level path
    wants the M x M sub-block identity, not the CHUNK x CHUNK one.
    """
    n = CHUNK if n is None else n
    key = (device, n)
    eye = _EYE_TILES.get(key)
    if eye is None:
        eye = torch.eye(n, device=device, dtype=torch.float32).contiguous()
        # torch_npu does not order every elementwise op against the raw
        # aclrtLaunchKernel calls, so publish the tile before any kernel reads it.
        torch.npu.synchronize()
        _EYE_TILES[key] = eye
    return eye


# Cached per-device core counts; the env vars override the device property.
_CORE_COUNTS: dict = {}


def _core_count(device, prop: str, env: str) -> int:
    entry = _CORE_COUNTS.setdefault(str(device), {})
    if prop not in entry:
        n = int(os.environ.get(env, "0") or 0)
        if not n:
            try:
                idx = device.index
                if idx is None:
                    idx = torch_npu.npu.current_device()
                props = torch_npu.npu.get_device_properties(idx)
                n = int(getattr(props, prop, 0) or 0)
            except Exception:
                n = 0
        entry[prop] = n
    return entry[prop]


def _aic_core_count(device) -> int:
    return _core_count(device, "cube_core_num", "KDA_AIC_CORES") or 20


def _aiv_core_count(device) -> int:
    n = _core_count(device, "vector_core_num", "KDA_AIV_CORES")
    return n if n else 2 * _aic_core_count(device)


def _solve_padded_chunks(c: int) -> int:
    """The chunk count the solve buffers and launches are padded to.

    The two-level solve launches whole wiring-unit groups (the lcm of the
    three kernels' chunks-per-block), so the padded count has to be a multiple
    of that unit; a short tail would leave a partial group unsolved.  The
    single-level path's NCHUNK (32 / 16) already divides this unit.
    """
    unit = math.lcm(SOLVE_WIDE_NCH, ASM_NCHUNK, WU_NCHUNK)
    return (c + unit - 1) // unit * unit


def _solve_slice_chunks(c_solve: int, unit: int, nch: int, aiv_cores: int,
                        target: int) -> int:
    """Two-level solve slice size: a whole number of AIV waves.

    One wave is `aiv_cores * nch` chunks, lcm'd with the wiring unit so a
    slice boundary is still a whole (wide / assemble / Cube) group.  The size
    is the multiple closest to `c_solve / target`; a smaller remainder slice
    carries the tail.
    """
    wave = math.lcm(unit, aiv_cores * nch)
    k = max(1, int(round(c_solve / float(target) / wave)))
    return k * wave


def _pre_gram_unroll(c: int, cores: int) -> int:
    """Balance chunk work and block startup across whole Cube waves."""
    if c < 256:
        return 1
    best_u, best_cost = 1, None
    for u in range(88, 0, -1):
        grid = -(-c // (2 * u))
        waves = -(-grid // cores)
        steps = waves * u
        # A wave also pays block initialization and pipeline fill/drain.
        # Approximate that fixed cost by one chunk step: minimizing steps
        # alone can replace a few long-lived blocks with thousands of tiny
        # blocks for a negligible arithmetic saving. Keep fewer steps on
        # equal cost so startup savings do not add needless padded work.
        cost = (steps + waves, steps)
        if best_cost is None or cost < best_cost:
            best_cost, best_u = cost, u
    return best_u


@lru_cache(maxsize=128)
def _persistent_varlen_blocks(counts: tuple[int, ...], heads: int, cores: int,
                              max_heads: int, default: int) -> int:
    """Estimate the existing stride map, including its single-head tails.

    A head alone loses the depth-one protocol's engine overlap. Charge its
    unmatched tail twice, then simulate blocks retiring on the available
    Cubes. Fixed/equal-length batches retain the established wave picker.
    Only compare a bounded set of head counts and nearby whole-wave grids.
    """
    if len(set(counts)) <= 1:
        return default
    bh = len(counts) * heads
    lower = (bh + max_heads - 1) // max_heads
    candidates = {default}
    for mh in range(2, max_heads + 1):
        nb = (bh + mh - 1) // mh
        candidates.update((nb, nb // cores * cores, (nb + cores - 1) // cores * cores))

    def cost(nb):
        slots = [0] * cores
        # The stride map changes only where a sequence boundary crosses a
        # block index. Evaluate each constant run once rather than sorting
        # the same head lengths separately for every block.
        boundaries = sorted({0, nb, *(seq * heads % nb
                                    for seq in range(1, len(counts) + 1))})
        for start, end in zip(boundaries, boundaries[1:]):
            work = sorted((counts[i // heads] for i in range(start, bh, nb)),
                          reverse=True)
            tail = work[0] - (work[1] if len(work) > 1 else 0)
            weight = sum(work) + tail
            remaining = end - start
            if weight == 0:
                continue
            # Once all cores are within one job of each other, a complete
            # wave adds the same duration to every core.
            while remaining and max(slots) - slots[0] > weight:
                ready = heapq.heappop(slots)
                heapq.heappush(slots, ready + weight)
                remaining -= 1
            waves, remaining = divmod(remaining, cores)
            if waves:
                slots = [ready + waves * weight for ready in slots]
            for _ in range(remaining):
                ready = heapq.heappop(slots)
                heapq.heappush(slots, ready + weight)
        return max(slots)

    return min((nb for nb in candidates if lower <= nb <= bh),
               key=lambda nb: (cost(nb), nb != default, nb))


def _persistent_fixed_blocks(bh: int, cores: int) -> int:
    best = None
    for mh in range(1, PERSIST_MAXH + 1):
        if mh == 1 and bh > cores:
            continue
        nb = (bh + mh - 1) // mh
        steps = -(-nb // cores) * mh
        if best is None or steps <= best[0]:
            best = (steps, mh)
    return bh if bh <= cores else (bh + best[1] - 1) // best[1]


@lru_cache(maxsize=128)
def _persistent_grouped_plan(counts: tuple[int, ...], heads: int, cores: int,
                             cap: int, default: int, override: int = 0,
                             force: bool = False):
    """Group similar recurrence lengths and launch long blocks first.

    Estimate the existing depth-one protocol's single-head tails and block
    startup on a greedy Cube schedule. Require a useful margin before paying
    for an explicit map. The returned rows append to the sequence metadata,
    so the schedule needs no additional allocation or device upload.
    """
    if len(set(counts)) <= 1 and not force:
        return default, ()
    bh = len(counts) * heads
    sequence_order = sorted(range(len(counts)), key=lambda seq: -counts[seq])
    order = [h for seq in sequence_order for h in range(seq * heads, (seq + 1) * heads)]
    ordered_work = [counts[h // heads] for h in order]
    prefix = [0]
    for work in ordered_work:
        prefix.append(prefix[-1] + work)

    def grouped_score(nb):
        # Prefix sums avoid rebuilding/sorting every candidate's head groups.
        slots = [0] * cores
        width, extra = divmod(bh, nb)
        start = 0
        for b in range(nb):
            size = width + (b < extra)
            end = start + size
            tail = ordered_work[start] - (ordered_work[start + 1] if size > 1 else 0)
            weight = 1 + prefix[end] - prefix[start] + tail
            ready = heapq.heappop(slots)
            heapq.heappush(slots, ready + weight)
            start = end
        return max(slots)

    slots = [0] * cores
    for b in range(default):
        work = sorted((counts[h // heads] for h in range(b, bh, default)), reverse=True)
        weight = 1 + sum(work) + work[0] - (work[1] if len(work) > 1 else 0)
        ready = heapq.heappop(slots)
        heapq.heappush(slots, ready + weight)
    baseline = max(slots)
    candidates = ({override} if override else
                  {default, *((bh + nh - 1) // nh for nh in range(2, cap + 1))})
    scores = {nb: grouped_score(nb) for nb in candidates}
    nb = min(candidates, key=lambda n: (scores[n], n))
    gain = baseline - scores[nb]
    if not force and gain < max(16, baseline / 20):
        return default, ()
    width, extra = divmod(bh, nb)
    start = 0
    rows = []
    for b in range(nb):
        size = width + (b < extra)
        rows.append(tuple(order[start:start + size] + [-1] * (4 - size)))
        start += size
    return nb, tuple(rows)


def _pack_ptrs(xs):
    return [struct.pack("<Q", 0 if x is None else int(x.data_ptr())) for x in xs]


def _i(x: int):
    return struct.pack("<i", int(x))


def _f(x: float):
    return struct.pack("<f", float(x))


def _launch(name: str, blocks: int, args: list[bytes], stream):
    _LAUNCH_COUNTS[name] = _LAUNCH_COUNTS.get(name, 0) + 1
    _LAUNCH_BLOCKS[name] = _LAUNCH_BLOCKS.get(name, 0) + int(blocks)
    launch_argsarray_engine(name, int(blocks), stream, args, 0)


# The two-level solve runs its AIV half and its AIC half on their own streams
# (see _launch_solve_two_level); one pair per device, created once.
_SOLVE_STREAMS: dict = {}


def _solve_streams(device: torch.device):
    key = (device, runtime_key())
    pair = _SOLVE_STREAMS.get(key)
    if pair is None:
        pair = (torch_npu.npu.Stream(device=device),
                torch_npu.npu.Stream(device=device))
        _SOLVE_STREAMS[key] = pair
    return pair


def _launch_solve_two_level(c_solve, c, nch, asm_nchunk, wu_nchunk, overlap, L, eye,
                            a16, lneg, rk, rv, W, U, stream) -> None:
    """R1 two-level solve: wide kernel (AIV), then assemble + Cube (AIC).

    The substitution runs on M = CHUNK/SB sub-blocks (fwd_kernel1_solve_wide.cpp) and
    the coupling block X21 = -X22 L21 X11 is left to the Cube
    (fwd_kernel1_solve_assemble.cpp).

    The wide half occupies the vector cores and the assemble/Cube half the
    Cube cores, so the chunk range is cut into ``overlap`` slices and the two
    halves run on separate streams behind a per-slice event; ``overlap = 0``
    runs everything on the caller's stream.  Slices hold whole ``unit``
    (wide block / assemble / Cube unit) groups, which keeps a padded tail
    from leaking into the next slice.  ``c`` is the real chunk count:
    rk/rv/W/U only carry the c real chunks, so the Cube launch - whose
    kernel clamps its loop to the count it is handed - is clamped to them.
    """
    unit = math.lcm(nch, asm_nchunk, wu_nchunk)
    ngrp = c_solve // unit
    # Slice sizes: whole AIV waves by default, but fat enough to amortise a
    # block's fixed cost - at most ngrp/8 of them, never more than one per
    # group.
    mode = int(os.environ.get("KDA_SOLVE_SLICE_CHUNKS", "0"))
    if overlap <= 0:
        pairs = [(0, ngrp)]
    elif mode < 0:
        # Legacy even split into SOLVE_OVERLAP slices.
        slices = min(overlap, max(1, ngrp // 8))
        pairs = [(i * ngrp // slices, (i + 1) * ngrp // slices)
                 for i in range(slices)]
    else:
        if mode > 0:
            sgrp = max(1, min(ngrp, mode // unit))
        else:
            sgrp = max(1, min(ngrp, _solve_slice_chunks(
                c_solve, unit, nch, _aiv_core_count(a16.device),
                SOLVE_SLICE_TARGET) // unit))
        slices = -(-ngrp // sgrp)
        pairs = [(i * sgrp, min((i + 1) * sgrp, ngrp)) for i in range(slices)]
        if slices > max(1, ngrp // 8):
            slices = max(1, ngrp // 8)
            pairs = [(i * ngrp // slices, (i + 1) * ngrp // slices)
                     for i in range(slices)]
    ovl = len(pairs) >= 2
    cur = torch_npu.npu.current_stream()
    if ovl:
        sa, sb = _solve_streams(a16.device)
        # The wide slices read what the caller's stream produced (L, the aqk
        # Gram's bf16 tile) and the Cube slices write W/U that K2 reads there.
        sa.wait_stream(cur)
    for glo, ghi in pairs:
        lo, n = glo * unit, (ghi - glo) * unit
        wargs = _pack_ptrs([L[lo:], eye, a16[lo:], lneg[lo:]]) + [_i(n)]
        ncube = min(n, c - lo)
        if SOLVE_FUSE:
            asm_name = "kda_solve_fused"
            aargs = _pack_ptrs([a16[lo:], lneg[lo:], rk[lo:], rv[lo:], W[lo:], U[lo:]]) + [_i(ncube)]
        else:
            asm_name = "kda_solve_assemble"
            aargs = _pack_ptrs([a16[lo:], lneg[lo:]]) + [_i(n)]
            cargs = _pack_ptrs([a16[lo:], rk[lo:], rv[lo:], W[lo:], U[lo:]]) + [_i(n)]
        asm_blocks = ((ncube + wu_nchunk - 1) // wu_nchunk if SOLVE_FUSE
                      else (n + asm_nchunk - 1) // asm_nchunk)
        if ovl:
            _launch("kda_solve_wu_wide", n // nch, wargs, sa.npu_stream)
            ev = torch_npu.npu.Event()
            ev.record(sa)
            sb.wait_event(ev)
            _launch(asm_name, asm_blocks,
                    aargs, sb.npu_stream)
            if ncube > 0 and not SOLVE_FUSE:
                _launch("kda_solve_wu_cube_kernel",
                        (ncube + wu_nchunk - 1) // wu_nchunk, cargs, sb.npu_stream)
        else:
            _launch("kda_solve_wu_wide", n // nch, wargs, stream)
            _launch(asm_name, asm_blocks,
                    aargs, stream)
            if ncube > 0 and not SOLVE_FUSE:
                _launch("kda_solve_wu_cube_kernel",
                        (ncube + wu_nchunk - 1) // wu_nchunk, cargs, stream)
    if ovl:
        cur.wait_stream(sb)


def _defines() -> str:
    """Source-side flags for the RTC compile.

    aclrtcCreateProg has no -D option, so the chunk size and the two
    chunk-dependent block sizes ride in front of the kernel source; every
    guarded kernel (KDA_CHUNK, KDA_SOLVE_WIDE_NCHUNK, KDA_WU_NCHUNK) picks
    them up through its own #ifndef default.

    Compiling a KDA_CHUNK kernel without this prefix is not a build error: the
    kernel silently keeps its ``#ifndef KDA_CHUNK 16`` default and answers a
    C=32/64 host call with 16 rows per chunk (fast, plausible, wrong).  Every
    compile in this package therefore goes through ``_rtc``.
    """
    return ("#define KDA_CHUNK %d\n#define KDA_SOLVE_WIDE_NCHUNK %d\n"
            "#define KDA_SOLVE_WIDE_SUBB %d\n#define KDA_ASM_NCHUNK %d\n"
            "#define KDA_WU_NCHUNK %d\n#define KDA_MAXH %d\n"
            % (CHUNK, SOLVE_WIDE_NCHUNK, SOLVE_WIDE_SUBB, ASM_NCHUNK,
               WU_NCHUNK, PERSIST_MAXH))


def compile_config() -> dict[str, int]:
    """The geometry the kernels were compiled with, as a plain dict; an RTC
    kernel carries no trace of its defines, so profiles carry them next to
    the timings."""
    return {
        "KDA_CHUNK": CHUNK,
        "KDA_MAXH": PERSIST_MAXH,
        "KDA_SOLVE_WIDE_NCHUNK": SOLVE_WIDE_NCHUNK,
        "KDA_SOLVE_WIDE_SUBB": SOLVE_WIDE_SUBB,
        "KDA_ASM_NCHUNK": WU_NCHUNK if SOLVE_FUSE else ASM_NCHUNK,
        "KDA_WU_NCHUNK": WU_NCHUNK,
        "KDA_SOLVE_OVERLAP": SOLVE_OVERLAP,
        "KDA_SOLVE_FUSE": int(SOLVE_FUSE),
        "nv": NV,
    }


def _rtc(rel: str, name: str, sizes: list[int]) -> None:
    """Compile ``csrc/ascendc/<rel>`` as ``name``, defines prefix included.

    The source is read as ``utf-8-sig``: a UTF-8 BOM in front of the first
    ``#include`` is a hard compile error, not a warning, and is invisible in
    an editor.
    """
    prefix = _defines()
    if name == "kda_solve_fused":
        prefix += ("#define KDA_SOLVE_FUSED 1\n#undef KDA_ASM_NCHUNK\n"
                   "#define KDA_ASM_NCHUNK KDA_WU_NCHUNK\n")
        sources = ("fwd_kernel1_solve_assemble.cpp", "fwd_kernel1_solve_cube.cpp",
                   "fwd_kernel1_solve_fused.cpp")
        source = "\n".join((PACKAGE_ROOT / "csrc/ascendc" / path).read_text(
            encoding="utf-8-sig") for path in sources)
    else:
        source = (ROOT / rel).read_text(encoding="utf-8-sig")
    rtc_compile(prefix + source, name, sizes)


_SOURCES: list[tuple[str, str, list[int]]] = [
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_pre_gram.cpp", "kda_pre_gram_mix", [8] * 23 + [4] * 8),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_wide.cpp", "kda_solve_wu_wide", [8] * 4 + [4]),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_assemble.cpp", "kda_solve_assemble", [8] * 2 + [4]),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_cube.cpp", "kda_solve_wu_cube_kernel", [8] * 5 + [4]),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel2_persistent.cpp", "kda_k2_persistent_loop", [8] * 17 + [4] * 8),
]
if SOLVE_FUSE:
    _SOURCES = [entry for entry in _SOURCES if entry[1] not in (
        "kda_solve_assemble", "kda_solve_wu_cube_kernel")]
    _SOURCES.append(("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_fused.cpp",
                     "kda_solve_fused", [8] * 6 + [4]))


def _compile_all() -> None:
    key = runtime_key()
    if key in _COMPILED:
        return
    for rel, name, sizes in _SOURCES:
        _rtc(rel, name, sizes)
    _COMPILED.add(key)


def _tri_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    masks = _GRAM_MASKS.get(device)
    if masks is None:
        idx = torch.arange(CHUNK, device=device)
        masks = ((idx[None, :] <= idx[:, None]).to(torch.float32).contiguous(),
                 (idx[None, :] < idx[:, None]).to(torch.float32).contiguous())
        # torch_npu does not order every elementwise op against the raw
        # aclrtLaunchKernel calls made by the launcher, so make sure the mask
        # tensors are on device before any kernel can read them.
        torch.npu.synchronize()
        _GRAM_MASKS[device] = masks
    return masks


def _check_inputs(q, k, v, g, beta, A_log, bias, initial_state, state_sequences=None):
    if q.device.type != "npu":
        raise ValueError("AscendC v1 requires NPU tensors")
    if any(x is not None and x.device != q.device
           for x in (k, v, g, beta, A_log, bias, initial_state)):
        raise ValueError("all inputs must be on q's NPU device")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("q/k/v must be BF16")
    if q.ndim != 4 or tuple(q.shape) != tuple(k.shape) or tuple(q.shape) != tuple(v.shape):
        raise ValueError("q/k/v must have identical [B,T,H,128] shape")
    b, t, h, d = q.shape
    if b < 1 or h < 1 or t < 1 or d != D:
        raise ValueError("support is B,H,T > 0, D=128")
    if g.shape != q.shape or g.dtype != torch.float32:
        raise ValueError("g must be FP32 [B,T,H,128]")
    if beta.shape != (b, t, h) or beta.dtype != torch.float32:
        raise ValueError("beta must be FP32 [B,T,H]")
    if A_log.shape != (h,) or A_log.dtype != torch.float32:
        raise ValueError("A_log must be FP32 [H]")
    if bias is not None and (bias.shape != (h, D) or bias.dtype != torch.float32):
        raise ValueError("bias must be FP32 [H,128]")
    sequences = b if state_sequences is None else state_sequences
    if initial_state is not None and (initial_state.shape != (sequences, h, D, D) or initial_state.dtype != torch.float32):
        raise ValueError("initial_state must be FP32 [N,H,128,128]")
    return b, t, h


def get_last_profile() -> dict[str, object]:
    profile = dict(_LAST_PROFILE)
    profile["launch_counts"] = dict(_LAUNCH_COUNTS)
    profile["launch_blocks"] = dict(_LAUNCH_BLOCKS)
    profile["launch_total"] = sum(_LAUNCH_COUNTS.values())
    # The compile geometry rides with every profile: an RTC kernel carries no
    # trace of the defines it was built with, and a timing without them cannot
    # be compared against another one.
    profile["compile"] = compile_config()
    return profile


def _merge_profiles(profiles, real_tokens, padded_tokens):
    """Publish all native groups belonging to one packed public call."""
    global _LAST_PROFILE, _LAUNCH_COUNTS, _LAUNCH_BLOCKS
    _LAST_PROFILE = dict(sequence_groups=len(profiles), real_tokens=real_tokens,
                         padded_tokens=padded_tokens, input_packing=bool(profiles))
    _LAUNCH_COUNTS, _LAUNCH_BLOCKS = {}, {}
    for profile in profiles:
        for key, value in profile.items():
            if key.endswith("_ms"):
                _LAST_PROFILE[key] = _LAST_PROFILE.get(key, 0.0) + value
        for key, value in profile["launch_counts"].items():
            _LAUNCH_COUNTS[key] = _LAUNCH_COUNTS.get(key, 0) + value
        for key, value in profile["launch_blocks"].items():
            _LAUNCH_BLOCKS[key] = _LAUNCH_BLOCKS.get(key, 0) + value


def _on_input_device(fn):
    @wraps(fn)
    def run(q, *args, **kwargs):
        if q.device.type == "npu" and q.device.index != torch_npu.npu.current_device():
            with torch_npu.npu.device(q.device):
                return fn(q, *args, **kwargs)
        return fn(q, *args, **kwargs)
    return run


@_on_input_device
def kda_bt16_fwd_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    A_log: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    out: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    valid_lengths: torch.Tensor | None = None,
    sequence_offsets: tuple[int, ...] | None = None,
):
    """Run an opt-in AscendC KDA path; all KDA arithmetic stays on device.

    ``q/k/v`` are BF16 ``[B, T, H, 128]``, ``g`` is FP32 ``[B, T, H, 128]``,
    ``beta`` and ``A_log``/``bias`` are FP32; returns ``(out, final_state)``,
    Any positive fixed sequence length is supported, including partial chunks.
    ``valid_lengths`` is internal packing metadata: counts in [0, T], with
    zero Q/K/V beyond each count. ``sequence_offsets`` selects direct packed
    input/output with compact per-sequence chunks; boundaries have already been
    copied to the host for public validation, never Q/K/V/g themselves.
    """
    # Builds outside the validated set are refused before anything is compiled
    # or launched: a silently wrong answer is the one failure mode this package
    # cannot afford.
    if CHUNK not in SUPPORTED_CHUNKS and os.environ.get(
            "KDA_ALLOW_UNSUPPORTED_CHUNK", "0") != "1":
        raise ValueError(
            "KDA_CHUNK=%d is an unsupported build: the kernels do follow the "
            "chunk size (M = KDA_CHUNK, and stage 3 contracts over K = M), but "
            "only the sizes in api.SUPPORTED_CHUNKS (16, 32, 64) are "
            "supported.  Anything else is unchecked, and "
            "a tile that is wrong for its chunk size is a silently wrong answer "
            "here (or a kernel-side aivec error), never a host-side report - "
            "C=128, for instance, sizes its L0C queue and its L0A allocation to "
            "exactly 128 KB and 64 KB, the whole of both buffers.  Build with "
            "KDA_CHUNK=16, 32 or 64; set KDA_ALLOW_UNSUPPORTED_CHUNK=1 to run it "
            "anyway (debugging only)." % CHUNK)
    global _LAST_PROFILE
    global _LAUNCH_COUNTS, _LAUNCH_BLOCKS
    _LAUNCH_COUNTS = {}
    _LAUNCH_BLOCKS = {}
    _LAST_PROFILE = {}
    profile = os.environ.get("KDA_PROFILE", "0") == "1"
    prof: dict[str, float] = {}
    def mark(name: str):
        if profile:
            torch.npu.synchronize()
            prof[name] = time.perf_counter()
    def finish(name: str, start_name: str):
        if profile:
            torch.npu.synchronize()
            prof[name] = (time.perf_counter() - prof[start_name]) * 1e3
    state_sequences = None
    if sequence_offsets is not None:
        if (len(sequence_offsets) < 2 or sequence_offsets[0] != 0
                or sequence_offsets[-1] != q.shape[1]
                or any(a > b for a, b in zip(sequence_offsets, sequence_offsets[1:]))
                or q.shape[0] != 1 or valid_lengths is not None):
            raise ValueError("invalid packed sequence offsets")
        state_sequences = len(sequence_offsets) - 1
    b, t, h = _check_inputs(q, k, v, g, beta, A_log, bias, initial_state, state_sequences)
    sequences = b if state_sequences is None else state_sequences
    if valid_lengths is not None and (
            valid_lengths.shape != (b,) or valid_lengths.dtype != torch.int32
            or valid_lengths.device != q.device or not valid_lengths.is_contiguous()):
        raise ValueError("valid_lengths must be contiguous INT32 [B] on q's NPU device")
    if out is not None and (
            out.shape != q.shape or out.dtype != torch.bfloat16
            or out.device != q.device or not out.is_contiguous()):
        raise ValueError("out must be contiguous BF16 with q/k/v's shape")
    if final_state is not None:
        if not output_final_state:
            raise ValueError("final_state requires output_final_state=True")
        if (final_state.shape != (sequences, h, D, D)
                or final_state.dtype not in (torch.float32, torch.bfloat16)
                or final_state.device != q.device or not final_state.is_contiguous()):
            raise ValueError(
                "final_state must be contiguous FP32 or BF16 [B,H,128,128]")
    if scale is None:
        scale = D ** -0.5
    nt = (t + CHUNK - 1) // CHUNK
    sequence_meta = None
    head_map = None
    planned_nblk = None
    total_chunks = b * nt
    if sequence_offsets is not None:
        records = []
        total_chunks = 0
        nt = 0
        for start, end in zip(sequence_offsets, sequence_offsets[1:]):
            count = (end - start + CHUNK - 1) // CHUNK
            records.append((start, end - start, total_chunks, count))
            total_chunks += count
            nt = max(nt, count)
        records.append((t, 0, total_chunks, 0))
        mapping_rows = ()
        mode = os.environ.get("KDA_PERSIST_HEAD_MAP", "auto")
        if mode not in ("auto", "stride", "grouped"):
            raise ValueError("KDA_PERSIST_HEAD_MAP must be auto, stride or grouped")
        if mode != "stride":
            counts = tuple(record[3] for record in records[:-1])
            cores = _aic_core_count(q.device)
            default = _persistent_varlen_blocks(
                counts, h, cores, PERSIST_MAXH, _persistent_fixed_blocks(sequences * h, cores))
            want_blocks = int(os.environ.get("KDA_PERSIST_LOOP_BLOCKS", "0"))
            override = want_blocks if 0 < want_blocks <= sequences * h else 0
            if override:
                override = max(override, (sequences * h + PERSIST_MAXH - 1) // PERSIST_MAXH)
                default = override
            planned_nblk, mapping_rows = _persistent_grouped_plan(
                counts, h, cores, PERSIST_MAXH, default, override, mode == "grouped")
        sequence_meta = torch.tensor(records + list(mapping_rows), dtype=torch.int64, device=q.device)
        if mapping_rows:
            head_map = sequence_meta[sequences + 1:]
    bh = sequences * h
    c = total_chunks * h
    tasks = bh * NV
    q, k, v, g, beta, A_log = [x.contiguous() for x in (q, k, v, g, beta, A_log)]
    if bias is not None:
        bias = bias.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    # The stage-1 inputs stay in their public [B, T, H, D] layout: pre_gram
    # gathers them with a byte-strided DataCopyPad (the block/stride form of
    # DataCopy cannot express that gather on this part).
    qk_row_bytes = h * D * 2 - D * 2
    g_row_bytes = h * D * 4 - D * 4
    if sequence_meta is not None:
        beta_pack = beta
    elif t % CHUNK == 0:
        beta_pack = beta.view(b, nt, CHUNK, h).permute(0, 3, 1, 2).contiguous().view(c, CHUNK)
    else:
        # Beta already needs a head-major copy. Pad only this small scalar
        # buffer; pre-Gram reads Q/K/V/g directly, padding their tails in UB.
        beta_pack = torch.empty((b, h, nt * CHUNK), dtype=beta.dtype, device=q.device)
        beta_pack[:, :, :t].copy_(beta.permute(0, 2, 1))
        beta_pack[:, :, t:].zero_()
        beta_pack = beta_pack.view(c, CHUNK)
    stream = torch_npu.npu.current_stream().npu_stream
    _compile_all()

    decay = torch.empty((c, D), dtype=torch.float32, device=q.device)
    rk = torch.empty((c, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    rv = torch.empty_like(rk)
    qg = torch.empty_like(rk); kg = torch.empty_like(rk)
    aqk32 = torch.empty((c, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    aqk16 = torch.empty((c, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    # Whole wiring groups include padded tail chunks. L's tail is zeroed so
    # the wide kernel can safely solve them as identities; W/U and K2 consume
    # only the c real chunks.
    c_solve = _solve_padded_chunks(c)
    L = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    if c_solve != c:
        # The wide kernel is launched in whole SOLVE_WIDE_NCH-chunk blocks, so
        # the chunk count is rounded up and the tail chunks are solved as well -
        # on the strength of L's tail being zero, which is what makes them come
        # out as a harmless identity (k1_solve_wu_wide.cpp).  The Gram only
        # writes the c real chunks, so the tail has to be zeroed here.
        L[c:].zero_()
    mask_s, mask_l = _tri_masks(q.device)
    pre_head = [q, k, v, g, beta_pack, A_log, bias, decay, rk, rv, qg, kg]
    pre_tail = [aqk32, aqk16, L, mask_s, mask_l]
    # One block walks `pre_unroll` consecutive chunks; the grid has to be a
    # whole number of AIC waves, and _pre_gram_unroll picks the walk length
    # from the core count.  Tiny grids (< 256 chunks) stay at 1.
    pre_unroll = (int(os.environ.get("KDA_PRE_UNROLL", "0"))
                  or _pre_gram_unroll(c, _aic_core_count(q.device)))
    mark("pre_gram_start")
    # The two intra-chunk Grams run on the paired Cube: the AIVs publish the
    # bf16 operands and the AIC does the Mmads, hidden behind the AIV's
    # remaining per-chunk work (see csrc/ascendc/fwd_kernel1_pre_gram.cpp).
    gram_ops = torch.empty((3, c, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    # The cross-band k side of the (1, 0) Gram block: a CHUNK = 64 gate
    # needs one decay reference per 32-row band, and the Gram block that
    # spans two bands is served by a second copy of band 0's k rows
    # published under band 1's centre (the kernel note).  A single-band
    # chunk writes no tile here, so CHUNK <= 32 keeps a one-row stand-in.
    gram_x = torch.empty((c, max(1, CHUNK // 2), D), dtype=torch.bfloat16,
                         device=q.device)
    pre_args = _pack_ptrs(pre_head + [gram_ops[0], gram_ops[1], gram_ops[2], gram_x]
                          + pre_tail + [valid_lengths, sequence_meta])
    pre_args += [_i(sequences), _i(t), _i(h), _f(lower_bound), _f(scale), _i(pre_unroll),
                 _i(qk_row_bytes), _i(g_row_bytes)]
    # One MIX block pairs an AIC with two AIV subcores, so it covers
    # 2 * pre_unroll chunks.
    _launch("kda_pre_gram_mix", (c + 2 * pre_unroll - 1) // (2 * pre_unroll),
            pre_args, stream)
    finish("pre_gram_ms", "pre_gram_start")

    a16 = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    W = torch.empty((c_solve, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    # U is consumed by vector subtraction, so retain its FP32 Cube sum.
    U = torch.empty(W.shape, dtype=torch.float32, device=q.device)
    mark("solve_start")
    if SOLVE_WIDE_SUBB > 1:
        # Two-level solve: the wide kernel writes A16's diagonal sub-blocks
        # and the negated coupling triangle, the
        # assemble kernel forms X21 on the Cube, and the Cube solve consumes
        # the assembled tile.  See _launch_solve_two_level.
        sub = CHUNK // SOLVE_WIDE_SUBB
        bf16 = torch.bfloat16
        # The 32-level build exports the single lower-left coupling block.
        lneg = torch.empty((c_solve, sub * sub), dtype=bf16, device=q.device)
        _launch_solve_two_level(c_solve, c, SOLVE_WIDE_NCH, ASM_NCHUNK, WU_NCHUNK,
                                SOLVE_OVERLAP, L,
                                _tri_eye(q.device, CHUNK // SOLVE_WIDE_SUBB),
                                a16, lneg, rk, rv, W, U, stream)
    else:
        # One AIV block solves SOLVE_WIDE_NCHUNK chunks with every vector
        # instruction (see the kernel header); the padded chunks are solved too
        # but land outside the first c.  The two-level operands are null here:
        # the kernel takes them in every build, and a short argument list would
        # leave it reading C out of the args array's tail.
        solve_args = _pack_ptrs([L, _tri_eye(q.device), a16, None])
        solve_args += [_i(c)]
        _launch("kda_solve_wu_wide", c_solve // SOLVE_WIDE_NCHUNK, solve_args, stream)
        # The Cube needs the bf16 A_inv the substitution just wrote, so the two
        # launches stay ordered on the stream.
        _launch("kda_solve_wu_cube_kernel", (c + WU_NCHUNK - 1) // WU_NCHUNK,
                _pack_ptrs([a16, rk, rv, W, U]) +
                [_i(c)],
                stream)
    finish("solve_ms", "solve_start")

    # One device-side chunk loop: each block owns up to MAXH heads, keeps
    # its fp32 state in UB across all chunks and only exports final state
    # when requested. The whole recurrence is a single MIX launch. The block
    # count must keep the head map total: nblk * MAXH >= bh.
    aic_cores = _aic_core_count(q.device)
    default_nblk = _persistent_fixed_blocks(bh, aic_cores)
    if sequence_offsets is not None:
        default_nblk = _persistent_varlen_blocks(
            tuple(record[3] for record in records[:-1]), h, aic_cores,
            PERSIST_MAXH, default_nblk)
    want = int(os.environ.get("KDA_PERSIST_LOOP_BLOCKS", "0"))
    nblk = want if 0 < want <= bh else default_nblk
    # The stride map is total as long as ceil(bh / nblk) fits the array
    # cap *PERSIST_MAXH*, not the heads the picker chose - so a `want`
    # below the picked heads-per-block stays legal.
    nblk = max(nblk, (bh + PERSIST_MAXH - 1) // PERSIST_MAXH)
    if planned_nblk is not None:
        nblk = planned_nblk
    state_output = None
    if output_final_state:
        state_output = (final_state.view(tasks, BV, D) if final_state is not None
                        else torch.empty((tasks, BV, D), dtype=torch.float32, device=q.device))
    s16 = torch.empty((tasks, BV, D), dtype=torch.bfloat16, device=q.device)
    # Keep Cube accumulators in FP32 across the vector handoff: rounding these
    # results to BF16 before residual/output addition amplifies cancellation.
    # The depth-one flag protocol retires each head's
    # previous chunk before its slot is overwritten, so these handoff buffers
    # need one chunk per head rather than a copy for every sequence chunk.
    d1 = torch.empty((tasks, CHUNK, BV), dtype=torch.float32, device=q.device)
    d2 = torch.empty_like(d1)
    d3 = torch.empty_like(d1)
    d4f = torch.empty((bh, D, D), dtype=torch.float32, device=q.device)
    # The loop writes the public [B, T, H, D] layout directly with a
    # strided store, so no host-side permute is needed.
    out_public = out if out is not None else torch.empty(
        (b, t, h, D), dtype=torch.bfloat16, device=q.device)
    vnew_t = torch.empty((tasks, BV, CHUNK), dtype=torch.bfloat16, device=q.device)
    h0 = None if initial_state is None else initial_state.view(bh, D, D)
    # kg goes to the loop in its public [c, CHUNK, D] layout: the AIC loads
    # it into L1 with Nd2Nz and transposes each 16x16 fractal on the way
    # into L0B (LoadDataWithTranspose).
    mark("k2_start")
    _launch("kda_k2_persistent_loop", nblk,
            _pack_ptrs([U, W, qg, aqk16, kg, decay, d1, d2, d3, d4f,
                        out_public, vnew_t, h0, state_output, s16, sequence_meta, head_map]) +
            [_i(bh), _i(nt), _i(NV), _i(nblk), _f(scale), _i(h),
             _i(state_output is not None and state_output.dtype == torch.bfloat16), _i(t)], stream)
    finish("k2_ms", "k2_start")
    final_state = None if not output_final_state else (
        final_state if final_state is not None else
        state_output.view(bh, NV, BV, D).reshape(sequences, h, D, D))
    if profile:
        prof["total_ms"] = sum(v for k, v in prof.items() if k.endswith("_ms"))
        _LAST_PROFILE = prof
    _LAST_PROFILE.update(sequence_groups=1, real_tokens=b * t,
                         padded_tokens=total_chunks * CHUNK, input_packing=False,
                         sequence_count=sequences, packed_direct=sequence_meta is not None,
                         pre_unroll=pre_unroll, persistent_blocks=nblk,
                         persistent_head_map="grouped" if head_map is not None else "stride")
    return out_public, final_state


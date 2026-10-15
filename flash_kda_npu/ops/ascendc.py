"""AscendC (RTC-compiled) KDA v1 forward."""
from __future__ import annotations

import math
import struct
import os
import time
from pathlib import Path

import torch
import torch_npu

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT.parent
EXT = PACKAGE_ROOT / "lib"
import sys
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))
from flash_kda_npu_launcher import launch_argsarray_engine, rtc_compile

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
D = 128
BV = 64
NV = 2
_COMPILED = False
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
    """Chunks per pre_gram block: search all sizes for the fewest wave steps."""
    if c < 256:
        return 1
    best_u, best_steps = 1, None
    for u in range(88, 0, -1):
        grid = -(-c // (2 * u))
        steps = -(-grid // cores) * u
        if best_steps is None or steps < best_steps:
            best_steps, best_u = steps, u
    return best_u


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
    pair = _SOLVE_STREAMS.get(device)
    if pair is None:
        pair = (torch_npu.npu.Stream(device=device),
                torch_npu.npu.Stream(device=device))
        _SOLVE_STREAMS[device] = pair
    return pair


def _launch_solve_two_level(c_solve, c, nch, asm_nchunk, wu_nchunk, overlap, L, eye,
                            a16, xb, lneg, rk, rv, W, U, stream) -> None:
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
        wargs = _pack_ptrs([L[lo:], eye, a16[lo:], xb[lo:], lneg[lo:]]) + [_i(n)]
        asm_name = "kda_solve_assemble"
        aargs = _pack_ptrs([a16[lo:], xb[lo:], lneg[lo:]]) + [_i(n)]
        cargs = _pack_ptrs([a16[lo:], rk[lo:], rv[lo:], W[lo:], U[lo:]]) + \
            [_i(n)]
        ncube = min(n, c - lo)
        if ovl:
            _launch("kda_solve_wu_wide", n // nch, wargs, sa.npu_stream)
            ev = torch_npu.npu.Event()
            ev.record(sa)
            sb.wait_event(ev)
            _launch(asm_name, (n + asm_nchunk - 1) // asm_nchunk,
                    aargs, sb.npu_stream)
            if ncube > 0:
                _launch("kda_solve_wu_cube_kernel",
                        (ncube + wu_nchunk - 1) // wu_nchunk, cargs, sb.npu_stream)
        else:
            _launch("kda_solve_wu_wide", n // nch, wargs, stream)
            _launch(asm_name, (n + asm_nchunk - 1) // asm_nchunk,
                    aargs, stream)
            if ncube > 0:
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
        "KDA_ASM_NCHUNK": ASM_NCHUNK,
        "KDA_WU_NCHUNK": WU_NCHUNK,
        "KDA_SOLVE_OVERLAP": SOLVE_OVERLAP,
        "nv": NV,
    }


def _rtc(rel: str, name: str) -> None:
    """Compile ``csrc/ascendc/<rel>`` as ``name``, defines prefix included.

    The source is read as ``utf-8-sig``: a UTF-8 BOM in front of the first
    ``#include`` is a hard compile error, not a warning, and is invisible in
    an editor.
    """
    rtc_compile(_defines() + (ROOT / rel).read_text(encoding="utf-8-sig"),
                name)


_SOURCES: list[tuple[str, str]] = [
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_pre_gram.cpp", "kda_pre_gram_mix"),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_wide.cpp", "kda_solve_wu_wide"),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_assemble.cpp", "kda_solve_assemble"),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel1_solve_cube.cpp", "kda_solve_wu_cube_kernel"),
    ("flash_kda_npu/csrc/ascendc/fwd_kernel2_persistent.cpp", "kda_k2_persistent_loop"),
]


def _compile_all() -> None:
    global _COMPILED
    if _COMPILED:
        return
    for rel, name in _SOURCES:
        _rtc(rel, name)
    _COMPILED = True


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


def _check_inputs(q, k, v, g, beta, A_log, bias, initial_state):
    if q.device.type != "npu":
        raise ValueError("AscendC v1 requires NPU tensors")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("q/k/v must be BF16")
    if q.ndim != 4 or tuple(q.shape) != tuple(k.shape) or tuple(q.shape) != tuple(v.shape):
        raise ValueError("q/k/v must have identical [B,T,H,128] shape")
    b, t, h, d = q.shape
    if d != D or t < CHUNK or t % CHUNK:
        raise ValueError("support is T>=%d, T%%%d==0, D=128" % (CHUNK, CHUNK))
    if g.shape != q.shape or g.dtype != torch.float32:
        raise ValueError("g must be FP32 [B,T,H,128]")
    if beta.shape != (b, t, h) or beta.dtype != torch.float32:
        raise ValueError("beta must be FP32 [B,T,H]")
    if A_log.shape != (h,) or A_log.dtype != torch.float32:
        raise ValueError("A_log must be FP32 [H]")
    if bias is not None and (bias.shape != (h, D) or bias.dtype != torch.float32):
        raise ValueError("bias must be FP32 [H,128]")
    if initial_state is not None and (initial_state.shape != (b, h, D, D) or initial_state.dtype != torch.float32):
        raise ValueError("initial_state must be FP32 [B,H,128,128]")
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
):
    """Run an opt-in AscendC KDA path; all KDA arithmetic stays on device.

    ``q/k/v`` are BF16 ``[B, T, H, 128]``, ``g`` is FP32 ``[B, T, H, 128]``,
    ``beta`` and ``A_log``/``bias`` are FP32; returns ``(out, final_state)``,
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
    b, t, h = _check_inputs(q, k, v, g, beta, A_log, bias, initial_state)
    if out is not None and (
            out.shape != q.shape or out.dtype != torch.bfloat16
            or out.device != q.device or not out.is_contiguous()):
        raise ValueError("out must be contiguous BF16 with q/k/v's shape")
    if final_state is not None:
        if not output_final_state:
            raise ValueError("final_state requires output_final_state=True")
        if (final_state.shape != (b, h, D, D)
                or final_state.dtype != torch.float32
                or final_state.device != q.device or not final_state.is_contiguous()):
            raise ValueError(
                "final_state must be contiguous FP32 [B,H,128,128]")
    if scale is None:
        scale = D ** -0.5
    nt = t // CHUNK
    bh = b * h
    c = bh * nt
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
    beta_pack = beta.view(b, nt, CHUNK, h).permute(0, 3, 1, 2).contiguous().view(c, CHUNK)
    stream = torch_npu.npu.current_stream().npu_stream
    _compile_all()

    decay = torch.empty((c, D), dtype=torch.float32, device=q.device)
    rk = torch.empty((c, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    rv = torch.empty_like(rk)
    qg = torch.empty_like(rk); kg = torch.empty_like(rk)
    aqk32 = torch.empty((c, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    aqk16 = torch.empty((c, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    # The wide solve kernel rounds the chunk count up to SOLVE_WIDE_NCHUNK and
    # walks whole chunk groups, so L and its two outputs are allocated with the
    # padded length: that keeps every gather and store of the last group inside
    # an allocation, and the values it computes for the padded chunks (never
    # read, and never handed out below) may be whatever uninitialised device
    # memory holds.  The debug dict hands out narrowed views so the shapes stay
    # ``[c, 16, 16]``.
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
                          + pre_tail)
    pre_args += [_i(b), _i(t), _i(h), _f(lower_bound), _f(scale), _i(pre_unroll),
                 _i(qk_row_bytes), _i(g_row_bytes)]
    # One MIX block pairs an AIC with two AIV subcores, so it covers
    # 2 * pre_unroll chunks.
    _launch("kda_pre_gram_mix", (c + 2 * pre_unroll - 1) // (2 * pre_unroll),
            pre_args, stream)
    finish("pre_gram_ms", "pre_gram_start")

    a16 = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    W = torch.empty((c_solve, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    U = torch.empty_like(W)
    mark("solve_start")
    if SOLVE_WIDE_SUBB > 1:
        # Two-level solve: the wide kernel writes the diagonal sub-blocks (Xb
        # next to the parent tile) and the negated coupling triangle, the
        # assemble kernel forms X21 on the Cube, and the Cube solve consumes
        # the assembled tile.  See _launch_solve_two_level.
        sub = CHUNK // SOLVE_WIDE_SUBB
        bf16 = torch.bfloat16
        xb = torch.empty((c_solve, SOLVE_WIDE_SUBB, sub, sub), dtype=bf16, device=q.device)
        # The 32-level build exports the single lower-left coupling block.
        lneg = torch.empty((c_solve, sub * sub), dtype=bf16, device=q.device)
        _launch_solve_two_level(c_solve, c, SOLVE_WIDE_NCH, ASM_NCHUNK, WU_NCHUNK,
                                SOLVE_OVERLAP, L,
                                _tri_eye(q.device, CHUNK // SOLVE_WIDE_SUBB),
                                a16, xb, lneg, rk, rv, W, U, stream)
    else:
        # One AIV block solves SOLVE_WIDE_NCHUNK chunks with every vector
        # instruction (see the kernel header); the padded chunks are solved too
        # but land outside the first c.  The two-level operands are null here:
        # the kernel takes them in every build, and a short argument list would
        # leave it reading C out of the args array's tail.
        solve_args = _pack_ptrs([L, _tri_eye(q.device), a16])
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
    # its fp32 state in UB across all chunks and never writes S32 until
    # the end, so the whole recurrence is a single MIX launch.  The block
    # count must keep the head map total: nblk * MAXH >= bh.
    aic_cores = _aic_core_count(q.device)
    # Pick the heads per block: minimize whole waves x heads, ties to the
    # larger block count; one head per block only while it fits a single
    # wave.
    best = None
    for mh in range(1, PERSIST_MAXH + 1):
        if mh == 1 and bh > aic_cores:
            continue
        nb = (bh + mh - 1) // mh
        steps = -(-nb // aic_cores) * mh
        if best is None or steps <= best[0]:
            best = (steps, mh)
    maxh = best[1]
    want = int(os.environ.get("KDA_PERSIST_LOOP_BLOCKS", "0"))
    nblk = want if 0 < want <= bh else (bh if bh <= aic_cores
                                        else (bh + maxh - 1) // maxh)
    # The stride map is total as long as ceil(bh / nblk) fits the array
    # cap *PERSIST_MAXH*, not the heads the picker chose - so a `want`
    # below the picked heads-per-block stays legal.
    nblk = max(nblk, (bh + PERSIST_MAXH - 1) // PERSIST_MAXH)
    s32 = (final_state.view(bh, NV, BV, D).reshape(tasks, BV, D)
           if final_state is not None else
           torch.empty((tasks, BV, D), dtype=torch.float32, device=q.device))
    s16 = torch.empty((tasks, BV, D), dtype=torch.bfloat16, device=q.device)
    # d1/d2/d3 cross to the vector side as bf16 (the loop's fixpipe rounds
    # them), so they are allocated bf16: an fp32 allocation would be
    # silently half-filled.
    d1 = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
    d2 = torch.empty_like(d1)
    d3 = torch.empty_like(d1)
    d4f = torch.empty((bh, D, D), dtype=torch.float32, device=q.device)
    # The loop writes the public [B, T, H, D] layout directly with a
    # strided store, so no host-side permute is needed.
    out_public = out if out is not None else torch.empty(
        (b, t, h, D), dtype=torch.bfloat16, device=q.device)
    vnew_t = torch.empty((tasks, nt, BV, CHUNK), dtype=torch.bfloat16, device=q.device)
    h0 = None if initial_state is None else initial_state.view(bh, D, D)
    # kg goes to the loop in its public [c, CHUNK, D] layout: the AIC loads
    # it into L1 with Nd2Nz and transposes each 16x16 fractal on the way
    # into L0B (LoadDataWithTranspose).
    mark("k2_start")
    _launch("kda_k2_persistent_loop", nblk,
            _pack_ptrs([U, W, qg, aqk16, kg, decay, d1, d2, d3, d4f,
                        out_public, vnew_t, h0, s32, s16]) +
            [_i(bh), _i(nt), _i(NV), _i(nblk), _f(scale), _i(h)], stream)
    finish("k2_ms", "k2_start")
    final_state = None if not output_final_state else (
        final_state if final_state is not None else
        s32.view(bh, NV, BV, D).reshape(b, h, D, D))
    if profile:
        prof["total_ms"] = sum(v for k, v in prof.items() if k.endswith("_ms"))
        _LAST_PROFILE = prof
    return out_public, final_state


"""AscendC (RTC-compiled) KDA v1 forward."""
from __future__ import annotations

import math
import struct
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch_npu

from .layout import pack_tokens

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "build/S02_clean/torch_extensions/kda_ascendc_v1_launcher"
import sys
if str(EXT) not in sys.path:
    sys.path.insert(0, str(EXT))
from kda_ascendc_v1_launcher import launch_argsarray_engine, rtc_compile

# Chunk size of the whole pipeline.  It has to match the kernels' KDA_CHUNK
# (see _defines below): the AIV/Cube block sizes, the triangular masks and the
# identity tile all follow it.  16 is the default because it is the widest
# supported shape (T % CHUNK == 0, so C=16 accepts four times the sequence
# lengths of C=64); 64 is what every R3 number and the production baseline are
# measured at - there the per-chunk setup, the gate cumsum and the K2 chunk
# hand-offs are amortised over four times the rows, and the two-level solve
# (SOLVE_WIDE_SUBB = 2) only exists at C >= 64.  Long sequences should build
# with KDA_CHUNK=64.
CHUNK = int(os.environ.get("KDA_CHUNK", "64"))
# Chunk sizes the pipeline is actually checked at.  C=16, C=32 and C=64 all
# pass the whole matrix of tests/test_chunk_shape_matrix.py (B = 1/2,
# H = 2/32/48/96, short and long T, with and without an initial state, plus
# determinism); anything else is refused before the RTC compile - see
# _kda_fwd_impl's guard.
SUPPORTED_CHUNKS = frozenset({16, 32, 64})
SOLVE_NCHUNK = 8
# The wide solve keeps four live [NC, CHUNK, CHUNK] tiles plus the Brcb
# expansion, so the 192 KB of UB cap NC at 32/16/4 chunks for CHUNK =
# 16/32/64 (the 16 and 4 settings are 184 KB: NC x [CHUNK, CHUNK] fp32 twice
# over, its bf16 rounding copy and the Brcb expansion).  The sweep at
# [1,8192,96,128] measured the solve stage at 2.70 (NC 8) -> 2.43 (NC 16) ms
# for CHUNK = 32.  The solve-Cube's L0C holds one slot per (pass, chunk) unit
# (2 * NC * CHUNK * 128 fp32), which is why NC drops to 4 at CHUNK = 32 (the
# whole 128 KB of L0C, 2.56 ms) and to 2 at CHUNK = 64 (also 128 KB).
# At CHUNK = 64 the two-level path (SB = 2) shrinks every live tile to
# [NC, M, M] with M = 32, so the wide kernel's own NC cap moves from 4 to 12
# (178 of the 192 KB; NC = 16 would need 234).  8 -> 12 takes the per-chunk
# instruction count 140 -> 93 and is bit-identical everywhere; it was left at
# 8 while the cube was the solve's wall, and flipped when the wide half became
# it: wide stream 1.52 -> 1.40 ms isolated, e2e -0.15 (docs 11.64).
SOLVE_WIDE_NCHUNK = (int(os.environ.get("KDA_SOLVE_WIDE_NCHUNK", "0"))
                     or (32 if CHUNK <= 16 else (16 if CHUNK <= 32 else 12)))
# Two-level solve (R1): the wide kernel runs the row recursion on M = CHUNK/SB
# sub-blocks and the assemble kernel forms the coupling block on the Cube, so
# the substitution's M^3/2 vector lanes per chunk drop with SB^2.  SB = 2
# needs M >= 16 for the 16-row fractal copies of the assemble kernel, so the
# C = 64 build is the one that gets the two-level path.
SOLVE_WIDE_SUBB = 2 if CHUNK >= 64 else 1
# Chunks one wide block solves (the wide kernel's tile runs over sub-blocks x
# chunks, see its header): SB sub-blocks of every chunk share one tile.
SOLVE_WIDE_NCH = SOLVE_WIDE_NCHUNK // SOLVE_WIDE_SUBB
# Chunks one assemble block forms the coupling block of.  The shipped queue
# form (KDA_ASM_LOADS=0) issues the L1 loads of a whole pass before its
# arithmetic, so its queue has to be exactly this deep; the explicit-buffer
# forms (1, 2 - production) have no such constraint, and 6/8/12/16 were
# re-tested on them (tools/probe_solve_assemble_nc.py): they run and are
# bit-identical, but e2e is flat inside the noise floor (NC = 4/6/8 ->
# 10.389/10.390/10.392 ms), so 4 stays - the knob is free, not profitable.
ASM_NCHUNK = int(os.environ.get("KDA_ASM_NCHUNK", "0")) or 4
# The wide part is AIV-only and the assemble/Cube part AIC-only, so the two
# can run at the same time: the chunk range is cut into this many slices, the
# wide slices go on one stream and the assemble+Cube slices on another behind
# a per-slice event.  Measured at [1,8192,96,128]/CHUNK=64: 3.22 -> 2.51 ms
# for the whole solve, bit-identical outputs.  0 keeps a single stream.
# R3: 16 -> 24 slices.  Two same-process interleaved sweeps (MIN of 3) put
# solve_ms at 2.650 (16) / 2.525 (24) / 2.742 (8) / 3.463 (48): the AIV and
# AIC halves overlap better with smaller slices, but only down to the point
# where the per-slice launch pair (and the AIC side's tail) starts to show.
SOLVE_OVERLAP = int(os.environ.get("KDA_SOLVE_OVERLAP", "24"))
# Two-level solve slice sizing (docs 11.59).  The default path sizes slices by
# whole AIV waves targeting this many slices (KDA_SOLVE_SLICE_CHUNKS=0 is
# auto; >0 fixes a chunk size, <0 goes back to the even split into
# SOLVE_OVERLAP slices that shipped through docs 11.57).
SOLVE_SLICE_TARGET = int(os.environ.get("KDA_SOLVE_SLICE_TARGET", "20"))
# Heads per block in the persistent K2 loop, and therefore the KDA_MAXH the
# kernel is compiled with (they must agree: the kernel's head map only walks
# MAXH heads per block, so a smaller define silently drops heads).
# The fp32 state is 32 KB of UB per head and stays resident, so all the
# staging has to fit in the rest of this part's 192 KB.  R3 halved that
# staging by aliasing it across the two phases the loop alternates (stage 2
# and stage 4, each of which opens with a PIPE_ALL drain): 112.5 KB -> 56.5 KB
# at C=64, 46.5 -> 26.5 KB at C=16 (kernels/v1/k2_persistent_loop.cpp carries
# the budget).  That is what makes 4 heads fit at every chunk size -
# 128 + 56.5 = 184.5 KB of 192 at C=64, 128 + 36.5 = 164.5 at C=32 - where
# the old layout capped C=64/32 at 2 (176.5 KB).  A UB overrun is a
# kernel-side aivec error, not a host-side allocation failure, so this is
# arithmetic, not a probe.
# Heads per block is not a free knob: it sets the block count, and the
# block count sets the waves, and the second wave costs a whole second
# pass over the chunks.  The kernel's head map strides by the runtime
# block count ("for (h = blk; h < BH && nh < MAXH; h += NBLK)"), so any
# heads-per-block in 1..PERSIST_MAXH is one nblk away.  The old form
# here chose "one wave per AIC" for the 24-core golden device (96 heads
# / 24 = 4 heads/block); this part has 20 AIC cores, and full-call
# device events read the three candidates at [1,8192,96,128] (interleaved
# MIN, tools/probe_k2_maxh_wave.py, bit-identical outputs incl. the fp32
# state): 24 blocks x 4 heads = 2 waves = 8 head-steps -> k2 4.025 ms;
# 32 x 3 = 2 waves = 6 -> 3.081 ms; 48 x 2 = 3 waves = 6 -> 3.276 ms.
# The picker below therefore searches whole-wave block counts and takes
# the smallest waves x heads, ties to the larger block (3 heads wins
# the 6/6 tie against 2 by retired work and prologue: 3.081 vs 3.276).
# On the 24-core part the same search keeps 4 heads/block, and one head
# per block stays available only while it fits a single wave (the old
# 4.38-vs-4.55 note at [2,4096,8]; past that a second wave of one-head
# blocks loses the engine overlap badly, 17.2 vs 13.8 at [1,8192,32]).
# nh = 1 is not just "one fewer head": the depth-one flag protocol only
# overlaps the two engines when a block has two or more heads in flight, so at
# C=64 the second head is worth k2 6.04 -> 3.91 ms and e2e 12.79 -> 10.74 ms
# at [1,8192,96,128] (MIN of 4), numerics unchanged to the bit (K2 stage gate
# 6.6e-36/9.6e-04/3.3e-08).  The env can lower MAXH, or raise it within the
# ceiling; C=16/32/64 are each checked end to end against the fp32 reference.
PERSIST_MAXH = max(1, min(4 if CHUNK <= 64 else 2,
                          int(os.environ.get("KDA_PERSIST_LOOP_MAXH", "0"))
                          or (4 if CHUNK <= 64 else 2)))
WU_NCHUNK = int(os.environ.get("KDA_WU_NCHUNK", "0")) or (4 if CHUNK <= 32 else 2)
D = 128
BV = 64
NV = 2
# The K2 recurrence has exactly one implementation: the device-side chunk
# loop walks KDA_CHUNK rows at a time and follows the build
# (kernels/v1/k2_persistent_loop.cpp).
PERSISTENT_LOOP = "persistent_loop"
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

    The wide kernel reads it as one dense n x n block (``DataCopyParams(n, n/8,
    0, 0)``), so the two-level path wants the M x M sub-block identity, not the
    CHUNK x CHUNK one: passing the chunk-sized tile there reads the first
    M*M floats of a CHUNK-wide identity packed M-per-row, which is a different
    matrix and puts the solve off by O(1) (measured 1.178e+00 against the
    fp64 inverse, vs 4.882e-04 with the right tile).
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


# The pre_gram launch's grid: one MIX block pairs an AIC with two AIV subcores
# and walks 2 * `pre_unroll` chunks, so the grid is ceil(c / (2u)) blocks and
# the stage's wall is (whole waves) x u block-steps.  The waves term is what a
# block count that is not a multiple of the AIC count costs.  At
# [1,8192,96,128]/C=64 this part has 20 AIC cores, so the shipped 96-block
# grid (u = 64) runs 5 waves of 64 = 320 steps, while the same 6144 steps
# land in 4 waves of 77 = 308 at 80 blocks - measured on device events
# through this very path (tools/probe_pg_grid_sweep.py, full calls, MIN of 4:
# 4.013 -> 3.881 ms, outputs and states bit-identical; 48/24-block arms
# confirm 20 and not 24 cores: 4.816 / 6.336 ms, the 24-core model would read
# 3.2 for both).  The old "4 waves of 24 AICs" reading came from the 24-core
# golden device.  A different part changes the arithmetic, so the picker
# reads cube_core_num (KDA_AIC_CORES overrides it for experiments) and picks
# the whole-wave block count by minimizing waves x u, fewest waves first, with
# u capped at 88 (the largest per-block walk whose price is measured; every
# real shape below the cap is exact).  u is a runtime arg: a tiling choice,
# not a recompile.
_AIC_CORES: dict = {}


def _aic_core_count(device) -> int:
    key = str(device)
    n = _AIC_CORES.get(key)
    if n is None:
        n = int(os.environ.get("KDA_AIC_CORES", "0") or 0)
        if not n:
            try:
                idx = device.index
                if idx is None:
                    idx = torch_npu.npu.current_device()
                n = int(torch_npu.npu.get_device_properties(idx).cube_core_num)
            except Exception:
                n = 20
        _AIC_CORES[key] = n
    return n


_AIV_CORES: dict = {}


def _aiv_core_count(device) -> int:
    """Vector (AIV) core count - the unit the wide solve schedules over.

    The two-level solve's wide half is an AIV-only kernel with one block per
    SOLVE_WIDE_NCH chunks, so its wave length is this many blocks.  The CANN
    property reads 40 on the 20-AIC part (one AIC + two AIV per MIX block);
    the fallbacks are the MIX ratio and the AIC count.
    """
    key = str(device)
    n = _AIV_CORES.get(key)
    if n is None:
        n = int(os.environ.get("KDA_AIV_CORES", "0") or 0)
        if not n:
            try:
                idx = device.index
                if idx is None:
                    idx = torch_npu.npu.current_device()
                props = torch_npu.npu.get_device_properties(idx)
                n = int(getattr(props, "vector_core_num", 0) or 0)
                if not n:
                    n = 2 * int(props.cube_core_num)
            except Exception:
                n = 2 * _aic_core_count(device)
        _AIV_CORES[key] = n
    return n


def _solve_padded_chunks(c: int) -> int:
    """The chunk count the solve buffers and launches are padded to.

    The two-level solve launches whole wiring-unit groups (the lcm of the
    three kernels' chunks-per-block), so the padded count has to be a multiple
    of that unit: rounding to SOLVE_WIDE_NCH alone leaves a trailing group
    unsolved whenever the rounded count is not a multiple of the unit - at
    CHUNK = 64 / NCHUNK = 12 (unit = lcm(6, 4, 2) = 12) that is every c with
    ceil(c / 6) odd, e.g. 4 chunks launched no solve at all and 64 chunks lost
    the last 4 (docs 11.64).  The single-level path's NCHUNK (32 / 16) already
    divides this unit, so C = 16/32 builds pad exactly as before.
    """
    unit = math.lcm(SOLVE_WIDE_NCH, ASM_NCHUNK, WU_NCHUNK)
    return (c + unit - 1) // unit * unit


def _solve_slice_chunks(c_solve: int, unit: int, nch: int, aiv_cores: int,
                        target: int) -> int:
    """Two-level solve slice size: a whole number of AIV waves.

    One wave is `aiv_cores * nch` chunks, lcm'd with the wiring unit so a
    slice boundary is still a whole (wide / assemble / Cube) group.  The size
    is the multiple closest to `c_solve / target`, so the slice count stays
    near `target` whatever c_solve is; a smaller remainder slice carries the
    tail.  Measured in tools/probe_solve_slices.py at [1,8192,96,128]: the
    old even split into 24 gave every slice a 128-block wide grid = 3.2 waves
    of 40 and the sum of wide spans read 2.15 ms; whole-wave slices read 1.85
    (summed assemble 0.60 -> 0.45, Cube flat), e2e paired MIN -0.07~-0.13 ms.
    """
    wave = math.lcm(unit, aiv_cores * nch)
    k = max(1, int(round(c_solve / float(target) / wave)))
    return k * wave


def _pre_gram_unroll(c: int, cores: int) -> int:
    """Chunk pairs per block whose launch is a whole number of AIC waves."""
    if c < 256:
        return 1
    best = None
    for k in range(1, 65):
        u = min(88, max(1, -(-c // (2 * cores * k))))
        grid = -(-c // (2 * u))
        steps = -(-grid // cores) * u
        if best is None or steps < best[0]:
            best = (steps, k, u)
    return best[2]


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


def a16_mode() -> int:
    """A16-store ablation mode for the wide solve (KDA_SOLVE_A16_MODE).

    0 is production: the wide kernel writes both the diagonal sub-blocks of the
    parent A_inv tile and its strict-upper blank.  1 drops the blank, 2 drops
    the whole parent tile - the Cube solve then reads a partially written
    operand, which is the point of the ablation (tools/probe_solve_a16_ablation.py
    prices the store) and never a production setting.  Read per call rather than
    frozen at import so a probe can flip it between two arms of one process; the
    kernel is unchanged either way, this is a runtime argument.
    """
    return int(os.environ.get("KDA_SOLVE_A16_MODE", "0"))


def asm_load_mode() -> int:
    """Load and intermediate shape for the coupling block (kda_solve_assemble).

    0 is the shipped form: one B1 queue per operand, six ND2NZ calls per chunk,
    and P through its own GM tile.  1 batches the same bytes - one call per
    block for the A operand, a chunk's two B bands merged, and pass 1's whole B
    operand in one call.  2 is 1 plus P on chip: pass 0's fixpipe writes P into
    L1 in NZ instead of GM and pass 1 builds L0B from it, which drops the tile's
    50.3 MB round trip (151.0 -> 100.7 MB per call of L1 traffic) and, since
    nothing touches the GM tile any more, the caller stops allocating its
    25.2 MB as well (section 11.40).  Measured in
    tools/probe_solve_assemble_loads.py at [1,8192,96,128], the three arms come
    out 0.902 / 0.692 / 0.512 ms isolated, AIC 2.614 / 2.305 / 2.197 and e2e
    10.624 / 10.445 / 10.408 ms.  All three land byte-identical
    operands (tools/probe_solve_assemble_coalesce.py and
    tools/probe_solve_assemble_l1p.py check P and A16 are equal before timing
    anything), so this is a scheduling knob, not a numeric one.  Read per call
    rather than frozen at import so a probe can flip it between arms of one
    process.  3 is 2 plus the batched pass-1 A16 store (section 11.42) and
    4 is 2 plus the whole-window Xb fill (section 11.50: both passes' Xb
    operands in one ND2NZ call, 2 MTE2 calls per block).  Both were measured
    as "isolated win, stage flat" while the AIV (wide) half was the wall; when
    section 11.59's whole-wave slices cut that half to 1.85 ms, the AIC
    (assemble + Cube) half became the floor and section 11.61 re-measured
    both on the production path in one process, alternating arms, MIN of 24:
    against mode 2's 9.736 / 9.719 wall, mode 3 lands -0.071 / -0.071 and
    mode 4 -0.049 / -0.047.  The two knobs compose - the pass*NC+ch slot
    order keeps the batched store's srcNdStride (MM/256 KB = 4) valid under
    the window fill and P still never leaves the chip - as mode 5
    (window fill + batched store), which measured -0.134 / -0.141 wall and
    -0.128 / -0.143 span, i.e. 9.602 / 9.578 against the same 9.736 / 9.719.
    5 is production; every arm is bit-identical (same file, and
    tools/probe_asm_mode_e2e.py checks out/state per round).
    """
    return int(os.environ.get("KDA_ASM_LOADS", "5"))


def cube_a16_resident() -> int:
    """A16 residency for the solve's Cube kernel (kda_solve_wu_cube_kernel).

    0 re-reads the block's A16 tile at the top of each of the two passes - the
    second read is an L2 hit (docs section 11.35 priced it at 0.070 ms).  1
    keeps the block's NC tiles in L1 for both passes, which is the same bytes
    of L1 as the queue it replaces.  1 is production: measured in
    tools/probe_solve_cube_a16_resident.py at [1,8192,96,128], the cube half
    goes 1.577 -> 1.511 ms and the stage 2.346 -> 2.327, with the pipeline
    bit-identical (a first attempt at this mode spun the block - its slots were
    still AllocTensor'd, which the kernel header records).  Read per call so a
    probe can flip it between two arms of one process; the arithmetic is
    identical either way.
    """
    return int(os.environ.get("KDA_CUBE_A16_RESIDENT", "1"))


def cube_load_mode() -> int:
    """RHS load shape for the solve's Cube kernel (kda_solve_wu_cube_kernel).

    0 keeps the shipped per-band loop: KF single-band Nd2Nz calls per
    chunk-pass, 16 calls of 4 KB per block.  1 merges a chunk's KF [16, D]
    bands into one strided ND run (ndNum = KF, source stride 16 * D elements),
    which lands byte-identical in L1 - the single-band calls already wrote at
    exactly those destination offsets.  1 is production: the on-board account
    (docs section 11.63, archive .../kda_msprof_20261008_cube) read the block
    wall as 4.382 us with MTE2 at 3.271 (74.6%), and the merged form measured
    -0.467 ms of the cube's 1.463 ms isolated replay with W/U bit-identical
    (tools/probe_solve_cube_knobs.py).  Read per call so a probe can flip it
    between two arms of one process; the arithmetic is identical either way.
    """
    return int(os.environ.get("KDA_CUBE_LOADS", "1"))


def pre_raw_mode() -> int:
    """Ablation arms for pre_gram's two raw fp32 Gram tiles (docs 11.48).

    ``kda_pre_gram_mix``'s paired Cube fixpipes the raw Grams into the Aqk32 and
    L slots and the AIV band loop reads both back to mask, scale and round them.
    Bit 0 drops the Aqk32 fixpipe, bit 1 the L one; a dropping arm computes a
    wrong answer by construction (the AIV still reads the slot), so this prices
    the store and nothing else - 0 is the only shipping value.  Read per call
    rather than frozen at import so tools/probe_pre_gram_rawmode.py can flip it
    between arms of one process, exactly like ``debugStores``.
    """
    return int(os.environ.get("KDA_PRE_RAW_MODE", "0"))


def _launch_solve_two_level(c_solve, c, nch, asm_nchunk, wu_nchunk, overlap, L, eye,
                            a32, a16, xb, lneg, pmid, rk, rv, W, U, stream,
                            debug_stores) -> None:
    """R1 two-level solve: wide kernel (AIV), then assemble + Cube (AIC).

    The substitution runs on M = CHUNK/SB sub-blocks (k1_solve_wu_wide.cpp) and
    the coupling block X21 = -X22 L21 X11 is left to the Cube
    (k1_solve_assemble.cpp), which cuts the row recursion's M^3/2 vector lanes
    per chunk by SB^2 and leaves the Cube the part it is actually good at.

    The wide half occupies the vector cores and the assemble/Cube half the Cube
    cores, so the chunk range is cut into ``overlap`` slices and the two halves
    run on their own streams behind a per-slice event - at [1,8192,96,128] and
    CHUNK = 64 that takes the stage from 3.22 to 2.51 ms with bit-identical
    outputs.  Slice sizes are whole AIV waves by default (docs 11.59,
    ``_solve_slice_chunks``); KDA_SOLVE_SLICE_CHUNKS overrides the size (>0)
    or restores the legacy even split (<0).  Slices hold whole ``unit``
    (wide block / assemble / Cube unit)
    groups, which is what keeps a padded tail from leaking into the next slice;
    ``overlap = 0`` runs everything on the caller's stream.  ``c`` is the real
    chunk count: the wide/assemble halves own c-sized-per-chunk buffers padded
    up to ``c_solve``, but rk/rv/W/U only carry the c real chunks, so the Cube
    launch - whose kernel clamps its loop to the count it is handed - has to be
    clamped to them (a padded count sends the tail block's stores past W/U).
    ``pmid`` (the assemble's P tile, 25.2 MB per call at [1,8192,96,128] - the
    50.3 MB in section 11.39 is its GM round trip, write plus read) is passed as
    None when the load mode keeps P on chip: the caller does not allocate it,
    ``_pack_ptrs`` writes a null pointer, and the kernel never dereferences it
    (docs section 11.40).  Modes 0/1 own the tile and need it non-null, so this
    is a property of the mode, not of the call site.
    """
    unit = math.lcm(nch, asm_nchunk, wu_nchunk)
    ngrp = c_solve // unit
    # Slice sizes.  Whole AIV waves by default (docs 11.59): an even split
    # into SOLVE_OVERLAP slices gives every slice a wide grid that is not a
    # multiple of the vector-core count, so each one pays a partial last wave.
    # Slices have to stay fat enough to keep a block's fixed cost amortised:
    # at most ngrp/8 of them, and never more than one per group.
    mode = int(os.environ.get("KDA_SOLVE_SLICE_CHUNKS", "0"))
    if overlap <= 0:
        pairs = [(0, ngrp)]
    elif mode < 0:
        # Legacy: the even split into SOLVE_OVERLAP slices (docs 11.38-11.57).
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
        wargs = _pack_ptrs([L[lo:], eye, a32[lo:], a16[lo:], xb[lo:],
                            lneg[lo:]]) + [_i(n), _i(a16_mode()), _i(1 if debug_stores else 0)]
        asm_name = "kda_solve_assemble"
        aargs = _pack_ptrs([a16[lo:], xb[lo:], lneg[lo:],
                            None if pmid is None else pmid[lo:]]) + \
            [_i(n), _i(asm_load_mode())]
        cargs = _pack_ptrs([a16[lo:], rk[lo:], rv[lo:], W[lo:], U[lo:]]) + \
            [_i(n), _i(cube_a16_resident()), _i(cube_load_mode())]
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
    compile in this package therefore goes through ``_rtc``, and
    ``tests/test_rtc_compile_config.py`` fails the build if a call site
    compiles a kernel without the prefix.
    """
    return ("#define KDA_CHUNK %d\n#define KDA_SOLVE_WIDE_NCHUNK %d\n"
            "#define KDA_SOLVE_WIDE_SUBB %d\n#define KDA_ASM_NCHUNK %d\n"
            "#define KDA_WU_NCHUNK %d\n#define KDA_MAXH %d\n"
            % (CHUNK, SOLVE_WIDE_NCHUNK, SOLVE_WIDE_SUBB, ASM_NCHUNK,
               WU_NCHUNK, PERSIST_MAXH))


def compile_config() -> dict[str, int]:
    """The geometry the kernels were compiled with, as a plain dict.

    ``get_last_profile()`` carries this next to the timings: the RTC compile
    leaves no trace in the .o of how big KDA_CHUNK / KDA_MAXH / the solve block
    sizes were, so a measurement without them cannot be compared against
    another one.
    """
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
    """Compile ``kernels/v1/<rel>`` as ``name``, defines prefix included.

    Every RTC compile of this package has to come through here (see
    ``_defines``); the C=16-only kernels of the S12-S15 experiments ignore the
    prefix, but they get it too so that the invariant is mechanical.

    The source is read as ``utf-8-sig``: a UTF-8 BOM in front of the first
    ``#include`` is not a warning here but a hard compile error
    (``unexpected character <U+FEFF>``, followed by every type name in the
    file turning unknown), and it is invisible in an editor - a BOM once made
    one of the kernels uncompilable with no trace of why.  Reading it
    away in the one place every compile goes through keeps that from coming
    back through any of the kernels.
    """
    rtc_compile(_defines() + (ROOT / rel).read_text(encoding="utf-8-sig"),
                name, "")


_SOURCES: list[tuple[str, str]] = [
    ("flash_kda_npu/csrc/kernels/v1/k1_pre_gram_mix.cpp", "kda_pre_gram_mix"),
    ("flash_kda_npu/csrc/kernels/v1/k1_solve_wu_wide.cpp", "kda_solve_wu_wide"),
    ("flash_kda_npu/csrc/kernels/v1/k1_solve_assemble.cpp", "kda_solve_assemble"),
    ("flash_kda_npu/csrc/kernels/v1/k1_solve_wu_cube.cpp", "kda_solve_wu_cube_kernel"),
    ("flash_kda_npu/csrc/kernels/v1/k2_persistent_loop.cpp", "kda_k2_persistent_loop"),
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
    return_intermediates: bool = False,
    k2_mode: str | None = None,
    out: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
):
    """Run an opt-in AscendC KDA path; all KDA arithmetic stays on device.

    ``q/k/v`` are BF16 ``[B, T, H, 128]``, ``g`` is FP32 ``[B, T, H, 128]``,
    ``beta`` and ``A_log``/``bias`` are FP32; returns ``(out, final_state)``,
    plus the stage tensors when ``return_intermediates`` is set.

    ``k2_mode`` names the K2 (state recurrence) implementation and defaults to
    ``"persistent_loop"``: one MIX launch runs the whole chunk loop and keeps
    the fp32 state in UB; it is the implementation this release ships.
    """
    if k2_mode is None:
        k2_mode = PERSISTENT_LOOP
    if k2_mode != PERSISTENT_LOOP:
        raise ValueError("unsupported k2_mode %r (only %r)"
                         % (k2_mode, PERSISTENT_LOOP))
    return _kda_fwd_impl(
        q, k, v, g, beta, scale=scale, initial_state=initial_state,
        output_final_state=output_final_state, A_log=A_log, bias=bias,
        lower_bound=lower_bound, return_intermediates=return_intermediates,
        k2_mode=k2_mode, out=out, final_state=final_state)


def _kda_fwd_impl(
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
    return_intermediates: bool = False,
    k2_mode: str = PERSISTENT_LOOP,
    out: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
):
    """The one implementation both entry points call; ``k2_mode`` is validated.
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
    if k2_mode != PERSISTENT_LOOP:
        raise ValueError("unsupported k2_mode %r (only %r)"
                         % (k2_mode, PERSISTENT_LOOP))
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
    # The four stage-1 inputs are no longer packed: pre_gram reads them out of
    # the public [B, T, H, D] layout with a byte-strided DataCopyPad, which
    # removes a 1.07 GB copy round trip per call (~0.6 ms at [1,8192,32]).  The
    # block/stride form of DataCopy cannot express that gather on this part.
    qk_row_bytes = h * D * 2 - D * 2
    g_row_bytes = h * D * 4 - D * 4
    beta_pack = beta.view(b, nt, CHUNK, h).permute(0, 3, 1, 2).contiguous().view(c, CHUNK)
    stream = torch_npu.npu.current_stream().npu_stream
    _compile_all()

    qn = torch.empty((c, CHUNK, D), dtype=torch.bfloat16, device=q.device)
    kn = torch.empty_like(qn)
    gate = torch.empty((c, CHUNK, D), dtype=torch.float32, device=q.device)
    gc = torch.empty_like(gate)
    beta_out = torch.empty((c, CHUNK), dtype=torch.float32, device=q.device)
    decay = torch.empty((c, D), dtype=torch.float32, device=q.device)
    rk = torch.empty_like(qn); rv = torch.empty_like(qn)
    qg = torch.empty_like(qn); kg = torch.empty_like(qn)
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
    # Stages 1+2 run as one AIV block per chunk ("k1_pre_gram.cpp"): the fused
    # kernel consumes Qn/Kn/Gc out of UB instead of round-tripping 32 KB per
    # chunk through GM.  It addresses every token in packed ``[c, CHUNK, D]``
    # order, so the public [B, T, H, D] tensors have to be packed first, and
    # it only writes Qn/Kn/Gate/Gc when those pointers are non-null (the
    # ``return_intermediates`` debug path).
    keep = return_intermediates
    # Three of the buffers a call allocates exist for the
    # ``return_intermediates`` views and are read by nothing on the device:
    # Aqk32's masked fp32 copy (the band loop reads the raw one; Aqk16 is
    # rounded from registers), A32 (the wide solve's fp32 A_inv - the Cube
    # solve reads A16) and BetaOut.  Writing them costs 405 MB per call at
    # C=64 and 0.19 ms at R3's measured store rate, so the two kernels take a
    # ``debugStores`` flag and production passes 0.  It is a flag and not a
    # nulled pointer because the raw Aqk32 slot stays live either way: the
    # pre_gram band loop reads it back before the masked copy exists.
    # Measured interleaved at [1,8192,96,128]/C=64 (tools/probe_dead_store.py,
    # 3 rounds x 800 ms): 10.765 -> 10.601 ms for the three stores together, so
    # the flag defaults to off and KDA_DEBUG_STORES=1 is the way back to the
    # old behaviour (docs 11.29).
    keep_debug = keep or os.environ.get("KDA_DEBUG_STORES", "0") == "1"
    pre_head = [q, k, v, g, beta_pack, A_log, bias,
                qn if keep else None, kn if keep else None,
                gate if keep else None, gc if keep else None,
                beta_out, decay, rk, rv, qg, kg]
    pre_tail = [aqk32, aqk16, L, mask_s, mask_l]
    # One block walks `pre_unroll` consecutive chunks.  The stage is issue-bound
    # and pays a fixed per-block setup cost, so unrolling is worth 8-12% from a
    # few hundred chunks up (measured 2.679 -> 2.342 ms at [1,8192,32]); tiny
    # grids (fewer than 256 chunks) stay at 1 because the loop wrapper itself
    # costs a few percent there.
    # R3: the block count is what costs - every extra wave re-pays the block
    # prologue - so the grid has to be a whole number of AIC waves, and the
    # device-event span (not the host-synced stage times the old sweeps used,
    # which read 0.7 ms low) is what prices it.  The R3 sweeps picked 96
    # blocks / u = 64 on the assumption of 4 waves of 24 AICs; this part has
    # 20 AIC cores, so that grid is 5 waves = 320 steps.  Full-call device
    # events at [1,8192,96,128] read 4.013 (96 x 64) -> 3.881 ms (80 x 77,
    # 4 waves = 308 steps), bit-identical, and _pre_gram_unroll finds the
    # 80 x 77 shape from the core count alone (tools/probe_pg_grid_sweep.py).
    pre_unroll = (int(os.environ.get("KDA_PRE_UNROLL", "0"))
                  or _pre_gram_unroll(c, _aic_core_count(q.device)))
    mark("pre_gram_start")
    # The two intra-chunk Grams run on the paired Cube: the AIVs publish
    # ga/gk1/gb in bf16 (12 KB per chunk) and the AIC does both 16x16x128
    # Mmads per step.  The Gram half is 1.05 ms of the vector-only block's
    # 2.21 ms and the vector pipe is the bottleneck of the whole stage, so
    # the Cube's work is hidden behind the AIV's remaining ~3.3 us per chunk
    # (see kernels/v1/k1_pre_gram_mix.cpp).
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
                 _i(qk_row_bytes), _i(g_row_bytes), _i(pre_raw_mode()),
                 _i(1 if keep_debug else 0)]
    # One MIX block pairs an AIC with two AIV subcores, so it covers
    # 2 * pre_unroll chunks.
    _launch("kda_pre_gram_mix", (c + 2 * pre_unroll - 1) // (2 * pre_unroll),
            pre_args, stream)
    finish("pre_gram_ms", "pre_gram_start")

    a32 = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.float32, device=q.device)
    a16 = torch.empty((c_solve, CHUNK, CHUNK), dtype=torch.bfloat16, device=q.device)
    W = torch.empty_like(qn)
    U = torch.empty_like(qn)
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
        # Load mode 2 keeps P on chip (section 11.39), so its GM tile is
        # neither written nor read: the 25.17 MB allocation at [1,8192,96,128]
        # (24 MiB = 12288 chunks of [32, 32] bf16 - the 50.3 MB section 11.39
        # quotes is the round trip that tile used to carry) is dead memory
        # under mode 2 and only modes 0/1 need it.  The mode is read per call,
        # so a probe that flips the knob back gets the tile allocated again on
        # the next call.
        pmid = (None if asm_load_mode() >= 2 else
                torch.empty((c_solve, sub, sub), dtype=bf16, device=q.device))
        _launch_solve_two_level(c_solve, c, SOLVE_WIDE_NCH, ASM_NCHUNK, WU_NCHUNK,
                                SOLVE_OVERLAP, L,
                                _tri_eye(q.device, CHUNK // SOLVE_WIDE_SUBB),
                                a32, a16,
                                xb, lneg, pmid, rk, rv, W, U, stream,
                                1 if keep_debug else 0)
    else:
        # One AIV block solves SOLVE_WIDE_NCHUNK chunks with every vector
        # instruction (see the kernel header); the padded chunks are solved too
        # but land outside the first c.  The two-level operands are null here:
        # the kernel takes them in every build, and a short argument list would
        # leave it reading C out of the args array's tail.
        solve_args = _pack_ptrs([L, _tri_eye(q.device), a32, a16, None, None])
        solve_args += [_i(c), _i(a16_mode()), _i(1 if keep_debug else 0)]
        _launch("kda_solve_wu_wide", c_solve // SOLVE_WIDE_NCHUNK, solve_args, stream)
        # The Cube needs the bf16 A_inv the substitution just wrote, so the two
        # launches stay ordered on the stream.
        _launch("kda_solve_wu_cube_kernel", (c + WU_NCHUNK - 1) // WU_NCHUNK,
                _pack_ptrs([a16, rk, rv, W, U]) +
                [_i(c), _i(cube_a16_resident()), _i(cube_load_mode())],
                stream)
    finish("solve_ms", "solve_start")

    if k2_mode == PERSISTENT_LOOP:
        # One device-side chunk loop.  Each block owns up to MAXH=4 heads (both
        # AIV subcores run the same flag sequence and split the value dim), keeps
        # its fp32 state in UB across all chunks and never writes S32 until the
        # end, so the whole recurrence is a single MIX launch.  The block count
        # must keep the head map total: nblk * MAXH >= bh.
        # One head per block is ~4% faster than two while the heads still fit in
        # the AIC count (measured at [2,4096,8]: 4.38 vs 4.55 ms); past that,
        # two heads per block keep every block resident instead of queueing a
        # second wave (32 blocks at [1,8192,32] cost 17.2 vs 13.8 ms).
        aic_cores = _aic_core_count(q.device)
        # Hunt over the heads-per-block candidates the kernel can carry (the
        # runtime head map strides by nblk, PERSIST_MAXH is only the array
        # cap): minimize whole waves x heads; ties go to the larger block.
        # One head per block only while it fits a single wave (see above).
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
        # below the picked heads-per-block stays legal (that is how the
        # KDA_PERSIST_LOOP_BLOCKS probe arm reproduces the 24 x 4 shape).
        nblk = max(nblk, (bh + PERSIST_MAXH - 1) // PERSIST_MAXH)
        s32 = (final_state.view(bh, NV, BV, D).reshape(tasks, BV, D)
               if final_state is not None else
               torch.empty((tasks, BV, D), dtype=torch.float32, device=q.device))
        # The fused loop publishes the bf16 state into one slot it overwrites
        # every chunk.
        s16 = torch.empty((tasks, BV, D), dtype=torch.bfloat16, device=q.device)
        # d1/d2/d3 cross to the vector side as bf16 (the loop's fixpipe rounds
        # them), so they are allocated bf16 here as well: the loop indexes the
        # buffers in bf16 elements and an fp32 allocation silently half-filled
        # them and returned garbage through ``return_intermediates``.
        d1 = torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
        d2 = torch.empty_like(d1)
        d3 = torch.empty_like(d1)
        d4f = torch.empty((bh, D, D), dtype=torch.float32, device=q.device)
        # The loop writes the public [B, T, H, D] layout directly: one 128 B
        # run per (chunk row) with a NH*D-element stride between rows, which
        # removes the 186 us `permute(0,3,4,1,2,5).contiguous()` (67 MB in +
        # 67 MB out) the api used to run over the task layout.  The strided
        # store costs 0.04 ms of device time, the same-store check is
        # bit-exact against the old layout + host permute.
        out_public = out if out is not None else torch.empty(
            (b, t, h, D), dtype=torch.bfloat16, device=q.device)
        # `vnew` is the row-major twin of `vnew_t`, and the kernel reads only
        # `vnew_t` - the row-major copy is written solely for
        # ``return_intermediates`` (the pQn/pKn pattern).  Production passes a
        # null pointer and skips both the store (0.095 ms of K2 at
        # [1,8192,96,128], measured interleaved) and the 201 MB allocation.
        vnew = (torch.empty((tasks, nt, CHUNK, BV), dtype=torch.bfloat16, device=q.device)
                if return_intermediates else None)
        vnew_t = torch.empty((tasks, nt, BV, CHUNK), dtype=torch.bfloat16, device=q.device)
        h0 = None if initial_state is None else initial_state.view(bh, D, D)
        # kg goes to the loop in its public [c, CHUNK, D] layout: the AIC loads
        # it into L1 with Nd2Nz and transposes each 16x16 fractal on the way
        # into L0B (LoadDataWithTranspose).
        mark("k2_start")
        _launch("kda_k2_persistent_loop", nblk,
                _pack_ptrs([U, W, qg, aqk16, kg, decay, d1, d2, d3, d4f,
                            out_public, vnew, vnew_t, h0, s32, s16]) +
                [_i(bh), _i(nt), _i(NV), _i(nblk), _f(scale), _i(h)], stream)
        finish("k2_ms", "k2_start")
        final_state = None if not output_final_state else (
            final_state if final_state is not None else
            s32.view(bh, NV, BV, D).reshape(b, h, D, D))
        if profile:
            prof["total_ms"] = sum(v for k, v in prof.items() if k.endswith("_ms"))
            _LAST_PROFILE = prof
        if not return_intermediates:
            return out_public, final_state
        debug = {"Qn": qn, "Kn": kn, "Gate": gate, "Gc": gc, "Beta": beta_out,
                 "Decay": decay, "Rk": rk, "Rv": rv, "Qg": qg, "Kg": kg,
                 "Aqk32": aqk32, "Aqk": aqk16, "L": L[:c], "A32": a32[:c], "A16": a16[:c],
                 "W": W, "U": U, "d1": d1, "d2": d2, "Vnew": vnew, "VnewT": vnew_t,
                 "d3": d3, "d4": d4f, "state_s32": s32}
        return out_public, final_state, debug


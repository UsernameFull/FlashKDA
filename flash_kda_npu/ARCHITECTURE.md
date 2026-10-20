# NPU code navigation

## Entry points and ownership

| File | Responsibility |
| --- | --- |
| `ops/kda_fwd.py` | Validate the public tensor contract, choose graph/eager execution, dispatch fixed or packed inputs. |
| `ops/auto_graph.py` | Optional thread-local LRU for automatic warmup, capture and replay. |
| `ops/graphs.py` | Explicit capture API and ownership of captured tensors/private scratch. |
| `ops/ascendc.py` | Backend selection (CATLASS by default), RTC compilation, scheduling, workspace allocation and native stage launches. |
| `csrc/catlass/` | Independent CATLASS block kernels and custom KDA vector blocks. |
| `csrc/third_party/catlass/` | Pinned upstream headers and license, also shipped in wheels. |
| `csrc/flash_kda.cpp` | RTC/launch bridge and native argument ABI checks. |
| `csrc/ascendc/fwd_kernel*.cpp` | Separate legacy backend for numerical/performance comparisons. |

The normal path is `fwd -> validation -> _run_ascendc -> native orchestration`.
With `KDA_AUTO_GRAPH=1`, validated fixed inputs first visit `try_auto_graph`.
A cache miss executes eager warmups; a later call captures and replays; hits
replay directly. Explicit `capture_fwd` validates inputs and calls
`_run_ascendc` directly to avoid reentering automatic capture.

Packed inputs have three paths: an empty token buffer only updates final
state; equal positive lengths aligned to CHUNK use a fixed-batch view; other
lengths use sequence metadata. The equal-length optimization requires no
Q/K/V copy. Packed calls currently bypass automatic graph caching.

## Scheduling and device stages

`ascendc.py` launches Pre-Gram, solve, then persistent recurrence. Solve may
overlap its vector and Cube slices on separate streams; stage timings must
account for that overlap when interpreting end-to-end latency.

Common scheduling names use these units:

| Name | Meaning |
| --- | --- |
| `CHUNK` | Tokens per recurrence chunk. |
| `counts` | Chunk count for each sequence, including partial tails. |
| `bh` | Number of sequence/head pairs. |
| `c` | Total sequence chunks times heads. |
| `nb` / `nblk` | Number of launched persistent blocks. |
| `cores` | Physical Cube cores used by the scheduling model. |
| `pre_unroll` | Iterations per mixed Pre-Gram block; two chunk/head items per iteration. |

`_pre_gram_unroll` balances arithmetic padding against block startup.
`_persistent_fixed_blocks` and `_persistent_varlen_blocks` model stride
mapping; `_persistent_grouped_plan` considers grouping similar recurrence
lengths. Their scores are relative work estimates, not measured milliseconds.
Empirical constants should be changed with target-device benchmarks.

Sequence metadata uses INT64 rows of `(token start, token count, chunk start,
chunk count)`, followed by a sentinel and optional four-head mapping rows.
Changes to this layout must update both host packing and device consumers.
The same applies to the positional launch argument lists and native ABI.

## CATLASS device implementation

The CATLASS sources own their schedule independently of `csrc/ascendc/`,
but deliberately preserve that backend's execution order and buffer layout.
`scheduled_tiles.hpp` adapts upstream `TileMmad`, `CopyGmToL1`,
`CopyL0CToGm`, contiguous vector add/multiply/scale/cast tiles and cross-core
flag objects. It does not allocate a full `BlockMmad` per product.

`fwd_kernel1_pre_gram.cpp` batches normalization and operand preparation in
16-row passes, keeps the complete gate prefix tile, and uses the same band
recentering and mask pipeline as AscendC. Cube operands stay in L1/L0 across
products; the vector epilogue uses queue-owned band buffers.

`fwd_kernel1_solve_wide.cpp` uses cross-chunk broadcast/repeat layouts and
fused multiply-add forward substitution. Assembly preserves its two passes,
L1 intermediate and batched final Fixpipe. W/U reuses resident inverse tiles.
The fused translation unit includes the CATLASS assembly/WU sources, with
matching chunk ownership macros, never the legacy source directory.

`fwd_kernel2_persistent.cpp` preserves the phased head loops, FP32 UB state,
BF16 state publication, residual fractal arrangement, operand reuse and
conditional next-step publication of the AscendC implementation. It no
longer serializes a complete predict/residual/update/finish cycle per head.
Partial chunks, packed metadata and empty-head finalization follow the same
ownership rules as the baseline.

Upstream tile coverage now includes aligned/strided GM/UB copies, byte-stride
row loads without padding, GM/L1 stored-fractal copies and the batched
`CopyGmToL1GMMPTD` interface. L1/L0 runs use upstream CopyL1ToL0A/B with
explicit fractal layouts, including BF16 transpose. `TileBroadcastOneBlk`,
`TileRowBroadcastAdd/Mul` and `TileOneBlkColumnBroadcastMul` replace compatible
block, row and column broadcasts without changing prefix/FMA dependencies.
The triangular inverse's count-mode scale/cast loops split full tiles and a
compile-time tail so all vector FP32-to-BF16 narrowing uses upstream TileCast.

Custom blocks retain two scalar-padding loads, one UB-to-UB fractal copy,
three strided prefix/reduction Add sites, fused MulAddDst, two specialized
Fixpipe sites (L0C->L1 NZ and batched GM output), nonlinear/reduction
primitives, BF16 widening and intra-core queue/event/barrier ownership.
These are exact-interface exceptions, not all classified as impossible
algorithms: decomposing them into several upstream calls could alter the
rounding, instruction batching or pipeline. Details and measurements are in
`benchmarks/results/catlass_extra_tiles_910b3.md`.

Upstream TileMmad adds its own small-tile PIPE_M barrier. The older
`common/prepare/solve/recurrence/vector_tiles.hpp` implementation was removed;
only the schedule-preserving kernels remain active.

## Graph lifetime and concurrency

Automatic cache keys include tensor addresses/layouts, scalar parameters,
stream identity and runtime scheduling overrides. Input values can change in
place if those writes precede replay on the stream. Replacing a tensor causes
a cache miss. Strong references prevent freed addresses from being reused
while their graphs remain cached.

Eviction and `clear_auto_graph_cache()` synchronize owning streams before
releasing references. Clearing affects the calling thread only. Disabling
automatic mode does not clear existing entries. Packed inputs, profiling,
external capture and input/output storage aliasing bypass automatic caching.
Capture and replay errors propagate to the caller.

Explicit capture retains its input/output buffers and private scratch. Use
separate objects and output buffers for concurrent executions. Warmup and
capture execute the forward and are setup costs; benchmark steady replay
separately from first-call latency.

## Validation after changes

Run commands from the repository root with the Ascend toolkit environment
loaded. Select a free NPU with `ASCEND_RT_VISIBLE_DEVICES` as appropriate.

```bash
python -m compileall -q flash_kda_npu/ops
git diff --check
python -m pytest -q flash_kda_npu/tests/test_fwd.py -k 'native_auto_graph or native_graph'
```

Scheduling or kernel changes also need the numerical tests and fixed/packed
benchmarks described in README.md. Compare warm steady-state timings with
the same geometry, state dtype and profiling settings. Historical measurements
and reproduction scripts are under `benchmarks/results/`.

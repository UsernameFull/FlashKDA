# CATLASS vector/copy/synchronization component migration — 910B3

> Historical implementation: superseded by the AscendC-scheduled CATLASS kernels on 2026-10-20. See [current report](catlass_scheduled_910b3.md). Measurements below describe the earlier sources.

Date: 2026-10-19. Branch: `npu_catlass`. This report supersedes the earlier
`catlass_rewrite_910b3.md` report for the current source state.

## Replacements

The entire default NPU path is independent of the old `csrc/ascendc/` sources.
The matrix pipeline remains upstream AtlasA2 `BlockMmad`. This change replaces
all compatible direct vector-stage operations with the following **unmodified
upstream** CATLASS components:

| Operation | Upstream component | Integration |
| --- | --- | --- |
| Elementwise FP32 addition | `TileElemWiseAdd` | Normalization, prefix accumulation, triangular substitution, output/state epilogues |
| Elementwise multiplication | `TileElemwiseMul` | Squared norms and gate-scaled operands |
| Scalar multiplication | `TileElemWiseMuls` | Normalization, beta/scale factors, triangular solve |
| FP32→BF16 rounding | `TileCast` | All rounded matrix operands, residuals, outputs and states |
| State decay broadcast | `TileRowBroadcastMul` | Replaces 32 row-wise multiplies per half-state with a broadcast tile |
| Contiguous GM↔UB transfers | `CopyGm2UbAligned`, `CopyUb2GmAligned` | Preprocessing, solve and recurrent scratch/state transfers |
| Strided GM↔UB transfers | `CopyGm2Ub`, `CopyUb2Gm` | Diagonal inverse stores, sliced residuals and public output strides |
| Cross-engine synchronization | `Arch::CrossCoreFlag`, `CrossCoreSetFlag`, `CrossCoreWaitFlag` | Named Gram and recurrence handoff objects |

`csrc/catlass/vector_tiles.hpp` contains the composition layer and KDA extension
tiles. `TileL2Normalize`, `TileGateScan`, `TileForwardSubstitution` and
`TileCausalMask` own the KDA-specific algorithms; their supported arithmetic
and transfers use the upstream tiles above. The pinned upstream include tree
(233 files) is byte-identical to its source checkout; no vendor patch is used.
Package-data coverage includes all 244 kernel/vendor files.

## Remaining direct AscendC operations and reasons

| Remaining operation | Reason |
| --- | --- |
| Exp, division, immediate scalar addition, subtraction, fill, square root and reduction | This upstream version has no standalone compatible tiles; these are explicitly implemented as KDA extension primitives, composed with upstream arithmetic. Existing SiLU/softmax blocks have different semantics. |
| BF16→FP32 conversion | Upstream `TileCast` hardcodes `CAST_RINT`. CANN 9.1's 910B conversion accepts only `CAST_NONE`; using the upstream tile here would select an unsupported branch rather than perform the required widening. `TileWiden` preserves the supported mode. |
| Local pipe barriers and MTE/V hard events | The upstream cross-core API covers engine handoffs, not intra-core DMA/vector/scalar ownership. Upstream block implementations also use these AscendC primitives internally. |
| Scalar UB access, metadata/indexing and kernel/device intrinsics | KDA masks/triangular dependencies and kernel execution control have no corresponding generic upstream replacement. |

The BF16 mode constraint is verified in CANN's
`asc/impl/basic_api/dav_c220/kernel_operator_vec_vconv_impl.h`, BF16→FP32
`CastIntrinsicsImpl`: only `CAST_NONE` invokes the conversion. Generic upstream
`BlockEpilogue` variants also cannot directly own the persistent live FP32 UB
state and mixed FP32/BF16, strided KDA epilogue: they assume separate UB storage,
homogeneous data types or fixed activation semantics. The custom KDA block
therefore composes compatible upstream tiles instead.

CATLASS is built on AscendC. This migration increases actual upstream component
reuse; it does not remove the language/runtime dependency or relabel extension
primitives as upstream implementations. The stage-source audit found zero
direct add/mul/muls/cast/GM-copy/cross-core calls; the documented exceptions
remain in the extension/common layers.

## Compilation and numerical validation

Environment: Ascend 910B3, CANN 9.1.0, driver 25.5.1, Python 3.12.13,
torch 2.10.0+cpu, torch_npu 2.10.0.post4. Default CHUNK64/MAXH4, fused solve,
overlap24. The host bridge and all exercised RTC variants compiled. Python
compilation, package-data coverage and `git diff --check` passed.

| Suite | Passed | Skipped | Failures/errors | Seconds |
| --- | ---: | ---: | ---: | ---: |
| full | 1015 | 12 | 0 | 727.87 |
| long | 8 | 0 | 0 | 340.51 |
| chunk16 | 309 | 104 | 0 | 80.39 |
| chunk32 | 341 | 72 | 0 | 89.08 |
| unfused | 91 | 0 | 0 | 63.93 |
| dual | 2 | 0 | 0 | 47.64 |

**1027 distinct test IDs passed** across the main and supplemental runs.
Original accuracy criteria were retained, including the existing large-state
initial-transient allowance and warning-only weak-gate final-state sweep.
All eight long tests passed, including 1M-token token-by-token and chunked FP64
references, packed 512K+512K input, and long partial tails. Alternate chunks,
unfused solve, graph replay/changed inputs and two-device behavior were tested.
The main run's conditional skips are covered by supplemental configurations.

- `output_relative_rms` maximum: 0.87364%, `test_fwd_fixed[in_only-bf16-T8192-H1]`.
- `output_max_window_relative_rms` maximum: 0.87368%, `test_fwd_varlen[in+out-fp32-seqs1300_547_2048_963_271_3063-H1]`.
- `output_post_transient_max_relative_rms` maximum: 0.44626%, `test_fwd_batched[in+out-bf16-B8_T128-H1]`.
- `state_relative_rms` maximum: 0.70277%, `test_cuda_gate_bias_accuracy[fixed-g=-8_bias=0]`.

The component migration additionally compares all output and requested state
elements against the immediately preceding full-rewrite snapshot for 18
identical-input cases (six geometries × BF16/FP32/no state). **Every element is
bitwise equal**, with maximum absolute difference 0. The snapshot is preserved
in `/tmp/kda-catlass-before-vector/`; the comparison runner and inputs/results
are in `/tmp/kda_component_equivalence.py` and `/tmp/kda-components-equivalence/`.
This is separate from comparison against the older AscendC backend.

An initial eight-worker attempt on card2 exhausted device memory and was
stopped. The final full matrix uses four workers on card5. The initial card1
benchmark attempt was discarded; the reported benchmark was rerun on card6.
No precision thresholds or algorithm were changed to accommodate these issues.
Logs for the interrupted attempts remain under `/tmp/kda-vector-*-card-pressure.log`.

## Performance

Both backends run in separate processes on card6, with the same fixed CPU
inputs, 30 warmups, 100 iterations ×3 repeats, NPU events, profiling off and
automatic graph caching off. Compilation/graph setup are excluded. Eager
includes host launch/allocation overhead; explicit graph replay better
isolates device execution. FP32-state cases are shown; JSON includes all 18
cases, their mean/min/max timings and full-tensor numerical differences.

| Shape | Old AscendC eager ms | CATLASS eager ms | Old graph ms | CATLASS graph ms |
| --- | ---: | ---: | ---: | ---: |
| short (T64, H4) | 1.2740 | 1.2099 | 0.1117 | 0.4271 |
| tail (T129, H4) | 1.2705 | 1.2551 | 0.1343 | 0.4602 |
| fixed_h16 (T8192, H16) | 2.8981 | 14.7471 | 1.9877 | 14.7204 |
| fixed_h96 (T8192, H96) | 8.2459 | 82.4367 | 8.2298 | 82.6643 |
| packed (T8192, H16) | 3.4004 | 15.6496 | — | — |
| empty_packed (T115, H4) | 1.5453 | 1.4960 | — | — |

**This component migration does not recover the legacy backend's long-input
performance.** The migration increases upstream reuse and does not demonstrate a
speedup. Small eager differences are dominated by host overhead. No controlled
same-card before/after timing of the preceding full rewrite is claimed here;
its numerical equivalence is validated separately as described above.

All benchmark outputs/states were finite. Fixed-length explicit graph replay
is bitwise equal to eager for all 12 shape/state cases. Differences versus the
legacy backend are numerical, not a requirement of bitwise equality.
Maximum relative RMS versus the legacy backend among all tensors: 0.08917%.

## Reproduction and artifacts

```bash
bash flash_kda_npu/csrc/build.sh
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 python -m pytest -q flash_kda_npu/tests/test_fwd.py \
 flash_kda_npu/tests/test_fwd_full.py -n 4 --dist=worksteal
ASCEND_RT_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 python -m pytest -q flash_kda_npu/tests/test_fwd.py \
 flash_kda_npu/tests/test_fwd_full.py --run-long -k long -n 4 --dist=worksteal
# Fresh processes: KDA_CHUNK=16 and 32 for alternate chunks,
# KDA_SOLVE_FUSE=0 for separate solve kernels; two visible devices for device tests.
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 python -m flash_kda_npu.benchmarks.bench_catlass --graphs \
 --json /tmp/catlass-components.json --warmup 30 --iters 100 --repeats 3
```

- `catlass_components_910b3.json`: legacy/current timings and numerical differences.
- `catlass_components_validation_910b3.json`: test totals and reference-error maxima.
- `catlass_components_equivalence_910b3.json`: 18-case before/after bitwise comparison.
- `/tmp/kda-vector-*.log` and `.xml`: raw logs and JUnit results in this workspace.

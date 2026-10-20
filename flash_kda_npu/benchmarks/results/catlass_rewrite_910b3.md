# Complete CATLASS kernel rewrite — Ascend 910B3

> Historical implementation: superseded by the AscendC-scheduled CATLASS kernels on 2026-10-20. See [current report](catlass_scheduled_910b3.md). Measurements below describe the earlier sources.

Date: 2026-10-19. Branch: `npu_catlass`.

**Historical baseline:** this report describes the initial full rewrite before
upstream vector/copy/synchronization component migration. Current results and
remaining AscendC operations are in [the component report](catlass_components_910b3.md).

## Scope

All NPU arithmetic stages are independently rewritten under `csrc/catlass/`:
normalization/gates/Gram, triangular inverse, coupling assembly, W/U and
persistent recurrent output/state updates. The default CATLASS path never
reads or includes the old AscendC kernel sources. Every matrix product uses
upstream `BlockMmad<MmadAtlasA2Pingpong<false>, ...>` for the entire GM/L1/L0/GM
pipeline, with upstream copy operators, matrix operations, buffer rotation,
events and accumulator stores. Custom KDA vector blocks use CATLASS
`Arch::Resource<AtlasA2>` UB storage and AscendC vector/scalar operations.
CATLASS itself is built on AscendC; Python scheduling, host ABI validation,
CANN RTC and the native launcher remain. This is not a claim that all code is
an upstream CATLASS operator or that AscendC dependencies have been removed.

The previous backend remains separately selectable with `KDA_BACKEND=ascendc`
for comparison. It was restored from the pre-migration working-tree snapshot,
preserving the user's pre-existing changes. No automatic fallback exists.
The old tile-only helper has been removed. Earlier `catlass_910b3.*` results
are historical and do not validate this implementation.

CATLASS revision: `41bf90da655bba3c66d0acd7e00abe33960ecfd6`, unmodified
vendored include tree, CANN Open Software License v2. Environment: CANN 9.1.0,
driver 25.5.1, Ascend910B3, Python 3.12.13, torch 2.10.0+cpu,
torch_npu 2.10.0.post4. CHUNK64, MAXH4, fused solve, overlap24.

## Compilation and precision

The host extension and all exercised RTC variants compiled successfully.
Original FP64-reference accuracy criteria were retained. Full tests cover
fixed/packed inputs, partial/empty sequences, initial/final BF16/FP32/no
state, extreme gates/bias, unroll/head mappings, graphs and device switching.
The long tests include 128K and 1M tokens and two packed 512K sequences.

| Suite | Passed | Skipped | Failures/errors | Seconds |
| --- | ---: | ---: | ---: | ---: |
| full | 1015 | 12 | 0 | 463.98 |
| long | 4 | 0 | 0 | 171.99 |
| native_long | 2 | 0 | 0 | 71.01 |
| native_scale_long | 2 | 0 | 0 | 319.99 |
| chunk16 | 309 | 104 | 0 | 81.52 |
| chunk32 | 341 | 72 | 0 | 94.74 |
| unfused | 91 | 0 | 0 | 63.55 |
| dual_device | 2 | 0 | 0 | 48.78 |

Across the main and supplemental runs, **1027 distinct test IDs passed**.
Chunk16/32 suites skip cases requiring a different chunk geometry and the
opt-in long/device cases; these are exercised in the corresponding runs.
Host build, Python compilation, `git diff --check` and package-data coverage
of all 243 new kernel/vendor header files also passed.

The reference tests retain the existing distinction between ordinary RMS
thresholds and large-arange-initial-state transients: an initial transient may
reach 1%, while subsequent output windows must remain below 0.5%. The CUDA
style weak-gate state sweep remains warning-only at its existing threshold;
warning maxima and test names are recorded in validation JSON rather than
being presented as strict state-threshold passes.

- Main `output_relative_rms` maximum: 0.0087363818 (0.87364%), `test_fwd_fixed[in_only-bf16-T8192-H1]`.
- Main `output_max_window_relative_rms` maximum: 0.0087367827 (0.87368%), `test_fwd_varlen[in+out-fp32-seqs1300_547_2048_963_271_3063-H1]`.
- Main `state_relative_rms` maximum: 0.0070277301 (0.70277%), `test_cuda_gate_bias_accuracy[fixed-g=-8_bias=0]`.
- Main `output_post_transient_max_relative_rms` maximum: 0.0044625681 (0.44626%), `test_fwd_batched[in_only-fp32-B8_T128-H1]`.

The benchmark independently compares every output and requested final-state
element for 18 identical-input cases (6 shapes × BF16/FP32/no state). All are
finite. Cross-backend bitwise equality is not expected because gate scan and
triangular-inverse arithmetic orders differ. Graph capture/replay is bitwise
equal to eager for all 12 fixed-length shape/state cases.
Maximum cross-backend relative RMS over these comparisons: **0.08917%**.

## Performance

Selected free card1; other cards handled accuracy runs. Backends run in
separate processes with identical fixed CPU seeds. Profiling and automatic
graph caching are disabled. Each case uses 30 warmups and 3 repeats of 100
iterations, with NPU events. Timings exclude RTC compilation and graph setup.
FP32 input/output state results are shown; JSON contains all 18 cases and timing summaries. Eager includes host allocation/launch overhead. Explicit graph
replay more closely isolates device execution.

| Shape | Old eager ms | CATLASS eager ms | Old graph ms | CATLASS graph ms |
| --- | ---: | ---: | ---: | ---: |
| short (T64, H4) | 1.2297 | 1.1912 | 0.1062 | 0.4244 |
| tail (T129, H4) | 1.2455 | 1.2159 | 0.1165 | 0.4562 |
| fixed_h16 (T8192, H16) | 2.6629 | 14.7025 | 2.0162 | 14.6698 |
| fixed_h96 (T8192, H96) | 8.2456 | 82.4547 | 8.2357 | 82.7036 |
| packed (T8192, H16) | 3.2416 | 15.5694 | — | — |
| empty_packed (T115, H4) | 1.4551 | 1.4599 | — | — |

**The complete rewrite is substantially slower on long inputs.** For T8192,
H16, graph latency increases about 7.3×; for H96, about 10.0×. Short eager calls
are dominated by host overhead and conceal their slower device execution.
This implementation does not demonstrate a speedup and should not be
represented as performance-equivalent to the prior backend.

The rewrite uses explicit per-row vector work, GM handoffs between complete
CATLASS GEMMs, and serialized per-head recurrence ownership. These are known
structural costs relative to the old specialized overlapping pipeline; this
run does not attribute timings through a per-stage hardware profile. Future
optimization would need vector batching, compatible GEMM fusion and overlap
without reverting to the old direct matrix primitives.

## Reproduction and artifacts

```bash
bash flash_kda_npu/csrc/build.sh
ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q flash_kda_npu/tests/test_fwd.py \
  flash_kda_npu/tests/test_fwd_full.py -n 8 --dist=worksteal
ASCEND_RT_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q flash_kda_npu/tests/test_fwd.py \
  flash_kda_npu/tests/test_fwd_full.py --run-long -k long -n 4 --dist=worksteal
ASCEND_RT_VISIBLE_DEVICES=5 KDA_CHUNK=16 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q flash_kda_npu/tests/test_fwd.py -k 'native and not auto_graph and not graph'
# Repeat with KDA_CHUNK=32; use two visible devices for the device tests.
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m flash_kda_npu.benchmarks.bench_catlass --graphs \
  --json /tmp/catlass-rewrite.json --warmup 30 --iters 100 --repeats 3
```

- `catlass_rewrite_910b3.json`: all performance summaries and full-tensor cross-backend differences.
- `catlass_rewrite_validation_910b3.json`: test totals and reference-error maxima with test IDs.
- Raw pytest logs/JUnit XML are retained in `/tmp/kda-rewrite-*.{log,xml}` in this workspace.

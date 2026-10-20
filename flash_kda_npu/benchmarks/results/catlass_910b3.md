# Historical tile-only migration validation — Ascend 910B3

Date: 2026-10-19. Branch: `npu_catlass`.

**Historical:** these results predate the complete device-kernel rewrite. See
[the current rewrite report](catlass_rewrite_910b3.md) for current validation.

All Cube matrix products and L1-to-L0 operand copies now use CATLASS AtlasA2 tile operators by default. KDA vector arithmetic, GM/L1 and L0C/GM transfers, event ownership, chunk scheduling and FP32 state recurrence remain custom AscendC. This is tile composition, not CATLASS block/device GEMM pipelining. The previous direct primitives remain explicitly selectable with `KDA_BACKEND=ascendc`; there is no fallback.

CATLASS: `41bf90da655bba3c66d0acd7e00abe33960ecfd6` (vendored unmodified headers and CANN Open Software License v2). Host launcher and all exercised RTC kernel variants compiled successfully with CANN 9.1.0. PyTorch 2.10.0+cpu, torch_npu 2.10.0.post4, Python 3.12.13; hardware: 910B3, driver 25.5.1. Default CHUNK=64, MAXH=4, fused solve and solve overlap enabled.

## Precision

Both benchmark runs compare the complete output and requested final state for 18 identical-input cases (6 geometries × BF16/FP32/no state). Every comparison is bitwise equal, with maximum absolute difference **0**. The graph run additionally checks graph replay against eager output/state for all 12 fixed-length cases. Inputs originate from a fixed CPU seed; backend workers run in separate processes.

- full_suite: **1015 passed, 12 skipped**, zero failures/errors (427.42 s).
- long_suite: **8 passed, 0 skipped**, zero failures/errors (657.75 s).
- CHUNK=16: **153 passed, 2 skipped** (54.91 s); CHUNK=32: **153 passed, 2 skipped** (55.31 s). Each covers state combinations, partial chunks and packed lengths; the skips are the other two configured-chunk cases.
- CHUNK=64, separate assembly/WU kernels (`KDA_SOLVE_FUSE=0`): **88 passed**, zero failures (50.39 s).
- Two-device context switching (physical NPUs 4/5): **1 passed** (40.38 s); cross-device input rejection: **1 passed** (10.42 s).
- All 12 main-suite skips were exercised in supplemental runs: 8 long cases, 2 configured-chunk cases and 2 two-device cases. Together these cover every case in the 1027-case suite.
- Package-data verification includes all 233 upstream headers, the KDA tile adapter and license. Python compilation and `git diff --check` pass.

The independent numerical oracle is CPU FP64. Existing criteria remain: ordinary output/state relative RMS <0.5%, packed output <0.6%, and up to 1% for the large initial-state transient, with <0.5% required after the first 64 tokens. Native diagnostics also retain pointwise checks. Gate-bias sweeps assert output RMS but retain their existing **warning-only** state threshold policy; bitwise baseline equality does not remove those pre-existing weak-decay numerical limitations.

### full_suite numerical maxima

| Metric | Maximum relative RMS | Worst case |
| --- | ---: | --- |
| gate_sweep/output_relative_rms | 0.5653% | `test_cuda_gate_bias_accuracy[varlen-g=-8_bias=0]` |
| gate_sweep/output_max_window_relative_rms | 0.6337% | `test_cuda_gate_bias_accuracy[varlen-g=-8_bias=0]` |
| gate_sweep/state_relative_rms | 0.7530% | `test_cuda_gate_bias_accuracy[fixed-g=-8_bias=0]` |
| matrix/output_relative_rms | 0.8736% | `test_fwd_fixed[in_only-fp32-T8192-H1]` |
| matrix/output_max_window_relative_rms | 0.8737% | `test_fwd_varlen[in_only-fp32-seqs1300_547_2048_963_271_3063-H1]` |
| matrix/state_relative_rms | 0.3164% | `test_fwd_fixed[out_only-bf16-T256-H1]` |
| matrix/output_post_transient_max_relative_rms | 0.4496% | `test_fwd_batched[in_only-fp32-B8_T128-H1]` |

### long_suite numerical maxima

| Metric | Maximum relative RMS | Worst case |
| --- | ---: | --- |
| matrix/output_relative_rms | 0.8412% | `test_fwd_long_varlen[seqs131072]` |
| matrix/output_max_window_relative_rms | 0.8414% | `test_fwd_long_varlen[seqs131072]` |
| matrix/output_post_transient_max_relative_rms | 0.4255% | `test_fwd_long_varlen[seqs524288_524288]` |
| matrix/state_relative_rms | 0.2886% | `test_fwd_long[T1048576]` |

## Performance

NPU event timings of the public API, including conversions, allocations and packed metadata handling; RTC compilation and input construction excluded. Both runs use 30 warmups, 100 iterations × 3 repeats per variant. Timing uses physical NPU 1; numerical tests use different cards. Other CPU reference processes run concurrently. Means/min/max and native profiles are retained in JSON. These are local measurements, not a cross-device performance guarantee.

### First eager comparison

| Case | State | AscendC ms | CATLASS ms | AscendC / CATLASS |
| --- | --- | ---: | ---: | ---: |
| short (T=64, H=4) | bf16 | 1.2185 | 1.3523 | 0.901× |
| short (T=64, H=4) | none | 1.1636 | 1.2873 | 0.904× |
| short (T=64, H=4) | fp32 | 1.2198 | 1.3287 | 0.918× |
| tail (T=129, H=4) | bf16 | 1.2316 | 1.3626 | 0.904× |
| tail (T=129, H=4) | none | 1.2011 | 1.3075 | 0.919× |
| tail (T=129, H=4) | fp32 | 1.2427 | 1.3422 | 0.926× |
| fixed_h16 (T=8192, H=16) | bf16 | 2.7740 | 3.1307 | 0.886× |
| fixed_h16 (T=8192, H=16) | none | 2.6186 | 3.0256 | 0.865× |
| fixed_h16 (T=8192, H=16) | fp32 | 2.6983 | 3.0804 | 0.876× |
| fixed_h96 (T=8192, H=96) | bf16 | 8.2414 | 8.2484 | 0.999× |
| fixed_h96 (T=8192, H=96) | none | 8.2365 | 8.2267 | 1.001× |
| fixed_h96 (T=8192, H=96) | fp32 | 8.2341 | 8.2441 | 0.999× |
| packed (T=8192, H=16) | bf16 | 3.1951 | 3.5367 | 0.903× |
| packed (T=8192, H=16) | none | 3.1165 | 3.3305 | 0.936× |
| packed (T=8192, H=16) | fp32 | 3.1918 | 3.4257 | 0.932× |
| empty_packed (T=115, H=4) | bf16 | 1.4523 | 1.6599 | 0.875× |
| empty_packed (T=115, H=4) | none | 1.4136 | 1.5855 | 0.892× |
| empty_packed (T=115, H=4) | fp32 | 1.4753 | 1.6135 | 0.914× |

### Repeat with graph replay (FP32 state)

| Case | Eager AscendC ms | Eager CATLASS ms | Graph AscendC ms | Graph CATLASS ms | Graph AscendC / CATLASS |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 1.2383 | 1.2720 | 0.1103 | 0.1093 | 1.009× |
| tail | 1.2410 | 1.2749 | 0.1250 | 0.1294 | 0.966× |
| fixed_h16 | 2.9557 | 2.9419 | 1.9926 | 2.0190 | 0.987× |
| fixed_h96 | 8.2520 | 8.2505 | 8.2304 | 8.2292 | 1.000× |
| packed | 3.3376 | 3.3970 | — | — | — |
| empty_packed | 1.5068 | 1.5257 | — | — | — |

CATLASS shows **no consistent speedup** in this tile-level migration. H=96 / T=8192 stays effectively unchanged (~8.23 ms replay). H=16 / T=8192 replay is 1.993 → 2.019 ms (~1.3% slower). The first eager run shows 7–16% slowdowns in several smaller cases; the repeat largely reduces that gap, indicating host/scheduling variance contributes. CATLASS TileMmad also inserts a PIPE_M barrier for small tiles, and this migration does not introduce a new block-level pipeline. Further performance work would require tile/block scheduling changes, with a new numerical and performance validation cycle.

Raw data: [eager comparison](catlass_910b3.json), [repeat and graph replay](catlass_graph_910b3.json), [numerical summary](catlass_validation_910b3.json).

## Reproduce

```bash
bash flash_kda_npu/csrc/build.sh
ASCEND_RT_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -xq -n 8 --dist=worksteal flash_kda_npu/tests/ \
  --junitxml=/tmp/kda-catlass-tests.xml
# -n/--dist require pytest-xdist; omit them for serial execution.
ASCEND_RT_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -xq flash_kda_npu/tests/test_fwd.py \
  -k "native_device_switch or native_rejects_cross_device_inputs"
ASCEND_RT_VISIBLE_DEVICES=6 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -xq flash_kda_npu/tests/ --run-long -k long \
  --junitxml=/tmp/kda-catlass-long.xml
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m flash_kda_npu.benchmarks.bench_catlass --graphs \
  --json /tmp/catlass-graph.json --warmup 30 --iters 100 --repeats 3
```

Run alternate chunks in fresh processes using `KDA_CHUNK=16` / `32`; use `KDA_SOLVE_FUSE=0` for the separate solve kernels. Select available cards appropriate to the host. The RTC target remains Ascend910B3.

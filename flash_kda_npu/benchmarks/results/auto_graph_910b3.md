# Automatic graph cache, 2026-10-17

Ascend910B3, H=96/D=128, FP32 initial/final state, BF16 public inputs.
Physical device 1, one host thread, 30 warmups, 200 iterations x 3 repeats,
regular/auto/regular/auto order (1200 samples per mode per shape).
Compilation, warmup and initial capture excluded. NPU event timings include
regular public validation and automatic cache-key construction on every call.

| T | Regular ms | Automatic replay ms | Speedup |
|---:|---:|---:|---:|
| 256 | 1.5808 | 0.3597 | 4.39x |
| 8192 | 8.2394 | 8.2310 | 1.00x |

Enable KDA_AUTO_GRAPH=1; no caller code changes are needed. Inputs/outputs must
reuse tensor addresses for cache hits. Cache is thread-local with stream in the
key and holds strong references to tensor buffers and private graph scratch.
Default eager threshold is 3 calls, capacity 8 entries; retired streams are
synchronized before LRU removal. Explicit clear_auto_graph_cache cleans this
thread's entries. Packed calls, profiling, external capture and input/output
storage aliases bypass the automatic path. Capture errors propagate rather
than silently retrying potentially failed device operations.

Tests: 7 graph/cache cases passed, then 1 stream-isolation/packed/profiling
bypass case passed. Covers BF16/FP32 state, changed input values, changed
scalar/address, LRU eviction, explicit capture with automatic mode enabled,
and independent graph workspace. Eager/auto outputs and final states match
bitwise. Python compileall and git diff --check passed.

Reproduce the saved benchmark script from the repository root with CANN loaded:

```bash
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python flash_kda_npu/benchmarks/results/auto_graph_910b3.py
```

The saved script writes raw JSON to /tmp/kda_auto_bench.json. Raw measurements
are also available in auto_graph_910b3.json. No FLA comparison was rerun.

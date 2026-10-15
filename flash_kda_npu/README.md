# FlashKDA Ascend NPU backend

This directory adds the Ascend NPU implementation for FlashKDA.  The native
path uses the AscendC v1 backend ported from the `kda-ascendc` release: the
host launcher exposes RTC compilation/launch APIs and
`flash_kda_npu/ops/ascendc.py` orchestrates the three production
stages (`pre_gram`, `solve`, and persistent-loop `k2`).

The native pipeline has three logical stages:

1. `pre_gram` normalizes Q/K, activates gates and beta, and forms the
   intra-chunk Gram matrices with paired vector/Cube work.
2. `solve` computes W/U using the wide vector solve and the Cube kernel.
   At chunk size 64 it uses two 32-row sub-blocks and an additional Cube
   assembly kernel; vector and Cube slices can overlap on separate streams.
3. Persistent-loop `k2` computes outputs and carries fp32 recurrent state
   across chunks, with up to four heads per block.

The diagnostic scheduling modes, debug
GM stores (`Qn/Kn/Gate/Gc/BetaOut/Vnew/A32`) and the
`return_intermediates` host path that the development build carried have
all been removed, along with their environment-variable switches
(`KDA_DEBUG_STORES`, `KDA_SOLVE_A16_MODE`, `KDA_ASM_LOADS`,
`KDA_CUBE_A16_RESIDENT`, `KDA_CUBE_LOADS`, `KDA_PRE_RAW_MODE`).

## Layout

```text
flash_kda_npu/
|-- csrc/
|   |-- flash_kda.cpp
|   |-- build.sh
|   `-- ascendc/
|       |-- fwd_kernel1_pre_gram.cpp
|       |-- fwd_kernel1_solve_wide.cpp
|       |-- fwd_kernel1_solve_assemble.cpp
|       |-- fwd_kernel1_solve_cube.cpp
|       `-- fwd_kernel2_persistent.cpp
|-- tests/
|   |-- conftest.py
|   |-- run_test_full.sh
|   |-- test.sh
|   |-- torch_ref.py
|   |-- test_fwd.py
|   `-- test_fwd_full.py
|-- benchmarks/
|   |-- bench.sh
|   |-- bench_fwd.py
|   `-- generate_benchmark_md.py
`-- ops/
    |-- ascendc.py
    `-- kda_fwd.py
```

## Build

Build on a Linux Ascend host with compatible PyTorch, `torch_npu`, CANN
headers/runtime (including `acl_rtc`), and a C++17 compiler. From the repository
root, configure the toolkit path for both the build and RTC compiler:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
source "$ASCEND_HOME_PATH/set_env.sh"
bash flash_kda_npu/csrc/build.sh
```

The launcher is packaged in `flash_kda_npu/lib`.  AscendC kernel sources in
`csrc/ascendc/` are RTC compiled into the launcher process on first use.  The
default chunk size is `KDA_CHUNK=64`; the native implementation also supports
16 and 32. The RTC target currently defaults to `Ascend910B3` in
`csrc/flash_kda.cpp`; this is not automatic hardware detection.

To install the NPU backend with pip from a repository checkout:

```bash
pip install --no-build-isolation .
```

`--no-build-isolation` uses the existing PyTorch installation; on an NPU host
this should already include `torch_npu`. `setup.py` invokes the NPU build only
when `ASCEND_HOME_PATH` is set. The build script accepts `PYTHON_BIN`,
`PYTHON_PREFIX`, `TORCH_PREFIX`, and `CXX` overrides for nonstandard environments.

## Usage

```python
import flash_kda

flash_kda.fwd(
    q, k, v, g, beta, scale, out,
    A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
    initial_state=h0, final_state=h1, cu_seqlens=None,
)
```

`flash_kda.fwd` sends non-CUDA tensors to `flash_kda_npu.ops.fwd`. The NPU
entry point executes native AscendC only, with no automatic PyTorch fallback.
CPU inputs are rejected. Validation, import, RTC compilation and kernel launch
errors propagate to the caller, as on the CUDA path.

The public API writes `out` and the optional `final_state` in place and returns
`None`. Q/K are normalized internally; pass raw gate values and beta logits,
since gate activation and sigmoid(beta) are also performed internally.

The native path through the public API currently requires:

- NPU tensors with matching `q/k/v` shape `[B,T,H,128]`, `T >= 64`,
  and `T % 64 == 0`;
- fixed-length batches (`cu_seqlens is None`);
- bf16 `q/k/v/g/out` of the same shape, and bf16 `beta` with shape `[B,T,H]`;
- fp32 `A_log` with shape `[H]` and fp32 `dt_bias` with shape `[H,128]`;
- bf16 or fp32 `final_state` with shape `[B,H,128,128]` when requested;
- bf16 or fp32 `initial_state`, when present, with shape `[B,H,128,128]`;
- matching initial/final state dtypes when both buffers are supplied, as on CUDA;
- all tensors contiguous and on the same NPU, with positive batch/head counts.

The wrapper validates public inputs before any conversion or native launch.
It converts the validated bf16 `g`/`beta` and optional initial state to fp32
for the internal AscendC interface. It does not repair noncontiguous inputs,
silently accept other public dtypes, or copy through a temporary final-state
buffer. The caller's output buffers are passed directly to AscendC.

The public dispatch gate always checks divisibility by 64, even when
`KDA_CHUNK` is 16 or 32. The lower-level
`flash_kda_npu.ops.ascendc.kda_bt16_fwd_ascendc` accepts lengths divisible
by the configured chunk size and returns `(out, final_state)`; it requires
fp32 gate/beta/parameters/initial state directly and uses `bias` instead of
`dt_bias`.

The public dtype, state-buffer and failure contracts match CUDA. Native shape
coverage is still narrower: CUDA supports varlen and partial chunks; NPU
currently raises `NotImplementedError` for NPU `cu_seqlens` and `RuntimeError`
for unsupported lengths or dimensions. Public tensor validation failures raise
`RuntimeError`; exceptions from the native implementation retain their types.

## Configuration and profiling

The default native chunk size is 64. The public API requires `T % 64 == 0`,
including when the native chunk size is configured to 16 or 32.

Warm up before benchmarking to exclude first-use RTC compilation and
mask/identity initialization. For native execution analysis, set `KDA_PROFILE=1`
and inspect `flash_kda_npu.ops.ascendc.get_last_profile()` after a call.
Profiling synchronizes the device and affects timing; leave it disabled for
end-to-end benchmarks.

Set compile options before importing `flash_kda_npu.ops.ascendc` and use a
fresh process when changing them. RTC binaries are compiled once and loaded
separately for each device/context. The native call selects the input tensor's
device and restores the caller's device; all inputs must share that device.
After resetting/destroying runtime contexts, use a fresh process.
For chunk 64, assembly and W/U share one Cube launch per slice; assembly
uses the W/U block's chunk count. `KDA_SOLVE_FUSE=0` restores separate
launches for comparison. Chunk 16/32 keeps the single-level solve.

## Benchmark

`benchmarks/` mirrors the CUDA suite's structure: `bench.sh` sources CANN,
installs the package and runs `bench_fwd.py` (three state variants, fixed
cases, `torch.npu.Event` timing); `generate_benchmark_md.py`
runs it twice (default `H` and `--H 64`) and writes `BENCHMARK_A3.md`.

```bash
bash flash_kda_npu/benchmarks/bench.sh
# Native-oriented fixed cases only:
bash flash_kda_npu/benchmarks/bench.sh --mode fixed --H 96
# Require native execution, using an fp32 initial/final state:
bash flash_kda_npu/benchmarks/bench.sh --mode fixed --state fp32 --require-native
# Short native smoke run, with machine-readable timings:
python flash_kda_npu/benchmarks/bench_fwd.py --T 128 --H 4 \
    --warmup 1 --iters 2 --repeats 1 --json /tmp/npu-bench.json
# Generate BENCHMARK_A3.md (after installation and CANN setup):
python flash_kda_npu/benchmarks/generate_benchmark_md.py --mode fixed
```

Fixed-length mode is the default. Varlen selectors and unsupported dimensions
or lengths produce an explicit error rather than a fallback benchmark.
All three state variants (bf16, no state, fp32) execute AscendC.
The script defaults to `H=96`, 30 warmups, 200 iterations, and 5 repeats;
`--gate-dtype` accepts `bf16`, matching the CUDA public contract.
`--require-native` remains accepted for compatibility; native execution is
always required. The report filename/device label does
not change the RTC hardware target.

State conversion and buffer allocation are outside the timed callable. Each
variant reports the backend (`ascendc`), device-event
latency, and mean wall latency per call. An untimed preflight checks that outputs
and requested final states are finite. `--state` selects one state variant;
`--T` overrides the fixed sequence length. JSON output preserves
backend and timing metadata, and Markdown reports label each variant's backend.

## Precision

The native K2 recurrence keeps inter-chunk state in fp32, including when the
final-state buffer is bf16. Only the final store rounds that state to bf16.
Calls without a requested final state skip its allocation and writeback; the
last chunk does not publish an unused intermediate state. Gate activation uses
the FlashKDA default lower bound `-5.0`.  Numerical tests compare against the
independent CPU fp64 token recurrence with `rtol=atol=5e-3` for outputs and
ordinary final-state cases. Like the CUDA gate sweep, gate-extreme final states
use relative RMS error below 0.5%, since weak decay accumulates bf16 rounding.
`lower_bound` is passed
through the API; `-5.0` is the default rather than a hardcoded kernel constant.

## Tests

```bash
python -m pytest -q flash_kda_npu/tests/
# Matching CUDA test entry points (from a repository checkout):
bash flash_kda_npu/tests/test.sh
bash flash_kda_npu/tests/run_test_full.sh
# CUDA-scale 128K/1M-token native cases (opt-in):
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
    flash_kda_npu/tests/test_fwd.py -k native_long --run-long
# Test alternate native geometries in separate processes:
KDA_CHUNK=16 python -m pytest -q flash_kda_npu/tests/test_fwd.py -k configured_chunk
KDA_CHUNK=32 python -m pytest -q flash_kda_npu/tests/test_fwd.py -k configured_chunk
```

`test_fwd.py` covers native state input/output combinations, bf16/fp32 initial
and final states, rejection of mixed state types, gates, batch/head mapping,
gate extremes, multiple chunks, alternate chunk sizes, and opt-in long
sequences. These tests verify
the persistent kernel launched. It also checks launcher ABI validation and
device switching when two NPUs are available. `test_fwd_full.py` checks the
public tensor contract, unsupported shapes, state dtype consistency, buffer
identity and exception propagation. Contract tests run with metadata doubles
on CPU hosts and with real NPU tensors when available; native numerical tests
are skipped on hosts without an NPU. The lower-level tests also cover FP32
gates, which are internal to AscendC rather than public API inputs.
Reference implementations and test helpers are in
`tests/torch_ref.py`. As with the CUDA suite, tests are run from a repository
checkout rather than installed as part of the package.

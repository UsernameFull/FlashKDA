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
|   |-- flashkda_reference.py
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

`flash_kda.fwd` sends non-CUDA tensors to `flash_kda_npu.ops.fwd`.  On NPU it
first checks whether the native AscendC path can run and otherwise falls back to
the chunked PyTorch implementation.

The public API writes `out` and the optional `final_state` in place and returns
`None`. Q/K are normalized internally; pass raw gate values and beta logits,
since gate activation and sigmoid(beta) are also performed internally.

The native path through the public API currently requires:

- NPU tensors with matching `q/k/v` shape `[B,T,H,128]`, `T >= 64`,
  and `T % 64 == 0`;
- fixed-length batches (`cu_seqlens is None`);
- bf16 `q/k/v`, and a contiguous bf16 `out` of the same shape;
- bf16 or fp32 `g` with shape `[B,T,H,128]`, `beta` with shape `[B,T,H]`,
  `A_log` with shape `[H]`, and `dt_bias` with shape `[H,128]`;
- fp32 `final_state` with shape `[B,H,128,128]` when requested;
- `initial_state`, when present, with shape `[B,H,128,128]`.

The wrapper makes input tensors contiguous and converts `g`, `beta`,
`A_log`, `dt_bias`, and `initial_state` to fp32 before calling the native
function. A noncontiguous fp32 final-state buffer is handled through a
contiguous temporary and copied back. Keep all tensors on the same NPU.

The public dispatch gate always checks divisibility by 64, even when
`KDA_CHUNK` is 16 or 32. The lower-level
`flash_kda_npu.ops.ascendc.kda_bt16_fwd_ascendc` accepts lengths divisible
by the configured chunk size and returns `(out, final_state)`; it requires
fp32 gate/beta/parameters/initial state directly and uses `bias` instead of
`dt_bias`.

Varlen inputs, lengths rejected by the dispatch gate, unsupported dimensions,
and native input-validation/import failures fall back to chunked PyTorch.
A bf16 final-state buffer also takes this fallback. RTC and launch
`RuntimeError`s propagate rather than silently falling back. The fallback can
run on CPU as well and uses 16-token chunks to keep inverse gates finite
at the default lower bound.

## Configuration and profiling

The default native chunk size is 64. The public API requires `T % 64 == 0`,
including when the native chunk size is configured to 16 or 32.

Warm up before benchmarking to exclude first-use RTC compilation and
mask/identity initialization. For native execution analysis, set `KDA_PROFILE=1`
and inspect `flash_kda_npu.ops.ascendc.get_last_profile()` after a call.
Profiling synchronizes the device and affects timing; leave it disabled for
end-to-end benchmarks.

## Benchmark

`benchmarks/` mirrors the CUDA suite's structure: `bench.sh` sources CANN,
installs the package and runs `bench_fwd.py` (three state variants, fixed
and varlen cases, `torch.npu.Event` timing); `generate_benchmark_md.py`
runs it twice (default `H` and `--H 64`) and writes `BENCHMARK_A3.md`.

```bash
bash flash_kda_npu/benchmarks/bench.sh
# Native-oriented fixed cases only:
bash flash_kda_npu/benchmarks/bench.sh --mode fixed --H 96
# Require native execution, using an fp32 initial/final state:
bash flash_kda_npu/benchmarks/bench.sh --mode fixed --state fp32 --require-native
# Short fixed/varlen smoke run, with machine-readable timings:
python flash_kda_npu/benchmarks/bench_fwd.py --T 128 --H 4 \
    --seq-lens 17 33 65 --warmup 1 --iters 2 --repeats 1 --json /tmp/npu-bench.json
# Generate BENCHMARK_A3.md (after installation and CANN setup):
python flash_kda_npu/benchmarks/generate_benchmark_md.py --mode fixed
```

Varlen cases exercise the PyTorch fallback path (the native AscendC path
requires fixed-length batches), so their timings are not comparable to
the fixed case. The bf16-state variant also uses fallback because its final
state is bf16; the no-state and fp32-state fixed variants can use AscendC.
The script defaults to `H=96`, 30 warmups, 200 iterations, and 5 repeats;
`--gate-dtype` accepts `bf16` or `fp32`. The report filename/device label does
not change the RTC hardware target.

State conversion and buffer allocation are outside the timed callable. Each
variant reports the observed backend (`ascendc` or `torch_fallback`), device-event
latency, and mean wall latency per call. An untimed preflight checks that outputs
and requested final states are finite. `--state` selects one state variant;
`--T` and `--seq-lens` override fixed and varlen lengths. JSON output preserves
backend and timing metadata, and Markdown reports label each variant's backend.

## Precision

The native K2 recurrence keeps inter-chunk state in fp32.  Gate activation uses
the FlashKDA default lower bound `-5.0`.  Numerical tests compare against the
independent CPU fp64 token recurrence with `rtol=atol=5e-3` for outputs and
ordinary final-state cases. Like the CUDA gate sweep, gate-extreme final states
use relative RMS error below 0.5%, since weak decay accumulates bf16 rounding.
`lower_bound` is passed
through the API; `-5.0` is the default rather than a hardcoded kernel constant.

## Tests

```bash
python -m pytest -q flash_kda_npu/tests/
# CUDA-scale 128K/1M-token native cases (opt-in):
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
    flash_kda_npu/tests/test_fwd.py -k native_long --run-long
# Test alternate native geometries in separate processes:
KDA_CHUNK=16 python -m pytest -q flash_kda_npu/tests/test_fwd.py -k configured_chunk
KDA_CHUNK=32 python -m pytest -q flash_kda_npu/tests/test_fwd.py -k configured_chunk
```

`test_fwd.py` covers native state input/output combinations, bf16/fp32 initial
states and gates, batch/head mapping, gate extremes, multiple chunks, alternate
chunk sizes, and opt-in long sequences. These tests reject fallback and verify
the persistent kernel launched. `test_fwd_full.py` checks fallback state layout,
sequence boundaries, strong decay, bf16 final states, and dispatch failure
policy. Fallback cases run on both CPU and NPU; NPU cases are skipped on hosts
without an NPU. The independent oracle is in `tests/helpers.py`; the existing
chunked reference remains in `tests/flashkda_reference.py`.

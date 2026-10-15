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
|   |-- cuda_cases.py
|   |-- run_test_full.sh
|   |-- test.sh
|   |-- torch_ref.py
|   |-- test_fwd.py
|   |-- test_fwd_full.py
|   |-- test_contract.py
|   `-- test_reference.py
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

- NPU tensors with matching `q/k/v` shape `[B,T,H,128]`;
- positive fixed-length batches, or packed variable-length inputs with `B=1`
  and contiguous NPU int64 `cu_seqlens` of shape `[N+1]`;
- bf16 `q/k/v/g/out` of the same shape, and bf16 `beta` with shape `[B,T,H]`;
- fp32 `A_log` with shape `[H]` and fp32 `dt_bias` with shape `[H,128]`;
- bf16 or fp32 `final_state` with shape `[N,H,128,128]` when requested;
- bf16 or fp32 `initial_state`, when present, with shape `[N,H,128,128]`
  (`N=B` for fixed-length batches);
- matching initial/final state dtypes when both buffers are supplied, as on CUDA;
- all tensors contiguous and on the same NPU, with positive batch/head counts.

The wrapper validates public inputs before any conversion or native launch.
It converts the validated bf16 `g`/`beta` and optional initial state to fp32
for the internal AscendC interface. It does not repair noncontiguous inputs,
or silently accept other public dtypes. All fixed-length inputs pass the
caller's input/output buffers directly to AscendC, including partial chunks.
The pre-Gram kernel reads only valid rows and zero-fills the tail in UB; K2
writes only valid output rows using the original batch stride. Only the small
head-major beta buffer needs tail padding. Unequal-length packed sequences
still use temporary native batches, then copy results into caller buffers.

Public sequence lengths need not be chunk-aligned and may be shorter than one
chunk. Packed boundaries must start at zero, end at `T`, and be nondecreasing;
empty packed sequences preserve their initial state (or return zero state).
The lower-level
`flash_kda_npu.ops.ascendc.kda_bt16_fwd_ascendc` accepts any positive fixed
sequence length and returns `(out, final_state)`; it requires
fp32 gate/beta/parameters/initial state directly and uses `bias` instead of
`dt_bias`.

The public dtype, state-buffer and failure contracts match CUDA, including
fixed and variable lengths with partial chunks. Public tensor validation failures raise
`RuntimeError`; exceptions from the native implementation retain their types.

Unequal-length sequences are grouped by length rounded up to `KDA_CHUNK` and
run as native AscendC batches. Each sequence adds fewer than one chunk of
padding; a short sequence is never padded to the longest sequence in the input.
For packed groups, Q/K/V padding is zero. The pre-Gram kernel explicitly sets padded gate
log-decay to zero before its prefix sum, so padded tokens do not decay or
update the final state. All KDA arithmetic stays in AscendC; PyTorch is used
for packing and copying, with no PyTorch KDA fallback. Equal aligned lengths
are reshaped directly into a native batch without packing.

The current packed dispatcher reads only `cu_seqlens` metadata on the host,
which synchronizes that transfer, and launches one native pipeline per rounded
length group. Packing, copies and metadata handling add overhead relative to
aligned fixed-length input, especially for many distinct lengths. Include
these costs when comparing end-to-end performance.

## Configuration and profiling

The default native chunk size is 64. Partial chunks use full computation tiles
of the configured size (16, 32 or 64), with neutral state transitions in
invalid rows. Fixed-length input tails are filled on chip, without packing
the full Q/K/V/g inputs or copying a temporary output back.

Warm up before benchmarking to exclude first-use RTC compilation and
mask/identity initialization. For native execution analysis, set `KDA_PROFILE=1`
and inspect `flash_kda_npu.ops.ascendc.get_last_profile()` after a call.
For packed calls it aggregates native launch counts and native stage timings
across all groups and reports real/padded token counts; stage timings exclude
packing and copying, whereas the benchmark includes them.
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
and variable-length cases, `torch.npu.Event` timing); `generate_benchmark_md.py`
runs it twice (default `H` and `--H 64`) and writes `BENCHMARK_A3.md`.

```bash
bash flash_kda_npu/benchmarks/bench.sh
# Native-oriented fixed cases only:
bash flash_kda_npu/benchmarks/bench.sh --mode fixed --H 96
# Require native execution, using an fp32 initial/final state:
bash flash_kda_npu/benchmarks/bench.sh --mode fixed --state fp32 --require-native
# Short native smoke run, with machine-readable timings:
python flash_kda_npu/benchmarks/bench_fwd.py --mode fixed --T 129 --H 4 \
    --warmup 1 --iters 2 --repeats 1 --json /tmp/npu-bench.json
# Packed native smoke run with partial chunks:
python flash_kda_npu/benchmarks/bench_fwd.py --mode varlen --seq-lens 17 33 65 \
    --H 4 --warmup 1 --iters 2 --repeats 1
# Generate BENCHMARK_A3.md (after installation and CANN setup):
python flash_kda_npu/benchmarks/generate_benchmark_md.py --mode fixed
```

The default mode runs both fixed and variable-length cases, as in the CUDA suite.
Use `--mode fixed` or `--mode varlen` to select one kind.
All three state variants (bf16, no state, fp32) execute AscendC.
The script defaults to `H=96`, 30 warmups, 200 iterations, and 5 repeats;
`--gate-dtype` accepts `bf16`, matching the CUDA public contract.
`--require-native` remains accepted for compatibility; native execution is
always required. The report filename/device label does
not change the RTC hardware target.

Caller state buffers and their dtype conversions are prepared outside the timed
callable; internal AscendC conversions and allocations remain inside it. Each
variant reports the backend (`ascendc`), device-event
latency, and mean wall latency per call. An untimed preflight checks that outputs
and requested final states are finite. `--state` selects one state variant;
`--T` overrides the fixed sequence length and `--seq-lens` overrides packed
lengths. JSON output preserves backend, timing, launch/group and padding
metadata, and Markdown reports label each variant's backend. Wrapper packing,
metadata transfer and copying are included in the timed callable.

## Precision

The native K2 recurrence keeps inter-chunk state in fp32, including when the
final-state buffer is bf16. Only the final store rounds that state to bf16.
Calls without a requested final state skip its allocation and writeback; the
last chunk does not publish an unused intermediate state. Gate activation uses
the FlashKDA default lower bound `-5.0`. Native diagnostic tests compare against
an independent CPU fp64 token recurrence with `rtol=atol=5e-3` for outputs and
ordinary final-state cases. Gate-extreme final states use relative RMS error
below 0.5%, since weak decay accumulates bf16 rounding.

The CUDA-size numerical matrix uses an independent CPU fp64 chunked solution
of the same recurrence. Its algebra is checked against the token recurrence,
including tiny/zero norms, empty sequences, non-default scale and extreme
gate/bias distributions. Output and final-state relative RMS error must be
below 0.5% (0.6% for packed outputs, matching the packed gate-sweep budget).
With CUDA's large `arange` initial states, output error may reach 1%
in the initial transient: BF16 Cube operands amplify
dot-product cancellation. The 0.5% output criterion still applies after the
first 64 tokens of each sequence, independently of native chunk geometry.
Output windows of at most 1024 tokens are checked separately, so a large
initial state cannot hide errors later in the sequence. Final states retain
the 0.5% criterion. The fixed/packed gate-bias sweeps use 0.5%/0.6% respectively
for outputs; final-state threshold violations produce warnings, as in CUDA.
Nonfinite outputs or states always fail.
As in CUDA's FLA comparison, the gate-bias sweep asserts global RMS and records
window errors as diagnostics.

The W/U solve exports U in FP32, and K2
keeps its Cube-to-vector D1/D2/D3 handoffs in FP32 until residual/output
addition. Cube matrix operands remain BF16. These precision changes add
FP32 U and handoff traffic; benchmark their end-to-end cost on target hardware.

These criteria measure error against mathematical fp64 results. CUDA's
`torch_ref.py` instead emulates CUDA instructions, reductions and BF16 state
rounding and requires `torch.equal`; it is not portable to Ascend. Matching
the case matrix does not establish bitwise CUDA/NPU numerical equality.
`lower_bound` is passed
through the API; `-5.0` is the default rather than a hardcoded kernel constant.

## Tests

```bash
python -m pytest -q flash_kda_npu/tests/
# Matching CUDA test entry points (from a repository checkout):
bash flash_kda_npu/tests/test.sh
bash flash_kda_npu/tests/run_test_full.sh
# Include all CUDA-scale 128K/1M-token fixed and packed cases (opt-in):
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
    flash_kda_npu/tests/ --run-long
# Test alternate native geometries in separate processes:
KDA_CHUNK=16 python -m pytest -q flash_kda_npu/tests/test_fwd.py -k configured_chunk
KDA_CHUNK=32 python -m pytest -q flash_kda_npu/tests/test_fwd.py -k configured_chunk
```

`test_fwd.py` covers native state input/output combinations, bf16/fp32 initial
and final states, rejection of mixed state types, gates, batch/head mapping,
gate extremes, multiple chunks, alternate chunk sizes, and opt-in long
sequences. It also covers partial chunks (including one token), independent
packed sequence states, empty sequences and invalid boundaries. These tests verify
the persistent kernel launched. It also checks launcher ABI validation and
device switching when two NPUs are available. It additionally mirrors CUDA's
ten gate/bias distributions with 8192-token fixed and six-sequence packed
inputs, using uniform Q/K/V and BF16-rounded random initial states.

`test_fwd_full.py` mirrors all 580 CUDA full-suite cases: 288 fixed-length,
192 packed variable-length, 96 batched, and four opt-in long cases. Parameter
names, IDs, state combinations and dtypes match CUDA. It also runs the two
seed-0 H=96 smoke scenarios from CUDA's `test_fwd.py`. Fixtures use CUDA's
distributions and generation order, including standard-normal gates, uniform
dt_bias and `arange -> BF16 -> requested dtype` states. They are generated on
CPU; CPU and CUDA random engines produce different bytes. Existing native
diagnostic fixtures and their pointwise/bitwise assertions remain in place.
Every numerical test checks that the persistent AscendC kernel launched.

`test_contract.py` checks the public tensor contract, unsupported shapes,
state dtype consistency, buffer identity and exception propagation. Contract
tests run with metadata doubles
on CPU hosts and with real NPU tensors when available; native numerical tests
are skipped on hosts without an NPU. The lower-level tests also cover FP32
gates, which are internal to AscendC rather than public API inputs.
`test_reference.py` checks the fp64 oracle and verifies the matrix constants
and pytest parametrization against CUDA's source through AST comparison,
without importing CUDA-only helpers. Reference implementations and test
helpers are in `tests/torch_ref.py`. As with the CUDA suite, tests run from a repository
checkout rather than installed as part of the package.

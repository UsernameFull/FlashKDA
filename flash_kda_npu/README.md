# FlashKDA Ascend NPU backend

This directory adds the Ascend NPU implementation for FlashKDA.  The native
path uses the AscendC v1 backend ported from the `kda-ascendc` release: the
host launcher exposes RTC compilation/launch APIs and
`flash_kda_npu/ops/ascendc.py` orchestrates the three production
stages (`pre_gram`, `solve`, and persistent-loop `k2`).

The NPU path is production-only: the diagnostic scheduling modes, debug
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
`-- ops/
    |-- ascendc.py
    `-- kda_fwd.py
```

## Build

On an Ascend host with CANN configured:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash flash_kda_npu/csrc/build.sh
```

The launcher is packaged in `flash_kda_npu/lib`.  AscendC kernel sources in
`csrc/ascendc/` are RTC compiled into the launcher process on first use.  The
production chunk size is fixed by `KDA_CHUNK=64`, which is also the default.

To install the NPU backend with pip from a repository checkout:

```bash
pip install --no-build-isolation .
```

`--no-build-isolation` uses the existing PyTorch installation; on an NPU host
this should already include `torch_npu`.  A Git checkout can also be installed
directly with `pip install --no-build-isolation git+https://github.com/UsernameFull/FlashKDA.git@npu`.

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

The native path currently requires:

- NPU tensors with `K = V = 128` and `T % 64 == 0`;
- fixed-length batches (`cu_seqlens is None`);
- contiguous bf16 `q/k/v/out`;
- bf16 or fp32 `g`, fp32 `A_log/dt_bias`, and a contiguous fp32
  `final_state` when one is requested;
- `initial_state`, when present, in fp32 or bf16 (converted for the kernel).

Unsupported shapes and varlen inputs fall back to PyTorch.

## Precision

The native K2 recurrence keeps inter-chunk state in fp32.  Gate activation uses
the FlashKDA default lower bound `-5.0`.  Numerical tests compare against the
fp64/fp32 PyTorch reference with `rtol=atol=5e-3`.

## Tests

```bash
python -m pytest -q flash_kda_npu/tests/
```

Naming mirrors the CUDA suite: `test_fwd.py` checks the native path
against the pure-PyTorch reference (`flashkda_reference.py`) on an NPU
host; `test_fwd_full.py` covers dispatch decisions and the PyTorch
fallback (varlen, unsupported dimensions), and runs on hosts without an
NPU as well.

## Benchmark

```bash
python tools/bench_npu_dispatch.py --batch 1 --steps 8192 --heads 96 --dim 128 --gate-dtype fp32
```

The default benchmark uses the FlashKDA bf16 gate.  Passing `--gate-dtype fp32`
measures the release-kernel aligned input.  On the container's Ascend A3
device with 10 warmup and 20 timed iterations:

- release-aligned fp32 `g`: `min 7.23 ms / median 7.27 ms`;
- standard bf16 `g` (one dtype conversion): `min 7.66 ms / median 7.68 ms`.

Timing uses `torch.npu.Event`; `triton.testing.do_bench` is not used
because this container's Triton-Ascend build fails kernel compilation.

# FlashKDA Ascend NPU backend

This directory adds the Ascend NPU implementation for FlashKDA.  The active
native path is the AscendC v1 backend ported from the `kda-ascendc` release:
the launcher exposes RTC compilation/launch APIs and
`flash_kda_npu/kda_ascendc_v1/api.py` orchestrates the three production
stages (`pre_gram`, `solve`, and persistent-loop `k2`).

The older `cpp_extension_asc` AOT scaffold and `csrc/CMakeLists.txt` are kept
for the follow-on CMake/AOT migration.  They are not used by the current
`ops/kda_fwd.py` dispatch path.

## Build

On an Ascend host with CANN configured:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export LAUNCHER_NAME=kda_ascendc_v1_launcher
bash flash_kda_npu/launcher/build_launcher.sh
```

The launcher is cached under `build/S02_clean/torch_extensions`.  AscendC
kernel sources live in `flash_kda_npu/csrc/kernels/v1/`.  They are RTC compiled
into the launcher process on first use.  The production chunk size is fixed by
`KDA_CHUNK=64`, which is also the default.

## Usage

```python
import torch
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
python -m pytest -q tests/test_npu_dispatch.py tests/test_npu_precision.py
```

The precision test covers the native path on an NPU host.  Dispatch/fallback
tests also run on hosts without NPU.

## Benchmark

```bash
python tools/bench_npu_dispatch.py --batch 1 --steps 8192 --heads 96 --dim 128 --gate-dtype fp32
```

The default benchmark uses the FlashKDA bf16 gate.  Passing `--gate-dtype fp32`
measures the release-kernel aligned input.  On the container's Ascend A3
device with 10 warmup and 20 timed iterations:

- release-aligned fp32 `g`: `min 7.32 ms / median 7.34 ms`;
- standard bf16 `g` (one dtype conversion): `min 7.62 ms / median 7.64 ms`.

The porting-source release path measured approximately `7.14 ms` with fp32
`g`.  Timing uses `torch.npu.Event`; `triton.testing.do_bench` is not used
because this container's Triton-Ascend build fails kernel compilation.

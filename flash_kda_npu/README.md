# FlashKDA Ascend NPU Backend

This package provides the Ascend NPU path for FlashKDA. It uses an AOT-built
`TORCH_LIBRARY` extension when CANN is available and falls back to a pure
PyTorch implementation for unsupported shapes or hosts without the extension.

## Requirements

- PyTorch 2.4 or newer
- CANN with the AscendC AOT toolchain (target: Ascend 910C/A3)
- `ASCEND_HOME_PATH` pointing to the CANN installation
- `torch_npu` when running on an NPU

## Build

Install from the repository root:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
pip install -v --no-build-isolation .
```

`setup.py` detects `ASCEND_HOME_PATH` and invokes
`flash_kda_npu/csrc/CMakeLists.txt`. CMake builds `kda_k1.cpp`, `kda_k2.cpp`,
and `kda_host.cpp` into `libflash_kda_npu_kernels`. The compiled library is
packaged under `flash_kda_npu/cpp_extension_asc/lib/`.

The default target architecture is `dav-c90`. Override it for a different CANN
toolchain with:

```bash
cmake -DCMAKE_ASC_ARCHITECTURES=dav-c90 ...
```

The kernel sources are ordinary `.cpp` files. They use the
`FLASH_KDA_ASC_OPERATOR` entry-point contract provided by the CANN ASC CMake
package. `kda_host.cpp` registers the operator through `TORCH_LIBRARY` and also
provides a complete ATen composite implementation for hosts where the AOT kernel
package is unavailable.

## API

The top-level API remains:

```python
import flash_kda

flash_kda.fwd(q, k, v, g, beta, scale, out, A_log=A_log, dt_bias=dt_bias,
              lower_bound=-5.0, initial_state=h0, final_state=h1,
              cu_seqlens=cu_seqlens)
```

CUDA tensors use the CUDA kernel. Other devices, including NPU, use this
backend. The NPU path supports the standard FlashKDA signatures and
fixed-length and `cu_seqlens` layouts.

## Precision

`lower_bound` is constrained to `[-5, 0]` by the native backend. State updates
accumulate in fp32. When neither `initial_state` nor `final_state` is fp32,
the chunk-to-chunk state is stored in bf16. Numerical validation uses
`atol=rtol=5e-3`.

## Tests

Run host-side dispatch and fallback tests with:

```bash
pytest tests/test_npu_dispatch.py
```

Precision, fallback, and A3 benchmark validation require an NPU host. The A3
baseline should use `B=1, T=8192, H=96, K=V=128` and compare against the
Triton-Ascend path before release.

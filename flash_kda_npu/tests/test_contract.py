"""CUDA-compatible public validation and native-only failure policy.

Metadata doubles exercise dispatch on CPU hosts. The same contract cases also
run with real NPU tensors; numerical coverage lives in test_fwd{,_full}.py.
"""
from types import SimpleNamespace

import pytest
import torch

import flash_kda
from flash_kda_npu.ops import kda_fwd


class _TensorMetadata:
    is_cuda = False

    def __init__(self, tensor, device):
        self.shape = tensor.shape
        self.ndim = tensor.ndim
        self.dtype = tensor.dtype
        self.device = device
        self._contiguous = tensor.is_contiguous()
        self._numel = tensor.numel()

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return self._numel


@pytest.fixture(params=["metadata", "npu"])
def contract(request, monkeypatch):
    real_npu = request.param == "npu"
    if real_npu and not kda_fwd.is_available():
        pytest.skip("requires an Ascend NPU")

    def tensor(shape, dtype=torch.bfloat16, *, contiguous=True, device="npu:0"):
        value = torch.zeros(shape, dtype=dtype, device=device if real_npu else "cpu")
        if not contiguous:
            value = value.transpose(-1, -2).contiguous().transpose(-1, -2)
            assert not value.is_contiguous()
        if real_npu or device == "cpu":
            return value
        return _TensorMetadata(value, SimpleNamespace(type="npu", index=int(device[-1])))

    args = {name: tensor((2, 128, 2, 128)) for name in ("q", "k", "v", "g", "out")}
    args.update(beta=tensor((2, 128, 2)), scale=128 ** -0.5,
                A_log=tensor((2,), torch.float32), dt_bias=tensor((2, 128), torch.float32),
                lower_bound=-5.0)

    def refuse_launch(*args, **kwargs):
        pytest.fail("invalid public input reached native execution")

    monkeypatch.setattr(kda_fwd, "_run_ascendc", refuse_launch)
    return args, tensor


@pytest.mark.parametrize("name", ["q", "k", "v", "g", "beta", "out", "A_log", "dt_bias",
                                  "initial_state", "final_state"])
def test_rejects_wrong_dtype(contract, name):
    args, tensor = contract
    shape = (2, 2, 128, 128) if name.endswith("state") else args[name].shape
    args[name] = tensor(shape, torch.float64)
    with pytest.raises(RuntimeError, match=f"{name} must be"):
        flash_kda.fwd(**args)


@pytest.mark.parametrize("name", ["q", "k", "v", "g", "beta", "out", "A_log", "dt_bias"])
def test_rejects_implicit_dtype_conversion(contract, name):
    args, tensor = contract
    dtype = torch.bfloat16 if name in ("A_log", "dt_bias") else torch.float32
    args[name] = tensor(args[name].shape, dtype)
    with pytest.raises(RuntimeError, match=f"{name} must be"):
        flash_kda.fwd(**args)


@pytest.mark.parametrize("name", ["q", "k", "v", "g", "beta", "out", "dt_bias",
                                  "initial_state", "final_state"])
def test_rejects_noncontiguous(contract, name):
    args, tensor = contract
    shape = (2, 2, 128, 128) if name.endswith("state") else args[name].shape
    dtype = torch.float32 if name in ("dt_bias", "initial_state", "final_state") else torch.bfloat16
    args[name] = tensor(shape, dtype, contiguous=False)
    with pytest.raises(RuntimeError, match=f"{name} must be contiguous"):
        flash_kda.fwd(**args)


@pytest.mark.parametrize("name,shape", [
    ("q", (128, 2, 128)), ("k", (2, 64, 2, 128)),
    ("v", (2, 128, 1, 128)), ("g", (2, 128, 2)),
    ("out", (2, 128, 2, 64)), ("beta", (2, 128, 2, 1)),
    ("A_log", (2, 1)), ("dt_bias", (128, 2)),
    ("initial_state", (1, 2, 128, 128)), ("final_state", (2, 2, 128)),
])
def test_rejects_wrong_shape(contract, name, shape):
    args, tensor = contract
    dtype = torch.float32 if name in ("A_log", "dt_bias", "initial_state", "final_state") else torch.bfloat16
    args[name] = tensor(shape, dtype)
    with pytest.raises(RuntimeError, match=f"{name} must"):
        flash_kda.fwd(**args)


@pytest.mark.parametrize("name", ["q", "k", "v", "g", "beta", "out", "A_log", "dt_bias",
                                  "initial_state", "final_state"])
def test_rejects_cpu_tensors(contract, name):
    args, tensor = contract
    shape = (2, 2, 128, 128) if name.endswith("state") else args[name].shape
    dtype = torch.float32 if name in ("A_log", "dt_bias", "initial_state", "final_state") else torch.bfloat16
    args[name] = tensor(shape, dtype, device="cpu")
    with pytest.raises(RuntimeError, match=f"{name} must be an NPU tensor"):
        flash_kda.fwd(**args)


@pytest.mark.parametrize("initial_dtype,final_dtype", [
    (torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16),
])
def test_rejects_mixed_state_dtypes(contract, initial_dtype, final_dtype):
    args, tensor = contract
    args["initial_state"] = tensor((2, 2, 128, 128), initial_dtype)
    args["final_state"] = tensor((2, 2, 128, 128), final_dtype)
    with pytest.raises(RuntimeError, match="must have the same dtype"):
        flash_kda.fwd(**args)


def test_rejects_empty_fixed_length(contract):
    args, tensor = contract
    for name in ("q", "k", "v", "g", "out"):
        args[name] = tensor((2, 0, 2, 128))
    args["beta"] = tensor((2, 0, 2))
    with pytest.raises(RuntimeError, match="T > 0 for fixed-length"):
        flash_kda.fwd(**args)


def test_rejects_unsupported_dimension(contract):
    args, tensor = contract
    for name in ("q", "k", "v", "g", "out"):
        args[name] = tensor((2, 128, 2, 64))
    args["dt_bias"] = tensor((2, 64), torch.float32)
    with pytest.raises(RuntimeError, match="only supports D == 128"):
        flash_kda.fwd(**args)


@pytest.mark.parametrize("shape,dtype,match", [
    ((3,), torch.int32, "must be int64"),
    ((2, 2), torch.int64, "must be 1D"),
    ((1,), torch.int64, "at least 2 elements"),
])
def test_rejects_invalid_cu_seqlens_contract(contract, shape, dtype, match):
    args, tensor = contract
    for name in ("q", "k", "v", "g", "out"):
        args[name] = tensor((1, 128, 2, 128))
    args["beta"] = tensor((1, 128, 2))
    args["cu_seqlens"] = tensor(shape, dtype)
    with pytest.raises(RuntimeError, match=match):
        flash_kda.fwd(**args)


def test_rejects_varlen_batch(contract):
    args, tensor = contract
    args["cu_seqlens"] = tensor((3,), torch.int64)
    with pytest.raises(RuntimeError, match="B must be 1"):
        flash_kda.fwd(**args)


@pytest.mark.parametrize("offsets,match", [
    ([1, 64, 128], "start at 0"), ([0, 64, 127], "end at T"),
    ([0, 90, 64, 128], "nondecreasing"), ([0, -1, 128], "nondecreasing"),
])
def test_invalid_cu_seqlens_values(offsets, match):
    with pytest.raises(RuntimeError, match=match):
        kda_fwd._sequence_offsets(torch.tensor(offsets, dtype=torch.int64), 128)


def test_rejects_cross_device_inputs(contract):
    args, tensor = contract
    if not isinstance(args["q"], _TensorMetadata) and torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    args["k"] = tensor(args["k"].shape, device="npu:1")
    with pytest.raises(RuntimeError, match="all inputs must be on q's NPU device"):
        flash_kda.fwd(**args)


@pytest.mark.parametrize("error_type", [ImportError, ValueError, TypeError, RuntimeError])
def test_native_failures_propagate(contract, monkeypatch, error_type):
    args, _ = contract
    error = error_type("injected native failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(kda_fwd, "_run_ascendc", fail)
    with pytest.raises(error_type, match="injected native failure") as caught:
        flash_kda.fwd(**args)
    assert caught.value is error


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_in,has_out", [(False, False), (True, False), (False, True), (True, True)])
def test_valid_state_dispatch(contract, monkeypatch, dtype, has_in, has_out):
    args, tensor = contract
    if has_in:
        args["initial_state"] = tensor((2, 2, 128, 128), dtype)
    if has_out:
        args["final_state"] = tensor((2, 2, 128, 128), dtype)
    calls = []
    monkeypatch.setattr(kda_fwd, "_run_ascendc", lambda *a: calls.append(a))
    assert flash_kda.fwd(**args) is None
    assert len(calls) == 1
    # The public wrapper passes the caller's buffers through unchanged.
    assert calls[0][6] is args["out"]
    assert calls[0][10] is args.get("initial_state")
    assert calls[0][11] is args.get("final_state")

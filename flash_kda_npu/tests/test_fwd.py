"""Native AscendC coverage; fallback is a failure, never a passing test."""
import pytest
import torch

import flash_kda
from flash_kda_npu.ops import kda_fwd
from flash_kda_npu.tests.torch_ref import (
    STATE_IDS, STATE_IO, assert_close, assert_relative_rms, make_inputs, make_state, recurrent_reference,
)

pytestmark = pytest.mark.skipif(not kda_fwd.is_available(), reason="requires an Ascend NPU")


@pytest.fixture
def native_only(monkeypatch):
    def refuse_fallback(*args, **kwargs):
        pytest.fail("native case unexpectedly selected the PyTorch fallback")
    monkeypatch.setattr(kda_fwd, "_torch_fallback", refuse_fallback)


def check_native(batch, steps, heads, state_dtype=torch.float32,
                 has_in=True, has_out=True, gate_dtype=torch.bfloat16,
                 gate_value=None, lower_bound=-5.0):
    inputs = make_inputs(batch, steps, heads, "npu", gate_dtype=gate_dtype)
    if gate_value is not None:
        inputs[3].fill_(gate_value)
    initial = make_state(batch, heads, 128, state_dtype, "npu") if has_in else None
    saved_initial = initial.clone() if initial is not None else None
    final = torch.full((batch, heads, 128, 128), float("nan"), dtype=state_dtype,
                       device="npu") if has_out else None
    out = torch.full_like(inputs[2], float("nan"))
    q, k, v, g, beta, a_log, bias = inputs
    result = flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out,
                           A_log=a_log, dt_bias=bias, lower_bound=lower_bound,
                           initial_state=initial, final_state=final)
    torch.npu.synchronize()
    assert result is None
    from flash_kda_npu.ops.ascendc import get_last_profile
    assert get_last_profile()["launch_counts"].get("kda_k2_persistent_loop", 0) == 1
    expected_out, expected_state = recurrent_reference(inputs, saved_initial, lower_bound=lower_bound)
    assert_close(out, expected_out)
    if final is not None:
        # CUDA gate sweeps use relative RMS: weak decay accumulates bf16
        # rounding across tokens, making per-element relative error unstable.
        if gate_value is not None:
            assert_relative_rms(final, expected_state)
        else:
            assert_close(final, expected_state)
    if initial is not None:
        torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)


@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IDS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,steps,heads", [(1, 64, 1), (1, 256, 4), (2, 128, 4)])
def test_native_state_io(native_only, batch, steps, heads, state_dtype, has_in, has_out):
    check_native(batch, steps, heads, state_dtype, has_in, has_out)


@pytest.mark.parametrize("heads", [1, 4, 32, 96])
def test_native_head_mapping(native_only, heads):
    check_native(1, 64, heads)


@pytest.mark.parametrize("batch,steps,heads", [(4, 256, 1), (8, 128, 4)])
def test_native_batched(native_only, batch, steps, heads):
    check_native(batch, steps, heads)


@pytest.mark.parametrize("unroll", [1, 3, 8])
@pytest.mark.parametrize("with_bias", [False, True])
def test_native_head_parameters(native_only, monkeypatch, unroll, with_bias):
    """Refresh cached metadata across heads, batches and the padded tail."""
    from flash_kda_npu.ops.ascendc import CHUNK, kda_bt16_fwd_ascendc
    monkeypatch.setenv("KDA_PRE_UNROLL", str(unroll))
    inputs = list(make_inputs(2, CHUNK * 5, 3, "npu", gate_dtype=torch.float32))
    inputs[5] = torch.tensor([-1.0, 0.25, 1.0], device="npu")
    inputs[6] = torch.linspace(-0.5, 0.5, 3 * 128, device="npu").reshape(3, 128)
    if not with_bias:
        inputs[6].zero_()
    q, k, v, g, beta, a_log, bias = inputs
    out, state = kda_bt16_fwd_ascendc(
        q, k, v, g, beta.float(), A_log=a_log,
        bias=bias if with_bias else None, output_final_state=True)
    torch.npu.synchronize()
    expected_out, expected_state = recurrent_reference(inputs)
    assert_close(out, expected_out)
    assert_close(state, expected_state)


@pytest.mark.parametrize("blocks", [1, 2, 4])
def test_native_reused_handoff_slots(native_only, monkeypatch, blocks):
    """Exercise repeated handoffs with one, two or four heads per block."""
    monkeypatch.setenv("KDA_PERSIST_LOOP_BLOCKS", str(blocks))
    check_native(1, 512, 4, gate_dtype=torch.float32)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_value", [-8.0, -4.0, 0.0, 8.0])
def test_native_gate_sweep(native_only, state_dtype, gate_dtype, gate_value):
    check_native(1, 128, 2, state_dtype=state_dtype, gate_dtype=gate_dtype, gate_value=gate_value)


@pytest.mark.parametrize("lower_bound", [-1.0, -3.0])
def test_native_lower_bound(native_only, lower_bound):
    check_native(1, 128, 2, lower_bound=lower_bound)


@pytest.mark.parametrize("steps", [1024, 8192])
def test_native_multichunk(native_only, steps):
    check_native(1, steps, 1)


@pytest.mark.parametrize("steps", [131072, 1048576])
def test_native_long(native_only, request, steps):
    if not request.config.getoption("--run-long"):
        pytest.skip("use --run-long for the CUDA-scale long-sequence cases")
    check_native(1, steps, 1)


@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_native_configured_chunk(native_only, chunk):
    from flash_kda_npu.ops.ascendc import CHUNK, kda_bt16_fwd_ascendc
    if CHUNK != chunk:
        pytest.skip(f"requires a fresh process with KDA_CHUNK={chunk}")
    inputs = make_inputs(1, chunk * 3, 2, "npu", gate_dtype=torch.float32)
    q, k, v, g, beta, a_log, bias = inputs
    out, state = kda_bt16_fwd_ascendc(q, k, v, g, beta.float(),
                                    A_log=a_log, bias=bias, output_final_state=True)
    torch.npu.synchronize()
    expected_out, expected_state = recurrent_reference(inputs)
    assert_close(out, expected_out)
    assert_close(state, expected_state)


@pytest.mark.parametrize("initial_dtype,final_dtype", [
    (torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32),
])
def test_native_mixed_state_dtypes(native_only, initial_dtype, final_dtype):
    # The public dispatch requires multiples of 64 at every native chunk size.
    inputs = make_inputs(2, 320, 3, "npu")
    q, k, v, g, beta, a_log, bias = inputs
    initial = make_state(2, 3, 128, initial_dtype, "npu")
    final = torch.empty_like(initial, dtype=final_dtype)
    out = torch.empty_like(q)
    flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, A_log=a_log,
                  dt_bias=bias, lower_bound=-5.0, initial_state=initial, final_state=final)
    torch.npu.synchronize()
    expected_out, expected_state = recurrent_reference(inputs, initial)
    assert_close(out, expected_out)
    assert_close(final, expected_state)


@pytest.mark.parametrize("chunks", [1, 5, 128])
def test_native_final_state_rounding(native_only, chunks):
    from flash_kda_npu.ops.ascendc import CHUNK, kda_bt16_fwd_ascendc
    q, k, v, g, beta, a_log, bias = make_inputs(
        1, CHUNK * chunks, 2, "npu", gate_dtype=torch.float32)
    kwargs = dict(A_log=a_log, bias=bias, output_final_state=True)
    out32, state32 = kda_bt16_fwd_ascendc(q, k, v, g, beta.float(), **kwargs)
    final16 = torch.empty_like(state32, dtype=torch.bfloat16)
    out16, state16 = kda_bt16_fwd_ascendc(
        q, k, v, g, beta.float(), final_state=final16, **kwargs)
    out_none, no_state = kda_bt16_fwd_ascendc(
        q, k, v, g, beta.float(), A_log=a_log, bias=bias)
    torch.npu.synchronize()
    assert state16.data_ptr() == final16.data_ptr()
    assert no_state is None
    torch.testing.assert_close(out16, out32, rtol=0, atol=0)
    torch.testing.assert_close(out_none, out32, rtol=0, atol=0)
    torch.testing.assert_close(state16, state32.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("bad_args,match", [
    ([], "expects 5 arguments"),
    ([b"\0"] + [b"\0" * 8] * 3 + [b"\0" * 4], "argument 0 expects 8 bytes"),
    ([b"\0" * 8] * 5, "argument 4 expects 4 bytes"),
])
def test_launcher_rejects_invalid_abi(bad_args, match):
    from flash_kda_npu.ops.ascendc import _compile_all, launch_argsarray_engine
    stream = torch.npu.current_stream().npu_stream
    _compile_all()
    with pytest.raises(ValueError, match=match):
        launch_argsarray_engine("kda_solve_wu_wide", 1, stream, bad_args, 0)


def test_launcher_cache_identity():
    from flash_kda_npu.ops.ascendc import _SOURCES, _defines, ROOT, rtc_compile
    torch.npu.current_stream()
    rel, name, sizes = next(entry for entry in _SOURCES if entry[1] == "kda_solve_wu_wide")
    source = _defines() + (ROOT / rel).read_text(encoding="utf-8-sig")
    first = rtc_compile(source, name, sizes)
    assert rtc_compile(source, name, sizes) == first
    with pytest.raises(ValueError, match="different source/options/ABI"):
        rtc_compile(source, name, sizes[:-1])
    with pytest.raises(ValueError, match="different source/options/ABI"):
        rtc_compile(source + "\n// changed source", name, sizes)


def test_native_device_switch(native_only):
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    from flash_kda_npu.ops.ascendc import CHUNK, _COMPILED, kda_bt16_fwd_ascendc
    previous = torch.npu.current_device()
    try:
        for device in (0, 1, 0):
            inputs = make_inputs(1, CHUNK * 3, 2, f"npu:{device}", gate_dtype=torch.float32)
            q, k, v, g, beta, a_log, bias = inputs
            # Call with a different current device; the function must restore it.
            torch.npu.set_device(1 - device)
            out, state = kda_bt16_fwd_ascendc(
                q, k, v, g, beta.float(), A_log=a_log, bias=bias, output_final_state=True)
            assert torch.npu.current_device() == 1 - device
            torch.npu.synchronize(device)
            expected_out, expected_state = recurrent_reference(inputs)
            assert_close(out, expected_out)
            assert_close(state, expected_state)
        assert {key[1] for key in _COMPILED}.issuperset({0, 1})
    finally:
        torch.npu.set_device(previous)


def test_native_rejects_cross_device_inputs():
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    from flash_kda_npu.ops.ascendc import CHUNK, kda_bt16_fwd_ascendc
    q, k, v, g, beta, a_log, bias = make_inputs(1, CHUNK, 1, "npu:0", gate_dtype=torch.float32)
    with pytest.raises(ValueError, match="all inputs must be on q's NPU device"):
        kda_bt16_fwd_ascendc(q, k.to("npu:1"), v, g, beta.float(), A_log=a_log, bias=bias)

"""Native NPU forward benchmarks with CUDA-compatible public inputs."""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # Register the NPU backend before using torch.npu.

import flash_kda

D_DEFAULT = 128
LOWER_BOUND = -5.0
VARLEN_CASES = [[1300, 547, 2048, 963, 271, 3063], [1024] * 8]


def bench_fn(fn, warmup, iters, repeats):
    for _ in range(max(warmup, 1)):
        fn()
    torch.npu.synchronize()
    device_ms, wall_ms = [], []
    for _ in range(repeats):
        starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
        torch.npu.synchronize()
        begin = time.perf_counter()
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.npu.synchronize()
        wall_ms.append((time.perf_counter() - begin) * 1000 / iters)
        device_ms.extend(start.elapsed_time(end) for start, end in zip(starts, ends))
    return {
        "mean_ms": sum(device_ms) / len(device_ms),
        "min_ms": min(device_ms), "max_ms": max(device_ms),
        "wall_mean_ms": sum(wall_ms) / len(wall_ms),
    }


def run_case(seq_lens, H, D, warmup, iters, repeats, gate_dtype,
             states=("bf16", "none", "fp32"), varlen=None):
    device = torch.device("npu")
    torch.manual_seed(42)
    varlen = len(seq_lens) > 1 if varlen is None else varlen
    if not seq_lens or any(n <= 0 for n in seq_lens) or (not varlen and len(seq_lens) != 1):
        raise ValueError("benchmark sequence lengths must be positive; use varlen for multiple sequences")
    if D != 128:
        raise ValueError("NPU benchmarks require D=128")
    if gate_dtype != torch.bfloat16:
        raise ValueError("the public API requires bfloat16 gates, as on CUDA")
    total, sequences = sum(seq_lens), len(seq_lens)
    extra = {}
    if varlen:
        extra["cu_seqlens"] = torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int64, device=device)
        print(f"varlen shape=[{total},{H},{D}] seq_lens={seq_lens} warmup={warmup} iters={iters} repeats={repeats}")
    else:
        print(f"shape=[{total},{H},{D}] warmup={warmup} iters={iters} repeats={repeats}")
    shape = (1, total, H, D)
    q, k = [F.normalize(torch.randn(shape, device=device), dim=-1).bfloat16() for _ in range(2)]
    v = torch.randn(shape, dtype=torch.bfloat16, device=device)
    g = torch.randn(shape, dtype=gate_dtype, device=device)
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device=device)
    a_log, bias = torch.rand(H, device=device), torch.rand(H, D, device=device)
    # Prepare all conversions and buffers outside the timed callable, as in CUDA.
    initial_fp32 = torch.arange(sequences * H * D * D, device=device, dtype=torch.float32).reshape(sequences, H, D, D)
    initial_bf16 = initial_fp32.bfloat16()
    buffers = {
        "bf16": (initial_bf16, torch.empty_like(initial_bf16)),
        "none": (None, None),
        "fp32": (initial_fp32, torch.empty_like(initial_fp32)),
    }
    out = torch.empty_like(v)
    rows = []
    for variant in states:
        initial, final = buffers[variant]

        def run():
            flash_kda.fwd(q, k, v, g, beta, 1 / math.sqrt(D), out,
                          A_log=a_log, dt_bias=bias, lower_bound=LOWER_BOUND,
                          initial_state=initial, final_state=final, **extra)

        run()
        torch.npu.synchronize()
        backend = "ascendc"
        finite = bool(torch.isfinite(out).all().item())
        if final is not None:
            finite = finite and bool(torch.isfinite(final).all().item())
        if not finite:
            raise RuntimeError(f"{variant} state ({backend}) produced nonfinite output/state; refusing to benchmark")
        timing = bench_fn(run, warmup, iters, repeats)
        label = "no state" if variant == "none" else f"{variant} state"
        print(f"  backend={backend} initial_state={None if initial is None else initial.dtype} final_state={None if final is None else final.dtype}")
        print(f"  flash_kda ({label}) : mean={timing['mean_ms']:.4f} ms, min={timing['min_ms']:.4f} ms, max={timing['max_ms']:.4f} ms")
        print(f"  wall_mean={timing['wall_mean_ms']:.4f} ms")
        from flash_kda_npu.ops.ascendc import get_last_profile
        profile = get_last_profile()
        rows.append(dict(kind="varlen" if varlen else "fixed", T=total, H=H, D=D,
                         seq_lens=seq_lens, state=variant, backend=backend,
                         gate_dtype=str(gate_dtype), warmup=warmup, iters=iters,
                         repeats=repeats, native_launches=profile["launch_total"],
                         sequence_groups=profile.get("sequence_groups", 1),
                         padded_tokens=profile.get("padded_tokens", total), **timing))
    return rows


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=positive_int, default=200)
    p.add_argument("--repeats", type=positive_int, default=5)
    p.add_argument("--mode", choices=["fixed", "varlen", "all"], default="all")
    p.add_argument("--H", type=positive_int, default=96)
    p.add_argument("--D", type=positive_int, default=D_DEFAULT)
    p.add_argument("--T", type=positive_int, default=8192, help="fixed sequence length")
    p.add_argument("--seq-lens", type=positive_int, nargs="+", help="override varlen lengths")
    p.add_argument("--state", choices=["all", "bf16", "none", "fp32"], default="all")
    p.add_argument("--gate-dtype", choices=["bf16"], default="bf16")
    p.add_argument("--require-native", action="store_true", help="compatibility flag; execution is always native")
    p.add_argument("--json", type=Path, help="save device and wall timings with backend metadata")
    args = p.parse_args()
    if args.warmup < 0:
        p.error("--warmup must be nonnegative")
    if args.seq_lens is not None and args.mode == "fixed":
        p.error("--seq-lens requires --mode varlen or all")
    if args.D != 128:
        p.error("NPU AscendC requires D=128")
    if not torch.npu.is_available():
        p.error("an Ascend NPU is required")
    cases = []
    if args.mode in ("fixed", "all"):
        cases.append(([args.T], False))
    if args.mode in ("varlen", "all"):
        cases.extend((lengths, True) for lengths in ([args.seq_lens] if args.seq_lens else VARLEN_CASES))
    rows = []
    for lengths, varlen in cases:
        rows.extend(run_case(lengths, args.H, args.D, args.warmup, args.iters,
                             args.repeats, torch.bfloat16,
                             states=("bf16", "none", "fp32") if args.state == "all" else (args.state,),
                             varlen=varlen))
    if args.json:
        device_index = torch.npu.current_device()
        payload = dict(device=torch.npu.get_device_name(device_index), device_index=device_index,
                       torch_version=torch.__version__, results=rows)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Compare CATLASS and AscendC with identical CPU-generated inputs.

Run as a module from the checkout. Separate worker processes guarantee RTC
handles and compile options cannot leak between backends. Outputs and states
are compared in full outside the timed region. Numerical accuracy against the
independent FP64 oracle is checked by the test suite.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import torch
import torch_npu

from flash_kda_npu.benchmarks.bench_fwd import bench_fn
from flash_kda_npu.tests.torch_ref import make_cuda_inputs, make_cuda_state

CASES = [
    ("short", 64, 4, None),
    ("tail", 129, 4, None),
    ("fixed_h16", 8192, 16, None),
    ("fixed_h96", 8192, 96, None),
    ("packed", 8192, 16, [1300, 547, 2048, 963, 271, 3063]),
    ("empty_packed", 115, 4, [0, 17, 33, 0, 65]),
]


def worker(args):
    import flash_kda
    from flash_kda_npu.ops.ascendc import BACKEND, compile_config, get_last_profile
    results = []
    for label, steps, heads, lengths in CASES:
        cpu_inputs = make_cuda_inputs(1, steps, heads, seed=42)
        q, k, v, g, beta, a_log, bias = [x.to("npu") for x in cpu_inputs]
        count = 1 if lengths is None else len(lengths)
        cu = (None if lengths is None else torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64, device="npu"))
        for state in ("bf16", "none", "fp32"):
            dtype = torch.bfloat16 if state == "bf16" else torch.float32
            initial = None if state == "none" else make_cuda_state(count, heads, dtype).to("npu")
            final = None if initial is None else torch.empty_like(initial)
            out = torch.empty_like(q)

            def run():
                flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out,
                              a_log, bias, -5.0, initial_state=initial,
                              final_state=final, cu_seqlens=cu)

            run()
            torch.npu.synchronize()
            tensors = [out.cpu()] + ([] if final is None else [final.cpu()])
            if not all(torch.isfinite(t).all() for t in tensors):
                raise RuntimeError(f"nonfinite {BACKEND} {label} {state}")
            torch.save(tensors, args.scratch / f"{BACKEND}-{label}-{state}.pt")
            timing = bench_fn(run, args.warmup, args.iters, args.repeats)
            row = dict(case=label, T=steps, H=heads, seq_lens=lengths,
                       state=state, backend=BACKEND, **timing)
            row["profile"] = get_last_profile()
            if args.graphs and lengths is None:
                from flash_kda_npu.ops import capture_fwd
                captured = capture_fwd(q, k, v, g, beta, 128 ** -0.5, out,
                                       a_log, bias, initial_state=initial, final_state=final)
                row["graph_replay"] = bench_fn(captured.replay, args.warmup, args.iters, args.repeats)
                torch.npu.synchronize()
                actual = [out.cpu()] + ([] if final is None else [final.cpu()])
                if not all(torch.equal(x, y) for x, y in zip(actual, tensors)):
                    raise RuntimeError(f"graph replay differs: {BACKEND} {label} {state}")
                del captured
            results.append(row)
            print(f"{BACKEND} {label} {state}: {timing['mean_ms']:.4f} ms", flush=True)
    metadata = dict(torch=torch.__version__, torch_npu=torch_npu.__version__,
                    device=torch.npu.get_device_name(0), compile=compile_config(),
                    warmup=args.warmup, iters=args.iters, repeats=args.repeats,
                    python=platform.python_version())
    (args.scratch / f"{BACKEND}.json").write_text(json.dumps(dict(metadata=metadata, rows=results), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path("/tmp/kda-catlass-comparison.json"))
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--graphs", action="store_true", help="also measure fixed-length graph replay to isolate device execution")
    parser.add_argument("--scratch", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.warmup, args.iters, args.repeats) < 1:
        parser.error("warmup, iters and repeats must be positive")
    if args.scratch:
        worker(args)
        return
    with tempfile.TemporaryDirectory(prefix="kda-catlass-") as directory:
        scratch = Path(directory)
        report = {}
        for backend in ("ascendc", "catlass"):
            env = dict(os.environ, KDA_BACKEND=backend, KDA_PROFILE="0", KDA_AUTO_GRAPH="0")
            subprocess.run([sys.executable, "-m", "flash_kda_npu.benchmarks.bench_catlass",
                            "--scratch", directory, "--warmup", str(args.warmup),
                            "--iters", str(args.iters), "--repeats", str(args.repeats),
                            *(["--graphs"] if args.graphs else [])],
                           env=env, check=True)
            report[backend] = json.loads((scratch / f"{backend}.json").read_text())
        comparisons = []
        for baseline, current in zip(report["ascendc"]["rows"], report["catlass"]["rows"]):
            label, state = baseline["case"], baseline["state"]
            a, b = [torch.load(scratch / f"{backend}-{label}-{state}.pt", weights_only=True)
                    for backend in ("ascendc", "catlass")]
            checks = [dict(bitwise_equal=torch.equal(x, y),
                           max_abs_diff=(x.float() - y.float()).abs().max().item(),
                           relative_rms=((x.double() - y.double()).square().mean().sqrt() /
                                         x.double().square().mean().sqrt().clamp_min(1e-30)).item())
                      for x, y in zip(a, b)]
            comparisons.append(dict(case=label, state=state, tensors=checks,
                                    speedup=baseline["mean_ms"] / current["mean_ms"]))
            if "graph_replay" in baseline:
                comparisons[-1]["graph_speedup"] = (baseline["graph_replay"]["mean_ms"] /
                                                      current["graph_replay"]["mean_ms"])
        report["comparisons"] = comparisons
        report["all_bitwise_equal"] = all(c["bitwise_equal"] for row in comparisons for c in row["tensors"])
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"All outputs/states bitwise equal: {report['all_bitwise_equal']}; {args.json}")


if __name__ == "__main__":
    main()

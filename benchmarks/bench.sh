set -e
cd "$(dirname "$0")/.."

# Ascend hosts need the CANN runtime on LD_LIBRARY_PATH before torch can
# even import: torch_npu auto-loads as a device backend and its _C ext fails
# to resolve the runtime symbols otherwise.
if [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

if python -c "import torch; assert torch.npu.is_available()" 2>/dev/null; then
    # Ascend host: benchmark the AscendC forward kernel directly.
    pip install -e . --no-build-isolation -q
    exec python - "$@" <<'EOF'
import argparse
import math
import statistics

import torch

from flash_kda_npu import fwd

parser = argparse.ArgumentParser(description="Benchmark FlashKDA on Ascend NPU")
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--steps", type=int, default=8192)
parser.add_argument("--heads", type=int, default=96)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--reps", type=int, default=20)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--gate-dtype", choices=("bf16", "fp32"), default="bf16")
args = parser.parse_args()
torch.npu.set_device(args.device)
device = torch.device("npu")
shape = (args.batch, args.steps, args.heads, args.dim)

def _empty(shape, dtype):
    return torch.empty(shape, dtype=dtype, device=device)

q = _empty(shape, torch.bfloat16).normal_()
k = _empty(shape, torch.bfloat16).normal_()
v = _empty(shape, torch.bfloat16).normal_()
g_dtype = torch.bfloat16 if args.gate_dtype == "bf16" else torch.float32
g = _empty(shape, g_dtype).normal_()
beta = _empty(shape[:3], torch.bfloat16).normal_()
out = torch.empty_like(v)
final_state = _empty((args.batch, args.heads, args.dim, args.dim), torch.float32)
A_log = torch.rand(args.heads, dtype=torch.float32, device=device)
dt_bias = torch.rand((args.heads, args.dim), dtype=torch.float32, device=device)
scale = 1.0 / math.sqrt(args.dim)

for _ in range(args.warmup):
    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, -5.0,
        initial_state=None, final_state=final_state)
torch.npu.synchronize()

times = []
for _ in range(args.reps):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, -5.0,
        initial_state=None, final_state=final_state)
    end.record()
    torch.npu.synchronize()
    times.append(start.elapsed_time(end))

times.sort()
p20 = times[max(0, math.ceil(0.20 * len(times)) - 1)]
p80 = times[min(len(times) - 1, math.ceil(0.80 * len(times)) - 1)]
print(f"shape B={args.batch} T={args.steps} H={args.heads} D={args.dim}")
print(f"g_dtype={args.gate_dtype}")
print(f"min={times[0]:.3f}ms median={statistics.median(times):.3f}ms")
print(f"p20={p20:.3f}ms p80={p80:.3f}ms")
EOF
else
    # CUDA host: run the FLA-comparison benchmark.
    pip install -e . -q
    pip install flash-linear-attention matplotlib -q
    python benchmarks/bench_fwd.py "$@"
fi

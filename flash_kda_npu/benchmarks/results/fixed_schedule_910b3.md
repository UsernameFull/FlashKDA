# Fixed-length persistent scheduling

Measured 2026-10-17 on Ascend910B3, 20 Cube cores, physical device 1.
BF16 public inputs, FP32 initial/final state, D=128, batch=1, CHUNK=64.
Each shape runs old/new then new/old, with 30 warmups and 200 iterations x
three repeats per configuration per round (1200 measurements per mode).
NPU event timings, solve overlap retained, profiling disabled during timing.
No correctness tests ran concurrently with these confirmation measurements.

| H | T | Old 32 blocks (ms) | New default 40 blocks (ms) | Reduction |
|---|---|---:|---:|---:|
| 96 | 8192 | 8.9660 | 8.6561 | 3.46% |
| 128 | 8192 | 12.2099 | 11.7929 | 3.42% |
| 96 | 256 | 1.5481 | 1.5471 | 0.07% (effectively unchanged) |

All outputs/final states are finite and bitwise identical to the old grid.
The raw JSON includes separate synchronized host stage medians, which include
launch overhead and must not be summed with event end-to-end timings.
The inputs use seed 42 and random FP32 initial states, independently of the
latest FLA benchmark's initial-state construction; no new FLA comparison was run.

The old picker considered ceil(heads / heads_per_block) grids and charged
all blocks for the maximum head count. The new picker also considers adjacent
whole-wave grids and simulates per-block retirement, charging single-head
blocks for lost AIC/AIV overlap. Strictly tied cost estimates retain the old
choice. Decisions are cached (128 entries); the kernel and recurrence math
are unchanged. These gains are steady-state and only measured on 20 cores.

Reproduce from the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
unset KDA_PROFILE KDA_PRE_UNROLL KDA_PERSIST_LOOP_BLOCKS KDA_PERSIST_HEAD_MAP
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python flash_kda_npu/benchmarks/results/fixed_schedule_910b3.py --json /tmp/fixed-schedule.json
```

Planner validation: 4000 random cases covering head caps 1..4 and core counts
8/20/24/32/48 match an explicit per-block heap simulation. Grid coverage is
checked for heads 1..1024 on 20/24/32 cores.

Regression checks: fixed schedule/head mapping/batch/state I/O selection:
44 passed; packed/variable-length and planner selection: 131 passed,
1 opt-in long test skipped. The two selections overlap in planner tests.
Python compileall and git diff --check passed.

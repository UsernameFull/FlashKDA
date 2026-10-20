# Variable-length scheduling measurements

Measured on Ascend910B3 (20 Cube cores), H=96, FP32 state, CHUNK=64.
Unequal sequence lengths: [1300, 547, 2048, 963, 271, 3063].
These timings are specific to this device; the original 32-core measurements
must not be compared directly with these absolute times.

| Configuration | Pre-Gram unroll / blocks | Persistent blocks | End-to-end ms | Persistent ms |
|---|---:|---:|---:|---:|
| Startup-aware Pre-Gram, old persistent | 63 / 100 | 192 | 10.441 | 3.573 |
| Forced Pre-Gram unroll=2, old persistent | 2 / 3144 | 192 | 11.793 | 3.593 |
| Forced Pre-Gram unroll=66, old persistent | 66 / 96 | 192 | 10.521 | 3.578 |
| New default, confirmation run | 63 / 100 | 144 | 9.994 | 3.137 |
| Old persistent override, same confirmation run | 63 / 100 | 192 | 10.444 | 3.581 |

The startup-aware Pre-Gram change was already present at revision 4cbc246.
The new persistent picker reduces end-to-end time by 4.3% and persistent stage
time by 12.4% in the confirmation run. It accounts for unequal per-head work
and single-head tails without changing the head mapping or recurrence.

Fixed [8192] timings before/after: 8.987 / 8.987 ms.
Equal packed [1024] x 8: 9.026 / 9.034 ms. Their default scheduling is unchanged.
An additional H=32 case [0, 1, 64, 7000, 63, 1064] selects 96 persistent
blocks; stage medians were 1.779 ms versus 1.836 ms for the old 64-block grid.
Its end-to-end timings showed host-side jitter, so no end-to-end speedup is
claimed for that case.

Every swept configuration matched its case's default output and final state
bitwise. Separate reference-based packed-buffer regression tests cover empty
sequences, guards, mixed head counts and BF16/FP32 final states.

End-to-end values use NPU events with profiling disabled and solve overlap
retained. Stage values are separate synchronized host measurements (median of
five calls), include launch overhead, and must not be added to the event times.
Initial sweeps used five warmups and 20 iterations x three repeats; confirmation
used five warmups and 50 iterations x five repeats. Compilation is excluded.
Raw results are in [schedule_910b3.json](schedule_910b3.json).

The scheduling model is a heuristic, not an on-device autotuner. Its first
uncached shape estimate costs about 0.7 ms of CPU time on this host; cache hits
cost below 1 us. The cache holds 128 shape/core/head-cap combinations. Workloads
with constantly changing shapes need separate latency validation; the reported
steady-state gains do not include an uncached decision on every invocation.
Performance on 24-/32-core hardware has not been measured here.

Reproduce after sourcing CANN's `set_env.sh` and building the launcher:

```bash
python -m flash_kda_npu.benchmarks.bench_schedule \
  --seq-lens 1300 547 2048 963 271 3063 --heads 96 \
  --iters 50 --repeats 5 --json /tmp/kda-schedule.json
```

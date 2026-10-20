# Persistent head grouping on Ascend910B3

H=96, CHUNK=64, FP32 state; lengths [1300,547,2048,963,271,3063].
All comparisons preserve FP32 recurrence order within each head.

| Mapping / metadata | End-to-end ms | Persistent stage ms |
|---|---:|---:|
| Original stride map | 9.9795 | 3.1315 |
| New kernel, stride map | 9.9738 | 3.1285 |
| Sorted groups, separate map upload | 9.7171 | 2.5739 |
| Sorted groups, shared metadata upload (auto) | 9.4352 | 2.5694 |

The shared-upload implementation reduces same-run end-to-end time by 5.5%. Persistent stage time drops from about 3.13 to 2.57 ms. Both retain 144 blocks; the improvement comes from head ownership/order.
A 192-block grouping performs similarly, while 288 blocks is slower (about 9.73 ms).
Unsorted contiguous groups and load-balanced mixed-length groups were also measured; neither matched sorted similar-length grouping.

The implementation groups heads by similar chunk counts, and launches long groups first. This keeps multiple heads active together instead of leaving a long single-head tail. AIC and both AIV subcores share the same explicit map. The map uses four INT64 slots per block, padded with -1, appended after the sequence metadata sentinel. The existing metadata tensor uploads both tables in one operation. Scratch/output/state buffers retain their original logical indices.

The cached host cost model compares candidate head counts with the existing stride map. It charges for single-head tails and block startup, and requires both a 16-step absolute and 5% relative predicted margin. Fixed inputs, equal chunk counts, and small cases without a predicted margin retain the existing mapping. `KDA_PERSIST_HEAD_MAP=stride` disables grouping; `grouped` forces it for controlled experiments. Block overrides remain supported.

The H=32 case [0,1,64,7000,63,1064] selects 96 blocks (two heads each). In the shared-upload comparison, persistent stage time falls from about 1.80 to 1.02 ms and end-to-end time from about 5.70 to 5.13 ms. Small-input end-to-end samples have more host jitter; exact values are in the raw data.

## Final public API benchmark

30 warmups, 200 iterations x five repeats, default scheduling, H=96. All input lengths total 8192.
This is a separate confirmation run; it must not be treated as the same-run A/B above.

| Input | BF16 state ms | No state ms | FP32 state ms | Mapping | Blocks |
|---|---:|---:|---:|---|---:|
| Fixed 8192 | 8.9708 | 8.9508 | 8.9593 | stride | 32 |
| Unequal six sequences | 9.5057 | 9.4272 | 9.4676 | grouped | 144 |
| Equal 8 x 1024 | 9.0455 | 8.9547 | 9.0082 | stride | 256 |

The final FP32 unequal/fixed gap is about 0.51 ms (previous public benchmark: about 1.03 ms).
Fixed/equal inputs retain their original scheduling. All final benchmark outputs/states passed finite-value checks.

Validation: all mapped A/B outputs and final states matched the original kernel bitwise. 159 reference/guard/ABI tests passed, one opt-in long test skipped. Coverage includes initial/final state presence, BF16/FP32 state, empty sequences, odd head counts, independent-sequence bitwise comparisons, fixed-input fallback and automatic/forced/disabled maps.
The host planner was subsequently accelerated with prefix sums; 1000 randomized decisions/maps matched the previous implementation exactly, followed by a planner regression rerun.

Planning results are cached in a bounded 128-entry cache. An isolated host-only measurement of the additional grouping planner took about 1 ms on a cold entry, below 1 us on a cache hit (in addition to the existing stride planner). This is not an end-to-end cold-call timing and may overlap preceding queued device work. Constantly changing shapes need separate latency validation. Only 20-core 910B3 performance was measured.

Reproduce the public API A/B after sourcing CANN and building the launcher:

```bash
KDA_PERSIST_HEAD_MAP=stride python -m flash_kda_npu.benchmarks.bench_fwd \
  --skip-fla --mode varlen --seq-lens 1300 547 2048 963 271 3063 \
  --H 96 --state fp32 --warmup 30 --iters 200 --repeats 5 --json /tmp/stride.json
KDA_PERSIST_HEAD_MAP=auto python -m flash_kda_npu.benchmarks.bench_fwd \
  --skip-fla --mode varlen --seq-lens 1300 547 2048 963 271 3063 \
  --H 96 --state fp32 --warmup 30 --iters 200 --repeats 5 --json /tmp/grouped.json
```

[A/B raw data](head_map_910b3.json) · [Final public API raw data](bench_910b3_grouped.json).
[Initial experiment diff](head_map_initial.patch) records the discarded separate-upload/alternative grouping implementation, not the production version.

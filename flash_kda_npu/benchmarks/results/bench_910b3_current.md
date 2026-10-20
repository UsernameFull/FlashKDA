# Current native benchmark (Ascend910B3)

H=96, D=128, total tokens=8192; 30 warmups, 200 iterations × 5 repeats.
NPU events, default scheduling, solve overlap enabled, no FLA comparison.

| Input | BF16 state (ms) | No state (ms) | FP32 state (ms) | Pre unroll | Persistent blocks |
|---|---:|---:|---:|---:|---:|
| Fixed 8192 | 8.9676 | 8.9516 | 8.9586 | 77 | 32 |
| Unequal [1300,547,2048,963,271,3063] | 10.0187 | 9.9353 | 9.9933 | 63 | 144 |
| Equal [1024] × 8 | 9.0124 | 8.9138 | 9.0029 | 77 | 256 |

All output and final-state tensors passed finite-value checks. This benchmark does not compare against a numerical reference.
Full timings and environment metadata: [bench_910b3_current.json](bench_910b3_current.json).

[Same-run FLA comparison](bench_910b3_fla.md).

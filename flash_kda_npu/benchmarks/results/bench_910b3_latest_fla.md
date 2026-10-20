# Latest NPU benchmark: FlashKDA and FLA

Measured 2026-10-16 on Ascend910B3 (20 Cube cores), H=96, D=128, total tokens=8192.
Default Pre-Gram/persistent scheduling, including automatic grouped head mapping.
30 warmups, 200 iterations x five repeats (1000 measurements per configuration).
NPU event timings; solve overlap enabled, KDA_PROFILE disabled. Compilation and warmup excluded.

## FP32-state comparison

| Input | FlashKDA ms | FLA KDA ms | KDA speedup | FLA GDN ms |
|---|---:|---:|---:|---:|
| Fixed 8192 | 8.9610 | 74.0847 | 8.27x | 39.1845 |
| Unequal [1300,547,2048,963,271,3063] | 9.4785 | 74.6236 | 7.87x | 41.9682 |
| Equal 8 x 1024 | 9.0376 | 74.1838 | 8.21x | 41.2995 |

## All FlashKDA state configurations

| Input | BF16 state ms | No state ms | FP32 state ms | Pre unroll | Persistent blocks | Head map |
|---|---:|---:|---:|---:|---:|---|
| Fixed 8192 | 8.9724 | 8.9522 | 8.9610 | 77 | 32 | stride |
| Unequal [1300,547,2048,963,271,3063] | 9.5049 | 9.4310 | 9.4785 | 63 | 144 | grouped |
| Equal 8 x 1024 | 9.0663 | 8.9810 | 9.0376 | 77 | 256 | stride |

The unequal/fixed FP32-state gap is 0.5175 ms (5.78%).
Environment: torch 2.10.0+cpu, torch_npu 2.10.0.post4, FLA 0.6.0, triton-ascend 3.2.2.
FLA is pinned to commit `15396e604df6a4699950d7f20457b391b0cd8e74`.
Physical NPU 1; OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, FLA_FLASH_KDA=0. No concurrent tests on this benchmark host.

FlashKDA and FLA KDA share raw Q/K/V/g/beta, parameters and FP32 initial state. FLA enables fused Q/K normalization, gate activation and beta sigmoid, lower_bound=-5, transpose_state_layout=True.
GDN uses scalar logsigmoid gates and preactivated sigmoid beta, prepared outside timing. It is a different operator and is included only as a performance reference.
All outputs/final states passed finite-value checks. This benchmark does not perform a numerical-reference comparison. Metadata handling, public dtype conversions, allocations and uploads remain inside end-to-end timing; planning caches are warmed.

Reproduce:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
unset KDA_PROFILE KDA_PRE_UNROLL KDA_PERSIST_LOOP_BLOCKS KDA_PERSIST_HEAD_MAP FLA_DISABLE_BACKEND_DISPATCH
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 FLA_FLASH_KDA=0 \
python -m flash_kda_npu.benchmarks.bench_fwd \
  --mode all --H 96 --state all --warmup 30 --iters 200 --repeats 5 \
  --json /tmp/bench_910b3_latest_fla.json
```

[Raw timings and metadata](bench_910b3_latest_fla.json)

# FlashKDA / FLA comparison on Ascend910B3

H=96, D=128, total tokens=8192; 30 warmups, 200 iterations × 5 repeats per configuration.
NPU event mean latency; solve overlap enabled. FlashKDA uses default scheduling.
All rows below use FP32 initial/final state and were measured in the same run.

| Input | FlashKDA ms | FLA chunk_kda ms | KDA speedup | FLA GDN ms |
|---|---:|---:|---:|---:|
| Fixed 8192 | 8.9664 | 74.0940 | 8.26× | 39.1699 |
| Unequal [1300,547,2048,963,271,3063] | 10.0309 | 74.6359 | 7.44× | 42.0511 |
| Equal [1024] × 8 | 9.0170 | 74.1889 | 8.23× | 41.3199 |

Environment: Ascend910B3, torch 2.10.0+cpu, torch_npu 2.10.0.post4, FLA 0.6.0, triton-ascend 3.2.2.
FLA source is pinned to commit `15396e604df6a4699950d7f20457b391b0cd8e74` as specified by `bench.sh`.

FlashKDA and FLA KDA share Q/K/V, raw gate/beta logits, A_log, bias and FP32 initial state. FLA enables fused Q/K normalization, gate activation and beta sigmoid, lower_bound=-5, and transpose_state_layout=True.
Gated Delta Rule uses scalar per-head logsigmoid gates and sigmoid beta prepared outside timing. It is a different operator; its timings are included for reference and are not a KDA-equivalent comparison.

All outputs and final states passed finite-value checks. This timing run does not establish numerical equivalence between implementations. First-use compilation is excluded.
The JSON also includes all FlashKDA BF16-state and no-state measurements, selected scheduling, timing ranges and environment metadata.

Reproduce after installing dependencies from `flash_kda_npu/benchmarks/bench.sh`:

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
unset KDA_PROFILE KDA_PRE_UNROLL KDA_PERSIST_LOOP_BLOCKS FLA_DISABLE_BACKEND_DISPATCH
ASCEND_RT_VISIBLE_DEVICES=1 python -m flash_kda_npu.benchmarks.bench_fwd \
  --mode all --H 96 --state all --warmup 30 --iters 200 --repeats 5 \
  --json /tmp/bench_910b3_fla.json
```

[Raw results](bench_910b3_fla.json)

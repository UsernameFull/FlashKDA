# NPU optimization experiments 1–6

Measured 2026-10-17, Ascend910B3, 20 Cube cores, D=128, FP32 state.
Physical device 1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1. No correctness tests
ran concurrently with these benchmarks. All accepted arithmetic retains the
existing FP32 gate/recurrence precision and BF16 rounding points.

## Accepted changes

1. Public BF16 gates are read and converted inside Pre-Gram UB. This removes
the full FP32 gate allocation/conversion/readback. Normalization rounds through
consumed Q/K landing tiles; the mask queue has one live slot. Together these
free space for the whole BF16 gate tile while retaining the CHUNK=64 UB budget
(~185.6 KiB). FP32 gates remain supported by the native entry. The Pre-Gram ABI
adds one INT32 dtype flag (32 arguments total); host and kernel changes must be
applied together. State and accumulators retain their existing precision.

2. An explicit `flash_kda_npu.ops.capture_fwd` API captures fixed-shape forward
with a private graph memory pool. It retains caller tensors, supports input
value updates and writes caller output/state buffers. Regular forward does not
capture automatically. Initial/final state storage must be separate; capture
executes warmups and synchronizes. Shape/address/scalar changes use regular
forward or a new capture. Separate objects/output buffers support concurrency.

## Production-path gate A/B

Each shape ran legacy/current then current/legacy, 30 warmups, 200 iterations x
3 repeats per mode per round (1200 measurements per mode). Both versions run
in the same process with the same inputs and the current persistent schedule.
The original pre-Gram source/dispatcher is loaded under a separate kernel name
for legacy mode. Compilation and setup are excluded. All output/final states
were finite and **bitwise identical** in every shape/mode.

| Input | H | Tokens | Previous ms | New ms | Reduction |
|---|---:|---:|---:|---:|---:|
| fixed | 96 | 8192 | 8.6562 | 8.2443 | 4.76% |
| fixed | 128 | 8192 | 11.8139 | 11.0751 | 6.25% |
| fixed | 96 | 256 | 1.6347 | 1.6390 | -0.27% |
| unequal packed | 96 | 8192 | 9.4762 | 9.3240 | 1.61% |
| equal packed | 96 | 8192 | 9.0164 | 8.9350 | 0.90% |

The short regular-call difference is small and does not establish a gain.
The unequal lengths are [1300,547,2048,963,271,3063]; equal packed is [1024]x8.
These use seed-42 random initial states and differ from the full FLA benchmark's
arange state fixture. No new FLA measurements were made in these trials.

## Production graph benchmark

30 warmups, 200 iterations x 5 repeats per configuration, NPU event timings.
Capture setup is excluded and separately recorded in raw data.

| H | T | Regular ms | Replay ms | Speedup |
|---:|---:|---:|---:|---:|
| 96 | 256 | 1.5138 | 0.3383 | 4.47x |
| 96 | 8192 | 8.2383 | 8.2374 | 1.00x |

The graph is useful for short sequences. Long-sequence device latency is
essentially unchanged. Tests verify in-place changes to Q, gates, beta and
initial state; prototype tests also changed these independently. Separate
captures retain independent scratch/output buffers and replay on two streams.

## Results for all six directions

| Direction | Implemented experiment | Result / decision |
|---|---|---|
| 1. Pre-Gram lookahead | Reclaimed mask queue space, Q/K/V ping-pong landing buffers; also early reuse without a larger ring | 8.663 baseline vs 8.683/8.704 ms; no gain, not retained |
| 2. Pre-Gram traffic/synchronization | Reduced mask queue depth and reused normalized Q/K landing storage; fused BF16 gate conversion | Retained the gate fusion and storage changes; production A/B above |
| 3. Solve tiling | Swept slice sizes 240/480/720/960/1440/1920/3072/6144/12288 | Default ~8.650 ms; best 960 ~8.645 ms is not a stable gain; larger/smaller slices regress, defaults unchanged |
| 4. Persistent lookahead | Double L1 queue trial, then explicit ping-pong W/Qg staging with MTE1/MTE2 fences | Queue variant stalled and was aborted. Explicit-buffer variant was bitwise correct, 8.677 vs 8.657 ms; not retained |
| 5. Workspace / graphs | Private per-call scratch pool and NPUGraph capture/replay | Pool-only long latency unchanged, short prototype 1.616→1.283 ms; retained graph-owned storage/replay instead of a global cache |
| 6. Cross-stage pipeline | Range-limited Pre-Gram ABI, event dependencies and Pre-Gram→solve overlap with 960/1920/3072/6144-chunk slices | All outputs/states bitwise correct, 13.333/11.686/11.029/10.129 vs 8.662 ms; not retained |

Directions 1/3/4/6 were experiments, not production switches. Persistent's
cross-core protocol remains depth one. The prefetch trial did not establish
that a deeper protocol can never help. The pipeline experiment also does not
rule out a larger kernel redesign; its current slicing implementation regresses.
Pipeline stage host timings in raw data are not meaningful individually because
the pre-launch is deferred into solve; compare its end-to-end event times only.

All prototype timings use 20 warmups, 80 iterations x 3 repeats, with baseline
rechecks (except initial graph input-change checks). Stage timings are separate
synchronized host medians including launch overhead; do not sum them with
end-to-end timings or interpret the solve sweep's shorter host time as a gain.
A three-call baseline device trace measured the large gate cast at
405.288/405.188/404.828 us, consistent with the fixed H=96 fusion benefit.
Only 20-core 910B3 performance has been measured; graph results are fixed-shape,
fixed-address, warm-cache results and exclude capture/compilation costs.

## Reproduce

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
unset KDA_PROFILE KDA_PRE_UNROLL KDA_PERSIST_LOOP_BLOCKS KDA_PERSIST_HEAD_MAP KDA_SOLVE_SLICE_CHUNKS
ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m flash_kda_npu.benchmarks.bench_graph --T 256 --H 96 --state fp32 \
  --warmup 30 --iters 200 --repeats 5 --json /tmp/kda-graph.json
```

Raw experiment/confirmation data: `optimization_1to6_910b3.json`.
Prototype sources, isolated harnesses, legacy sources and device trace are
archived separately with patch 0004 under the requested model/profile folder;
experimental kernel sources are not applied to production.

## Final verification

- CHUNK=64: 413 passed, 6 skipped, 23 deselected. Excluded the CUDA/FLA gate
  comparison group and the subprocess configuration test; opt-in long and
  unavailable extra-device cases are skipped as usual.
- CHUNK=16 and CHUNK=32: 77 passed each (BF16/FP32 gate equivalence, graph replay,
  head mapping and packed state I/O).
- The default-chunk run covers all partial-tail sizes, guard buffers, weak and
  extreme gates, mixed state I/O, persistent reused slots, empty sequences,
  packed/direct pointers, launcher ABI and graph input updates/private pools.
- compileall and git diff --check passed.

The full native benchmark also ran all three state modes on fixed, unequal
and equal-packed inputs (30 warmups, 200 iterations x 3 repeats). Its FP32 state
results are 8.2363 / 9.3001 / 8.9774 ms respectively, using the standard
benchmark fixtures. These are independent of the legacy/current A/B fixture
above; raw data is under `native_final` in the JSON. FLA was explicitly skipped
in this run, since its implementation is unchanged.

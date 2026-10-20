# Pre-Gram candidate optimization experiments

Ascend910B3, 20 Cube cores, H=96, CHUNK=64, FP32 state.
Lengths: [1300, 547, 2048, 963, 271, 3063], total 8192 tokens.
The previous startup-aware Pre-Gram and length-aware persistent scheduling
remain enabled throughout.

| Trial | End-to-end mean ms | Pre-Gram host median, averaged over rounds, ms |
|---|---:|---:|
| Baseline (trial ABI) | 9.9993 | 4.1861 |
| Sequence descriptor cache | 9.9854 | 4.1712 |
| Cache + contiguous beta packing | 10.4902 | 4.1318 |
| Baseline (production ABI confirmation) | 10.0191 | 4.1959 |
| Cache only (production ABI confirmation) | 10.0218 | 4.1852 |

**Neither candidate was retained in production.** The apparent ~0.014 ms cache
benefit in the first isolated experiment did not repeat after removing the
beta trial argument from the kernel ABI. This is too small to claim a stable
end-to-end improvement. Beta packing slowed end-to-end execution by ~0.491 ms,
despite reducing the Pre-Gram stage by ~0.054 ms.

The cache held the current sequence descriptor in scalar registers and only
performed a new binary search when the monotonically increasing chunk index
crossed a sequence boundary. It skipped repeated prefixes from empty sequences.

The beta experiment allocated a zero-padded [chunks, CHUNK] FP32 buffer and
copied each sequence's small beta tensor into head-major order. Pre-Gram then
used a contiguous DataCopy instead of strided scalar DataCopyPad plus Gather.
Allocation and all six sequence copies were included in end-to-end timing.
This result rules out this host-driven packing implementation; it does not
prove that a fused/device-side beta layout transformation could never help.

Each table entry averages three rounds, with baseline/candidate order reversed
between rounds. Each round used 30 warmups and 200 iterations x three repeats
(1800 measured calls per mode across the experiment). Compilation is excluded.
End-to-end timing uses NPU events with KDA_PROFILE disabled and solve overlap
retained. Stage timing is the median of ten separate synchronized host calls;
these values include launch overhead and must not be added to event times.
OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. No correctness tests ran concurrently
with the isolated experiments. An initial concurrent-test sweep was discarded
because it showed host scheduling jitter.

All variants matched baseline output and final state bitwise on the main case.
The preliminary sweep also checked fixed length, equal packed sequences, and
[0,1,64,7000,63,1064] at H=32. The trial passed 30 reference/guard tests including
empty sequences, BF16/FP32 states and unroll=66 across sequence boundaries.

[Raw isolated and confirmation results](pre_gram_trials_910b3.json).
[Trial implementation diff](pre_gram_trials.patch) is retained only as an
experimental artifact; it was generated against the working tree before this
experiment and is not applied to production. Baseline kernels used the original
Pre-Gram source at revision 4cbc246. Host persistent scheduling was identical
between modes. The confirmation used the original 31-argument kernel ABI.

// K1 stages 1+2 fused: input normalisation, gate/decay preparation and the
// intra-chunk Gram matrices in one AIV block per (batch, head, chunk).
//
// The standalone pair costs 3.7 ms at [1,8192,32] and is DMA bound: the
// preprocess half alone moves 61 KB per chunk (1.0 GB per pass) at ~570 GB/s,
// and a stub that keeps every DataCopy but drops the whole compute chain still
// takes 1.79 of its 1.96 ms.  "Qn"/"Kn"/"Gc" are 32 KB per chunk of that
// traffic and their only consumer is the Gram kernel on the next launch, so
// folding the two stages together deletes the round trip.
//
// The Gram half is unchanged arithmetic and bit-identical to the old
// kernel's "Aqk32"/"Aqk"/"L" outputs: qf/kf here are the same bf16-rounded,
// l2-normalised rows the old kernel loaded back from "Qn"/"Kn", and the
// chunk-centred gate is still the same fp32 value.
//
// The GM copies of Qn/Kn/Gate/Gc are debug-only and skipped when the pointer
// is null ("api.py" passes them only for "return_intermediates").
//
// Per-row scalars (the l2 norms, beta, the chunk-centred gate) are broadcast
// with "Brcb" instead of a 16-iteration "Muls"/"Duplicate" loop: Brcb turns the
// 16 values into a tile with eight 32B blocks per row, and the consumer then
// runs 64 lanes x 16 repeats with "src1RepStride = 1" block (i.e. one scalar
// per repeat; "dstRepStride = 16" blocks walks the 128-lane rows), once for the
// low half and once for the high half.  That is 3 instructions instead of 16
// plus the 16 scalar "GetValue" reads, and it is bit-identical (measured
// max(abs(diff)) = 0.000e+00 on all 13 outputs): 3.114 -> 2.679 ms at
// [1,8192,32] and 1.590 -> 1.373 ms at [1,4096,32].
//
// The kernel is issue-bound, not FLOP-bound (msprof ArithmeticUtilization: the
// fp32 vector ALU is busy 20% of the block, while a micro-benchmark puts a
// fixed ~25-35 cycles on every vector instruction, plus ~1-7 cycles per repeat
// depending on the form).  Three changes follow from that, all bit-identical
// and worth 2.69 -> 2.59 ms at [1,8192,32]:
//   * every input load is issued up front behind its own MTE2->V flag and the
//     wait is deferred to the consumer, so the q/k norms run while G/V/the
//     masks are still in flight (the block used to wait for all five copies);
//   * Qg/Kg/Rk/Rv/BetaOut are stored as soon as they exist, so their MTE3
//     traffic drains behind the Gram loop instead of at the end of the block;
//   * the two "V_S" sync pairs around the l2 norms are dead (nothing reads
//     those reductions with the scalar unit any more) and cost ~2%.
//
// The block walks `unroll` consecutive chunks (the body is byte-identical,
// just wrapped in the loop) because the whole kernel is issue-bound and every
// block pays a fixed setup cost worth ~10% of one chunk's work: 2.679 ->
// 2.342 ms at [1,8192,32] with unroll 8, 0.333 -> 0.308 ms at [1,1024,32]
// with unroll 4, 0.0953 -> 0.0932 ms at [2,1024,4] with unroll 2, all
// bit-identical.  Back-to-back-launch measurements (host launch overhead
// hidden) put the optimum at >= 256 blocks; at chunk counts under 256 the
// loop wrapper itself costs ~6%, so `api.py` leaves those at unroll 1.
//
// A last pass deleted four more instructions per chunk, again bit-identical
// (0.000e+00 on all 13 outputs, 2.289 -> 2.262 ms at [1,8192,32] with the same
// back-to-back harness):
//   * "Muls(gf, gf, aexp)" followed by "Muls(gf, gf, -1)" is one "Muls" with
//     the negated scalar (a sign flip is exact, so this is not a rounding
//     change);
//   * "Muls(t2, t2, -1.0f)" after "gf - gf[15]" is the same as computing
//     "gf[15] - gf" directly - swap the two "Sub" operands *and* their repeat
//     strides (src0 is the fixed row, so its stride is the 0 one);
//   * the A-side triangular mask multiply is a no-op: the Gram loop writes
//     exactly the lower triangle and "Duplicate" already put zeros above it,
//     so "Muls(ga32, redA, scale)" is the whole post-processing - and the
//     "MaskS" tile no longer has to be fetched.  "MaskL" has to stay, because
//     the K-side loop does write L's diagonal and the mask is what zeroes it.
//
// The fast path has no "PipeBarrier<PIPE_ALL>" left except the one after the
// last "post_gram" (the "Gate"/"Gc" ones only run when the caller asks for
// intermediates).  The "Decay" store needs only the MTE3->V half:
// "SetFlag<HardEvent::MTE3_V>" after the copy, "WaitFlag<HardEvent::MTE3_V>"
// before the "Mul" that overwrites the tile - 2.259 -> 2.209 ms at
// [1,8192,32] (R=10, unroll 8, bit-identical), in-pipeline "pre_gram"
// 2.285 -> 2.237 ms of a 7.463 -> 7.433 ms pass.
//
// The pass boundary (and with it the chunk boundary) is two *self-paired*
// pairs, one per hazard, not a drain and not a loop-carried flag chain: the
// MTE3->V drain at the end of the pass body covers "MTE3 read of a stored
// tile -> V write of that tile", and the V->MTE2 marker at the last
// landing-buffer read (the rv cast) covers "V read of qnb/knb/rvb -> the next
// pass's DataCopyPad".  Self-paired means no state crosses the iteration, so
// the priming problem that hangs loop-carried pairs here does not arise; a
// WaitFlag only orders its own pipe's queue, which is why the two hazards need
// two pairs and why a single MTE3->V drain left the loads a pass ahead.  See
// docs/ASCENDC_V1_REFACTOR_PLAN_20260913.md sections 11.12/11.13.
//
// Cost model from probes on this kernel (+8 instructions per chunk, launch
// time at [1,8192,32]): +0.040 ms for 8 one-repeat "Adds" => ~27 cycles per
// vector instruction, +0.056 ms for 8 "Mul"s with 16 repeats => ~0.6 cycles
// per extra repeat, +0.051 ms for 8 one-repeat "Exp" => the special-function
// unit is dearer per repeat but does not dominate.  msprof agrees: the vector
// pipe is busy 70% of this kernel and the instruction count, not the lane
// count, is what it waits on.
//
// Negative results, worth not retrying: dropping the 15 barriers of the gate
// cumsum and/or the 64 of the Gram loop changes nothing (the vector pipe is
// in-order, so they are free); giving the Gram loop a second product tile so
// the A and K halves stop serialising on one buffer changes nothing (the pipe
// is issue-limited, not latency-limited); 256B UB padding on "gb" or "redA"
// costs 4.5% and on "ga"/"gtb" nothing, so the natural layout is already the
// good one; unroll 4/16/32 are all worse than 8.  (An earlier revision of this
// note priced "move the Gram to the Cube" as a wash at this kernel's own
// ~370 GB/s; that was wrong - the publishes ride an idle MTE3, see below.)
//
// ---------------------------------------------------------------------------
// MIX variant (this file): the two intra-chunk Grams move to the paired Cube.
//
// A stub of the vector-only block with just the 32-iteration Gram loop deleted
// runs 1.158 ms against 2.205 ms for the whole block at [1,8192,32]
// (back-to-back launches, MIN of 10), so the Gram half is 1.05 ms, 47% of the
// stage, and it is the one part of the chunk the vector unit is bad at: 32
// row-broadcast multiplies plus 32 whole-row reductions against a handful of
// instructions for every other step.  msprof agrees the block has no vector
// headroom left (aiv_vec_ratio 0.845).
//
// The split needs three things:
//   * the operands (ga = qf*exp2(gz), gk1 = ga's K side times beta,
//     gb = kf*exp2(-gz)) rounded to bf16 and published to GM: 12 KB per chunk.
//     A store-headroom probe measured 24 KB/chunk of extra MTE3 at 0.047 ms of
//     the 2.2 ms block, so the traffic is nearly free;
//   * the Cube side: Aqk = ga @ gb^T and L = gk1 @ gb^T as two 16x16x128 Mmads
//     that share one L0B operand.  Neither side needs a transpose: Nd2Nz of the
//     [16,128] tile plus a *plain* LoadData, read as B[K=128,N=16] with
//     MmadParams(16,16,128), is the transposed operand (probe in
//     /tmp/kdaval/cubeprobe);
//   * a software pipeline so the cross-core round trip is off the critical
//     path.  The AIV masks chunk u-1's Gram (which the Cube finished while it
//     computed chunk u) and only then publishes chunk u, so the Cube has a
//     whole chunk of vector work to answer in; the AIC is paired with two AIVs
//     (MIX_AIC_1_2) and computes both subcores' Grams per step, which leaves it
//     at ~25% duty.  The Aqk32/L slots double as the
//     Cube->AIV channel (2 KB read + 2 KB write per chunk) so no new buffer is
//     needed for the Gram results; the mask, the scale and the bf16 cast stay
//     on the AIV.
//
// Precision: the Cube forms the same products the vector loop did (both
// operands are exactly bf16 after the cast) with an fp32 accumulator, so the
// only difference is the operand rounding - measured end to end through the
// whole KDA pipeline in docs/ASCENDC_V1_KERNELS.md.
//
// The AIV publishes through a two-deep UB ring: the slot written for chunk u
// was last read by the Cube at step u-2, and the DONE wait for chunk u-1 sits
// between them, so the WAR is covered by the handshake rather than by an extra
// loop-carried flag (those hang in this runtime, see the note below).
//
// NOTE: the body below is sensitive to how it is written - an equivalent
// rewrite that only reformats the buffer declarations or moves the
// "PipeBarrier<PIPE_V>" after the gate re-centring loop trips an aivec error
// (mte error info 0x8030860ef, pc offset ~0x3d8) on this CANN runtime.  Keep
// the layout of this file; verified working on 910_9382 / CANN 9.1.0.
//
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 16
#endif
constexpr int32_t M = KDA_CHUNK;   // rows in one chunk
constexpr int32_t D = 128;
constexpr int32_t N = M * D;
// The vector side walks the chunk in NP passes of MT rows so that every
// staging buffer keeps the size it has at KDA_CHUNK = 16: at C = 32 the
// whole-chunk tiles do not fit in the 192 KB UB (the eleven fp32 [M, D]
// tiles alone are 176 KB).  Only the gate is chunk-global - its cumsum runs
// along the rows - so it stays one [M, D] tile computed before the passes.
// The Cube side is unchanged: the AIV still publishes whole [M, D] operands
// through the ring, so it is the ring rather than the Gram that grows.
constexpr int32_t MT = M > 16 ? 16 : M;
constexpr int32_t NP = M / MT;      // passes over one chunk
constexpr int32_t NG = MT * D;      // elements in one pass tile
constexpr int32_t KF = M / 16;      // 16-row fractal bands of a chunk tile
constexpr int32_t DF = D / 16;
// The gate reference has to sit inside the *band* its rows belong to.  One
// reference for the whole chunk gives both Gram operands exponents of +-M/2
// rows' worth of gate; the gate cumsum can drop the full lower bound (-5) per
// row, so at CHUNK = 64 that reaches exp(+-160) - past the fp32 and bf16 exp
// range of exp(88) - and the operands of every element of the A/L tiles turn
// into inf or 0 (measured: Aqk32 -> inf -> W/U/out NaN on a legal raw gate,
// C=16 finite on the same tensors, docs section 11.22).  Bands of BS rows cap
// the exponent at BS/2 rows' worth, which at BS = 32 is exp(+-80) = 5.5e34 and
// keeps ~2000x of headroom.  The Gram's (1, 0) block is the one block whose two
// operands come from *different* bands, so its k side is published a second
// time under band 1's centre (see the header note).  At M <= 32 there is a
// single band, the reference row is M/2 = MID0 exactly as before, and every
// branch below compiles away - the C=16 build is bit-identical.
constexpr int32_t BS = M > 32 ? 32 : M;   // rows per gate band
constexpr int32_t MID0 = BS / 2;          // band 0's reference row
constexpr bool XBAND = M > BS;            // is there a second gate band?
constexpr float RCP_LN2 = 1.4426950216f;
constexpr float LN2 = 0.6931471805599453f;
constexpr float EPS = 1e-6f;
// Cross-core flag channels (mode 2 = this AIC and its two AIVs).  The
// AIV->AIC direction is an *AND over the group's subcores*: a wait on the AIC
// is satisfied only once both AIVs have set that flagId, and it consumes both
// bits at once (probe /tmp/miniflag.cpp: one subcore setting, or two channels
// with one set each, never satisfies a wait; both setting the same id always
// does).  So a per-subcore channel cannot carry a step, and an AIC that waits
// twice per step instead of once deadlocks.  Both subcores therefore set one
// shared READY per step and the AIC waits it once.  The flag is also a level,
// not a count: the publish has to sit *after* the previous step's DONE wait,
// otherwise a subcore that runs a step ahead donates its next set to the
// current wait, the pairing drifts one step per chunk and the long shapes hang
// (which is what the original version did).
// A mode-2 AIV->AIC flag fires only when *both* subcores of the group have set
// it (a per-subcore channel can never be satisfied by one producer alone, and
// the flag is a level rather than a count), so the handshake uses one shared
// channel per step with both subcores setting it, and the AIC waits it once.
constexpr uint16_t FL_READY = 8;    // both AIVs -> AIC: step u's operands are in GM
constexpr uint16_t FL_DONE = 9;     // AIC -> both AIVs: step u's raw Gram is in GM

// Row-wise sum of an [MT, D] fp32 pass tile: rs[i] holds sum_d tile[i, d].
// The first Add halves every row in place (strided repeats keep each row's
// data inside its own 128-element slot); WholeReduceSum then collapses the
// remaining 64 elements per row.
static __aicore__ inline void RowReduce(LocalTensor<float> rs, LocalTensor<float> tmp,
                                        const LocalTensor<float> tile) {
    Add(tmp, tile, tile[64], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
    PipeBarrier<PIPE_V>();
    WholeReduceSum(rs, tmp, 64, MT, 1, 1, 16);
}



// AIV side of the handoff: the Cube has just written the raw fp32 Gram of
// chunk `c` into the Aqk32/L slots.  Mask, scale and the Aqk16 rounding stay
// on the vector unit (2 KB read + 2 KB write per chunk, no extra GM buffer).
//
// R3: the band staging is a two-deep TQue ring rather than one set of tiles
// plus a trailing "PipeBarrier<PIPE_ALL>".  That barrier only kept the next
// band's MTE2 loads out of the UB the current band's MTE3 stores were still
// reading; a queue's slot is one the compiler tracks a consumer for (MTE3 for
// "qout"/"qo16"), so the WAR is covered by the queue's own events and the
// drain disappears.  Measured at [1, 8192, 96, 128] / CHUNK = 64: deleting
// the two drains outright is worth 0.078 (masks) + 0.24 (band) ms, and this
// rewrite lands 4.039 -> 3.931 ms, bit-identical on all 13 outputs.
static __aicore__ inline void post_gram(TQue<TPosition::VECIN, 2>& qin,
                                        TQue<TPosition::VECIN, 2>& qmk,
                                        TQue<TPosition::VECOUT, 2>& qout,
                                        TQue<TPosition::VECOUT, 2>& qo16,
                                        const LocalTensor<uint8_t> mbitsAll,
                                        const GlobalTensor<float> Aqk32,
                                        const GlobalTensor<float> L,
                                        const GlobalTensor<bfloat16_t> Aqk16,
                                        const GlobalTensor<float> MaskS,
                                        const GlobalTensor<float> MaskL,
                                        int32_t c, float scale, bool buildMasks,
                                        bool keepAqk32) {
    const uint64_t m0 = static_cast<uint64_t>(c) * M * M;
    CrossCoreWaitFlag(FL_DONE);
    // P2-1: the two triangular masks are the same [M, M] tile for every chunk
    // of every block, so their *bit* form - the only thing the select consumes
    // - is built once per block into mbitsAll and read back band by band.  The
    // old form re-read 2 x 16 x M fp32 from GM and re-ran the two Compares for
    // every band of every chunk, which at KDA_CHUNK = 64 is 8 DataCopy and 128
    // vector instructions per chunk; the bits are the same values, so the
    // outputs stay bit-identical.  mbitsAll holds the S bits first (M * M / 8
    // bytes), then the L bits, each band's slice contiguous.
    constexpr int32_t BB = 16 * M / 8;   // bit-mask bytes of one band
    // One 16-row band at a time.  At KDA_CHUNK = 16 (a single band) this is
    // exactly the whole-chunk form this function has always used; at
    // KDA_CHUNK = 64 the whole-chunk staging (two fp32 masks, two fp32 Gram
    // tiles and the bf16 rounding tile = 72 KB) is a third of the 192 KB of
    // UB and the kernel faulted with a VEC out-of-bounds.  Every element goes
    // through the same compare/select/scale/round sequence either way, so the
    // outputs are bit-identical.
    // One-band-ahead (docs 11.60): band 0 is issued once before the loop and
    // every iteration dequeues the current band first, then issues the NEXT
    // band's loads before computing on the current one, so one MTE2 latency is
    // covered by one band body (the queue is FIFO; depth 2 is enough, and the
    // slot order - alloc next before free current - keeps two tensors live).
    {
        LocalTensor<float> gpre = qin.AllocTensor<float>();
        DataCopy(gpre, Aqk32[m0], 16 * M);
        DataCopy(gpre[16 * M], L[m0], 16 * M);
        qin.EnQue(gpre);
    }
    for (int32_t mm = 0; mm < KF; ++mm) {
    const uint64_t o = m0 + static_cast<uint64_t>(mm) * 16 * M;
    const uint64_t mo = static_cast<uint64_t>(mm) * 16 * M;
    constexpr int32_t NB = 16 * M;   // elements in one band
    LocalTensor<float> gA = qin.DeQue<float>();
    if (mm + 1 < KF) {
        LocalTensor<float> gnx = qin.AllocTensor<float>();
        const uint64_t on = m0 + static_cast<uint64_t>(mm + 1) * 16 * M;
        DataCopy(gnx, Aqk32[on], NB);
        DataCopy(gnx[NB], L[on], NB);
        qin.EnQue(gnx);
    }
    LocalTensor<float> gmk, gmaskS, gmaskL;
    if (buildMasks) {
        gmk = qmk.AllocTensor<float>();
        gmaskS = gmk; gmaskL = gmk[NB];
        DataCopy(gmaskS, MaskS[mo], NB);
        DataCopy(gmaskL, MaskL[mo], NB);
        qmk.EnQue(gmk);
    }
    LocalTensor<float> ga32 = gA, gl32 = gA[NB];
    LocalTensor<float> gM;
    if (buildMasks) {
        gM = qmk.DeQue<float>();
        gmaskS = gM; gmaskL = gM[NB];
    }
    LocalTensor<float> gout = qout.AllocTensor<float>();
    LocalTensor<float> ga32o = gout, gl32o = gout[NB];
    LocalTensor<uint8_t> gmaskBits = mbitsAll[mm * BB];
    LocalTensor<uint8_t> gmaskBitsL = mbitsAll[M * M / 8 + mm * BB];
    // The two masks are applied with a select instead of a multiply.  The raw
    // Gram is the Cube's fp32 accumulation of bf16 gated operands, and inside
    // the region the mask drops the two exponents are the far ends of the
    // 2 * CHUNK-row gate cumsum: at KDA_CHUNK = 32 that product overflows fp32
    // (exp2 of +-230), so `x * 0` turns an Inf/NaN into a NaN and the rounding
    // below then writes NaN into Aqk16 - which is where a C = 32 run picked up
    // NaNs in the output (measured with the fp32 torch reference: 50 NaNs in
    // the Aqk32 upper triangle at [1, 64, 2, 128], none at chunk 16, where the
    // doubled range still fits: exp2(116) = 8e34 < 3.4e38).  A select keeps the
    // kept region bit-identical and makes the dropped region exactly 0.
    if (buildMasks) {
        Compares(gmaskBits, gmaskS, 0.5f, CMPMODE::GT, NB);
        PipeBarrier<PIPE_V>();
        Compares(gmaskBitsL, gmaskL, 0.5f, CMPMODE::GT, NB);
        PipeBarrier<PIPE_V>();
    }
    Select(ga32o, gmaskBits, ga32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);
    Muls(ga32o, ga32o, scale, NB);
    PipeBarrier<PIPE_V>();
    Select(gl32o, gmaskBitsL, gl32, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, NB);
    PipeBarrier<PIPE_V>();
    LocalTensor<bfloat16_t> g16 = qo16.AllocTensor<bfloat16_t>();
    Cast(g16, ga32o, RoundMode::CAST_RINT, NB);
    qout.EnQue(gout);
    qo16.EnQue(g16);
    LocalTensor<float> gA2 = qout.DeQue<float>();
    LocalTensor<float> ga32s = gA2, gl32s = gA2[NB];
    LocalTensor<bfloat16_t> g16s = qo16.DeQue<bfloat16_t>();
    // The masked fp32 copy has no device consumer: the only reader of Aqk32 is
    // the band loop above (the raw Gram the Cube wrote), and Aqk16 - the tile
    // the next launch actually consumes - is rounded from this register value,
    // not from the store.  It exists for the ``return_intermediates`` views, so
    // ``debugStores == 0`` skips the 201 MB (C=64) store (docs 11.29).  The raw
    // Aqk32 pointer has to stay valid either way - this function reads it.
    if (keepAqk32) DataCopy(Aqk32[o], ga32s, NB);
    DataCopy(L[o], gl32s, NB);
    DataCopy(Aqk16[o], g16s, NB);
    qout.FreeTensor(gA2);
    qo16.FreeTensor(g16s);
    qin.FreeTensor(gA);
    if (buildMasks) qmk.FreeTensor(gM);
    }
}

// The paired Cube: for every step u it computes the two Grams of the two
// chunks the AIVs of this group just published (Aqk = ga @ gb^T and
// L = gk1 @ gb^T), writing the raw fp32 results into the Aqk32/L slots that
// the AIV then masks, scales and rounds.
static __aicore__ inline void run_gram_aic(GM_ADDR pGa, GM_ADDR pGk, GM_ADDR pGb,
                                           GM_ADDR pGx,
                                           GM_ADDR pAqk32, GM_ADDR pL,
                                           int32_t nchunk, int32_t unroll,
                                           int32_t rawMode) {
    const int32_t base = GetBlockIdx() * 2 * unroll;
    // Ablation arms that price the two raw fp32 Gram tiles (docs 11.48): bit 0
    // drops the Aqk32 Fixpipe, bit 1 the L one.  Both slots stay allocated and
    // the AIV band loop still reads them, so a dropping arm computes a wrong
    // answer by construction - it only ever prices the store, it is never a
    // shipping mode.  The step's Mmad/M_FIX pairing and the trailing
    // CrossCoreSetFlag(FL_DONE) are untouched, so post_gram's
    // CrossCoreWaitFlag still has its partner and nothing can hang.
    const bool keepAqk32Raw = (rawMode & 1) == 0;
    const bool keepLRaw = (rawMode & 2) == 0;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
    TQue<QuePosition::B1, 4> qa, qb;
    pipe.InitBuffer(qa, 4, M * D * 2);
    pipe.InitBuffer(qb, 4, M * D * 2);
    TQue<QuePosition::CO1, 2> qc;
    pipe.InitBuffer(qc, 2, M * M * 4);
    // The cross-band k side of the (1, 0) block: BS rows of one chunk, only
    // ever touched when the gate has two bands (see the constants).
    TQue<QuePosition::B1, 2> qx;
    pipe.InitBuffer(qx, 2, BS * D * 2);
    // L0A/L0B slots: two per chunk (ga, gk1) and one shared gb, plus the two
    // cross-band tiles (one per chunk) in L0B.
    LocalTensor<uint8_t> a8(TPosition::A2, 0, 4 * M * D * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, 2 * M * D * 2 + (XBAND ? 2 * BS * D * 2 : 0));
    GlobalTensor<bfloat16_t> Ga, Gk, Gb, Gx;
    GlobalTensor<float> Aqk32, L;
    Ga.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGa));
    Gk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGk));
    Gb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGb));
    Gx.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGx));
    Aqk32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAqk32));
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));

    for (int32_t u = 0; u < unroll; ++u) {
        // One wait per AIV channel: neither wait can be satisfied by the
        // other AIV.  The handshake stays strictly paired (backlog <= 1)
        // because the AIV that is about to publish step u+1 must first pass
        // the FL_DONE of step u's Gram, which this AIC can only send after it
        // has consumed that AIV's step-u flag.
        // One shared channel: both AIV subcores set it once per step, and a
        // mode-2 AIV->AIC flag only fires when *both* of them have (see the
        // header note), so this is the only channel that can carry a step.
        CrossCoreWaitFlag(FL_READY);
        // Both subcores clamp an out-of-range chunk to the last one, so the
        // pair stays in step; the duplicate work is idempotent.
        int32_t c0 = base + 2 * u;
        int32_t c1 = c0 + 1;
        if (c0 >= nchunk) c0 = nchunk - 1;
        if (c1 >= nchunk) c1 = nchunk - 1;
        const uint64_t o0 = static_cast<uint64_t>(c0) * N;
        const uint64_t o1 = static_cast<uint64_t>(c1) * N;
        auto ta0 = qa.AllocTensor<bfloat16_t>();
        auto tk0 = qa.AllocTensor<bfloat16_t>();
        auto ta1 = qa.AllocTensor<bfloat16_t>();
        auto tk1 = qa.AllocTensor<bfloat16_t>();
        auto tb0 = qb.AllocTensor<bfloat16_t>();
        auto tb1 = qb.AllocTensor<bfloat16_t>();
        // The two cross-band tiles (BS rows each) ride their own queue so the
        // two shared L0B slots keep their size; at one gate band this alloc /
        // load / free pair compiles away.
        LocalTensor<bfloat16_t> tx0, tx1;
        if (XBAND) {
            tx0 = qx.AllocTensor<bfloat16_t>();
            tx1 = qx.AllocTensor<bfloat16_t>();
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(ta0[mm * DF * 256], Ga[o0 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(tk0[mm * DF * 256], Gk[o0 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(tb0[mm * DF * 256], Gb[o0 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(ta1[mm * DF * 256], Ga[o1 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(tk1[mm * DF * 256], Gk[o1 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; mm < KF; ++mm) {
            DataCopy(tb1[mm * DF * 256], Gb[o1 + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        for (int32_t mm = 0; XBAND && mm < BS / 16; ++mm) {
            DataCopy(tx0[mm * DF * 256], Gx[static_cast<uint64_t>(c0) * BS * D + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
            DataCopy(tx1[mm * DF * 256], Gx[static_cast<uint64_t>(c1) * BS * D + mm * 16 * D],
                     Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
        }
        qa.EnQue(ta0);
        qa.EnQue(tk0);
        qa.EnQue(ta1);
        qa.EnQue(tk1);
        qb.EnQue(tb0);
        qb.EnQue(tb1);
        if (XBAND) {
            qx.EnQue(tx0);
            qx.EnQue(tx1);
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
        ta0 = qa.DeQue<bfloat16_t>();
        tk0 = qa.DeQue<bfloat16_t>();
        ta1 = qa.DeQue<bfloat16_t>();
        tk1 = qa.DeQue<bfloat16_t>();
        tb0 = qb.DeQue<bfloat16_t>();
        tb1 = qb.DeQue<bfloat16_t>();
        if (XBAND) {
            tx0 = qx.DeQue<bfloat16_t>();
            tx1 = qx.DeQue<bfloat16_t>();
        }
        LocalTensor<bfloat16_t> la0 = a8[0].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lk0 = a8[M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> la1 = a8[2 * M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lk1 = a8[3 * M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lb0 = b8[0].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lb1 = b8[M * D * 2].ReinterpretCast<bfloat16_t>();
        // The cross-band k side: BS rows, so the L0B fractal order is
        // [k block][16-row block] exactly as the full tiles' (n/16 = 2).
        constexpr int32_t KX = BS / 16;
        LocalTensor<bfloat16_t> lx0 =
            b8[2 * M * D * 2].ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> lx1 =
            b8[2 * M * D * 2 + (XBAND ? BS * D * 2 : 0)].ReinterpretCast<bfloat16_t>();
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(la0[(mm * DF + dd) * 256], ta0[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(lk0[(mm * DF + dd) * 256], tk0[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(la1[(mm * DF + dd) * 256], ta1[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(lk1[(mm * DF + dd) * 256], tk1[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(lb0[(dd * KF + mm) * 256], tb0[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(lb1[(dd * KF + mm) * 256], tb1[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; XBAND && dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KX; ++mm) {
                LoadData(lx0[(dd * KX + mm) * 256], tx0[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        for (int32_t dd = 0; XBAND && dd < DF; ++dd) {
            for (int32_t mm = 0; mm < KX; ++mm) {
                LoadData(lx1[(dd * KX + mm) * 256], tx1[(mm * DF + dd) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
        SetFlag<HardEvent::MTE1_M>(e1m);
        WaitFlag<HardEvent::MTE1_M>(e1m);
        // srcStride counts C0 (16-element) units between the n-blocks of one
        // L0C row: at KF = 2 the two n-blocks of a row sit KF * 256 elements
        // apart, i.e. KF * 16 = M units (probe /tmp/cubeprobe.py STYLE=fract
        // FXMODE=one FXSTRIDE=32 is exact; M / 16 = 2 is not).  With a
        // single n-block the field is not read, so the KDA_CHUNK = 16 build
        // keeps the value it has always shipped.
        constexpr int32_t FX_SRC_STRIDE = M > 16 ? M : 1;
        auto ip = FixpipeParamsV220(M, M, FX_SRC_STRIDE, M, false);
        ip.quantPre = QuantMode_t::NoQuant;
        ip.unitFlag = 0;
        // The (1, 0) block is a BS x BS square of the same [M, M] tile: same
        // destination row stride (M) and the accumulator's n-blocks BS C0
        // units apart - FX_SRC_STRIDE's rule at m = BS.
        auto ipx = FixpipeParamsV220(BS, BS, BS, M, false);
        ipx.quantPre = QuantMode_t::NoQuant;
        ipx.unitFlag = 0;
        for (int32_t s = 0; s < 2; ++s) {
            LocalTensor<float> cf0 = qc.AllocTensor<float>();
            LocalTensor<float> cf1 = qc.AllocTensor<float>();
            if (s == 0) {
                Mmad(cf0, la0, lb0, MmadParams(M, M, D, 0, false, true));
                Mmad(cf1, lk0, lb0, MmadParams(M, M, D, 0, false, true));
            } else {
                Mmad(cf0, la1, lb1, MmadParams(M, M, D, 0, false, true));
                Mmad(cf1, lk1, lb1, MmadParams(M, M, D, 0, false, true));
            }
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            const uint64_t m0 = static_cast<uint64_t>(s == 0 ? c0 : c1) * M * M;
            if (keepAqk32Raw) Fixpipe<float, float, CFG_ROW_MAJOR>(Aqk32[m0], cf0, ip);
            if (keepLRaw) Fixpipe<float, float, CFG_ROW_MAJOR>(L[m0], cf1, ip);
            SetFlag<HardEvent::FIX_M>(efm);
            WaitFlag<HardEvent::FIX_M>(efm);
            qc.FreeTensor(cf0);
            qc.FreeTensor(cf1);
            if (XBAND) {
                // Band 1's q / k rows (referenced to band 1's centre, as
                // published) against band 0's k rows, which the AIV published
                // a second time under that same centre.  This is the one Gram
                // block a single reference cannot serve: with band 0's own
                // centre the band-1 rows would run past the fp32 exp range.
                // The Fixpipe below lands after the full tile's and so
                // overwrites the block the band split gets wrong.
                LocalTensor<bfloat16_t> laX = s == 0 ? la0[2 * DF * 256] : la1[2 * DF * 256];
                LocalTensor<bfloat16_t> lkX = s == 0 ? lk0[2 * DF * 256] : lk1[2 * DF * 256];
                LocalTensor<bfloat16_t> lx = s == 0 ? lx0 : lx1;
                LocalTensor<float> cf2 = qc.AllocTensor<float>();
                LocalTensor<float> cf3 = qc.AllocTensor<float>();
                Mmad(cf2, laX, lx, MmadParams(BS, BS, D, 0, false, true));
                Mmad(cf3, lkX, lx, MmadParams(BS, BS, D, 0, false, true));
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                if (keepAqk32Raw) {
                    Fixpipe<float, float, CFG_ROW_MAJOR>(Aqk32[m0 + BS * M], cf2, ipx);
                }
                if (keepLRaw) {
                    Fixpipe<float, float, CFG_ROW_MAJOR>(L[m0 + BS * M], cf3, ipx);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qc.FreeTensor(cf2);
                qc.FreeTensor(cf3);
            }
        }
        qa.FreeTensor(ta0);
        qa.FreeTensor(tk0);
        qa.FreeTensor(ta1);
        qa.FreeTensor(tk1);
        qb.FreeTensor(tb0);
        qb.FreeTensor(tb1);
        if (XBAND) {
            qx.FreeTensor(tx0);
            qx.FreeTensor(tx1);
        }
        CrossCoreSetFlag<2, PIPE_FIX>(FL_DONE);
    }
}

extern "C" __global__ __aicore__ void kda_pre_gram_mix(
    GM_ADDR pQ, GM_ADDR pK, GM_ADDR pV, GM_ADDR pG, GM_ADDR pBeta,
    GM_ADDR pAlog, GM_ADDR pBias,
    GM_ADDR pQn, GM_ADDR pKn, GM_ADDR pGate, GM_ADDR pGc, GM_ADDR pBetaOut,
    GM_ADDR pDecay, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pQg, GM_ADDR pKg,
    GM_ADDR pGa, GM_ADDR pGk, GM_ADDR pGb, GM_ADDR pGx,
    GM_ADDR pAqk32, GM_ADDR pAqk16, GM_ADDR pL, GM_ADDR pMaskS, GM_ADDR pMaskL,
    int32_t B, int32_t T, int32_t H, float lower_bound, float scale, int32_t unroll,
    // ``debugStores`` stays the trailing int32 on purpose: tests/test_dead_store.py
    // reads args[-1] as the flag, so the probe-only ``rawMode`` goes in front of it.
    int32_t xRowBytes, int32_t gRowBytes, int32_t rawMode, int32_t debugStores) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIC {
        run_gram_aic(pGa, pGk, pGb, pGx, pAqk32, pL, B * H * (T / M), unroll,
                     rawMode);
    }
    if ASCEND_IS_AIV {
    const int32_t nt = T / M;
    const DataCopyPadExtParams<bfloat16_t> padNop(false, 0, 0, 0);
    const DataCopyPadExtParams<float> padNopF(false, 0, 0, 0);
    const int32_t nchunk = B * H * nt;

    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vq = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vk = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vg = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vv = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e2vs = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TEventID evs = pipe.AllocEventID<HardEvent::V_S>();
    TEventID e3d = pipe.AllocEventID<HardEvent::MTE3_V>();
    // The pass boundary.  "e3p" is a *self-paired* MTE3 -> V drain (set and
    // wait adjacent, no state carried across iterations): the V pipe stalls
    // until every MTE3 op issued before it has landed.  See the pass loop.
    TEventID e3p = pipe.AllocEventID<HardEvent::MTE3_V>();
    // The cross-band k side's landing tile is "rvbo", whose next writer is
    // the *next* pass's rv cast - after the pass-boundary drain, which
    // therefore cannot close that WAR (see the store below).
    TEventID e3x = pipe.AllocEventID<HardEvent::MTE3_V>();
    // The other half of the pass boundary: a self-paired V -> MTE2 marker
    // that keeps the next pass's loads off the three landing buffers
    // until this pass's V has read them (see the rv block).
    TEventID em2 = pipe.AllocEventID<HardEvent::V_MTE2>();
    // R3: the band staging of "post_gram" is a two-deep ring, one queue per
    // stream (Gram tiles, masks, the fp32 result, the bf16 rounding), so that
    // the band drain can go (see there).  TQue storage is not merged with the
    // scratch buffers by the TPipe allocator, which is why the ring has to be
    // paid for in UB rather than overlaid on dead scratch: at CHUNK = 64 the
    // four queues are 52 KB against the 18 KB of single-buffered tiles they
    // replace, and the kernel then has under 8 KB of UB headroom left
    // (measured with a live dummy buffer; the exact budget depends on the
    // allocator's per-position slabs, not on the nominal sizes).
    TQue<TPosition::VECIN, 2> qgin, qgmk;
    TQue<TPosition::VECOUT, 2> qgout, qgo16;
    TBuf<TPosition::VECCALC> bQf, bKf, bT0, bT2, bEf, bRed,
        bQnb, bKnb, bRkb, bRvb, bQgb, bKgb, bBias, bBeta, bBb, bAlog, bZz,
        bGef, bQK, bRvo, bMfull;
    pipe.InitBuffer(bQf, NG * 4); pipe.InitBuffer(bKf, NG * 4);
    pipe.InitBuffer(bT0, N * 4); pipe.InitBuffer(bT2, NG * 4);
    pipe.InitBuffer(bEf, NG * 4); pipe.InitBuffer(bRed, 384 * 4);
    pipe.InitBuffer(bQnb, NG * 2); pipe.InitBuffer(bKnb, NG * 2); pipe.InitBuffer(bRkb, NG * 2);
    pipe.InitBuffer(bRvb, NG * 2); pipe.InitBuffer(bQgb, NG * 2); pipe.InitBuffer(bKgb, NG * 2);
    pipe.InitBuffer(bBias, D * 4); pipe.InitBuffer(bBeta, M * 4);
    // The beta broadcast only needs one 32 B block per row: Brcb leaves row k
    // at bb[8 * k] and the consumers walk it with srcRepStride = 1.
    pipe.InitBuffer(bBb, M * 8 * 4); pipe.InitBuffer(bAlog, 8 * 4);
    pipe.InitBuffer(bZz, NG * 4);
    // One tile for the first exponential of "zz" (the second one is built
    // over it once the first has no readers left, see the Gram stage), and
    // the two bf16 tiles that must not be the MTE2 landing buffers (below).
    pipe.InitBuffer(bGef, NG * 4);
    pipe.InitBuffer(bQK, 2 * NG * 2);
    pipe.InitBuffer(bRvo, NG * 2);
    // 2 x 16 x M fp32 per queue slot = 8 KB at CHUNK = 64 for the Gram and
    // mask tiles; the fp32 result and its bf16 rounding are a third pair.
    pipe.InitBuffer(qgin, 2, 2 * 16 * M * 4);
    pipe.InitBuffer(qgmk, 2, 2 * 16 * M * 4);
    pipe.InitBuffer(qgout, 2, 2 * 16 * M * 4);
    pipe.InitBuffer(qgo16, 2, 16 * M * 2);
    // The select's bit masks are V-only (compare then select, same pipe).  P2-1
    // hoisted them out of the per-band loop: the two triangular masks are the
    // same [M, M] tile for every chunk, so the whole chunk's bits (M * M / 8
    // bytes per mask, plus padding) are built once per block and read back band
    // by band (see post_gram).  This replaces the per-band bMbits scratch.
    pipe.InitBuffer(bMfull, 2 * M * M / 8 + 64);
    // Published Gram operands: one pass band's worth of UB per operand.
    // These used to be a two-deep whole-chunk ring (3 x [M, D] bf16 = 48 KB
    // at KDA_CHUNK = 64) which, with the whole-chunk staging above, put this
    // kernel past the 192 KB of UB (measured: the C = 64 build faulted with a
    // VEC out-of-bounds).  The stores now leave for GM with each pass band,
    // and the FL_READY handoff below already rides PIPE_MTE3, so the Cube
    // still only sees a chunk once its last band has drained.
    TBuf<TPosition::VECCALC> bPga0, bPgk0, bPgb0;
    pipe.InitBuffer(bPga0, NG * 2);
    pipe.InitBuffer(bPgk0, NG * 2);
    pipe.InitBuffer(bPgb0, NG * 2);
    LocalTensor<float> qf = bQf.Get<float>(), kf = bKf.Get<float>();
    LocalTensor<float> gf = bT0.Get<float>(), t2 = bT2.Get<float>();
    LocalTensor<float> ef = bEf.Get<float>(), red = bRed.Get<float>();
    LocalTensor<float> bias = bBias.Get<float>(), beta = bBeta.Get<float>();
    LocalTensor<float> bb = bBb.Get<float>(), alog = bAlog.Get<float>();
    LocalTensor<float> zz = bZz.Get<float>();
    LocalTensor<float> gef = bGef.Get<float>();
    LocalTensor<bfloat16_t> qnb2 = bQK.Get<bfloat16_t>(), knb2 = bQK.Get<bfloat16_t>()[NG];
    LocalTensor<bfloat16_t> rvbo = bRvo.Get<bfloat16_t>();
    LocalTensor<uint8_t> mbitsAll = bMfull.Get<uint8_t>();
    bool masksBuilt = false;

    LocalTensor<bfloat16_t> qnb = bQnb.Get<bfloat16_t>(), knb = bKnb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> rkb = bRkb.Get<bfloat16_t>(), rvb = bRvb.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> qgb = bQgb.Get<bfloat16_t>(), kgb = bKgb.Get<bfloat16_t>();

    GlobalTensor<bfloat16_t> Q, K, V, Qn, Kn, Rk, Rv, Qg, Kg, Ga, Gk, Gb, Gx;
    GlobalTensor<float> G, Beta, Alog, Bias, Gate, Gc, BetaOut, Decay;
    Q.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQ));
    K.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pK));
    V.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pV));
    Qn.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQn));
    Kn.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKn));
    Rk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRk));
    Rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRv));
    Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
    Kg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKg));
    Ga.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGa));
    Gk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGk));
    Gb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGb));
    Gx.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pGx));
    G.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pG));
    Beta.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pBeta));
    Alog.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAlog));
    Bias.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pBias));
    Gate.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pGate));
    Gc.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pGc));
    BetaOut.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pBetaOut));
    Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));

    GlobalTensor<bfloat16_t> Aqk16;
    GlobalTensor<float> Aqk32, L, MaskS, MaskL;
    Aqk16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk16));
    Aqk32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pAqk32));
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    MaskS.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pMaskS));
    MaskL.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pMaskL));
    // One block walks `unroll` consecutive chunks: the kernel is issue-bound
    // (~40 cycles per vector instruction) and pays a fixed per-block cost that
    // is a tenth of a block's work, so amortizing it over 2-8 chunks is worth
    // 2.68 -> 2.34 ms at [1,8192,32] (unroll 8) with the body unchanged and
    // bit-identical output.  `api.py` picks the unroll from the chunk count.
    // In MIX_AIC_1_2 the AIV sees 2N raw blocks for N AIC blocks: the raw index
    // picks the group and GetSubBlockIdx() the subcore.  The two subcores walk
    // interleaved chunks (2u + sub) so the paired Cube reads two adjacent
    // operand tiles per step.  Out-of-range chunks clamp to the last one, which
    // keeps both subcores' flag counts balanced and the duplicate work
    // idempotent.
    const int32_t ratio = static_cast<int32_t>(GetTaskRation());
    const int32_t group = ratio == 0 ? static_cast<int32_t>(GetBlockIdx())
                                     : static_cast<int32_t>(GetBlockIdx()) / ratio;
    const int32_t sub = static_cast<int32_t>(GetSubBlockIdx());
    int32_t cprev = -1;
    for (int32_t u = 0; u < unroll; ++u) {
    int32_t c = group * 2 * unroll + 2 * u + sub;
    if (c >= nchunk) c = nchunk - 1;
    const int32_t bh = c / nt;
    const int32_t head = bh % H;
    const uint64_t m0 = static_cast<uint64_t>(c) * M * M;
    const uint64_t x0 = static_cast<uint64_t>(c) * N;
    // The cross-band k side's ring is BS rows per chunk (see the constants).
    const uint64_t xg = static_cast<uint64_t>(c) * BS * D;
    const uint64_t cm = static_cast<uint64_t>(c) * M;
    // The four stage-1 inputs are read straight out of the public [B, T, H, D]
    // layout instead of a packed [c, M, D] copy: one token is D contiguous
    // elements and consecutive tokens of a chunk sit H*D elements apart, so a
    // strided DataCopy replaces the pack (which cost a 0.67 ms round trip of
    // 1.07 GB at [1,8192,32]).  The downstream buffers keep the packed order.
    const int32_t b = bh / H;
    const int32_t ck = c - bh * nt;
    const uint64_t xb = (static_cast<uint64_t>(b) * T + static_cast<uint64_t>(ck) * M) *
                            static_cast<uint64_t>(H) * D +
                        static_cast<uint64_t>(head) * D;

    // The block/stride form of DataCopy is broken for a strided GM source on
    // this part (halves the bursts land unwritten, probe
    // /tmp/kdaval/probe_stride4.py); DataCopyPad's byte-stride form reads the
    // same gather correctly, so the four stage-1 inputs use it.
    // ---- full-chunk gate -------------------------------------------------
    // The cumsum runs along the rows, so the whole chunk has to be in one
    // tile; everything else below is row-local and is walked in NP passes of
    // MT rows, which is what keeps the staging at its KDA_CHUNK = 16 size.
    DataCopyPad(gf, G[xb], DataCopyExtParams(M, D * 4, gRowBytes, 0, 0), padNopF);
    SetFlag<HardEvent::MTE2_V>(e2vg);
    DataCopy(alog, Alog[head], 8);
    DataCopy(beta, Beta[cm], DataCopyParams(1, M / 8, 0, 0));
    // One set/one wait: the beta sigmoid below needs the G/log/beta loads in
    // UB, and the wait is the only consumer of this channel ("e2vs" used to
    // also carry post_gram's mask loads, which the band queues own now).
    SetFlag<HardEvent::MTE2_V>(e2vs);
    WaitFlag<HardEvent::MTE2_V>(e2vg);

    // ---- beta sigmoid ----------------------------------------------------
    WaitFlag<HardEvent::MTE2_V>(e2vs);
    Muls(beta, beta, -1.0f, M);
    Exp(beta, beta, M);
    Adds(beta, beta, 1.0f, M);
    Duplicate(red, 1.0f, M);
    PipeBarrier<PIPE_V>();
    Div(beta, red, beta, M);
    PipeBarrier<PIPE_V>();
    // ---- beta broadcast over the D axis ----------------------------------
    // Brcb leaves row k's beta at bb[8 * k] (one 32 B block apart), which is
    // the stride the beta products below read with srcRepStride = 1.
    Brcb(bb, beta, M / 8, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();

    // ---- gate (cumsum carried in the log2 domain) ------------------------
    if (pBias != nullptr) {
        DataCopy(bias, Bias[head * D], DataCopyParams(1, 16, 0, 0));
        SetFlag<HardEvent::MTE2_V>(e2v);
        WaitFlag<HardEvent::MTE2_V>(e2v);
        Add(gf, gf, bias, 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        Add(gf[64], gf[64], bias[64], 64, M, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        PipeBarrier<PIPE_V>();
    }
    Exp(alog, alog, 8);
    SetFlag<HardEvent::V_S>(evs);
    WaitFlag<HardEvent::V_S>(evs);
    const float aexp = -alog.GetValue(0);
    // The elementwise part of the sigmoid runs in the same NP passes as the
    // rest of the chunk: it needs a full-tile "1.0" operand, and the staging
    // only holds an MT-row tile (see the header note).  The cumsum below is
    // the only row-coupled op, so it keeps the whole chunk.
    Duplicate(t2, 1.0f, NG);
    for (int32_t hp = 0; hp < NP; ++hp) {
    LocalTensor<float> gfs = gf[hp * NG];
    Muls(gfs, gfs, aexp, NG);
    Exp(gfs, gfs, NG);
    Adds(gfs, gfs, 1.0f, NG);
    PipeBarrier<PIPE_V>();
    Div(gfs, t2, gfs, NG);
    Muls(gfs, gfs, lower_bound, NG);
    PipeBarrier<PIPE_V>();
    }
    // Blocked (radix-8) prefix sum, docs 11.55.  The 63-step serial chain
    // (Add + PipeBarrier per row, ~0.3 ms of the pre_gram launch at
    // [1,8192,96,128]) is replaced by three phases whose aliasing read/write
    // pairs are split across instructions - a naive in-place log-scan reads
    // rows the same instruction wrote, repeat-level RAW that the hardware
    // does not order, and comes out NaN:
    //   A) per 8-row block, the serial prefix in 7 levels; a level's two
    //      calls write rows {8b+j} and read rows {8b+j-1}, disjoint mod 8;
    //   B) the S block totals (rows 8b+7) scanned serially - 7 dependent
    //      count-form Adds, the shipped loop's shape over 8 rows;
    //   C) rows 8b..8b+6 add the previous blocks' cumulative total
    //      (broadcast source, src1RepStride = 0); row 8b+7 already holds it
    //      and is skipped, so no phase-C call reads what phase-C wrote.
    // Same prefix, different summation order: last-ulp changes (measured at
    // [1,8192,96,128]: out rel 2.5e-6 / max|d| 3.05e-5 against the gate's
    // own 8.6e-3 margin, e2e -0.112 ms), so this is a tolerance-gated change
    // rather than a bit-identical one.
    {
        constexpr int32_t BS = 8;         // rows per block
        constexpr int32_t NB = M / BS;    // blocks (8 at CHUNK = 64)
        constexpr int32_t ROW8 = 8 * 16;  // 8 rows of D fp32 = 128 blocks
        for (int32_t j = 1; j < BS; ++j) {
            Add(gf[D * j], gf[D * j], gf[D * (j - 1)], 64, NB,
                BinaryRepeatParams(1, 1, 1, ROW8, ROW8, ROW8));
            Add(gf[D * j + 64], gf[D * j + 64], gf[D * (j - 1) + 64], 64, NB,
                BinaryRepeatParams(1, 1, 1, ROW8, ROW8, ROW8));
            PipeBarrier<PIPE_V>();
        }
        for (int32_t b = 1; b < NB; ++b) {
            Add(gf[D * (BS * b + BS - 1)], gf[D * (BS * b + BS - 1)],
                gf[D * (BS * (b - 1) + BS - 1)], 128);
            PipeBarrier<PIPE_V>();
        }
        for (int32_t b = 1; b < NB; ++b) {
            Add(gf[D * BS * b], gf[D * BS * b], gf[D * (BS * (b - 1) + BS - 1)],
                64, BS - 1, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
            Add(gf[D * BS * b + 64], gf[D * BS * b + 64],
                gf[D * (BS * (b - 1) + BS - 1) + 64], 64, BS - 1,
                BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        }
        PipeBarrier<PIPE_V>();
    }
    Muls(gf, gf, RCP_LN2, N);
    if (pGate != nullptr) {
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        DataCopy(Gate[x0], gf, DataCopyParams(M, 16, 0, 0));
        PipeBarrier<PIPE_ALL>();
    }

    // ---- decay = exp2(gate_last) ----------------------------------------
    Muls(t2, gf[(M - 1) * D], LN2, D);
    Exp(t2, t2, D);
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Decay[static_cast<uint64_t>(c) * D], t2, DataCopyParams(1, 16, 0, 0));
    // "t2" is reused by the first pass below, so this one store needs an
    // MTE3->V flag instead of a full PIPE_ALL.  The wait sits above the pass
    // loop rather than at the first consumer: one SetFlag pairs with exactly
    // one WaitFlag (a second wait on the same event never fires and hangs the
    // block), and the passes after the first already have the store behind
    // them.
    SetFlag<HardEvent::MTE3_V>(e3d);
    // Same as the masked Aqk32 store: BetaOut feeds the debug views only, so
    // production switches it off with the same flag.  The flag pair below is
    // unconditional - it orders the tile's reuse, not this store.
    if (debugStores != 0) DataCopy(BetaOut[cm], beta, DataCopyParams(1, M / 8, 0, 0));
    LocalTensor<bfloat16_t> pga = bPga0.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> pgk = bPgk0.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> pgb = bPgb0.Get<bfloat16_t>();

    WaitFlag<HardEvent::MTE3_V>(e3d);
    // ---- the row-local rest of the chunk, NP passes of MT rows -----------
    for (int32_t hp = 0; hp < NP; ++hp) {
    const int32_t gh = hp * NG;                              // pass offset in UB
    const uint64_t xh = x0 + gh;                             // ... and in GM
    const uint64_t xbh = xb + static_cast<uint64_t>(hp) * MT * H * D;
    LocalTensor<float> gfp = gf[gh];
    LocalTensor<float> bbp = bb[hp * MT * 8];
    DataCopyPad(qnb, Q[xbh], DataCopyExtParams(MT, D * 2, xRowBytes, 0, 0), padNop);
    SetFlag<HardEvent::MTE2_V>(e2vq);
    DataCopyPad(knb, K[xbh], DataCopyExtParams(MT, D * 2, xRowBytes, 0, 0), padNop);
    SetFlag<HardEvent::MTE2_V>(e2vk);
    DataCopyPad(rvb, V[xbh], DataCopyExtParams(MT, D * 2, xRowBytes, 0, 0), padNop);
    SetFlag<HardEvent::MTE2_V>(e2vv);
    WaitFlag<HardEvent::MTE2_V>(e2vq);
    Cast(qf, qnb, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();

    // ---- q l2 norm -------------------------------------------------------
    Mul(t2, qf, qf, NG);
    PipeBarrier<PIPE_V>();
    RowReduce(red, ef, t2);
    Adds(red, red, EPS, MT);
    Rsqrt(red, red, MT);
    PipeBarrier<PIPE_V>();
    Brcb(red[64], red, 2, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
    Mul(qf, qf, red[64], 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(qf[64], qf[64], red[64], 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(qnb2, qf, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();
    Cast(qf, qnb2, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();

    // ---- k l2 norm -------------------------------------------------------
    WaitFlag<HardEvent::MTE2_V>(e2vk);
    Cast(kf, knb, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, kf, kf, NG);
    PipeBarrier<PIPE_V>();
    RowReduce(red, ef, t2);
    Adds(red, red, EPS, MT);
    Rsqrt(red, red, MT);
    PipeBarrier<PIPE_V>();
    Brcb(red[64], red, 2, BrcbRepeatParams(1, 8));
    PipeBarrier<PIPE_V>();
    Mul(kf, kf, red[64], 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(kf[64], kf[64], red[64], 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(knb2, kf, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();
    Cast(kf, knb2, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();

    // ---- gc = gate - gate[the pass's band reference] ---------------------
    // The reference is the middle row of the pass's *gate band* (M/2 at a
    // single band, 16 / 48 at CHUNK = 64): the exponents below are the only
    // place the gate's absolute size enters, and a band-wide reference is what
    // keeps them inside fp32 (see the constants).
    const int32_t mid = MID0 + (hp * MT / BS) * BS;
    Sub(zz, gfp, gf[mid * D], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    Sub(zz[64], gfp[64], gf[mid * D + 64], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
    if (pGc != nullptr) {
        // "Gc" is documented as the chunk-centred gate, so at CHUNK = 64 -
        // where the reference above is band-centred - the debug tile is
        // rebuilt against the chunk's middle row instead of storing the
        // band's (the debug path only; the Gram never reads this, and "gef"
        // is this pass's free fp32 scratch at this point).
        if (mid != M / 2) {
            Sub(gef, gfp, gf[(M / 2) * D], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
            Sub(gef[64], gfp[64], gf[(M / 2) * D + 64], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        DataCopy(Gc[xh], mid == M / 2 ? zz : gef, DataCopyParams(MT, 16, 0, 0));
        PipeBarrier<PIPE_ALL>();
    }

    // ---- exp2(gate) ------------------------------------------------------
    Muls(ef, gfp, LN2, NG);
    Exp(ef, ef, NG);
    PipeBarrier<PIPE_V>();

    // ---- qg = qn * exp2(gate) --------------------------------------------
    Mul(t2, qf, ef, NG);
    PipeBarrier<PIPE_V>();
    Cast(qgb, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();

    // ---- rk = kn * beta * exp2(gate) -------------------------------------
    Mul(t2, kf, ef, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(t2[64], t2[64], bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(rkb, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();

    // ---- rv = v * beta ---------------------------------------------------
    WaitFlag<HardEvent::MTE2_V>(e2vv);
    Cast(t2, rvb, RoundMode::CAST_NONE, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(t2[64], t2[64], bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(rvbo, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();
    // Pass boundary, load side.  This is the last read of the three MTE2
    // landing buffers (qnb and knb are read by the two norms above, rvb by the
    // cast just before), so a *self-paired* V -> MTE2 marker here holds every
    // later DataCopyPad - the next pass's three loads - until the V queue is
    // past this point.  It has to be a marker and not the MTE3 drain's
    // transitive effect: a WaitFlag on a pipe's event queue does not stall the
    // scalar unit, so with the drain alone the loads ran a pass ahead and the
    // rv tile - the late reader of the three - came back holding the next
    // pass's V (6136/8192 elements wrong at CHUNK = 64, probe /tmp/pgqK.py;
    // with the marker: bit-exact, and 0.004 ms = noise).
    SetFlag<HardEvent::V_MTE2>(em2); WaitFlag<HardEvent::V_MTE2>(em2);

    // ---- kg = kn * exp2(gate_last - gate) --------------------------------
    Sub(t2, gf[(M - 1) * D], gfp, 64, MT, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
    Sub(t2[64], gf[(M - 1) * D + 64], gfp[64], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 0, 16));
    PipeBarrier<PIPE_V>();
    Muls(t2, t2, LN2, NG);
    Exp(t2, t2, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, kf, NG);
    PipeBarrier<PIPE_V>();
    Cast(kgb, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();

    // early stores: let MTE3 drain behind the Gram work
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Qg[xh], qgb, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Kg[xh], kgb, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Rk[xh], rkb, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Rv[xh], rvbo, DataCopyParams(MT, 8, 0, 0));
    if (pQn != nullptr) DataCopy(Qn[xh], qnb2, DataCopyParams(MT, 8, 0, 0));
    if (pKn != nullptr) DataCopy(Kn[xh], knb2, DataCopyParams(MT, 8, 0, 0));

    // ---- Gram half: bf16 operands for the paired Cube ---------------------
    // Each pass casts its own MT rows and stores them straight to GM, so the
    // operands never need a whole-chunk UB ring (see the buffer note).
    // Neither the product scratch nor the second exponential needs a tile of
    // its own: "t2" is dead after the kg cast above, and the -zz exponential
    // is built over "gef" once its last reader has gone.  Both are pure
    // reorganisations - same arithmetic, same rounding points.
    Muls(gef, zz, LN2, NG);
    Exp(gef, gef, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, qf, gef, NG);
    PipeBarrier<PIPE_V>();
    Cast(pga, t2, RoundMode::CAST_RINT, NG);
    Mul(t2, kf, gef, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, t2, bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    Mul(t2[64], t2[64], bbp, 64, MT, BinaryRepeatParams(1, 1, 0, 16, 16, 1));
    PipeBarrier<PIPE_V>();
    Cast(pgk, t2, RoundMode::CAST_RINT, NG);
    Muls(gef, zz, -LN2, NG);
    Exp(gef, gef, NG);
    PipeBarrier<PIPE_V>();
    Mul(t2, kf, gef, NG);
    PipeBarrier<PIPE_V>();
    Cast(pgb, t2, RoundMode::CAST_RINT, NG);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE3>(ev3);
    WaitFlag<HardEvent::V_MTE3>(ev3);
    DataCopy(Ga[xh], pga, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Gk[xh], pgk, DataCopyParams(MT, 8, 0, 0));
    DataCopy(Gb[xh], pgb, DataCopyParams(MT, 8, 0, 0));
    // Pass boundary, store side: every tile the next pass overwrites is
    // written by V (qgb/kgb/rkb/rvbo/pga/pgk/pgb), so a self-paired MTE3 -> V
    // drain - V waits until the stores that read those tiles have landed -
    // closes the MTE3-read -> V-write half of the boundary.  The MTE2 half
    // (V-read -> load-write) is a separate pair at the last landing-buffer
    // read, see the rv block; the three store-only bf16 tiles are what makes
    // the two halves independent, and the MTE3-read -> MTE2-write path that
    // needed the PIPE_ALL (probe /tmp/pgq8.py, 13 debug tensors diffed: only
    // Qn/Kn/Rv moved) no longer exists.  A full PipeBarrier<PIPE_ALL> here is
    // 0.49 ms of the stage; a loop-carried MTE3->MTE2 / MTE3->V *pair*
    // instead hangs or traps even when primed (3 attempts, /tmp/pgq5.py +
    // pgq6.py + pgq7.py).  At KDA_CHUNK = 16 there is one pass and both
    // branches compile away, as before.
    SetFlag<HardEvent::MTE3_V>(e3p); WaitFlag<HardEvent::MTE3_V>(e3p);

    // ---- cross-band k side of the (1, 0) Gram block -----------------------
    // Band 0's rows are read against band 1's q rows, whose exponents are
    // referenced to band 1's centre (the mid above); a Gram block is only
    // what it should be if both of its operands carry the *same* reference
    // (any fixed one is exact - the exponentials cancel, only their
    // difference means anything), so band 0's k side is published a second
    // time under band 1's centre.  Band 1's centre is the reference that
    // keeps both sides inside the fp32/bf16 exp range: with band 0's own
    // centre the band-1 operands would run to exp(-160) and vanish.  The two
    // band-0 passes each publish their own 16 rows, and "rvbo" - the pass's
    // V->MTE3 store tile, whose Rv copy is behind the drain just closed -
    // doubles as their landing buffer.
    if (XBAND && mid == MID0) {
        Sub(t2, gfp, gf[(MID0 + BS) * D], 64, MT, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        Sub(t2[64], gfp[64], gf[(MID0 + BS) * D + 64], 64, MT,
            BinaryRepeatParams(1, 1, 1, 16, 16, 0));
        PipeBarrier<PIPE_V>();
        Muls(t2, t2, -LN2, NG);
        Exp(t2, t2, NG);
        PipeBarrier<PIPE_V>();
        Mul(t2, t2, kf, NG);
        PipeBarrier<PIPE_V>();
        Cast(rvbo, t2, RoundMode::CAST_RINT, NG);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        DataCopy(Gx[xg + gh], rvbo, DataCopyParams(MT, 8, 0, 0));
        // "rvbo" carries the next pass's rv cast, and the pass-boundary
        // drain above sits *before* this store, so this WAR needs a pair of
        // its own: without it the next pass's V write can land in the tile
        // before this MTE3 read has taken it.  Measured before the pair was
        // added: the published tile came back 2.7e-1 wide instead of 8.8e-5
        // (rows 0..23 holding the next band's v * beta), the (1, 0) Gram
        // block and the solve's L with it, and the whole pipeline's output
        // at 1.2e+3 relative instead of 5.5e-3 - on one launch layout and
        // not on another, which is what a landing-buffer race looks like.
        SetFlag<HardEvent::MTE3_V>(e3x);
        WaitFlag<HardEvent::MTE3_V>(e3x);
    }
    }
    // ---- previous chunk's Gram: mask, scale, round, store -----------------
    if (cprev >= 0) {
        post_gram(qgin, qgmk, qgout, qgo16, mbitsAll, Aqk32, L, Aqk16,
                  MaskS, MaskL, cprev, scale, !masksBuilt, debugStores != 0);
        masksBuilt = true;
    }
    // ---- publish + hand off ---------------------------------------------
    // The publish sits *after* the previous step's DONE wait so that at most
    // one publication per subcore is outstanding: the flag is a level, not a
    // count, so a subcore that runs a step ahead would otherwise donate its
    // next set to the current wait and the pairings would drift (which is what
    // hangs the long shapes).
    CrossCoreSetFlag<2, PIPE_MTE3>(FL_READY);
    cprev = c;
    // No PipeBarrier<PIPE_ALL> here any more (0.13 ms).  The chunk boundary is
    // the pass boundary one level up and is covered by the same two pairs: the
    // V->MTE2 marker in the rv block orders the next chunk's three loads after
    // this chunk's last landing-buffer read, and the MTE3->V drain at the end
    // of the pass body orders the next chunk's V writes after the stores that
    // read the tiles they overwrite.  Both are unconditional so that the C=16
    // build - one pass per chunk, where the boundary *is* the chunk boundary -
    // keeps the same two guarantees.  Measured at CHUNK=64 (probe /tmp/r3a.py,
    // 3 interleaved rounds x 8 chunks, 25 intermediates diffed): bit-exact,
    // pre_gram 3.441 -> 3.310 ms.
    }
    // Drain the pipeline: the last chunk's Gram has no later chunk to hide
    // behind (~1/unroll of the stage, and the Cube is idle by then).
    post_gram(qgin, qgmk, qgout, qgo16, mbitsAll, Aqk32, L, Aqk16,
              MaskS, MaskL, cprev, scale, !masksBuilt, debugStores != 0);
    masksBuilt = true;
    PipeBarrier<PIPE_ALL>();
    }
}

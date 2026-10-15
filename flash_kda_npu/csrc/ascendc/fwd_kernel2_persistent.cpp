// Persistent K2: one MIX_AIC_1_2 launch runs the whole chunk recurrence on
// the device.
//
// Each block owns one or more heads (bh) and every AIV subcore owns one value
// tile (iv), so a block has nh * 2 independent state tiles in flight and the
// AIC is never blocked by its own chain.  Per chunk the two engines run
//     AIC: [wait R; d12(h); set C1] x nh   then   [wait V; d34(h); set C2] x nh
//     AIV: [wait C1; vnew(h); set V] x nh  then   [wait C2; out(h); set R] x nh
// with both subcores executing the same flag sequence (the unit of the
// protocol is a head, because that is the only split both subcores share).
// Four flag ids carry the whole loop, the protocol is depth one, and the AIV
// pre-sets R once per head before the loop so the first chunk needs no special
// case.
//
// The AIV stores `out` straight into the caller's [B, T, H, D] tensor: the M
// chunk rows of a 64-wide value tile are M separate 128 B runs, NH * D
// elements apart, which the block form of DataCopy expresses with dstGap.
//
// The fp32 state never leaves the AIV: it is loaded from H0 at start-up, kept
// in UB across the whole loop, with an optional FP32/BF16 final export. Only
// intermediate chunks publish S16 to GM for the next chunk's Cube operands.
//
// Each stage issues the loads that do not depend on its cross-core flag
// *before* the wait: the two MTE2 reads of W/Qg in stage 1, the flag-free Aqk
// and Kg^T reads in stage 3, U in stage 2 and D2/Decay in stage 4.  The waits
// they precede only guard the state-dependent tiles (S16, Vt, D1, D3/D4), so
// the earlier of the two loads is always private to this core.  Hoisting those
// reads lets their latency overlap work that does not depend on the cross-core
// result.  The WAR `PipeBarrier<PIPE_ALL>` of stages 2
// and 4 must stay *before* the hoisted load (it orders the previous
// iteration's vector reads against the UB staging buffers).
//
// D1/D2/D3 cross to the AIV in fp32. Keep the Cube accumulation precision until
// residual/output addition, particularly for weak decay and large initial
// states where BF16 handoff rounding would amplify cancellation.
// Each head owns one GM slot for D1/D2/D3/Vt, reused across chunks.  Before
// stage 1 overwrites D1/D2, FL_R confirms that the previous stage 4 consumed
// D2/D3 (and stage 2 consumed D1).  Before stage 2 overwrites Vt, FL_C1 from
// that next stage 1 provides the same ordering after the previous stage 3's
// Vt loads.  No slot is shared between heads or AIV subcores.
//
// Every tile follows the chunk size.  M = KDA_CHUNK, NG = 2 * BV / M and
// stage 3's contraction dim is the chunk's row count, so the whole kernel is
// one set of formulas over (M, NG) with no branch on the chunk size.  The
// staging budget is what caps the head count: at M = 64 it is 56.5 KB of this
// part's 192 KB UB, so two heads of 32 KB state fit together with nh = 2.
//
// The [BV, M] v_new^T tile is stored *once*, in the block order the d4 A
// operand reads - packed, contiguous, one call - and the d34 B operand's order
// (the same 16 x 16 blocks, block-transposed) is built by its load with
// explicit strides.  The A operand walks R / 16 bands row-block-major, while a
// `dstNzC0Stride = R` Nd2Nz packs the source C0 blocks column-block-major,
// which is the B operand's order.
//
// The 16-row fractal stays the unit of every L0 load, and the three operand
// layouts this kernel needs are all built from the two idioms below:
//   * L0A [R, C] (W/Qg, Aqk, v_new^T as the d4 A operand): R / 16 bands, each
//     `Nd2NzParams(1, 16, C, 0, C, 16, 1, 0)` - dstNzC0Stride = R = 16 = one
//     band - then one `LoadData2dParams(0, C / 16, 1, 0, 0, false, 0)`.
//     v_new^T is stored in that band order, so the read is one plain burst.
//   * L0B [C, R] whose source has the rows on the n dim (S16, v_new^T):
//     `Nd2NzParams(1, R, C, 0, C, R, 1, 0)`, one call, then
//     `LoadData2dParams(0, R * C / 256, 1, 0, 0, false, 0)`.  dstNzC0Stride =
//     R packs the source C0 blocks column-block-major, which *is* the L0B
//     fractal order (the 16-row and 64-row forms of this call are the production
//     S16 and v_new loads).
//   * L0B [C, R] whose source has the rows on the *k* dim (kg, which the api
//     hands over in its public [c, CHUNK, D] layout): the same band loop as an
//     L0A, then `LoadDataWithTranspose` per band, whose destination fractals
//     are consecutive.  The band's fractals land at band * (R / 16), which is
//     the L0B k-block order.
// A chunk-sized tile therefore costs 4 small calls but covers four times
// fewer chunks.  The C=16 path keeps a plain-burst shortcut, see the
// `K == FR` branches in stage 3.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 16
#endif
#ifndef KDA_MAXH
#define KDA_MAXH 4
#endif
constexpr int32_t M = KDA_CHUNK;     // rows in one chunk (the tile's m dim)
constexpr int32_t FR = 16;           // fractal side - the unit of every L0 load
constexpr int32_t NB = M / FR;       // 16-row bands in one chunk-sized tile
constexpr int32_t D = 128;
constexpr int32_t BV = 64;
constexpr int32_t K = M;             // stage 3 contracts over the chunk rows
constexpr int32_t N = 64;
constexpr int32_t N_D4 = 128;
constexpr int32_t TILE = M * BV;
constexpr int32_t S_TILE = BV * D;
constexpr int32_t NG = 2 * BV / M;   // 16-row groups in the whole d4 tile
constexpr int32_t MAXH = KDA_MAXH;   // heads owned by one block
// L0A holds W and Qg side by side (2*M*D bf16) and is then reused by d34's
// NG*M x K v_new operand plus the M x K Aqk tile, so the raw allocation is
// whichever is larger.
constexpr int32_t L0A_ELEMS =
    (2 * M * D > NG * M * K + M * K) ? 2 * M * D : NG * M * K + M * K;
// Depth-one loop protocol, ids stay inside the usable range (<= 7).
constexpr uint16_t FL_C1 = 0;
constexpr uint16_t FL_V = 1;
constexpr uint16_t FL_C2 = 2;
constexpr uint16_t FL_R = 3;

extern "C" __global__ __aicore__ void kda_k2_persistent_loop(
    GM_ADDR pU, GM_ADDR pW, GM_ADDR pQg, GM_ADDR pAqk, GM_ADDR pKgT, GM_ADDR pDecay,
    GM_ADDR pD1, GM_ADDR pD2, GM_ADDR pD3, GM_ADDR pD4,
    GM_ADDR pOut, GM_ADDR pVnewT,
    GM_ADDR pH0, GM_ADDR pFinal, GM_ADDR pS16,
    int32_t BH, int32_t NT, int32_t NV, int32_t NBLK, float scale, int32_t NH,
    int32_t finalBF16, int32_t T) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    const int32_t nv = NV;

    if ASCEND_IS_AIC {
        const int32_t blk = static_cast<int32_t>(GetBlockIdx());
        int32_t heads[MAXH];
        int32_t nh = 0;
        for (int32_t h = blk; h < BH && nh < MAXH; h += NBLK) heads[nh++] = h;
        if (nh == 0) return;

        GlobalTensor<bfloat16_t> W, Qg, Aqk, Vt, Kt, S16;
        GlobalTensor<float> D4;
        GlobalTensor<float> D1, D2, D3;
        W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
        Qg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pQg));
        Aqk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pAqk));
        Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
        Kt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pKgT));
        S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
        D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1));
        D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD2));
        D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
        D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));

        TPipe pipe;
        TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
        TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
        TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
        TEventID efm = pipe.AllocEventID<HardEvent::FIX_M>();
        TEventID em1 = pipe.AllocEventID<HardEvent::M_MTE1>();
        TQue<QuePosition::B1, 1> qw, qg, qs, qa, qv, qx, qk;
        pipe.InitBuffer(qw, 1, M * D * 2);
        pipe.InitBuffer(qg, 1, M * D * 2);
        pipe.InitBuffer(qs, 1, nv * BV * D * 2);
        pipe.InitBuffer(qa, 1, M * K * 2);
        pipe.InitBuffer(qv, 1, NG * M * K * 2);
        pipe.InitBuffer(qx, 1, nv * BV * K * 2);
        pipe.InitBuffer(qk, 1, D * K * 2);
        TQue<QuePosition::CO1, 1> qc;
        pipe.InitBuffer(qc, 1, (NG * M * N_D4 + nv * M * N) * 4);
        LocalTensor<float> cf = qc.AllocTensor<float>();
        LocalTensor<uint8_t> a8(TPosition::A2, 0, L0A_ELEMS * 2);
        // L0B: the two state tiles (with rows on the n dim), reused by d34's
        // kg and v_new operands.
        LocalTensor<uint8_t> b8(TPosition::B2, 0, nv * BV * D * 2);
        LocalTensor<bfloat16_t> l0a = a8.ReinterpretCast<bfloat16_t>();
        LocalTensor<bfloat16_t> l0b = b8.ReinterpretCast<bfloat16_t>();

        for (int32_t chunk = 0; chunk < NT; ++chunk) {
            // Stage 1: d1 = W @ S16^T, d2 = Qg @ S16^T for every head.
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const uint64_t a0 = static_cast<uint64_t>(bh * NT + chunk) * M * D;
                auto lw = qw.AllocTensor<bfloat16_t>();
                auto lg = qg.AllocTensor<bfloat16_t>();
                auto ls = qs.AllocTensor<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    DataCopy(lw[b * FR * D], W[a0 + b * FR * D],
                             Nd2NzParams(1, FR, D, 0, D, FR, 1, 0));
                    DataCopy(lg[b * FR * D], Qg[a0 + b * FR * D],
                             Nd2NzParams(1, FR, D, 0, D, FR, 1, 0));
                }
                CrossCoreWaitFlag(FL_R);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const uint64_t s0 = static_cast<uint64_t>(bh * nv + iv) * BV * D;
                    DataCopy(ls[iv * BV * D], S16[s0], Nd2NzParams(1, BV, D, 0, D, BV, 1, 0));
                }
                SetFlag<HardEvent::MTE2_MTE1>(e21);
                WaitFlag<HardEvent::MTE2_MTE1>(e21);
                qw.EnQue(lw);
                qg.EnQue(lg);
                qs.EnQue(ls);
                lw = qw.DeQue<bfloat16_t>();
                lg = qg.DeQue<bfloat16_t>();
                ls = qs.DeQue<bfloat16_t>();
                for (int32_t b = 0; b < NB; ++b) {
                    LoadData(l0a[b * FR * D], lw[b * FR * D],
                             LoadData2dParams(0, D / FR, 1, 0, 0, false, 0));
                    LoadData(l0a[M * D + b * FR * D], lg[b * FR * D],
                             LoadData2dParams(0, D / FR, 1, 0, 0, false, 0));
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    LoadData(l0b[iv * BV * D], ls[iv * BV * D],
                             LoadData2dParams(0, BV * D / 256, 1, 0, 0, false, 0));
                }
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    Mmad(cf[iv * M * N], l0a, l0b[iv * BV * D],
                         MmadParams(M, N, D, 0, false, true));
                    Mmad(cf[(nv + iv) * M * N], l0a[M * D], l0b[iv * BV * D],
                         MmadParams(M, N, D, 0, false, true));
                }
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const uint64_t o0 = static_cast<uint64_t>(bh * nv + iv) * TILE;
                    auto ip1 = FixpipeParamsV220(N, M, M, N, false);
                    ip1.quantPre = QuantMode_t::NoQuant;
                    ip1.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(D1[o0], cf[iv * M * N], ip1);
                    auto ip2 = FixpipeParamsV220(N, M, M, N, false);
                    ip2.quantPre = QuantMode_t::NoQuant;
                    ip2.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(D2[o0], cf[(nv + iv) * M * N], ip2);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qw.FreeTensor(lw);
                qg.FreeTensor(lg);
                qs.FreeTensor(ls);
                // L0A/L0B/L0C are reused by the next stage on this same core.
                // The FIX_M event above only covers L0C; the L0A/L0B WAR is an
                // Mmad-read vs LoadData-write hazard, so this needs an
                // M -> MTE1 order, not a full drain.  PipeBarrier<PIPE_M> only
                // orders M against M, so it does not keep the next LoadData
                // from overwriting L0B under the Mmad.  The M_MTE1 event pair
                // is the actual required order.
                SetFlag<HardEvent::M_MTE1>(em1);
                WaitFlag<HardEvent::M_MTE1>(em1);
                CrossCoreSetFlag<2, PIPE_FIX>(FL_C1);
            }
            // Stage 3: d3 = Aqk @ v_new, d4 = v_new^T @ kg for every head.
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t c = bh * NT + chunk;
                auto la = qa.AllocTensor<bfloat16_t>();
                auto lv = qv.AllocTensor<bfloat16_t>();
                auto lx = qx.AllocTensor<bfloat16_t>();
                auto lk = qk.AllocTensor<bfloat16_t>();
                if (K == FR) {
                    // One C0 block per row: the ND tile already *is* the
                    // fractal order, so one plain burst loads it.
                    DataCopy(la, Aqk[static_cast<uint64_t>(c) * M * K], M * K);
                } else {
                    for (int32_t b = 0; b < NB; ++b) {
                        DataCopy(la[b * FR * K],
                                 Aqk[static_cast<uint64_t>(c) * M * K + b * FR * K],
                                 Nd2NzParams(1, FR, K, 0, K, FR, 1, 0));
                    }
                }
                for (int32_t b = 0; b < NB; ++b) {
                    DataCopy(lk[b * FR * D], Kt[static_cast<uint64_t>(c) * M * D + b * FR * D],
                             Nd2NzParams(1, FR, D, 0, D, FR, 1, 0));
                }
                CrossCoreWaitFlag(FL_V);
                for (int32_t iv = 0; iv < nv; ++iv) {
                    const int32_t task = bh * nv + iv;
                    const uint64_t t0 = static_cast<uint64_t>(task) * BV * K;
                    if (K == FR) {
                        // Same ND == fractal-order shortcut as `la` above: the
                        // [BV, K] tile is one C0 block per row, and the d4 A
                        // operand (which reads BV / FR bands per value tile,
                        // iv-major - the band count is *not* NB: at C=16 it is
                        // 4 against NB = 1) and the d34 B operand both read it
                        // straight.  One burst per operand and per tile.
                        DataCopy(lv[iv * BV * K], Vt[t0], BV * K);
                        DataCopy(lx[iv * BV * K], Vt[t0], BV * K);
                    } else {
                        // The GM tile is in the A operand's block order (one
                        // plain burst), and the B operand's is its block
                        // transpose, done here with explicit block strides
                        // instead of an Nd2Nz that re-packs the row-major
                        // tile.  See the stage-2 store.
                        // The B operand's [K, N] fractal order is row-major
                        // like the A operand's [M, K]: k band major, value
                        // band minor, inner stride BV / FR.  So Vt's block
                        // (nb, kb) has to land at kb * (BV / FR) + nb - the
                        // walk is over the chunk's K / FR k bands with BV / FR
                        // bursts each, and *not* over BV / FR groups with a
                        // BV / FR source stride.  C=16 (NB = 1) and C=64
                        // (K = BV = 64) cannot tell those two apart; C=32
                        // can, and confusing them writes outside the qx slot.
                        DataCopy(lv[iv * BV * K], Vt[t0], BV * K);
                        for (int32_t c = 0; c < K / FR; ++c) {
                            DataCopy(lx[iv * BV * K + c * (BV / FR) * FR * FR],
                                     Vt[t0 + c * FR * FR],
                                     DataCopyParams(BV / FR, FR * FR / 16,
                                                    (K / FR) * FR * FR / 16 - FR * FR / 16, 0));
                        }
                    }
                }
                SetFlag<HardEvent::MTE2_MTE1>(e21);
                WaitFlag<HardEvent::MTE2_MTE1>(e21);
                qa.EnQue(la);
                qv.EnQue(lv);
                qx.EnQue(lx);
                qk.EnQue(lk);
                la = qa.DeQue<bfloat16_t>();
                lv = qv.DeQue<bfloat16_t>();
                lx = qx.DeQue<bfloat16_t>();
                lk = qk.DeQue<bfloat16_t>();
                // The d4 A operand is NG * M rows: both value tiles, iv-major,
                // i.e. NG * NB bands of FR rows (2 * BV / FR of them, whatever
                // the chunk size).  It is *not* NB bands: at C=16 the two
                // coincide only because lv's row-major [BV, K] tile is already
                // the L0A fractal order when K = FR = 16, and the load above
                // then fills the whole NG * M * K - which is why the shipped
                // single LoadData(.., NG, ..) covered it.
                for (int32_t b = 0; b < NG * NB; ++b) {
                    LoadData(l0a[b * FR * K], lv[b * FR * K],
                             LoadData2dParams(0, K / FR, 1, 0, 0, false, 0));
                }
                for (int32_t b = 0; b < NB; ++b) {
                    LoadData(l0a[NG * M * K + b * FR * K], la[b * FR * K],
                             LoadData2dParams(0, K / FR, 1, 0, 0, false, 0));
                    LoadDataWithTranspose(l0b[b * (D / FR) * 256], lk[b * FR * D],
                                          LoadData2dTransposeParams(0, D / FR, 1, 0, 0));
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    LoadData(l0b[D * K + iv * BV * K], lx[iv * BV * K],
                             LoadData2dParams(0, BV * K / 256, 1, 0, 0, false, 0));
                }
                SetFlag<HardEvent::MTE1_M>(e1m);
                WaitFlag<HardEvent::MTE1_M>(e1m);
                // L0C, L0A and L0B are free again once the d12 fixpipes
                // retired (stage 1 ends with an FIX_M wait and a barrier).
                // One 128 x 128 Mmad and one Fixpipe write the whole d4 tile.
                // A single Mmad leaves L0C as one contiguous run of 16 x 16
                // C0 fractals, which is the layout the row-major Fixpipe walk
                // with srcStride = mSize reads.
                Mmad(cf, l0a, l0b, MmadParams(NG * M, N_D4, K, 0, false, true));
                for (int32_t iv = 0; iv < nv; ++iv) {
                    Mmad(cf[NG * M * N_D4 + iv * M * N], l0a[NG * M * K],
                         l0b[D * K + iv * BV * K], MmadParams(M, N, K, 0, false, true));
                }
                SetFlag<HardEvent::M_FIX>(emf);
                WaitFlag<HardEvent::M_FIX>(emf);
                {
                    auto ip = FixpipeParamsV220(N_D4, NG * M, NG * M, N_D4, false);
                    ip.quantPre = QuantMode_t::NoQuant;
                    ip.unitFlag = 0;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(
                        D4[static_cast<uint64_t>(bh) * D * D], cf[0], ip);
                }
                for (int32_t iv = 0; iv < nv; ++iv) {
                    auto ip = FixpipeParamsV220(N, M, M, N, false);
                    ip.unitFlag = 0;
                    ip.quantPre = QuantMode_t::NoQuant;
                    Fixpipe<float, float, CFG_ROW_MAJOR>(
                        D3[static_cast<uint64_t>(bh * nv + iv) * TILE],
                        cf[NG * M * N_D4 + iv * M * N], ip);
                }
                SetFlag<HardEvent::FIX_M>(efm);
                WaitFlag<HardEvent::FIX_M>(efm);
                qa.FreeTensor(la);
                qv.FreeTensor(lv);
                qx.FreeTensor(lx);
                qk.FreeTensor(lk);
                // Same M -> MTE1 order as stage 1: d34's operands overwrite the
                // L0A/L0B bytes the three mmads above read.
                SetFlag<HardEvent::M_MTE1>(em1);
                WaitFlag<HardEvent::M_MTE1>(em1);
                CrossCoreSetFlag<2, PIPE_FIX>(FL_C2);
            }
        }
        qc.FreeTensor(cf);
        return;
    }

    if ASCEND_IS_AIV {
        const int32_t ratio = static_cast<int32_t>(GetTaskRation());
        const int32_t raw = static_cast<int32_t>(GetBlockIdx());
        const int32_t blk = ratio == 0 ? raw : raw / ratio;
        // Both subcores execute the same flag sequence; only the value tile
        // they work on differs.
        const int32_t iv = static_cast<int32_t>(GetSubBlockIdx());
        int32_t heads[MAXH];
        int32_t nh = 0;
        for (int32_t h = blk; h < BH && nh < MAXH; h += NBLK) heads[nh++] = h;
        if (nh == 0) return;

        GlobalTensor<bfloat16_t> Vt, S16, Out, Final16;
        GlobalTensor<float> U, D4, Decay, H0, Final32;
        GlobalTensor<float> D1, D2, D3;
        U.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pU));
        Vt.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pVnewT));
        S16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pS16));
        D1.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD1));
        D2.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD2));
        D3.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD3));
        D4.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pD4));
        Decay.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pDecay));
        Out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pOut));
        H0.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pH0));
        Final32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pFinal));
        Final16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pFinal));

        TPipe pipe;
        TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
        TEventID ev3 = pipe.AllocEventID<HardEvent::V_MTE3>();
        TEventID evm2 = pipe.AllocEventID<HardEvent::V_MTE2>();
        TEventID e3v = pipe.AllocEventID<HardEvent::MTE3_V>();
        // UB budget.  The fp32 state is 32 KB per head and all MAXH of them
        // stay resident for the whole loop, so the staging has to fit in the
        // rest of this part's 192 KB UB.  The staging is *aliased by
        // phase* - stage 2 and stage 4 never overlap (each stage opens with a
        // PipeBarrier<PIPE_ALL>, which is exactly the drain that makes the
        // reuse safe), so one set of buffers carries both:
        //   A  fp32 TILE     : vf (2)   | d3f, then of (4)
        //   E  fp32 TILE     : d1f, then sc/vt (2) | d2f (4)
        //   B  TILE bf16     : unused (2) | s16's half (4)
        //   C  TILE bf16     : unused (2) | ob (4)
        //   D  TILE bf16     : vb (2)   | d4's 16-row quarter (4)
        // B/D are sized max(TILE, S_TILE / 2) bytes because at C = 16 the
        // bf16 state half (8 KB) is four times a 16-row tile (2 KB).
        // D3 loads directly into the output accumulator. D2 stays in E until
        // the scaled addition; neither aliases the BF16 output/state stores.
        // In stage 2 the V_MTE2 event after Sub protects d1f before sc/vt
        // overwrite its E buffer through the MTE2 pipe.
        // The aliased layout needs 56.5 KB at C = 64, so MAXH = 4 fits:
        // 128 + 56.5 = 184.5 KB of 192.
        constexpr int32_t HALF_BYTES = (TILE > S_TILE / 2) ? TILE * 2 : S_TILE;
        TBuf<TPosition::VECCALC> uA, uB, uC, uD, uE, udec, us;
        pipe.InitBuffer(uA, TILE * sizeof(float));
        pipe.InitBuffer(uB, HALF_BYTES);
        pipe.InitBuffer(uC, TILE * sizeof(bfloat16_t));
        pipe.InitBuffer(uD, HALF_BYTES);
        pipe.InitBuffer(uE, TILE * sizeof(float));
        pipe.InitBuffer(udec, D * sizeof(float));
        pipe.InitBuffer(us, MAXH * S_TILE * sizeof(float));

        LocalTensor<float> vf = uA.Get<float>();
        LocalTensor<float> of = uA.Get<float>();
        LocalTensor<bfloat16_t> sc = uE.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> vt = sc[TILE];
        LocalTensor<float> d1f = uE.Get<float>();
        LocalTensor<float> d2f = uE.Get<float>();
        LocalTensor<float> d3f = uA.Get<float>();
        LocalTensor<bfloat16_t> s16 = uB.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> ob = uC.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> vb = uD.Get<bfloat16_t>();
        LocalTensor<float> d4 = uD.Get<float>();
        LocalTensor<float> dec = udec.Get<float>();
        LocalTensor<float> st = us.Get<float>();

        for (int32_t s = 0; s < nh; ++s) {
            const int32_t task = heads[s] * nv + iv;
            LocalTensor<float> state = st[s * S_TILE];
            if (pH0 == nullptr) {
                Duplicate(state, 0.0f, S_TILE);
                PipeBarrier<PIPE_V>();
            } else {
                DataCopy(state, H0[static_cast<uint64_t>(task) * S_TILE],
                         DataCopyParams(BV, D / 8, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
            }
            // Published in two halves: a full S_TILE bf16 buffer on top of the
            // fp32 state does not fit at MAXH = 4.  The second half's Cast
            // rewrites s16 while the first half's MTE3 copy may still be
            // reading it, so the two halves are separated by an MTE3 drain
            // (exactly the hazard the in-loop publish guards the same way).
            for (int32_t hf = 0; hf < 2; ++hf) {
                // NB: PipeBarrier<PIPE_MTE3> does NOT order a later V-pipe
                // Cast after an MTE3 read of the same buffer.  A global
                // barrier does.
                if (hf == 1) PipeBarrier<PIPE_ALL>();
                LocalTensor<float> sh = state[hf * (S_TILE / 2)];
                Cast(s16, sh, RoundMode::CAST_RINT, S_TILE / 2);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(ev3);
                WaitFlag<HardEvent::V_MTE3>(ev3);
                DataCopy(S16[static_cast<uint64_t>(task) * S_TILE + hf * (BV / 2) * D], s16,
                         DataCopyParams(BV / 2, D / 16, 0, 0));
            }

            PipeBarrier<PIPE_ALL>();
        }
        for (int32_t s = 0; s < nh; ++s) {
            CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);   // prologue: first d12 waits
        }

        for (int32_t chunk = 0; chunk < NT; ++chunk) {
            // Stage 2: v_new = u - d1 (and its transpose) per head.
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t task = bh * nv + iv;
                const int32_t c = bh * NT + chunk;
                // The previous iteration's vector work may still be reading the
                // UB staging buffers (ub/vf/vb/sc) that this iteration loads
                // into.  A one-chunk-per-launch kernel never sees this WAR
                // hazard; a loop does.
                PipeBarrier<PIPE_ALL>();
                const uint64_t u0 = static_cast<uint64_t>(c) * M * D;
                const uint64_t out0 = static_cast<uint64_t>(task) * TILE;
                // Only this subcore's value half of U is read: the block form
                // gathers the iv-th 64-column run of each of the M rows.
                DataCopy(vf, U[u0 + iv * BV], DataCopyParams(M, BV / 8, (D - BV) / 8, 0));
                CrossCoreWaitFlag(FL_C1);
                DataCopy(d1f, D1[out0], DataCopyParams(M, BV / 8, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                PipeBarrier<PIPE_V>();
                // v_new = u - d1 for all M rows in one instruction: both the
                // dst and the d1 operand step by one BV-float row (BV / 8
                // blocks), the u operand is the same-shape fp32 tile.
                Sub(vf, vf, d1f, BV, M,
                    BinaryRepeatParams(1, 1, 1, BV / 8, BV / 8, BV / 8));
                Cast(vb, vf, RoundMode::CAST_RINT, TILE);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(evm2);
                WaitFlag<HardEvent::V_MTE2>(evm2);
                // One 16 x 16 block per call: the source block of the
                // destination's (j band j0, m band m0) is vb's (m band m0,
                // j band j0) - the m index is vb's *row* - and the rows of
                // both are BV / 16 blocks apart.  The destination is the
                // packed block order of the [BV, M] tile the transpose below
                // writes, i.e. its first index (j) is the row.
                for (int32_t j0 = 0; j0 < BV / FR; ++j0) {
                    for (int32_t m0 = 0; m0 < NB; ++m0) {
                        DataCopy(sc[(j0 * NB + m0) * FR * FR], vb[m0 * FR * BV + j0 * FR],
                                 DataCopyParams(FR, 1, BV / FR - 1, 0));
                    }
                }
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                for (int32_t bl = 0; bl < (BV / FR) * NB; ++bl) {
                    AscendC::Transpose(vt[bl * FR * FR], sc[bl * FR * FR]);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(ev3);
                WaitFlag<HardEvent::V_MTE3>(ev3);
                // The transposes above leave the
                // [BV, M] tile in packed 16 x 16 block order, exactly the A
                // operand's band walk rather than a row-major tile.  Vt gets
                // that packed order in one call; the B operand's block
                // transpose is done on the MTE2 side in stage 3.
                DataCopy(Vt[out0], vt, BV * M);
                CrossCoreSetFlag<2, PIPE_MTE3>(FL_V);
            }
            // Stage 4: out = d2 * scale + d3 and the state recurrence.
            for (int32_t s = 0; s < nh; ++s) {
                const int32_t bh = heads[s];
                const int32_t task = bh * nv + iv;
                const int32_t c = bh * NT + chunk;
                // Same WAR hazard as stage 2: d2/d3/d4/dec, ob and s16 are all
                // rewritten here while the previous iteration's reads of them
                // (and its MTE3 copies out of ob/s16) may still be in flight.
                PipeBarrier<PIPE_ALL>();
                LocalTensor<float> state = st[s * S_TILE];
                const uint64_t t0 = static_cast<uint64_t>(task) * TILE;
                const uint64_t d4base = static_cast<uint64_t>(bh) * D * D +
                                        static_cast<uint64_t>(iv) * BV * D;
                DataCopy(d2f, D2[t0], DataCopyParams(M, BV / 8, 0, 0));
                DataCopy(dec, Decay[static_cast<uint64_t>(c) * D], DataCopyParams(1, D / 8, 0, 0));
                CrossCoreWaitFlag(FL_C2);
                DataCopy(d3f, D3[t0], DataCopyParams(M, BV / 8, 0, 0));
                DataCopy(d4, D4[d4base], DataCopyParams(BV / 4, D / 8, 0, 0));
                SetFlag<HardEvent::MTE2_V>(e2v);
                WaitFlag<HardEvent::MTE2_V>(e2v);
                // out = d2 * scale + d3 is accumulated in fp32 and only then
                // rounded to bf16, exactly like the separated outstate kernel.
                Muls(d2f, d2f, scale, TILE);
                Add(of, d3f, d2f, TILE);
                Cast(ob, of, RoundMode::CAST_RINT, TILE);
                PipeBarrier<PIPE_V>();
                // The state recurrence walks the tile in four 16-row quarters:
                // a quarter of the bf16 staging (S_TILE / 4) is what
                // lets the d4 tile share the stage-2 v_new buffer, and 16 rows
                // is the smallest slice the 16-block repeat stride still walks
                // as whole rows.  Each next quarter's load and its MTE2->V
                // wait hide behind the current quarter's vector work exactly
                // as the two halves did.
                const bool nextChunk = chunk + 1 < NT;
                const bool publish = nextChunk || (pFinal != nullptr && finalBF16);
                for (int32_t hf = 0; hf < 4; ++hf) {
                    if (hf > 0) {
                        WaitFlag<HardEvent::MTE2_V>(e2v);
                        // The previous quarter's S16 store has to have read
                        // s16 before this quarter's Cast rewrites it.  That is
                        // an MTE3 -> V dependency across pipes, which a
                        // PipeBarrier<PIPE_MTE3> does NOT provide (it only
                        // orders MTE3 against MTE3), so it needs the event
                        // pair.
                        if (publish) WaitFlag<HardEvent::MTE3_V>(e3v);
                    }
                    LocalTensor<float> sh = state[hf * (S_TILE / 4)];
                    Mul(sh, sh, dec, 64, BV / 4, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
                    Mul(sh[64], sh[64], dec[64], 64, BV / 4, BinaryRepeatParams(1, 1, 1, 16, 16, 0));
                    Add(sh, sh, d4, S_TILE / 4);
                    PipeBarrier<PIPE_V>();
                    if (publish) Cast(s16, sh, RoundMode::CAST_RINT, S_TILE / 4);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE3>(ev3);
                    WaitFlag<HardEvent::V_MTE3>(ev3);
                    if (hf == 0) {
                        // [b, t, h, D] directly: one 4-block (128 B) run per
                        // chunk row, the next row NH * D elements further on.
                        const int32_t hh = bh - (bh / NH) * NH;
                        const uint64_t token = static_cast<uint64_t>(bh / NH) * T +
                                               static_cast<uint64_t>(chunk) * M;
                        const uint64_t obase = token * NH * D +
                                               static_cast<uint64_t>(hh) * D + iv * BV;
                        const int32_t remaining = T - chunk * M;
                        const int32_t rows = remaining < M ? remaining : M;
                        DataCopy(Out[obase], ob,
                                 DataCopyParams(rows, BV / 16, 0,
                                                static_cast<uint16_t>(NH * D / 16 - BV / 16)));
                    }
                    if (publish) {
                        const uint64_t offset = static_cast<uint64_t>(task) * S_TILE + hf * (BV / 4) * D;
                        if (nextChunk) {
                            DataCopy(S16[offset], s16, DataCopyParams(BV / 4, D / 16, 0, 0));
                        } else {
                            DataCopy(Final16[offset], s16, DataCopyParams(BV / 4, D / 16, 0, 0));
                        }
                    }
                    if (hf < 3) {
                        // ... and release the buffer for the next quarter once
                        // this store has actually read it.
                        if (publish) SetFlag<HardEvent::MTE3_V>(e3v);
                        SetFlag<HardEvent::V_MTE2>(evm2);
                        WaitFlag<HardEvent::V_MTE2>(evm2);
                        DataCopy(d4, D4[d4base + (hf + 1) * (BV / 4) * D],
                                 DataCopyParams(BV / 4, D / 8, 0, 0));
                        SetFlag<HardEvent::MTE2_V>(e2v);
                    }
                }
                if (nextChunk) CrossCoreSetFlag<2, PIPE_MTE3>(FL_R);
            }
        }
        SetFlag<HardEvent::V_MTE3>(ev3);
        WaitFlag<HardEvent::V_MTE3>(ev3);
        for (int32_t s = 0; pFinal != nullptr && !finalBF16 && s < nh; ++s) {
            const int32_t task = heads[s] * nv + iv;
            DataCopy(Final32[static_cast<uint64_t>(task) * S_TILE], st[s * S_TILE],
                     DataCopyParams(BV, D / 8, 0, 0));
        }
        PipeBarrier<PIPE_ALL>();
        return;
    }
}

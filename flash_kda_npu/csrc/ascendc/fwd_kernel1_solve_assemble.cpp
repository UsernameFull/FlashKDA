// K1 stage 3c: the coupling block of the two-level solve.
//
//   X21 = X22 @ Lneg21 @ X11                (all three [M, M], M = KDA_CHUNK/2)
//
// With KDA_SOLVE_WIDE_SUBB = 2, the wide kernel solves the two diagonal M x M
// triangles.  This kernel forms the remaining lower-left coupling block of the
// 2x2 block inverse
//
//   A_inv = [[X11, 0], [-X22 L21 X11, X22]]
//
// which holds for I + L split as [[I+L11, 0], [L21, I+L22]] with L strictly
// lower.  X11/X22 arrive in Xb, and Lneg21 is the negated bf16 export.  This
// kernel is the only writer of the parent tile's lower-left block.
// This kernel's output is the only writer of the parent tile's lower-left
// block; the wide kernel writes its diagonal and zeroes its strict upper
// triangle (the Cube solve reads all of it).
//
// P = Lneg21 @ X11 goes through GM because L0C has no path back into
// L0A/L0B.  The two Mmads are split into passes, with each pass's L1 loads
// issued before its arithmetic.  `loadMode` selects the load/store path:
//
//   0  one B1 queue per operand and per-chunk calls
//   1  explicit B1 buffers with batched A/B loads
//   2  mode 1 plus P retained on chip in NZ; pass 1 builds L0B from that
//      region with LoadDataWithTranspose
//   3  mode 2 plus a batched pass-1 A16 store
//   4  mode 2 plus one whole-window Xb fill for both passes
//   5  mode 3 and mode 4 combined
//
// Mode 0 requires the NC-deep queue shape; NC=4 is the supported depth.
#include "kernel_operator.h"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 64
#endif
constexpr int32_t PC = KDA_CHUNK;  // parent chunk size (the wide kernel's PC)
constexpr int32_t M = PC / 2;      // diagonal sub-block size
constexpr int32_t MM = M * M;
constexpr int32_t KF = M / 16;     // 16-row fractal bands
constexpr int32_t BANDE = KF * 256;  // elements of one [16, M] band (512 at M=32)
constexpr int32_t BAND = BANDE * 2;  // bytes of one band (1 KB at M=32)
#ifndef KDA_ASM_NCHUNK
#define KDA_ASM_NCHUNK 4
#endif
constexpr int32_t NC = KDA_ASM_NCHUNK;
constexpr int32_t PASSES = 2;
constexpr int32_t LBSZ = MM * 2;   // one [M, M] bf16 operand
constexpr int32_t SLOTS = PASSES * NC;  // one L0 slot per (pass, chunk) unit

// One chunk's L0 chain and store, shared by both load paths.  `defer_pass1`
// suppresses pass 1's A16 store when the caller batches the block's four
// lower-left stores after the chunk loop.
static __aicore__ inline void assemble_chunk(
    int32_t pass, int32_t slot, bool onchip, bool window,
    LocalTensor<bfloat16_t> la, LocalTensor<bfloat16_t> lb,
    LocalTensor<bfloat16_t> lp, LocalTensor<uint8_t>& a8,
    LocalTensor<uint8_t>& b8, LocalTensor<float>& cfall,
    GlobalTensor<bfloat16_t>& P, GlobalTensor<bfloat16_t>& A16,
    int32_t c0, int32_t ch, TEventID e1m, TEventID emf, bool defer_pass1) {
    LocalTensor<bfloat16_t> a = a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    if (pass == 1 && window) {
        // The whole-window fill lands pass 1's A in the 16-row (n-block-major)
        // order that the four crossed calls undo, so it reads straight through
        // in one call and the permutation is the fill's, not the lift's.
        LoadData(a, la, LoadData2dParams(0, KF * KF, 1, 0, 0, false, 0));
    } else {
        for (int32_t dd = 0; dd < KF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(a[(mm * KF + dd) * 256], la[(dd * KF + mm) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
    }
    if (pass == 0 || !onchip) {
        LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, KF * KF, 1, 0, 0));
    } else {
        // P out of L1 in NZ order: the four 512-byte fractals are ordered
        // [n-block][k-block] instead of the Nd2Nz source's [k-block][n-block],
        // so L0B's slots take source blocks 0, 2, 1, 3.
        const int32_t src = ch * MM;
        LoadDataWithTranspose(b, lp[src], LoadData2dTransposeParams(0, 1, 1, 0, 0));
        LoadDataWithTranspose(b[2 * 256], lp[src + 256],
                              LoadData2dTransposeParams(0, 1, 1, 0, 0));
        LoadDataWithTranspose(b[1 * 256], lp[src + 512],
                              LoadData2dTransposeParams(0, 1, 1, 0, 0));
        LoadDataWithTranspose(b[3 * 256], lp[src + 768],
                              LoadData2dTransposeParams(0, 1, 1, 0, 0));
    }
    SetFlag<HardEvent::MTE1_M>(e1m);
    WaitFlag<HardEvent::MTE1_M>(e1m);
    LocalTensor<float> cf = cfall[slot * MM];
    Mmad(cf, a, b, MmadParams(M, M, M, 0, false, true));
    SetFlag<HardEvent::M_FIX>(emf);
    WaitFlag<HardEvent::M_FIX>(emf);
    // Pass 0 writes the intermediate row-major; pass 1 writes the parent
    // tile's lower-left block, whose rows are PC elements apart instead of M.
    auto ip = FixpipeParamsV220(M, M, M, (pass == 0) ? M : PC, false);
    ip.quantPre = QuantMode_t::F322BF16;
    ip.unitFlag = 0;
    if (pass == 0 && onchip) {
        auto ipnz = FixpipeParamsV220(M, M, M, M, false);
        ipnz.quantPre = QuantMode_t::F322BF16;
        ipnz.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_NZ>(lp[ch * MM], cf, ipnz);
    } else if (pass == 0) {
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
            P[static_cast<uint64_t>(c0 + ch) * MM], cf, ip);
    } else if (!defer_pass1) {
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
            A16[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], cf, ip);
    }
}

extern "C" __global__ __aicore__ void kda_solve_assemble(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, GM_ADDR pP, int32_t C,
    int32_t loadMode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    const bool batched = (loadMode != 0);
    const bool onchip = (loadMode >= 2);
    // Mode 3's batched pass-1 store needs the L0C slot spacing expressible in
    // the nd words' 1 KB units (srcNdStride = MM / 256): true for M = 32 and
    // M = 16, false for M = 8, where the mode falls back to per-chunk stores.
    // The assemble only launches at C = 64 in production, so the fallback is
    // for the compile-only legs and any future geometry.
    constexpr bool CAN_BATCH_STORE = (MM % 256) == 0;
    // Mode 3 adds a batched store, mode 4 adds the whole-window fill, and
    // mode 5 composes both.
    const bool batched_store = (loadMode == 3 || loadMode == 5) && CAN_BATCH_STORE;
    const bool window = (loadMode >= 4);
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    // Mode 0 uses one queue per chunk; NC=4 is required because a whole pass's
    // loads are issued before its arithmetic begins.
    TQue<QuePosition::B1, NC> qa, qb;
    pipe.InitBuffer(qa, NC, LBSZ);
    pipe.InitBuffer(qb, NC, LBSZ);
    // Mode 1's L1 shape: the same 2 * NC * LBSZ bytes as explicit buffers, one
    // per operand, written by one batched call each.  Both shapes are declared
    // because the mode is a runtime argument; together they are 32 KB of L1.
    TBuf<TPosition::B1> bufA, bufB;
    pipe.InitBuffer(bufA, NC * LBSZ);
    pipe.InitBuffer(bufB, NC * LBSZ);
    // Mode 2's P tiles, one [M, M] bf16 per chunk of the block (8 KB at NC = 4,
    // M = 32): written by pass 0's fixpipe, read by pass 1's L0B fill.  It
    // replaces the GM round trip, not L1 - the buffer only exists when the mode
    // is on, so mode 0/1 leave it as 8 KB of unused L1.
    TBuf<TPosition::B1> bufP;
    pipe.InitBuffer(bufP, NC * LBSZ);
    // Mode 4's whole-Xb window: one 16-row band per band of every chunk's two
    // [M, M] blocks (4 bands of 512 elements at M = 32), 16 KB at NC = 4.  It
    // replaces nothing - modes 0-3 leave it as unused L1.
    TBuf<TPosition::B1> bufX;
    pipe.InitBuffer(bufX, NC * 2 * MM * 2);
    // One L0 slot per (pass, chunk) unit: 2 * NC * 4 KB = 32 KB of the 128 KB
    // L0C, 16 KB of each of the 64 KB L0A/L0B.
    LocalTensor<float> cfall(TPosition::CO1, 0, SLOTS * MM);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, SLOTS * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, SLOTS * MM * 2);
    GlobalTensor<bfloat16_t> A16, Xb, Lneg, P;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    P.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pP));
    LocalTensor<bfloat16_t> laAll = bufA.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lbAll = bufB.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lpAll = bufP.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lxAll = bufX.Get<bfloat16_t>();

    if (window) {
        // Both passes' Xb operands in one call: 4*nch ND matrices of 16 rows at
        // the uniform band stride.  Band 0 of a chunk's window is pass 0's B
        // (read with LoadDataWithTranspose, exactly the layout mode 2's
        // per-chunk band-merged call produced) and band 1 is pass 1's A.
        DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
                 Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
        DataCopy(lxAll, Xb[static_cast<uint64_t>(c0) * 2 * MM],
                 Nd2NzParams(4 * nch, 16, M, BANDE, M, 16, 1, BANDE));
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
    }
    for (int32_t pass = 0; pass < PASSES; ++pass) {
        if (batched && !window) {
            // Same bytes, fewer calls: the A tiles of the block in one Nd2Nz
            // (the source stride is MM for Lneg and 2*MM for Xb's pass-1
            // block), a chunk's two B bands merged, and in pass 1 the whole
            // block's B operand in one call (P is chunk-contiguous, so its
            // eight 1 KB bands are one 8 KB transfer).  1 + NC calls for pass
            // 0, 2 for pass 1.
            if (pass == 0) {
                DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
                         Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
                for (int32_t ch = 0; ch < nch; ++ch) {
                    DataCopy(lbAll[ch * MM],
                             Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM],
                             Nd2NzParams(KF, 16, M, BANDE, M, 16, 1, BANDE));
                }
            } else {
                DataCopy(laAll, Xb[static_cast<uint64_t>(c0) * 2 * MM + MM],
                         Nd2NzParams(nch, M, M, 2 * MM, M, M, 1, MM));
                if (!onchip) {
                    // Mode 2 has no P in GM: pass 1's B operand comes from the
                    // L1 tiles pass 0's fixpipe just wrote.
                    DataCopy(lbAll, P[static_cast<uint64_t>(c0) * MM],
                             Nd2NzParams(KF * nch, 16, M, BANDE, M, 16, 1, BANDE));
                }
            }
            SetFlag<HardEvent::MTE2_MTE1>(e21);
            WaitFlag<HardEvent::MTE2_MTE1>(e21);
        }
        if (batched) {
            if (window) {
                for (int32_t ch = 0; ch < nch; ++ch) {
                    LocalTensor<bfloat16_t> la =
                        (pass == 0) ? laAll[ch * MM] : lxAll[ch * 2 * MM + MM];
                    LocalTensor<bfloat16_t> lb =
                        (pass == 0) ? lxAll[ch * 2 * MM] : lpAll;
                    assemble_chunk(pass, pass * NC + ch, true, true, la, lb,
                                   lpAll, a8, b8, cfall, P, A16, c0, ch, e1m,
                                   emf, batched_store);
                }
                if (pass == 1 && batched_store) {
                    // Mode 5: the window fill and the batched store composed
                    // (docs 11.61).  Same nd call as mode 3's - here the
                    // pass-1 slots are one region too (slot = pass * NC + ch)
                    // and every Mmad is already M_FIX-confirmed by its
                    // chunk's wait.
                    auto ipb = FixpipeParamsV220(
                        M, M, M, PC, false, QuantMode_t::F322BF16, 0,
                        static_cast<uint16_t>(nch),
                        static_cast<uint16_t>(MM / 256),
                        static_cast<uint16_t>(PC * PC), 0);
                    Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                        A16[static_cast<uint64_t>(c0) * PC * PC + M * PC],
                        cfall[NC * MM], ipb);
                }
                PipeBarrier<PIPE_ALL>();
                continue;
            }
            for (int32_t ch = 0; ch < nch; ++ch) {
                assemble_chunk(pass, pass * NC + ch, onchip, false, laAll[ch * MM],
                               lbAll[ch * MM], lpAll, a8, b8, cfall, P, A16,
                               c0, ch, e1m, emf, batched_store);
            }
            if (pass == 1 && batched_store) {
                // Section 11.42: the four lower-left blocks are one nd call -
                // srcNdStride is the L0C slot spacing (MM floats = MM/256 KB),
                // dstNdStride the parent tile (PC * PC elements).  The four
                // Mmads are already M_FIX-confirmed by their chunk's wait, so
                // the FIX pipe can read the slots directly.
                auto ipb = FixpipeParamsV220(
                    M, M, M, PC, false, QuantMode_t::F322BF16, 0,
                    static_cast<uint16_t>(nch), static_cast<uint16_t>(MM / 256),
                    static_cast<uint16_t>(PC * PC), 0);
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(
                    A16[static_cast<uint64_t>(c0) * PC * PC + M * PC],
                    cfall[NC * MM], ipb);
            }
        } else {
            for (int32_t ch = 0; ch < nch; ++ch) {
                auto la = qa.AllocTensor<bfloat16_t>();
                auto lb = qb.AllocTensor<bfloat16_t>();
                // The A operand is one whole-tile Nd2Nz call (the layout the
                // crossed LoadData indices below undo, as in k1_solve_wu_cube);
                // the B operand goes through LoadDataWithTranspose, which wants
                // the band-major order the per-band calls here produce.
                if (pass == 0) {
                    DataCopy(la, Lneg[static_cast<uint64_t>(c0 + ch) * MM],
                             Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                    for (int32_t mm = 0; mm < KF; ++mm) {
                        DataCopy(lb[mm * KF * 256],
                                 Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM + mm * 16 * M],
                                 Nd2NzParams(1, 16, M, 0, M, 16, 1, 0));
                    }
                } else {
                    DataCopy(la, Xb[static_cast<uint64_t>(c0 + ch) * 2 * MM + MM],
                             Nd2NzParams(1, M, M, 0, M, M, 1, 0));
                    for (int32_t mm = 0; mm < KF; ++mm) {
                        DataCopy(lb[mm * KF * 256],
                                 P[static_cast<uint64_t>(c0 + ch) * MM + mm * 16 * M],
                                 Nd2NzParams(1, 16, M, 0, M, 16, 1, 0));
                    }
                }
                qa.EnQue(la);
                qb.EnQue(lb);
            }
            SetFlag<HardEvent::MTE2_MTE1>(e21);
            WaitFlag<HardEvent::MTE2_MTE1>(e21);
            for (int32_t ch = 0; ch < nch; ++ch) {
                auto la = qa.DeQue<bfloat16_t>();
                auto lb = qb.DeQue<bfloat16_t>();
                assemble_chunk(pass, pass * NC + ch, false, false, la, lb,
                               lpAll, a8, b8, cfall, P, A16, c0, ch, e1m, emf,
                               false);
                qa.FreeTensor(la);
                qb.FreeTensor(lb);
            }
        }
        // Pass 1 reads the P tiles pass 0 wrote.
        PipeBarrier<PIPE_ALL>();
    }
    PipeBarrier<PIPE_ALL>();
}

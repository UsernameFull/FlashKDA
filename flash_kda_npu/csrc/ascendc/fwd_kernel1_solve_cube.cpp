// K1 stage 3b: the two solve right-hand sides on the Cube.
//
//   w = bf16(A_inv) @ (k * beta * exp2(gate))   A16 [16,16] bf16, rk [16,128] bf16
//   u = bf16(A_inv) @ (v * beta)                A16 [16,16] bf16, rv [16,128] bf16
//
// This replaces the vector row-broadcast + column reduction (MatVec2) that used
// to run inside kda_solve_wu_kernel.  rk/rv are consumed in their natural
// [16,128] layout: Nd2Nz turns them into 16x16 fractals in L1 and
// LoadDataWithTranspose transposes each fractal into the B2 zN layout the Cube
// wants, so no transposed staging buffer is needed.
//
// One AIC block handles NCHUNK chunks: the matmul itself is far too small to
// hide a block's fixed cost (measured 1.26 us per block at [1,8192,32]), so the
// loads of every chunk are issued up front and the mmads follow.
//
// A16 residency (docs section 11.38): both passes of a block read the same
// A16 tiles, so `a16Mode` (api.cube_a16_resident(), KDA_CUBE_A16_RESIDENT,
// read per call) picks whether the second read happens at all.  0 keeps the
// shipped per-pass load through the qa queue; 1 fills one L1 buffer of
// NC * M * K * 2 bytes once, before the pass loop, and both passes read it.
// The candidate is the second pass's read only - section 11.35 measured it as
// an L2 hit at 1439 GB/s marginal against the first read's 839 GB/s cold rate,
// which is why the transcription priced the whole thing at 0.070 ms rather
// than the 0.16 a flat bandwidth projection gave.
//
// Load shape (docs section 11.63): the on-board PipeUtilization account reads
// the block wall as 4.382 us, of which MTE2 is 3.271 (74.6%) - 16 calls of
// 4 KB (this file's per-band RHS loop) plus 2 of 8 KB (A16).  A chunk's KF
// [16, D] bands are one strided ND run, so `loadMode` (api.cube_load_mode(),
// KDA_CUBE_LOADS, read per call) picks how they travel: 0 keeps the KF single
// band calls, 1 issues one Nd2Nz with ndNum = KF, which lands byte-identical
// in L1 (same per-matrix nValue/dValue/dstNzC0Stride, bands spanned by the
// srcNdMatrixStride/dstNzMatrixStride = 16 * D elements that the single calls
// used as their destination offsets).  Measured 2026-10-08 in
// tools/probe_solve_cube_knobs.py: W/U bit-identical and the cube's isolated
// replay 1.463 -> 0.996 ms.
#include "kernel_operator.h"
using namespace AscendC;
#ifndef KDA_CHUNK
#define KDA_CHUNK 16
#endif
constexpr int32_t M = KDA_CHUNK, K = KDA_CHUNK, D = 128;
constexpr int32_t KF = K / 16;   // 16-row bands of a chunk-sized tile
constexpr int32_t DF = D / 16;
#ifndef KDA_WU_NCHUNK
#define KDA_WU_NCHUNK 4
#endif
constexpr int32_t NC = KDA_WU_NCHUNK;

extern "C" __global__ __aicore__ void kda_solve_wu_cube_kernel(
    GM_ADDR pA16, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pW, GM_ADDR pU, int32_t C,
    int32_t a16Mode, int32_t loadMode) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    // a16Mode (docs section 11.38): 0 re-reads the block's A16 tile at the top
    // of each of the two passes, 1 keeps the block's NC tiles in L1 for both.
    // tools/probe_solve_cube_a16.py priced the candidate at 0.070 ms in a
    // transcription of this kernel; tools/probe_solve_cube_a16_resident.py is
    // the same candidate in the kernel itself, which is what decides it.
    const bool a16res = (a16Mode != 0);
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TQue<QuePosition::B1, NC> qa, qb;
    pipe.InitBuffer(qa, NC, M * K * 2);
    pipe.InitBuffer(qb, NC, D * K * 2);
    // a16Mode 1's resident copy: the same NC * M * K * 2 bytes as the qa queue
    // it replaces, filled once per block instead of once per pass.
    TBuf<TPosition::B1> bufA16;
    pipe.InitBuffer(bufA16, NC * M * K * 2);
    // One L0C slot per (pass, chunk) unit: 2 * NC * 8 KB = 64 KB of the 128 KB
    // L0C.  Slots are never reused inside a block, so no FIX_M -> M ordering is
    // needed between a Fixpipe and the next Mmad; with the old NC-deep queue
    // the two passes shared slots and that drain cost 0.14 ms of the 1.94 ms
    // stage (removing it is bit-identical: d_out 0.00e+00).
    LocalTensor<float> cfall(TPosition::CO1, 0, 2 * NC * M * D);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, 2 * NC * M * K * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, 2 * NC * D * K * 2);
    LocalTensor<bfloat16_t> a16l1 = bufA16.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> A16, Rk, Rv, W, U;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Rk.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRk));
    Rv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pRv));
    W.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pW));
    U.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pU));

    if (a16res) {
        // One read of the block's A16 for both passes - the second pass's read
        // is the L2 hit section 11.35 priced at 0.070 ms.
        for (int32_t ch = 0; ch < nch; ++ch) {
#if KDA_CHUNK > 16
            DataCopy(a16l1[ch * M * K], A16[static_cast<uint64_t>(c0 + ch) * M * K],
                     Nd2NzParams(1, M, K, 0, K, M, 1, 0));
#else
            DataCopy(a16l1[ch * M * K], A16[static_cast<uint64_t>(c0 + ch) * M * K], M * K);
#endif
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);
    }
    for (int32_t pass = 0; pass < 2; ++pass) {
        // Issue every chunk's L1 load before touching L0: the GM round trips of
        // one chunk then overlap the arithmetic of the previous one.  Each pass
        // loads its own right-hand side (pass 0 = rk -> W, pass 1 = rv -> U).
        for (int32_t ch = 0; ch < nch; ++ch) {
            // a16Mode 1 must not draw from qa at all: slots that are
            // AllocTensor'd but never EnQueued are still taken out of the
            // queue, and the second pass then blocks on it (measured: the
            // kernel spins at 100% AICore instead of failing).
            if (!a16res) {
                auto la = qa.AllocTensor<bfloat16_t>();
#if KDA_CHUNK > 16
                // One whole-tile call, whose dstNzC0Stride = nValue layout is
                // column-block-major (probe /tmp/nzprobe.py), so the L0A read
                // below crosses its indices.  That crossing is free; the
                // per-band form would be KF calls and a "Nd2Nz-shaped" copy
                // costs ~600 ns more per call than the plain one on this part
                // (measured at KDA_CHUNK = 16: 8 such calls per block are worth
                // 2.6 ms of the 4.4 ms solve), and this kernel is wave-bound.
                DataCopy(la, A16[static_cast<uint64_t>(c0 + ch) * M * K],
                         Nd2NzParams(1, M, K, 0, K, M, 1, 0));
#else
                // KF == 1 (the 16-row chunk): the band loop below would run
                // once and convert exactly one fractal, but the
                // Nd2Nz-parameterised copy costs ~600 ns more per call than
                // the plain one on this part and this kernel is wave-bound,
                // not bandwidth-bound: measured at [1,8192,96,128], the
                // per-band form costs 4.39 ms of the solve against 1.75 for
                // the plain form (both correct).  So the 16-row build keeps
                // the original single calls.
                DataCopy(la, A16[static_cast<uint64_t>(c0 + ch) * M * K], M * K);
#endif
                qa.EnQue(la);
            }
            auto lb = qb.AllocTensor<bfloat16_t>();
#if KDA_CHUNK > 16
            GlobalTensor<bfloat16_t> &rhs = (pass == 0) ? Rk : Rv;
            if (loadMode == 1) {
                // The chunk's KF bands in one call (docs section 11.63).
                DataCopy(lb, rhs[static_cast<uint64_t>(c0 + ch) * M * D],
                         Nd2NzParams(KF, 16, D, 16 * D, D, 16, 1, 16 * D));
            } else {
                for (int32_t mm = 0; mm < KF; ++mm) {
                    DataCopy(lb[mm * DF * 256],
                             rhs[static_cast<uint64_t>(c0 + ch) * M * D + mm * 16 * D],
                             Nd2NzParams(1, 16, D, 0, D, 16, 1, 0));
                }
            }
#else
            if (pass == 0) {
                DataCopy(lb, Rk[static_cast<uint64_t>(c0 + ch) * M * D],
                         Nd2NzParams(1, M, D, 0, D, M, 1, 0));
            } else {
                DataCopy(lb, Rv[static_cast<uint64_t>(c0 + ch) * M * D],
                         Nd2NzParams(1, M, D, 0, D, M, 1, 0));
            }
#endif
            qb.EnQue(lb);
        }
        SetFlag<HardEvent::MTE2_MTE1>(e21);
        WaitFlag<HardEvent::MTE2_MTE1>(e21);

        for (int32_t ch = 0; ch < nch; ++ch) {
            LocalTensor<bfloat16_t> la;
            if (a16res) {
                la = a16l1[ch * M * K];
            } else {
                la = qa.DeQue<bfloat16_t>();
            }
            auto lb = qb.DeQue<bfloat16_t>();
            // L0A/L0B are indexed by the pass as well: the previous pass's mmads
            // may still be reading the same chunk slot.
            const int32_t slot = pass * NC + ch;
            LocalTensor<bfloat16_t> a = a8[slot * M * K * 2].ReinterpretCast<bfloat16_t>();
            LocalTensor<bfloat16_t> b = b8[slot * D * K * 2].ReinterpretCast<bfloat16_t>();
            // A16 arrives column-block-major (see the load above) while L0A
            // wants its fractal grid (band, k-block) - the same crossing the
            // K2 loop's W/Qg loads undo.  L0B keeps the band-major L1 order
            // with the 16x16 fractal transposed by the load.
            for (int32_t dd = 0; dd < K / 16; ++dd) {
                for (int32_t mm = 0; mm < KF; ++mm) {
                    LoadData(a[(mm * (K / 16) + dd) * 256],
                             la[(dd * KF + mm) * 256],
                             LoadData2dParams(0, 1, 1, 0, 0, false, 0));
                }
            }
            LoadDataWithTranspose(b, lb, LoadData2dTransposeParams(0, KF * DF, 1, 0, 0));
            SetFlag<HardEvent::MTE1_M>(e1m);
            WaitFlag<HardEvent::MTE1_M>(e1m);
            LocalTensor<float> cf = cfall[slot * M * D];
            Mmad(cf, a, b, MmadParams(M, D, K, 0, false, true));
            SetFlag<HardEvent::M_FIX>(emf);
            WaitFlag<HardEvent::M_FIX>(emf);
            auto ip = FixpipeParamsV220(D, M, M, D, false);
            ip.quantPre = QuantMode_t::F322BF16;
            ip.unitFlag = 0;
            if (pass == 0) {
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(W[static_cast<uint64_t>(c0 + ch) * M * D], cf, ip);
            } else {
                Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(U[static_cast<uint64_t>(c0 + ch) * M * D], cf, ip);
            }
            if (!a16res) {
                qa.FreeTensor(la);
            }
            qb.FreeTensor(lb);
        }
    }
}

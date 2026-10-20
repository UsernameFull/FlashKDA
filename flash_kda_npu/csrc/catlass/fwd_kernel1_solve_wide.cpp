// CATLASS tile implementation; schedule and buffer ownership follow the AscendC backend.
// K1 stage 3: 16x16 forward-substitution solve, WIDE_NCHUNK chunks per vector
// instruction.
//
//   A_inv = (I + L)^{-1},  A_inv[i,:] = e_i + sum_{j<i} (-L[i][j]) A_inv[j,:]
//
// Same recursion as the scalar solve, but the vector lane width is the chunk
// axis.  One MulAddDst updates row i across many chunks at once.  The layout
// requirements are:
//   1. kda::scheduled::BroadcastBlocks<8 * (2*NC)>{}(dst, src) reads one 32B block (eight
//      columns of one chunk) per repeat and writes eight blocks, so block
//      (16*ch + j) of the expansion holds -L[ch][i][j] eight times over.
//   2. A binary vector op with src1BlkStride = 0 reads that one block for both
//      32B halves of a 16-lane repeat, i.e. it broadcasts the coefficient over
//      exactly the sixteen lanes of one chunk's row, and src1RepStride = 16
//      blocks walks the chunks.
//   3. The coefficients come from DataCopyParams(NC, 2, 30, 0), a gather of
//      two-block rows 32 blocks apart, which turns the chunk-major L of GM
//      into the per-row layout Brcb wants.  The transposed layout only exists
//      in UB: GM traffic is unchanged (2 KB per row of 32 chunks = the same
//      1 KB per chunk that L occupies).
//
// The upper triangle of L is zero (the Gram kernel writes only the lower one
// and the mask zeroes the diagonal), so the j >= i terms contribute nothing;
// they are simply not issued.
//
// NC is bounded by the live-tile UB budget; the caller rounds C up to a whole
// block and zeroes padded L rows so they solve to identity.
//
#include "scheduled_tiles.hpp"
using namespace AscendC;

#ifndef KDA_CHUNK
#define KDA_CHUNK 16
#endif
// The parent chunk (one column block of the pipeline) and its size.  M is the
// *sub-block* the recursion runs on: the two-level solve below splits every
// PC x PC chunk into SB diagonal M x M sub-blocks, so at
// KDA_SOLVE_WIDE_SUBB = 2 and KDA_CHUNK = 64 the substitution sees 32-wide
// blocks.  SB = 1 keeps the single-level kernel (M = PC).
#ifndef KDA_SOLVE_WIDE_SUBB
#define KDA_SOLVE_WIDE_SUBB 1
#endif
constexpr int32_t SB = KDA_SOLVE_WIDE_SUBB;  // diagonal sub-blocks per chunk
constexpr int32_t PC = KDA_CHUNK;            // the parent (whole-chunk) size
constexpr int32_t M = PC / SB;               // the diagonal sub-block size
constexpr int32_t MM = M * M;
#ifndef KDA_SOLVE_WIDE_NCHUNK
#define KDA_SOLVE_WIDE_NCHUNK 32
#endif
constexpr int32_t NC = KDA_SOLVE_WIDE_NCHUNK;
// Two-level solve (SB > 1).  The recursion costs M^3/2 vector lanes per
// chunk, so it is the *depth* that has to come down: solving SB diagonal
// M x M sub-blocks of each PC-sized chunk instead of the whole PC x PC
// triangle divides the vector work by SB^2, and the coupling block
// X21 = X22 @ Lneg21 @ X11 is left to the Cube (k1_solve_assemble.cpp).
// The instances of one block's tile are then (sub-block, chunk) pairs: a row
// of the tile runs over SB groups of NCH chunks, and the gather stride
// between two instances is one whole parent chunk (PC * PC floats).
constexpr int32_t NCH = NC / SB;             // chunks per block
// The block's A_inv tile is laid out [row][chunk][lane] rather than the GM's
// [chunk][row][lane]: the repeat axis of every instruction is the chunk, so
// its stride has to fit the 8-bit repeat-stride field, and at KDA_CHUNK = 64
// the chunk stride of the GM order is MM/8 = 512 blocks (the field saturates
// at 255 - the compiler warns "512 to 0" and the solve then reads the wrong
// rows).  RW is both the stride of one row in this tile and the length of one
// gathered row of L, which is chunk-major for exactly the same reason.
constexpr int32_t RW = NC * M;    // floats in one row of the tile / of lraw
constexpr int32_t CH = NC * MM;   // floats in one block's A_inv
constexpr int32_t CM = 64 * 255;  // largest count-mode calCount
constexpr int32_t CE = MM;        // one [M, M] block per chunk
// The L21 tile is [chunk][row][lane]: one contiguous [M, M] block per chunk,
// which is exactly how the Lneg export is laid out in GM.  The per-chunk
// stride is MM and the tile totals NCH * MM elements.

extern "C" __global__ __aicore__ void kda_solve_wu_wide(
    GM_ADDR pL, GM_ADDR pEye, GM_ADDR pA16,
    GM_ADDR pLneg, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const int32_t c0 = GetBlockIdx() * NCH;
    if (c0 >= C) return;
    TPipe pipe;
    TEventID e2v = pipe.AllocEventID<HardEvent::MTE2_V>();
    TEventID e3 = pipe.AllocEventID<HardEvent::V_MTE3>();
    TBuf<TPosition::VECCALC> bLraw, bCexp, bAf, bAb, bEye;
    pipe.InitBuffer(bLraw, CH * 4);
#if KDA_SOLVE_WIDE_SUBB > 1
    // A whole M x M tile of zeros: the parent tile's strict upper triangle is
    // zero in A_inv but no kernel writes it (this one stores the diagonal
    // sub-blocks, the assemble kernel the lower-left one) while the Cube solve
    // reads the whole parent tile, so this kernel has to blank it.  The source
    // has to be the full tile: a DataCopyParams source gap of 0 means *no gap*
    // between bursts, i.e. the rows are read contiguously.  With a one-row
    // buffer the copy walks off its end and stores UB garbage; a broadcast
    // would have to be a repeat stride inside a vector op, not a DMA
    // parameter.
    TBuf<TPosition::VECCALC> bZf, bZb;
    pipe.InitBuffer(bZf, MM * 4);
    pipe.InitBuffer(bZb, MM * 2);
#endif
    pipe.InitBuffer(bCexp, NC * M * 8 * 4);
    pipe.InitBuffer(bAf, CH * 4);
    pipe.InitBuffer(bAb, CH * 2);
    pipe.InitBuffer(bEye, MM * 4);
#if KDA_SOLVE_WIDE_SUBB > 1
    TBuf<TPosition::VECCALC> bL21f, bL21b;
    pipe.InitBuffer(bL21f, NCH * CE * 4);
    pipe.InitBuffer(bL21b, NCH * CE * 2);
#endif
    LocalTensor<float> lraw = bLraw.Get<float>(), cexp = bCexp.Get<float>();
    LocalTensor<float> af = bAf.Get<float>(), eye = bEye.Get<float>();
    LocalTensor<bfloat16_t> ab = bAb.Get<bfloat16_t>();
    GlobalTensor<float> L, Eye;
    GlobalTensor<bfloat16_t> A16;
#if KDA_SOLVE_WIDE_SUBB > 1
    LocalTensor<float> l21f = bL21f.Get<float>();
    LocalTensor<bfloat16_t> l21b = bL21b.Get<bfloat16_t>();
    LocalTensor<float> zf = bZf.Get<float>();
    LocalTensor<bfloat16_t> zrow = bZb.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> Lneg;
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
#endif
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    Eye.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pEye));
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));

    kda::scheduled::LoadVector(eye, Eye[0], DataCopyParams(M, M / 8, 0, 0));
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t s = 0; s < SB; ++s) {
            kda::scheduled::LoadVector(lraw[i * RW + s * NCH * M], L[static_cast<uint64_t>(c0) * PC * PC + (s * M + i) * PC + s * M], DataCopyParams(NCH, M / 8, (PC * PC - M) / 8, 0));
        }
    }
#if KDA_SOLVE_WIDE_SUBB > 1
    // The coupling block wants L21 negated so the Cube can fold the sign into
    // its operand.  One 2D copy per chunk: the M rows of that chunk's [M, M]
    // lower-left block (row stride PC) land contiguous, ready for the store.
    for (int32_t ch = 0; ch < NCH; ++ch) {
        kda::scheduled::LoadVector(l21f[ch * MM], L[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC], DataCopyParams(M, M / 8, (PC - M) / 8, 0));
    }
#endif
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    // Fold the sign of the coefficient into the load: one instruction for the
    // whole gathered tile (split: a count-mode call stops at CM elements).
    for (int32_t off = 0; off + CM <= CH; off += CM) {
        kda::scheduled::Scale<CM>{}(lraw[off], lraw[off], -1.0f);
    }
    if constexpr (CH % CM != 0) {
        kda::scheduled::Scale<CH % CM>{}(lraw[CH / CM * CM], lraw[CH / CM * CM], -1.0f);
    }
#if KDA_SOLVE_WIDE_SUBB > 1
    kda::scheduled::Scale<NCH * CE>{}(l21f, l21f, -1.0f);
    kda::scheduled::Round<NCH * CE>{}(l21b, l21f);
#endif
#if KDA_SOLVE_WIDE_SUBB > 1
    Duplicate(zf, 0.0f, MM);
    PipeBarrier<PIPE_V>();
    kda::scheduled::Round<MM>{}(zrow, zf);
#endif
    PipeBarrier<PIPE_V>();

    for (int32_t i = 0; i < M; ++i) {
        // Row i starts from e_i for every chunk: one instruction whose repeat
        // stride walks the chunks of the tile layout, while the source re-reads
        // the same identity row (srcRepStride = 0) for each of them.
        Adds(af[i * RW], eye[i * M], 0.0f, M, NC, UnaryRepeatParams(1, 1, M / 8, 0));
        kda::scheduled::BroadcastBlocks<8 * (NC * M / 8)>{}(cexp, lraw[i * RW]);
        PipeBarrier<PIPE_V>();
        for (int32_t j = 0; j < i; ++j) {
            MulAddDst(af[i * RW], af[j * RW], cexp[j * 8], M, NC,
                      BinaryRepeatParams(1, 1, 0, M / 8, M / 8, M));
        }
        PipeBarrier<PIPE_V>();
    }
    for (int32_t off = 0; off + CM <= CH; off += CM) {
        kda::scheduled::Round<CM>{}(ab[off], af[off]);
        PipeBarrier<PIPE_V>();
    }
    if constexpr (CH % CM != 0) {
        kda::scheduled::Round<CH % CM>{}(ab[CH / CM * CM], af[CH / CM * CM]);
        PipeBarrier<PIPE_V>();
    }

    SetFlag<HardEvent::V_MTE3>(e3);
    WaitFlag<HardEvent::V_MTE3>(e3);
#if KDA_SOLVE_WIDE_SUBB > 1
    // Both exports get the blank: the Cube solve reads A16, and A32 is the
    // fp32 twin the debug views hand out - leaving it uninitialised made the
    // two disagree on a block that is zero in A_inv.
    // The A16 store next to each one is the one that has to stay.
    for (int32_t s = 0; s + 1 < SB; ++s) {
        for (int32_t s2 = s + 1; s2 < SB; ++s2) {
            for (int32_t ch = 0; ch < NCH; ++ch) {
                const uint64_t d = static_cast<uint64_t>(c0 + ch) * PC * PC
                                   + (s * M) * PC + s2 * M;
                kda::scheduled::StoreVector(A16[d], zrow, DataCopyParams(M, M / 16, 0, (PC - M) / 16));
            }
        }
    }
#endif
    // One call per chunk gathers that chunk's rows out of the [row][chunk]
    // tile: blockCount = M rows, srcStride = the NC - 1 other chunks' rows.
    for (int32_t s = 0; s < SB; ++s) {
        for (int32_t ch = 0; ch < NCH; ++ch) {
            const int32_t t = s * NCH + ch;
            const uint64_t d =
                static_cast<uint64_t>(c0 + ch) * PC * PC + (s * M) * PC + s * M;
            kda::scheduled::StoreVector(A16[d], ab[t * M], DataCopyParams(M, M / 16, (NC - 1) * (M / 16), (PC - M) / 16));
        }
    }
#if KDA_SOLVE_WIDE_SUBB > 1
    // One contiguous 2 KB copy per chunk (the tile is already [chunk][row]).
    for (int32_t ch = 0; ch < NCH; ++ch) {
        kda::scheduled::StoreVector(Lneg[static_cast<uint64_t>(c0 + ch) * CE], l21b[ch * CE], DataCopyParams(1, CE / 16, 0, 0));
    }
#endif
    PipeBarrier<PIPE_ALL>();
}

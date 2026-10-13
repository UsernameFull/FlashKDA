// K1 stage 3: 16x16 forward-substitution solve, WIDE_NCHUNK chunks per vector
// instruction.
//
//   A_inv = (I + L)^{-1},  A_inv[i,:] = e_i + sum_{j<i} (-L[i][j]) A_inv[j,:]
//
// Same recursion as k1_solve_wu.cpp, but the vector lane width is the chunk
// axis instead of the 16-element row.  On this core a count-mode vector
// instruction costs ~25-30 cycles of issue however few lanes it fills (three
// independent probes agree: +8 one-repeat Adds per chunk in the fused kernel,
// the unroll 4/8/16 sweep there, and the variant timings here), so the old
// per-chunk Axpy loop paid that 120 times per chunk.  Here one MulAddDst
// updates row i of *thirty-two* chunks at once: 120 instructions per 32 chunks
// instead of 120 per chunk.
//
// The three pieces of layout that make it work (all verified on hardware with
// a probe kernel, see docs/ASCENDC_V1_KERNELS.md):
//   1. Brcb(dst, src, 2*NC, BrcbRepeatParams(1, 8)) reads one 32B block (eight
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
// Measured at [1,8192,32] against the per-chunk version, bit-identical on both
// outputs (fp32 max diff 0.000e+00, every bf16 value equal):
//   NC =  8  0.2235 ms      NC = 16  0.1285 ms      NC = 32  0.1006 ms
//   the old kernel                 0.9120 ms
// NC = 64 does not fit: the four live tiles would need ~190 KB of the 192 KB.
// What is left is memory traffic - 40 MB per pass (16 MB L read, 16 MB fp32
// A_inv write, 8 MB bf16 A_inv write) at ~370 GB/s is ~0.11 ms, so the kernel
// is now at the bandwidth floor rather than the issue floor.
//
// C is rounded up to a multiple of NC by the caller, which zeroes the tail rows
// of L so the padded chunks solve to a harmless identity (api.py, right after
// the buffer is allocated - an uninitialised tail is not harmless, it is
// garbage that the substitution turns into inf).
#include "kernel_operator.h"
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
#if KDA_SOLVE_WIDE_SUBB == 4
// The SB = 4 coupling export: the six strictly-lower 16-blocks of every chunk
// in one contiguous per-chunk bundle, negated for the Cube.  Layout per chunk:
//   [ (1,0) 256 | (3,2) 256 | rows 32..63 x cols 0..31  1024 ]  = 1536 bf16
// The first two are the 16-level couplings, the 32x32 tail is exactly the
// [2,0|2,1 ; 3,0|3,1] rectangle the assemble's second-level products consume
// (k1_solve_assemble4.cpp); at SB = 2 the bundle is the single lower-left
// block (M^2 = 1024 at M = 32), i.e. this constant generalises that export.
constexpr int32_t CE = 6 * MM;
#else
constexpr int32_t CE = MM;        // SB <= 2: one [M, M] block per chunk
#endif
// The L21 tile is [chunk][row][lane]: one contiguous [M, M] block per chunk,
// which is exactly how the Lneg export is laid out in GM.  The per-chunk
// stride is MM and the tile totals NCH * MM elements; the old [row][chunk]
// form walked the tile one coupling row at a time, NCH bursts per call, and
// cost 32 gather + 32 scatter descriptors per block against these 4 + 4.

// ``a16Mode`` is the A16-store ablation (2026-09-23, plan 11.34): the kernel's
// own share of the solve stage is 55% row recursion and ~45% DMA (L gathers,
// the bf16 round trip, and the A16/Xb/Lneg exports), so the first thing to
// price is what dropping the parent-tile writes buys.  0 = production (both
// stores), 1 = no strict-upper blank, 2 = no parent tile at all.  Modes 1/2
// leave the Cube solve with a partially written operand *on purpose* - they
// exist to measure the store, not to be correct - and the single-level build
// (SB == 1), where that tile is the kernel's only output, ignores the mode.
extern "C" __global__ __aicore__ void kda_solve_wu_wide(
    GM_ADDR pL, GM_ADDR pEye, GM_ADDR pA32, GM_ADDR pA16, GM_ADDR pXb,
    GM_ADDR pLneg, int32_t C, int32_t a16Mode, int32_t debugStores) {
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
    // between bursts, i.e. the rows are read contiguously (measured - with a
    // one-row buffer the copy walks off its end and stores UB garbage), so the
    // broadcast the header used to assume would have to be a repeat stride
    // inside a vector op, not a DMA parameter.
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
#if KDA_SOLVE_WIDE_SUBB > 1
    LocalTensor<float> zf = bZf.Get<float>();
    LocalTensor<bfloat16_t> zrow = bZb.Get<bfloat16_t>();
    // The two modes of the ablation (see the signature).  Declared here, not
    // at the stores: the blank's *source* is built above the recursion, and an
    // unguarded Duplicate/Cast there would pay for a store that never happens.
    const bool writeBlank = (a16Mode == 0);
    const bool writeDiagA16 = (a16Mode <= 1);
#else
    const bool writeBlank = false;
    const bool writeDiagA16 = true;   // SB == 1: the parent tile is the output
#endif
    GlobalTensor<float> L, Eye, A32;
    GlobalTensor<bfloat16_t> A16;
#if KDA_SOLVE_WIDE_SUBB > 1
    LocalTensor<float> l21f = bL21f.Get<float>();
    LocalTensor<bfloat16_t> l21b = bL21b.Get<bfloat16_t>();
    GlobalTensor<bfloat16_t> Xb, Lneg;
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
#endif
    L.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pL));
    Eye.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pEye));
    A32.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(pA32));
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));

    DataCopy(eye, Eye[0], DataCopyParams(M, M / 8, 0, 0));
    for (int32_t i = 0; i < M; ++i) {
        for (int32_t s = 0; s < SB; ++s) {
            DataCopy(lraw[i * RW + s * NCH * M],
                     L[static_cast<uint64_t>(c0) * PC * PC + (s * M + i) * PC + s * M],
                     DataCopyParams(NCH, M / 8, (PC * PC - M) / 8, 0));
        }
    }
#if KDA_SOLVE_WIDE_SUBB > 1
    // The coupling block wants L21 negated so the Cube can fold the sign into
    // its operand.  One 2D copy per chunk: the M rows of that chunk's [M, M]
    // lower-left block (row stride PC) land contiguous, ready for the store.
    for (int32_t ch = 0; ch < NCH; ++ch) {
#if KDA_SOLVE_WIDE_SUBB == 4
        // Three gathers into the chunk's 1536-element slot: the two 16-level
        // couplings and the 32x32 rectangle.  Each is a straight run of rows
        // (row stride PC) so the destination is the contiguous bundle.
        const uint64_t base = static_cast<uint64_t>(c0 + ch) * PC * PC;
        DataCopy(l21f[ch * CE + 0 * MM],
                 L[base + M * PC],
                 DataCopyParams(M, M / 8, (PC - M) / 8, 0));
        DataCopy(l21f[ch * CE + 1 * MM],
                 L[base + 3 * M * PC + 2 * M],
                 DataCopyParams(M, M / 8, (PC - M) / 8, 0));
        DataCopy(l21f[ch * CE + 2 * MM],
                 L[base + 2 * M * PC],
                 DataCopyParams(2 * M, (2 * M) / 8, (PC - 2 * M) / 8, 0));
#else
        DataCopy(l21f[ch * MM],
                 L[static_cast<uint64_t>(c0 + ch) * PC * PC + M * PC],
                 DataCopyParams(M, M / 8, (PC - M) / 8, 0));
#endif
    }
#endif
    SetFlag<HardEvent::MTE2_V>(e2v);
    WaitFlag<HardEvent::MTE2_V>(e2v);
    // Fold the sign of the coefficient into the load: one instruction for the
    // whole gathered tile (split: a count-mode call stops at CM elements).
    for (int32_t off = 0; off < CH; off += CM) {
        const int32_t n = (CH - off < CM) ? (CH - off) : CM;
        Muls(lraw[off], lraw[off], -1.0f, n);
    }
#if KDA_SOLVE_WIDE_SUBB > 1
    Muls(l21f, l21f, -1.0f, NCH * CE);
    Cast(l21b, l21f, RoundMode::CAST_RINT, NCH * CE);
#endif
#if KDA_SOLVE_WIDE_SUBB > 1
    if (writeBlank) {
        Duplicate(zf, 0.0f, MM);
        PipeBarrier<PIPE_V>();
        Cast(zrow, zf, RoundMode::CAST_RINT, MM);
    }
#endif
    PipeBarrier<PIPE_V>();

    for (int32_t i = 0; i < M; ++i) {
        // Row i starts from e_i for every chunk: one instruction whose repeat
        // stride walks the chunks of the tile layout, while the source re-reads
        // the same identity row (srcRepStride = 0) for each of them.
        Adds(af[i * RW], eye[i * M], 0.0f, M, NC, UnaryRepeatParams(1, 1, M / 8, 0));
        Brcb(cexp, lraw[i * RW], NC * M / 8, BrcbRepeatParams(1, 8));
        PipeBarrier<PIPE_V>();
        for (int32_t j = 0; j < i; ++j) {
            MulAddDst(af[i * RW], af[j * RW], cexp[j * 8], M, NC,
                      BinaryRepeatParams(1, 1, 0, M / 8, M / 8, M));
        }
        PipeBarrier<PIPE_V>();
    }
    // A32 has no device consumer at all (the Cube solve reads A16, W and U and
    // the debug views are host-side), so production skips its stores - 201 MB
    // per call at C=64 (docs 11.29).  The flag is declared here because the
    // bf16 round trip below reads it: the fp32 refresh exists only so the A32
    // view is the rounded twin of A16, and with no A32 reader it is dead work
    // (docs 11.64).  It has to sit outside the SUBB > 1 block below: the
    // second store it guards is the single-level path's, which is exactly the
    // C=16/32 builds.
    const bool keepA32 = (debugStores != 0);
    for (int32_t off = 0; off < CH; off += CM) {
        const int32_t n = (CH - off < CM) ? (CH - off) : CM;
        Cast(ab[off], af[off], RoundMode::CAST_RINT, n);
        PipeBarrier<PIPE_V>();
        if (keepA32) {
            Cast(af[off], ab[off], RoundMode::CAST_NONE, n);
            PipeBarrier<PIPE_V>();
        }
    }

    SetFlag<HardEvent::V_MTE3>(e3);
    WaitFlag<HardEvent::V_MTE3>(e3);
#if KDA_SOLVE_WIDE_SUBB > 1
    // Both exports get the blank: the Cube solve reads A16, and A32 is the
    // fp32 twin the debug views hand out - leaving it uninitialised made the
    // two disagree on a block that is zero in A_inv.
    // The A16 store next to each one is the one that has to stay.
    if (writeBlank) {
        for (int32_t s = 0; s + 1 < SB; ++s) {
            for (int32_t s2 = s + 1; s2 < SB; ++s2) {
                for (int32_t ch = 0; ch < NCH; ++ch) {
                    const uint64_t d = static_cast<uint64_t>(c0 + ch) * PC * PC
                                       + (s * M) * PC + s2 * M;
                    if (keepA32) {
                        DataCopy(A32[d], zf, DataCopyParams(M, M / 8, 0, (PC - M) / 8));
                    }
                    DataCopy(A16[d], zrow,
                             DataCopyParams(M, M / 16, 0, (PC - M) / 16));
                }
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
            if (keepA32) {
                DataCopy(A32[d], af[t * M],
                         DataCopyParams(M, M / 8, (NC - 1) * (M / 8), (PC - M) / 8));
            }
            if (writeDiagA16) {
                DataCopy(A16[d], ab[t * M],
                         DataCopyParams(M, M / 16, (NC - 1) * (M / 16), (PC - M) / 16));
            }
#if KDA_SOLVE_WIDE_SUBB > 1
            // The same tile again, contiguous, for the coupling kernel: it
            // reads X11 / X22 as plain [M, M] operands.
            DataCopy(Xb[static_cast<uint64_t>(c0 + ch) * SB * MM + s * MM], ab[t * M],
                     DataCopyParams(M, M / 16, (NC - 1) * (M / 16), 0));
#endif
        }
    }
#if KDA_SOLVE_WIDE_SUBB > 1
    // One contiguous 2 KB copy per chunk (the tile is already [chunk][row]):
    // the per-row scatter this replaces cost 32 small descriptors per block.
    for (int32_t ch = 0; ch < NCH; ++ch) {
        DataCopy(Lneg[static_cast<uint64_t>(c0 + ch) * CE], l21b[ch * CE],
                 DataCopyParams(1, CE / 16, 0, 0));
    }
#endif
    PipeBarrier<PIPE_ALL>();
}

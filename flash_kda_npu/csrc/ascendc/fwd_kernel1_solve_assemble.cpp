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
// kernel's output is the only writer of the parent tile's lower-left block;
// the wide kernel writes its diagonal and zeroes its strict upper triangle
// (the Cube solve reads all of it).
//
// P = Lneg21 @ X11 stays on chip in NZ (written by pass 0's fixpipe, read by
// pass 1's L0B fill).  Both passes' A/B operands are filled by one whole-window
// call before the pass loop, and pass 1's A16 stores are batched into one nd
// Fixpipe call after the chunk loop.
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
    int32_t pass, int32_t slot,
    LocalTensor<bfloat16_t> la, LocalTensor<bfloat16_t> lb,
    LocalTensor<bfloat16_t> lp, LocalTensor<uint8_t>& a8,
    LocalTensor<uint8_t>& b8, LocalTensor<float>& cfall,
    int32_t ch, TEventID e1m, TEventID emf) {
    LocalTensor<bfloat16_t> a = a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    if (pass == 1) {
        LoadData(a, la, LoadData2dParams(0, KF * KF, 1, 0, 0, false, 0));
    } else {
        for (int32_t dd = 0; dd < KF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                LoadData(a[(mm * KF + dd) * 256], la[(dd * KF + mm) * 256],
                         LoadData2dParams(0, 1, 1, 0, 0, false, 0));
            }
        }
    }
    if (pass == 0) {
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
    if (pass == 0) {
        auto ipnz = FixpipeParamsV220(M, M, M, M, false);
        ipnz.quantPre = QuantMode_t::F322BF16;
        ipnz.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_NZ>(lp[ch * MM], cf, ipnz);
    }
    // Pass 1's A16 stores are batched after the chunk loop.
}

extern "C" __global__ __aicore__ void kda_solve_assemble(
    GM_ADDR pA16, GM_ADDR pXb, GM_ADDR pLneg, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TBuf<TPosition::B1> bufA;
    pipe.InitBuffer(bufA, NC * LBSZ);
    TBuf<TPosition::B1> bufP;
    pipe.InitBuffer(bufP, NC * LBSZ);
    TBuf<TPosition::B1> bufX;
    pipe.InitBuffer(bufX, NC * 2 * MM * 2);
    // One L0 slot per (pass, chunk) unit: 2 * NC * 4 KB = 32 KB of the 128 KB
    // L0C, 16 KB of each of the 64 KB L0A/L0B.
    LocalTensor<float> cfall(TPosition::CO1, 0, SLOTS * MM);
    LocalTensor<uint8_t> a8(TPosition::A2, 0, SLOTS * MM * 2);
    LocalTensor<uint8_t> b8(TPosition::B2, 0, SLOTS * MM * 2);
    GlobalTensor<bfloat16_t> A16, Xb, Lneg;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Xb.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pXb));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    LocalTensor<bfloat16_t> laAll = bufA.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lpAll = bufP.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lxAll = bufX.Get<bfloat16_t>();

    // Both passes' Xb operands in one call: 4*nch ND matrices of 16 rows at
    // the uniform band stride.  Band 0 of a chunk's window is pass 0's B
    // (read with LoadDataWithTranspose) and band 1 is pass 1's A.
    DataCopy(laAll, Lneg[static_cast<uint64_t>(c0) * MM],
             Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
    DataCopy(lxAll, Xb[static_cast<uint64_t>(c0) * 2 * MM],
             Nd2NzParams(4 * nch, 16, M, BANDE, M, 16, 1, BANDE));
    SetFlag<HardEvent::MTE2_MTE1>(e21);
    WaitFlag<HardEvent::MTE2_MTE1>(e21);
    for (int32_t pass = 0; pass < PASSES; ++pass) {
        for (int32_t ch = 0; ch < nch; ++ch) {
            LocalTensor<bfloat16_t> la =
                (pass == 0) ? laAll[ch * MM] : lxAll[ch * 2 * MM + MM];
            LocalTensor<bfloat16_t> lb =
                (pass == 0) ? lxAll[ch * 2 * MM] : lpAll;
            assemble_chunk(pass, pass * NC + ch, la, lb,
                           lpAll, a8, b8, cfall, ch, e1m, emf);
        }
        if (pass == 1) {
            // One nd call: srcNdStride is the L0C slot spacing (MM / 256 KB),
            // dstNdStride the parent tile (PC * PC elements).  The nch Mmads
            // are already M_FIX-confirmed by their chunk's wait, so the FIX
            // pipe reads the slots directly.
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
    }
    PipeBarrier<PIPE_ALL>();
}

// CATLASS tile implementation; schedule and buffer ownership follow the AscendC backend.
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
// lower.  X11/X22 are read directly from A16; Lneg21 is the negated bf16 export. This
// kernel's output is the only writer of the parent tile's lower-left block;
// the wide kernel writes its diagonal and zeroes its strict upper triangle
// (the Cube solve reads all of it).
//
// P = Lneg21 @ X11 stays on chip in NZ (written by pass 0's fixpipe, read by
// pass 1's L0B fill).  Both passes' A/B operands are filled by one whole-window
// call before the pass loop, and pass 1's A16 stores are batched into one nd
// Fixpipe call after the chunk loop.
#include "scheduled_tiles.hpp"
using namespace AscendC;
namespace kda_assembly {

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

// One chunk's L0 chain. Pass 1's lower-left stores are batched by the caller.
static __aicore__ inline void assemble_chunk(
    int32_t pass, int32_t slot,
    LocalTensor<bfloat16_t> la, LocalTensor<bfloat16_t> lb,
    LocalTensor<bfloat16_t> lp, LocalTensor<uint8_t>& a8,
    LocalTensor<uint8_t>& b8, LocalTensor<float>& cfall,
    int32_t ch, TEventID e1m, TEventID emf) {
    LocalTensor<bfloat16_t> a = a8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    LocalTensor<bfloat16_t> b = b8[slot * MM * 2].ReinterpretCast<bfloat16_t>();
    if (pass == 1) {
        kda::scheduled::LoadFractals<KF * KF, false>(a, la);
    } else {
        for (int32_t dd = 0; dd < KF; ++dd) {
            for (int32_t mm = 0; mm < KF; ++mm) {
                kda::scheduled::LoadFractals<1, false>(a[(mm * KF + dd) * 256], la[(dd * KF + mm) * 256]);
            }
        }
    }
    if (pass == 0) {
        kda::scheduled::TransposeFractals<KF * KF>(b, lb);
    } else {
        // P out of L1 in NZ order: the four 512-byte fractals are ordered
        // [n-block][k-block] instead of the Nd2Nz source's [k-block][n-block],
        // so L0B's slots take source blocks 0, 2, 1, 3.
        const int32_t src = ch * MM;
        kda::scheduled::TransposeFractals<1>(b, lp[src]);
        kda::scheduled::TransposeFractals<1>(b[2 * 256], lp[src + 256]);
        kda::scheduled::TransposeFractals<1>(b[1 * 256], lp[src + 512]);
        kda::scheduled::TransposeFractals<1>(b[3 * 256], lp[src + 768]);
    }
    SetFlag<HardEvent::MTE1_M>(e1m);
    WaitFlag<HardEvent::MTE1_M>(e1m);
    LocalTensor<float> cf = cfall[slot * MM];
    kda::scheduled::MmadTile(cf, a, b, M, M, M, 0, false, true);
    SetFlag<HardEvent::M_FIX>(emf);
    WaitFlag<HardEvent::M_FIX>(emf);
    // Pass 0 keeps the BF16 intermediate in L1 NZ layout.
    if (pass == 0) {
        auto ipnz = FixpipeParamsV220(M, M, M, M, false);
        ipnz.quantPre = QuantMode_t::F322BF16;
        ipnz.unitFlag = 0;
        Fixpipe<bfloat16_t, float, CFG_NZ>(lp[ch * MM], cf, ipnz);
    }
    // Pass 1's A16 stores are batched after the chunk loop.
}

static __aicore__ inline void run(
    GM_ADDR pA16, GM_ADDR pLneg, int32_t C) {
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
    GlobalTensor<bfloat16_t> A16, Lneg;
    A16.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pA16));
    Lneg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(pLneg));
    LocalTensor<bfloat16_t> laAll = bufA.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lpAll = bufP.Get<bfloat16_t>();
    LocalTensor<bfloat16_t> lxAll = bufX.Get<bfloat16_t>();

    // Keep the existing band-major L1 layout while gathering the two
    // diagonal blocks from their strided rows in the parent A16 tile.
    kda::scheduled::LoadOperandBatch(laAll, Lneg[static_cast<uint64_t>(c0) * MM], Nd2NzParams(nch, M, M, MM, M, M, 1, MM));
    for (int32_t ch = 0; ch < nch; ++ch) {
        for (int32_t diag = 0; diag < 2; ++diag) {
            kda::scheduled::LoadOperandBatch(lxAll[ch * 2 * MM + diag * MM],
                     A16[static_cast<uint64_t>(c0 + ch) * PC * PC + diag * (M * PC + M)], Nd2NzParams(KF, 16, M, 16 * PC, PC, 16, 1, BANDE));
        }
    }
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
}  // namespace kda_assembly

#ifndef KDA_SOLVE_FUSED
extern "C" __global__ __aicore__ void kda_solve_assemble(
    GM_ADDR pA16, GM_ADDR pLneg, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    kda_assembly::run(pA16, pLneg, C);
}
#endif

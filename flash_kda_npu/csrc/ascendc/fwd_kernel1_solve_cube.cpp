// K1 stage 3b: the two solve right-hand sides on the Cube.
//
//   w = bf16(A_inv) @ (k * beta * exp2(gate))   A16 [16,16] bf16, rk [16,128] bf16
//   u = bf16(A_inv) @ (v * beta)                A16 [16,16] bf16, rv [16,128] bf16
//
// This is the Cube implementation of the solve right-hand sides.  rk/rv are
// consumed in their natural [16,128] layout: Nd2Nz turns them into 16x16
// fractals in L1 and
// LoadDataWithTranspose transposes each fractal into the B2 zN layout the Cube
// wants, so no transposed staging buffer is needed.
//
// One AIC block handles NCHUNK chunks: loads for every chunk are issued up
// front and the Mmads follow.
//
#include "kernel_operator.h"
using namespace AscendC;
namespace kda_wu {
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

static __aicore__ inline void run(
    GM_ADDR pA16, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pW, GM_ADDR pU, int32_t C) {
    const int32_t c0 = GetBlockIdx() * NC;
    if (c0 >= C) return;
    const int32_t nch = ((C - c0) < NC) ? (C - c0) : NC;
    TPipe pipe;
    TEventID e21 = pipe.AllocEventID<HardEvent::MTE2_MTE1>();
    TEventID e1m = pipe.AllocEventID<HardEvent::MTE1_M>();
    TEventID emf = pipe.AllocEventID<HardEvent::M_FIX>();
    TQue<QuePosition::B1, NC> qb;
    pipe.InitBuffer(qb, NC, D * K * 2);
    // The resident A16 copy: one L1 buffer filled once per block, both passes read it.
    TBuf<TPosition::B1> bufA16;
    pipe.InitBuffer(bufA16, NC * M * K * 2);
    // One L0C slot per (pass, chunk) unit.  Slots are never reused inside a
    // block, so no FIX_M -> M ordering is needed between a Fixpipe and the
    // next Mmad.
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
    for (int32_t pass = 0; pass < 2; ++pass) {
        // Issue every chunk's L1 load before touching L0 so one chunk's GM
        // round trips overlap the previous chunk's arithmetic.
        for (int32_t ch = 0; ch < nch; ++ch) {
            auto lb = qb.AllocTensor<bfloat16_t>();
#if KDA_CHUNK > 16
            GlobalTensor<bfloat16_t> &rhs = (pass == 0) ? Rk : Rv;
            DataCopy(lb, rhs[static_cast<uint64_t>(c0 + ch) * M * D],
                     Nd2NzParams(KF, 16, D, 16 * D, D, 16, 1, 16 * D));
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
            LocalTensor<bfloat16_t> la = a16l1[ch * M * K];
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
            qb.FreeTensor(lb);
        }
    }
}
}  // namespace kda_wu

#ifndef KDA_SOLVE_FUSED
extern "C" __global__ __aicore__ void kda_solve_wu_cube_kernel(
    GM_ADDR pA16, GM_ADDR pRk, GM_ADDR pRv, GM_ADDR pW, GM_ADDR pU, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    kda_wu::run(pA16, pRk, pRv, pW, pU, C);
}
#endif

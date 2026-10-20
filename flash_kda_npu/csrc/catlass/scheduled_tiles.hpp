#pragma once
#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/gemm/tile/copy_gm_to_l1.hpp"
#include "catlass/gemm/tile/copy_l0c_to_gm.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_elemwise_mul.hpp"
#include "catlass/epilogue/tile/tile_elemwise_muls.hpp"
#include "catlass/epilogue/tile/tile_cast.hpp"

// Custom KDA block schedules own the buffers, events and inter-head overlap.
// TileMmad consumes the already resident fractal operands, unlike BlockMmad,
// which would reload each operand and drain the pipeline per product.
namespace kda::scheduled {
using Arch = Catlass::Arch::AtlasA2;
using Operand = Catlass::Gemm::GemmType<bfloat16_t, Catlass::layout::RowMajor>;
using Accumulator = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
CATLASS_DEVICE void MmadTile(const AscendC::LocalTensor<float> &c,
    const AscendC::LocalTensor<bfloat16_t> &a,
    const AscendC::LocalTensor<bfloat16_t> &b,
    int m, int n, int k, uint8_t unit, bool align, bool init) {
    // BF16 kernels always pass align=false; upstream TileMmad sets its own
    // architecture-specific alignment for the element/layout combination.
    Catlass::Gemm::Tile::TileMmad<Arch, Operand, Operand, Accumulator>{}(
        c, a, b, m, n, k, init, unit);
}
template<uint8_t Mode, pipe_t Pipe>
CATLASS_DEVICE void Publish(uint16_t id) {
    Catlass::Arch::CrossCoreFlag flag(id);
    Catlass::Arch::CrossCoreSetFlag<Mode, Pipe>(flag);
}
CATLASS_DEVICE void Wait(uint16_t id) {
    Catlass::Arch::CrossCoreFlag flag(id);
    Catlass::Arch::CrossCoreWaitFlag(flag);
}
}

namespace kda::scheduled {
template<int Count> struct Shape { static constexpr uint32_t ROW=1, COLUMN=Count, COUNT=Count; };
using Vector = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
template<int Count> using Add = Catlass::Epilogue::Tile::TileElemWiseAdd<Arch, Vector, Count>;
template<int Count> using Mul = Catlass::Epilogue::Tile::TileElemwiseMul<Arch, Vector, Shape<Count>>;
template<int Count> using Scale = Catlass::Epilogue::Tile::TileElemWiseMuls<Arch, Vector, Count>;
template<int Count> using Round = Catlass::Epilogue::Tile::TileCast<Arch,
    Catlass::Gemm::GemmType<bfloat16_t, Catlass::layout::RowMajor>, Vector, Shape<Count>>;
CATLASS_DEVICE void LoadOperand(const AscendC::LocalTensor<bfloat16_t> &dst,
    const AscendC::GlobalTensor<bfloat16_t> &src, const AscendC::Nd2NzParams &p) {
    // Only single-matrix calls are routed here. Batched fractal strip copies
    // retain the original single DMA instruction in the custom block.
    const auto local = Catlass::layout::zN(p.nValue, p.dValue, 16,
        (p.nValue+15)/16, 16, (p.dValue+15)/16,
        p.dstNzNStride*16, 256, 1, p.dstNzC0Stride*16);
    Catlass::Gemm::Tile::CopyGmToL1<Arch, Operand>{}(
        dst, src, local, Catlass::layout::RowMajor(p.nValue, p.dValue, p.srcDValue));
}
template<class Output>
CATLASS_DEVICE void StoreProduct(const AscendC::GlobalTensor<Output> &dst,
    const AscendC::LocalTensor<float> &src, const AscendC::FixpipeParamsV220 &p) {
    using O = Catlass::Gemm::GemmType<Output, Catlass::layout::RowMajor>;
    const auto local = Catlass::layout::zN(p.mSize, p.nSize, 16,
        (p.mSize+15)/16, 16, (p.nSize+15)/16, 16, 256, 1, p.srcStride*16);
    Catlass::Gemm::Tile::CopyL0CToGm<Arch, float, O>{}(
        dst, src, Catlass::layout::RowMajor(p.mSize, p.nSize, p.dstStride), local, p.unitFlag);
}
}

#include "catlass/epilogue/tile/copy_gm_to_ub.hpp"
#include "catlass/epilogue/tile/copy_ub_to_gm.hpp"
#include "catlass/epilogue/tile/tile_broadcast_add.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
namespace kda::scheduled {
template<class T> using RowType = Catlass::Gemm::GemmType<T, Catlass::layout::RowMajor>;
template<class T> CATLASS_DEVICE void LoadVector(AscendC::LocalTensor<T> dst,
    AscendC::GlobalTensor<T> src, uint32_t count) {
    Catlass::Epilogue::Tile::CopyGm2UbAligned<Arch, RowType<T>>{}(
        dst, src, Catlass::layout::RowMajor(1,count), Catlass::layout::RowMajor(1,count));
}
template<class T> CATLASS_DEVICE void LoadVector(AscendC::LocalTensor<T> dst,
    AscendC::GlobalTensor<T> src, const AscendC::DataCopyParams &p) {
    constexpr int E=32/sizeof(T); const int cols=p.blockLen*E;
    Catlass::Epilogue::Tile::CopyGm2UbAligned<Arch, RowType<T>>{}(dst,src,
        Catlass::layout::RowMajor(p.blockCount,cols,cols+p.dstStride*E),
        Catlass::layout::RowMajor(p.blockCount,cols,cols+p.srcStride*E));
}
template<class T> CATLASS_DEVICE void StoreVector(AscendC::GlobalTensor<T> dst,
    AscendC::LocalTensor<T> src, uint32_t count) {
    Catlass::Epilogue::Tile::CopyUb2GmAligned<Arch, RowType<T>>{}(
        dst,src,Catlass::layout::RowMajor(1,count),Catlass::layout::RowMajor(1,count));
}
template<class T> CATLASS_DEVICE void StoreVector(AscendC::GlobalTensor<T> dst,
    AscendC::LocalTensor<T> src, const AscendC::DataCopyParams &p) {
    constexpr int E=32/sizeof(T); const int cols=p.blockLen*E;
    Catlass::Epilogue::Tile::CopyUb2GmAligned<Arch, RowType<T>>{}(dst,src,
        Catlass::layout::RowMajor(p.blockCount,cols,cols+p.dstStride*E),
        Catlass::layout::RowMajor(p.blockCount,cols,cols+p.srcStride*E));
}
// No-padding byte-stride loads; scalar zero-padding remains a separate contract.
template<class T> CATLASS_DEVICE void LoadRows(AscendC::LocalTensor<T> dst,
    AscendC::GlobalTensor<T> src, const AscendC::DataCopyExtParams &p) {
    const int cols=p.blockLen/sizeof(T);
    Catlass::Epilogue::Tile::CopyGm2Ub<Arch, RowType<T>>{}(dst,src,
        Catlass::layout::RowMajor(p.blockCount,cols,cols+p.dstStride*(32/sizeof(T))),
        Catlass::layout::RowMajor(p.blockCount,cols,cols+p.srcStride/sizeof(T)));
}
template<int Rows,int Cols> struct MatrixShape { static constexpr uint32_t ROW=Rows,COLUMN=Cols,COUNT=Rows*Cols; };
template<int Rows,int Cols> CATLASS_DEVICE void BroadcastAdd(
    AscendC::LocalTensor<float> dst,AscendC::LocalTensor<float> x,AscendC::LocalTensor<float> row) {
    Catlass::Epilogue::Tile::TileRowBroadcastAdd<Arch,Vector,MatrixShape<Rows,Cols>>{}(
        dst,x,row,Catlass::MatrixCoord(static_cast<uint32_t>(Rows),static_cast<uint32_t>(Cols)));
}
template<int Rows,int Cols> CATLASS_DEVICE void BroadcastMul(
    AscendC::LocalTensor<float> dst,AscendC::LocalTensor<float> x,AscendC::LocalTensor<float> row) {
    Catlass::Epilogue::Tile::TileRowBroadcastMul<Arch,Vector,MatrixShape<Rows,Cols>>{}(dst,x,row);
}
}

#include "catlass/gemm/tile/copy_l1_to_l0a.hpp"
#include "catlass/gemm/tile/copy_l1_to_l0b.hpp"
namespace kda::scheduled {
template<int Fractals, bool B> CATLASS_DEVICE void LoadFractals(
    AscendC::LocalTensor<bfloat16_t> dst, AscendC::LocalTensor<bfloat16_t> src) {
    if constexpr (B) {
        using S=Catlass::Gemm::GemmType<bfloat16_t,Catlass::layout::zN,AscendC::TPosition::B1>;
        using O=Catlass::Gemm::GemmType<bfloat16_t,Catlass::layout::zN,AscendC::TPosition::B2>;
        const auto layout=Catlass::layout::zN(16*Fractals,16,16,Fractals,16,1,16,256,1,256*Fractals);
        Catlass::Gemm::Tile::CopyL1ToL0B<Arch,S,O>{}(dst,src,layout,layout);
    } else {
        using S=Catlass::Gemm::GemmType<bfloat16_t,Catlass::layout::zN,AscendC::TPosition::A1>;
        using O=Catlass::Gemm::GemmType<bfloat16_t,Catlass::layout::zZ,AscendC::TPosition::A2>;
        const auto local=Catlass::layout::zZ(16,16*Fractals,16,1,16,Fractals,16,256*Fractals,1,256);
        const auto source=Catlass::layout::zN(16,16*Fractals,16,1,16,Fractals,16,256*Fractals,1,256);
        Catlass::Gemm::Tile::CopyL1ToL0A<Arch,S,O>{}(dst,src,local,source);
    }
}
template<int Fractals> CATLASS_DEVICE void TransposeFractals(
    AscendC::LocalTensor<bfloat16_t> dst,AscendC::LocalTensor<bfloat16_t> src) {
    using S=Catlass::Gemm::GemmType<bfloat16_t,Catlass::layout::zN,AscendC::TPosition::A1>;
    const auto source=Catlass::layout::zN(16,16*Fractals,16,1,16,Fractals,16,256*Fractals,1,256);
    const auto local=Catlass::layout::nZ(16,16*Fractals,16,1,16,Fractals,1,256*Fractals,16,256);
    Catlass::Gemm::Tile::CopyL1ToL0B<Arch,S>{}(dst,src,local,source);
}
}

#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
namespace kda::scheduled {
template<int Count> using BroadcastBlocks = Catlass::Epilogue::Tile::TileBroadcastOneBlk<Arch,Vector,Count>;
// Fractal GM operands are copied as stored, without ND->NZ conversion.
CATLASS_DEVICE void LoadStoredOperand(AscendC::LocalTensor<bfloat16_t> dst,
    AscendC::GlobalTensor<bfloat16_t> src, const AscendC::DataCopyParams &p) {
    using T=Catlass::Gemm::GemmType<bfloat16_t,Catlass::layout::zN>;
    const auto from=Catlass::layout::zN(p.blockLen,p.blockCount*16,16,1,16,p.blockCount,
        16,256,1,(p.blockLen+p.srcStride)*16);
    const auto to=Catlass::layout::zN(p.blockLen,p.blockCount*16,16,1,16,p.blockCount,
        16,256,1,(p.blockLen+p.dstStride)*16);
    Catlass::Gemm::Tile::CopyGmToL1<Arch,T>{}(dst,src,to,from);
}
CATLASS_DEVICE void LoadStoredOperand(AscendC::LocalTensor<bfloat16_t> dst,
    AscendC::GlobalTensor<bfloat16_t> src,uint32_t count) {
    LoadStoredOperand(dst,src,AscendC::DataCopyParams(1,count/16,0,0));
}
}

namespace kda::scheduled {
CATLASS_DEVICE void LoadOperandBatch(AscendC::LocalTensor<bfloat16_t> dst,
    AscendC::GlobalTensor<bfloat16_t> src,const AscendC::Nd2NzParams &p) {
    // The upstream GMM tile exposes batched ND->NZ without splitting the DMA.
    const auto local=Catlass::layout::zN(p.nValue,p.dValue,16,(p.nValue+15)/16,
        16,(p.dValue+15)/16,p.dstNzNStride*16,256,1,p.dstNzC0Stride*16);
    Catlass::Gemm::Tile::CopyGmToL1GMMPTD<Arch,Operand>{}(dst,src,local,
        Catlass::layout::RowMajor(p.nValue,p.dValue,p.srcDValue),p.ndNum,
        p.srcNdMatrixStride,p.dstNzNStride,p.dstNzMatrixStride,p.dstNzC0Stride);
}
}

namespace kda::scheduled {
template<int Rows,int Cols> CATLASS_DEVICE void ColumnBroadcastMul(
    AscendC::LocalTensor<float> dst,AscendC::LocalTensor<float> x,AscendC::LocalTensor<float> blocks) {
    Catlass::Epilogue::Tile::TileOneBlkColumnBroadcastMul<Arch,Vector,MatrixShape<Rows,Cols>>{}(dst,x,blocks);
}
}

// RTC prepends the assembly and W/U implementations with KDA_SOLVE_FUSED.
// Both phases own the same WU_NCHUNK chunks. Their scoped TPipe lifetimes
// drain and release events/buffers before L0 is reused, so the 128 KB W/U
// L0C allocation is a peak, rather than a sum with assembly's allocation.
// A16 stays in GM at the phase boundary to preserve the existing BF16
// rounding/layout contract; fusion saves one host launch per solve slice.
static_assert(kda_assembly::NC == kda_wu::NC, "fused chunk ownership must match");
extern "C" __global__ __aicore__ void kda_solve_fused(
    GM_ADDR pA16, GM_ADDR pLneg, GM_ADDR pRk, GM_ADDR pRv,
    GM_ADDR pW, GM_ADDR pU, int32_t C) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    kda_assembly::run(pA16, pLneg, C);
    kda_wu::run(pA16, pRk, pRv, pW, pU, C);
}

// FlashKDA K1: normalize q/k, activate gates, and emit the chunk-local
// operands consumed by K2.  KDA_CHUNK is fixed to 64 by CMake.
#include "kda_kernel_interface.h"

FLASH_KDA_ASC_OPERATOR(flash_kda_npu_kda_k1, KDA_CHUNK) {
  FLASH_KDA_ASC_LOAD_BF16(q, KDA_CHUNK * D);
  FLASH_KDA_ASC_LOAD_BF16(k, KDA_CHUNK * D);
  FLASH_KDA_ASC_LOAD_BF16(g, KDA_CHUNK * D);
  FLASH_KDA_ASC_LOAD_BF16(beta, KDA_CHUNK);
  FLASH_KDA_ASC_LOAD_FP32(A_log, H);
  FLASH_KDA_ASC_LOAD_FP32(dt_bias, H * D);

  FLASH_KDA_ASC_L2_NORMALIZE(q, q_normalized);
  FLASH_KDA_ASC_L2_NORMALIZE(k, k_normalized);
  FLASH_KDA_ASC_GATE_CUMSUM(g, A_log, dt_bias, lower_bound, gate_cumsum);
  FLASH_KDA_ASC_SIGMOID(beta, beta_activated);

  FLASH_KDA_ASC_STORE_BF16(q_normalized, q_normalized);
  FLASH_KDA_ASC_STORE_BF16(k_normalized, k_normalized);
  FLASH_KDA_ASC_STORE_FP32(gate_cumsum, gate_cumsum);
  FLASH_KDA_ASC_STORE_BF16(beta_activated, beta_activated);
}

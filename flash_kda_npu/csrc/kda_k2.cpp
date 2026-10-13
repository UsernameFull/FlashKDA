// FlashKDA K2: chunk recurrence and output projection.  State updates use
// fp32 accumulation; the state handed between chunks is rounded to bf16 when
// neither initial_state nor final_state is fp32.
#include "kda_kernel_interface.h"

FLASH_KDA_ASC_OPERATOR(flash_kda_npu_kda_k2, KDA_CHUNK) {
  FLASH_KDA_ASC_LOAD_BF16(q_normalized, KDA_CHUNK * D);
  FLASH_KDA_ASC_LOAD_BF16(k_normalized, KDA_CHUNK * D);
  FLASH_KDA_ASC_LOAD_BF16(v, KDA_CHUNK * D);
  FLASH_KDA_ASC_LOAD_FP32(gate_cumsum, KDA_CHUNK * D);
  FLASH_KDA_ASC_LOAD_BF16(beta_activated, KDA_CHUNK);
  FLASH_KDA_ASC_LOAD_STATE(state);

  FLASH_KDA_ASC_STRICT_LOWER_GRAM(k_normalized, gate_cumsum, beta_activated, lower);
  FLASH_KDA_ASC_INVERSE_LOWER(lower, chunk_inverse);
  FLASH_KDA_ASC_DELTA_RULE(
      q_normalized, k_normalized, v, gate_cumsum, beta_activated,
      chunk_inverse, state, output, new_state);

  FLASH_KDA_ASC_STORE_BF16(output, output);
  FLASH_KDA_ASC_STORE_STATE(new_state, state);
}

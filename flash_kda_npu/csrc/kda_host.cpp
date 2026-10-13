#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace flash_kda_npu {
namespace {

constexpr int64_t kDim = 128;
constexpr int64_t kChunk = 64;
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kEps = 1.0e-6f;

at::Tensor normalized_bf16(const at::Tensor& input) {
  auto value = input.to(at::kFloat);
  auto sum = value.pow(2).sum(-1, true);
  return (value * at::rsqrt(sum + kEps)).to(at::kBFloat16);
}

at::Tensor pad_tokens(const at::Tensor& input, int64_t start, int64_t end) {
  const int64_t length = end - start;
  auto tokens = input.narrow(0, start, length);
  if (length == kChunk) {
    return tokens;
  }
  auto padding = at::zeros({kChunk - length, input.size(1), input.size(2)},
                           input.options());
  return at::cat({tokens, padding}, 0);
}

at::Tensor pad_beta(const at::Tensor& input, int64_t start, int64_t end) {
  const int64_t length = end - start;
  auto tokens = input.narrow(0, start, length);
  if (length == kChunk) {
    return tokens;
  }
  auto padding = at::zeros({kChunk - length, input.size(1)}, input.options());
  return at::cat({tokens, padding}, 0);
}

std::vector<int64_t> sequence_boundaries(const std::optional<at::Tensor>& cu_seqlens,
                                         int64_t batch_size, int64_t length) {
  std::vector<int64_t> boundaries;
  if (cu_seqlens.has_value()) {
    auto cpu = cu_seqlens->to(at::kCPU, /*non_blocking=*/false);
    TORCH_CHECK(cpu.scalar_type() == at::kLong, "cu_seqlens must be int64");
    auto* data = cpu.data_ptr<int64_t>();
    boundaries.assign(data, data + cpu.numel());
  } else {
    boundaries.reserve(batch_size + 1);
    for (int64_t index = 0; index <= batch_size; ++index) {
      boundaries.push_back(index * length);
    }
  }
  return boundaries;
}

void validate(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
              const at::Tensor& g, const at::Tensor& beta, const at::Tensor& out,
              const at::Tensor& A_log, const at::Tensor& dt_bias,
              float lower_bound, const std::optional<at::Tensor>& initial_state,
              const std::optional<at::Tensor>& final_state,
              const std::optional<at::Tensor>& cu_seqlens) {
  TORCH_CHECK(q.dim() == 4, "q must have shape [B, T, H, ", kDim, "]");
  TORCH_CHECK(q.size(-1) == kDim && v.size(-1) == kDim,
              "FlashKDA NPU only supports K=V=", kDim);
  TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes() && q.sizes() == g.sizes(),
              "q, k, v, and g must have identical [B, T, H, 128] shapes");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                  v.scalar_type() == at::kBFloat16 && g.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16 && beta.scalar_type() == at::kBFloat16,
              "q, k, v, g, beta, and out must be bf16");
  TORCH_CHECK(beta.dim() == 3 && beta.sizes() == q.sizes().slice(0, 3),
              "beta must have shape [B, T, H]");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat,
              "A_log and dt_bias must be fp32");
  TORCH_CHECK(A_log.numel() == q.size(2), "A_log must have H elements");
  TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.size(0) == q.size(2) &&
                  dt_bias.size(1) == kDim,
              "dt_bias must have shape [H, 128]");
  TORCH_CHECK(lower_bound >= -5.0f && lower_bound <= 0.0f,
              "lower_bound must be in [-5, 0]");
  TORCH_CHECK(out.sizes() == v.sizes(), "out must have the same shape as v");

  if (cu_seqlens.has_value()) {
    TORCH_CHECK(q.size(0) == 1, "B must be 1 when cu_seqlens is provided");
    TORCH_CHECK(cu_seqlens->dim() == 1 && cu_seqlens->numel() >= 2,
                "cu_seqlens must have shape [N + 1]");
  }

  const int64_t sequences =
      cu_seqlens.has_value() ? cu_seqlens->numel() - 1 : q.size(0);
  auto check_state = [&](const std::optional<at::Tensor>& state, const char* name) {
    if (!state.has_value()) return;
    TORCH_CHECK(state->dim() == 4, name, " must have shape [N, H, 128, 128]");
    TORCH_CHECK(state->size(0) == sequences && state->size(1) == q.size(2) &&
                    state->size(2) == kDim && state->size(3) == kDim,
                name, " has an incompatible shape");
  };
  check_state(initial_state, "initial_state");
  check_state(final_state, "final_state");
  if (initial_state.has_value() && final_state.has_value()) {
    TORCH_CHECK(initial_state->scalar_type() == final_state->scalar_type(),
                "initial_state and final_state must have the same dtype");
  }
}

void fwd(const at::Tensor& q_, const at::Tensor& k_, const at::Tensor& v_,
         const at::Tensor& g_, const at::Tensor& beta_, double scale,
         at::Tensor& out_, const at::Tensor& A_log, const at::Tensor& dt_bias,
         double lower_bound, const std::optional<at::Tensor>& initial_state,
         const std::optional<at::Tensor>& final_state,
         const std::optional<at::Tensor>& cu_seqlens) {
  const float scale_value = static_cast<float>(scale);
  const float bound = static_cast<float>(lower_bound);
  validate(q_, k_, v_, g_, beta_, out_, A_log, dt_bias, bound, initial_state,
           final_state, cu_seqlens);

  auto q = normalized_bf16(q_.contiguous());
  auto k = normalized_bf16(k_.contiguous());
  auto v = v_.contiguous();
  auto beta = beta_.contiguous();
  auto out = at::empty_like(v);

  auto gate_raw = g_.contiguous().to(at::kFloat) + dt_bias.unsqueeze(0);
  auto gate_scale = at::special_exp2(A_log * kLog2E).unsqueeze(0).unsqueeze(-1);
  auto gate = bound * kLog2E * at::sigmoid(gate_scale * gate_raw);

  const int64_t batch_size = q.size(0);
  const int64_t length = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t sequences =
      cu_seqlens.has_value() ? cu_seqlens->numel() - 1 : batch_size;
  auto boundaries = sequence_boundaries(cu_seqlens, batch_size, length);

  auto state_dtype =
      (initial_state.has_value() && initial_state->scalar_type() == at::kFloat) ||
              (final_state.has_value() && final_state->scalar_type() == at::kFloat)
          ? at::kFloat
          : at::kBFloat16;
  auto state = at::zeros({sequences, heads, kDim, kDim}, q.options().dtype(state_dtype));
  if (initial_state.has_value()) {
    state.copy_(initial_state->to(state->device()));
  }

  auto scale_bf16 = at::scalar_tensor(scale_value, q.options());
  auto activated_beta = at::sigmoid(beta.to(at::kFloat)).to(at::kBFloat16);

  for (int64_t sequence = 0; sequence < sequences; ++sequence) {
    const int64_t start = boundaries[sequence];
    const int64_t end = boundaries[sequence + 1];
    for (int64_t token = start; token < end; token += kChunk) {
      const int64_t actual_length = std::min(kChunk, end - token);
      const int64_t stop = token + actual_length;

      auto qc = pad_tokens(q, token, stop).permute({1, 0, 2});
      auto kc = pad_tokens(k, token, stop).permute({1, 0, 2});
      auto vc = pad_tokens(v, token, stop).permute({1, 0, 2});
      auto beta_c = pad_beta(activated_beta, token, stop)
                        .permute({1, 0})
                        .unsqueeze(-1);
      auto gate_c = pad_tokens(gate, token, stop)
                        .permute({1, 0, 2})
                        .cumsum(1);
      auto gate_total = gate_c.narrow(1, kChunk - 1, 1);

      auto k_decay = kc * at::special_exp2(gate_c).to(at::kBFloat16);
      auto q_decay =
          qc * at::special_exp2(gate_c).to(at::kBFloat16) * scale_bf16;
      auto k_inverse = kc * at::special_exp2(-gate_c).to(at::kBFloat16);
      auto k_restored =
          k_inverse * at::special_exp2(gate_total).to(at::kBFloat16);

      auto inverse_key = k_inverse.transpose(-1, -2);
      auto lower = at::matmul(k_decay, inverse_key);
      auto attention = at::matmul(q_decay, inverse_key);
      lower = lower *
              at::tril(at::ones_like(lower, lower.options().dtype(at::kFloat)), -1)
                  .to(at::kBFloat16) *
              beta_c;
      attention = at::tril(attention);
      auto identity = at::eye(kChunk, lower.options());
      auto chunk_inverse = at::inverse(identity + lower);

      auto state_slice = state.narrow(0, sequence, 1).squeeze(0).to(at::kFloat);
      auto state_transposed = state_slice.transpose(-1, -2);
      auto value =
          (vc - at::matmul(k_decay, state_transposed)) * beta_c;
      value = at::matmul(chunk_inverse, value);
      auto result = at::matmul(q_decay, state_transposed) +
                    at::matmul(attention, value);
      auto delta = at::matmul(k_restored.transpose(-1, -2), value);
      auto decay =
          at::special_exp2(gate_total.to(at::kFloat)).squeeze(1).unsqueeze(-1);
      auto state_new = (delta.to(at::kFloat) + state_transposed * decay)
                           .to(state_dtype);
      state.narrow(0, sequence, 1).copy_(state_new.unsqueeze(0));

      out.narrow(0, token, actual_length)
          .copy_(result.narrow(1, 0, actual_length).permute({1, 0, 2}));
    }
  }

  out_.copy_(out);
  if (final_state.has_value()) {
    final_state->copy_(state);
  }
}

}  // namespace

TORCH_LIBRARY(flash_kda_npu, m) {
  m.def("fwd(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, "
        "Tensor(a!) out, Tensor A_log, Tensor dt_bias, float lower_bound, "
        "Tensor? initial_state=None, Tensor? final_state=None, "
        "Tensor? cu_seqlens=None) -> ()");
}

TORCH_LIBRARY_IMPL(flash_kda_npu, CompositeExplicitAutograd, m) {
  m.impl("fwd", &fwd);
}

}  // namespace flash_kda_npu

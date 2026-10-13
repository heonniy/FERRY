#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/index.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <memory>
#include <vector>

at::Tensor silu_mul_cuda(const at::Tensor& gate, const at::Tensor& up);

// ---------------------------------------------------------------------------
// Routed-expert MLPs of every (slot, rows, cols) group in one host call.
// cache[slot] holds [gate (I x H) | up (I x H) | down (H x I)] in BF16.
// Output i is down(silu(x W_g^T) * (x W_u^T)) scaled by the routing weight of each row.
std::vector<at::Tensor> execute_experts(
    const at::Tensor& cache, const at::Tensor& input, const at::Tensor& routing_weights,
    const std::vector<int64_t>& slots, const std::vector<at::Tensor>& rows,
    const std::vector<at::Tensor>& cols, int64_t hidden, int64_t intermediate) {
  TORCH_CHECK(cache.is_cuda() && input.is_cuda() && routing_weights.is_cuda());
  TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && input.scalar_type() == at::kBFloat16);
  TORCH_CHECK(routing_weights.scalar_type() == at::kBFloat16);
  TORCH_CHECK(cache.dim() == 2 && cache.is_contiguous() && cache.size(1) == 3 * hidden * intermediate);
  TORCH_CHECK(input.dim() == 2 && input.size(1) == hidden);
  TORCH_CHECK(routing_weights.dim() == 2 && routing_weights.size(0) == input.size(0));
  TORCH_CHECK(slots.size() == rows.size() && slots.size() == cols.size());
  c10::cuda::CUDAGuard guard(cache.device());
  at::NoGradGuard no_grad;
  const int64_t part = hidden * intermediate;
  std::vector<at::Tensor> outputs;
  outputs.reserve(slots.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    TORCH_CHECK(slots[i] >= 0 && slots[i] < cache.size(0));
    TORCH_CHECK(rows[i].scalar_type() == at::kLong && cols[i].sizes() == rows[i].sizes());
    auto w = cache.select(0, slots[i]);
    auto gate = w.narrow(0, 0, part).view({intermediate, hidden});
    auto up = w.narrow(0, part, part).view({intermediate, hidden});
    auto down = w.narrow(0, 2 * part, part).view({hidden, intermediate});
    auto x = input.index_select(0, rows[i]);
    auto activated = silu_mul_cuda(at::mm(x, gate.t()), at::mm(x, up.t()));
    auto y = at::mm(activated, down.t());
    outputs.push_back(y * at::index(routing_weights, {rows[i], cols[i]}).unsqueeze(1));
  }
  return outputs;
}

// ---------------------------------------------------------------------------
// Dispatch packet: [hidden (2H bytes) | top-k routing weights (2k bytes) | expert IDs (k bytes)].
// selected/weights are this rank's router outputs; send_idx/send_ids describe the packets.
at::Tensor pack_dispatch(const at::Tensor& hidden, const at::Tensor& selected,
                         const at::Tensor& weights, const at::Tensor& send_idx,
                         const at::Tensor& send_ids, int64_t experts) {
  c10::cuda::CUDAGuard guard(hidden.device());
  at::NoGradGuard no_grad;
  TORCH_CHECK(hidden.scalar_type() == at::kBFloat16 && weights.scalar_type() == at::kBFloat16);
  const int64_t n = send_idx.size(0), h = hidden.size(1), k = send_ids.size(1);
  auto dense = at::zeros({hidden.size(0), experts}, hidden.options().dtype(at::kFloat))
                   .scatter_add_(1, selected, weights.to(at::kFloat)).to(at::kBFloat16);
  auto valid = send_ids.ge(0);
  auto packet_weights = at::index(dense, {send_idx.unsqueeze(1), send_ids.clamp_min(0)}) * valid;
  auto payload = at::empty({n, 2 * h + 3 * k}, hidden.options().dtype(at::kByte));
  payload.narrow(1, 0, 2 * h).copy_(hidden.index_select(0, send_idx).view(at::kByte));
  payload.narrow(1, 2 * h, 2 * k).copy_(packet_weights.contiguous().view(at::kByte));
  payload.narrow(1, 2 * h + 2 * k, k).copy_(send_ids.to(at::kByte));
  return payload;
}

std::vector<at::Tensor> unpack_dispatch(const at::Tensor& raw, int64_t hidden, int64_t topk) {
  const int64_t m = raw.size(0);
  auto x = raw.narrow(1, 0, 2 * hidden).contiguous().view(at::kBFloat16).view({m, hidden});
  auto w = raw.narrow(1, 2 * hidden, 2 * topk).contiguous().view(at::kBFloat16).view({m, topk});
  return {x, w};
}

// Sum expert outputs into one partial per received packet (FP32 accumulation, BF16 wire).
at::Tensor reduce_partials(const std::vector<at::Tensor>& parts, const std::vector<at::Tensor>& rows,
                           int64_t received, int64_t hidden, const at::Tensor& like) {
  c10::cuda::CUDAGuard guard(like.device());
  at::NoGradGuard no_grad;
  auto partial = at::zeros({received, hidden}, like.options().dtype(at::kFloat));
  for (size_t i = 0; i < parts.size(); ++i) partial.index_add_(0, rows[i], parts[i].to(at::kFloat));
  return partial.to(at::kBFloat16);
}

// Add returned partials back to the tokens that sent them (FP32 accumulation).
at::Tensor scatter_returned(const at::Tensor& returned, const at::Tensor& send_idx, int64_t tokens) {
  c10::cuda::CUDAGuard guard(returned.device());
  at::NoGradGuard no_grad;
  auto output = at::zeros({tokens, returned.size(1)}, returned.options().dtype(at::kFloat));
  if (returned.size(0)) output.index_add_(0, send_idx, returned.to(at::kFloat));
  return output.to(returned.scalar_type());
}

// ---------------------------------------------------------------------------
// H2D expert fetches from a pinned host pool into GPU cache slots on a dedicated stream.
// A copy into a slot waits for the last compute that read the slot.
class H2DEngine {
 public:
  H2DEngine(at::Tensor cache, at::Tensor pool)
      : cache_(std::move(cache)), pool_(std::move(pool)),
        stream_(c10::cuda::getStreamFromPool(false, cache_.device().index())),
        copy_done_(cache_.size(0)), used_(cache_.size(0)) {
    TORCH_CHECK(pool_.is_pinned() && pool_.size(1) == cache_.size(1));
    for (auto& e : copy_done_) e = std::make_shared<at::cuda::CUDAEvent>();
  }

  void fetch(const std::vector<int64_t>& slots, const std::vector<int64_t>& keys) {
    c10::cuda::CUDAGuard guard(cache_.device());
    c10::cuda::CUDAStreamGuard stream_guard(stream_);
    for (size_t i = 0; i < slots.size(); ++i) {
      const int64_t slot = slots[i];
      if (used_[slot]) used_[slot]->block(stream_);
      cache_.select(0, slot).copy_(pool_.select(0, keys[i]), /*non_blocking=*/true);
      copy_done_[slot]->record(stream_);
      ++copies_;
    }
  }

  void wait_host(const std::vector<int64_t>& slots) {
    pybind11::gil_scoped_release release;
    for (auto slot : slots) copy_done_[slot]->synchronize();
  }

  void mark_used(const std::vector<int64_t>& slots) {
    auto event = std::make_shared<at::cuda::CUDAEvent>();
    event->record(c10::cuda::getCurrentCUDAStream(cache_.device().index()));
    for (auto slot : slots) used_[slot] = event;
  }

  void synchronize() {
    pybind11::gil_scoped_release release;
    stream_.synchronize();
  }

  int64_t copies() const { return copies_; }

 private:
  at::Tensor cache_, pool_;
  c10::cuda::CUDAStream stream_;
  std::vector<std::shared_ptr<at::cuda::CUDAEvent>> copy_done_, used_;
  int64_t copies_ = 0;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("execute_experts", &execute_experts, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("pack_dispatch", &pack_dispatch, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("unpack_dispatch", &unpack_dispatch);
  m.def("reduce_partials", &reduce_partials, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("scatter_returned", &scatter_returned, pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<H2DEngine>(m, "H2DEngine")
      .def(pybind11::init<at::Tensor, at::Tensor>())
      .def("fetch", &H2DEngine::fetch, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("wait_host", &H2DEngine::wait_host)
      .def("mark_used", &H2DEngine::mark_used)
      .def("synchronize", &H2DEngine::synchronize)
      .def_property_readonly("copies", &H2DEngine::copies);
}

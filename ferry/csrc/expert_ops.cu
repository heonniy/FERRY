#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/BFloat16.h>

__global__ void silu_mul(const c10::BFloat16* gate, const c10::BFloat16* up,
                         c10::BFloat16* output, int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float g = static_cast<float>(gate[i]);
    const float u = static_cast<float>(up[i]);
    output[i] = c10::BFloat16((g / (1.0f + expf(-g))) * u);
  }
}

at::Tensor silu_mul_cuda(const at::Tensor& gate, const at::Tensor& up) {
  auto output = at::empty_like(gate);
  if (gate.numel()) {
    silu_mul<<<(gate.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        gate.data_ptr<c10::BFloat16>(), up.data_ptr<c10::BFloat16>(),
        output.data_ptr<c10::BFloat16>(), gate.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}

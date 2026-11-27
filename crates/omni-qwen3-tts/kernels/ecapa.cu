// ECAPA-TDNN's convolutions use frame-major im2col and cuBLASLt. Keep the
// reference model's BF16 boundaries between the remaining operations.
#include "common.cuh"
#include <math_constants.h>

extern "C" __global__ void ecapa_columns(
    const bf16* x, const bf16* previous, bf16* columns, int frames, int stride,
    int start, int width, int previous_start, int taps, int dilation, int packed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= frames * packed) return;
  int tap_width = width * taps, c = i % packed;
  if (c >= tap_width) {
    columns[i] = to_bf16(c == tap_width ? 1.f : 0.f);
    return;
  }
  int t = i / packed + (c / width - taps / 2) * dilation;
  t = t < 0 ? -t : t >= frames ? 2 * (frames - 1) - t : t;
  int at = t * stride + c % width;
  bf16 v = x[at + start];
  if (previous_start >= 0) v = to_bf16(f32(v) + f32(previous[at + previous_start]));
  columns[i] = v;
}

extern "C" __global__ void ecapa_context(
    const bf16* x, const bf16* stats, bf16* columns, int frames, int width, int packed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= frames * packed) return;
  int c = i % packed;
  columns[i] = c < width ? x[(i / packed) * width + c]
      : c < 3 * width ? stats[c - width] : to_bf16(c == 3 * width ? 1.f : 0.f);
}

extern "C" __global__ void ecapa_place(
    const bf16* x, bf16* out, int frames, int in_stride, int in_start, int out_stride,
    int out_start, int width, int activation) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= frames * width) return;
  int t = i / width, c = i % width;
  float v = f32(x[t * in_stride + in_start + c]);
  if (activation == 1) v = fmaxf(v, 0.f);
  if (activation == 2) v = tanhf(fmaxf(v, 0.f));
  if (activation == 3) v = 1.f / (1.f + expf(-v));
  out[t * out_stride + out_start + c] = to_bf16(v);
}

extern "C" __global__ void ecapa_excite(
    const bf16* x, const bf16* gate, bf16* residual, bf16* layers, int frames,
    int width, int layer_width, int start) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= frames * width) return;
  int t = i / width, c = i % width;
  bf16 v = to_bf16(f32(residual[i]) + round_bf16(f32(x[i]) * f32(gate[c])));
  residual[i] = v;
  layers[t * layer_width + start + c] = v;
}

// A warp reads adjacent channels from the same frame. Eight warps accumulate
// separate time stripes and share their reduced result across the block.
constexpr int time_threads = 256;
constexpr int time_warps = time_threads / 32;

template <typename Term, typename Combine>
__device__ float reduce_time(int frames, float identity, Term term, Combine combine) {
  __shared__ float sums[time_threads];
  int thread = threadIdx.x, warp = thread / 32, lane = thread % 32;
  float sum = identity;
  for (int t = warp; t < frames; t += time_warps) sum = combine(sum, term(t));
  sums[thread] = sum;
  __syncthreads();
  if (warp == 0) {
    for (int w = 1; w < time_warps; ++w) sum = combine(sum, sums[w * 32 + lane]);
    sums[lane] = sum;
  }
  __syncthreads();
  sum = sums[lane];
  __syncthreads();
  return sum;
}

__device__ float add(float a, float b) { return a + b; }
__device__ float maximum(float a, float b) { return fmaxf(a, b); }

extern "C" __global__ void ecapa_squeeze(const bf16* x, bf16* mean, int frames, int width) {
  int c = blockIdx.x * 32 + threadIdx.x % 32;
  float sum = reduce_time(frames, 0.f, [&](int t) { return c < width ? f32(x[t * width + c]) : 0.f; }, add);
  if (threadIdx.x < 32 && c < width) mean[c] = to_bf16(sum / frames);
}

extern "C" __global__ void ecapa_softmax(bf16* attention, int frames, int width) {
  int c = blockIdx.x * 32 + threadIdx.x % 32;
  auto value = [&](int t) { return c < width ? f32(attention[t * width + c]) : 0.f; };
  float peak = reduce_time(frames, -CUDART_INF_F, value, maximum);
  auto exponent = [&](int t) { return expf(value(t) - peak); };
  float sum = reduce_time(frames, 0.f, exponent, add);
  if (c >= width) return;
  for (int t = threadIdx.x / 32; t < frames; t += time_warps)
    attention[t * width + c] = to_bf16(exponent(t) / sum);
}

extern "C" __global__ void ecapa_statistics(
    const bf16* x, const bf16* attention, bf16* stats, int frames, int width, int weighted) {
  int c = blockIdx.x * 32 + threadIdx.x % 32;
  // The reference first sums a BF16 mask, then divides it by that sum.
  float uniform = round_bf16(1.f / round_bf16(float(frames)));
  auto value = [&](int t) { return c < width ? f32(x[t * width + c]) : 0.f; };
  auto weight = [&](int t) { return weighted && c < width ? f32(attention[t * width + c]) : uniform; };
  float mean = round_bf16(reduce_time(frames, 0.f, [&](int t) { return round_bf16(weight(t) * value(t)); }, add));
  float var = reduce_time(frames, 0.f, [&](int t) {
    float delta = round_bf16(value(t) - mean);
    return round_bf16(weight(t) * round_bf16(delta * delta));
  }, add);
  if (threadIdx.x < 32 && c < width) {
    stats[c] = to_bf16(mean);
    stats[width + c] = to_bf16(sqrtf(fmaxf(round_bf16(var), round_bf16(1e-12f))));
  }
}

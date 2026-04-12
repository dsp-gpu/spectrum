
// NOTE: Reference copy of HIP kernel source.
// Runtime source is embedded in include/kernels/fft_processor_kernels_rocm.hpp (GetHIPKernelSource).

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

struct float2_t {
    float x;
    float y;
};

// 2D grid: blockIdx.y == beam_id — eliminates int div/mod per thread.
// Zeros in fft_input handled by hipMemsetAsync before launch — no divergent else-branch.
// __launch_bounds__(256) improves occupancy.
__launch_bounds__(256)
extern "C" __global__ void pad_data(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ output,
    unsigned int n_point,
    unsigned int nFFT)
{
    unsigned int beam_id = blockIdx.y;
    unsigned int pos     = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_point) return;  // Only copy valid points; rest already zero

    output[beam_id * nFFT + pos] = input[beam_id * n_point + pos];
}

// Interleaved {mag, phase} output — caller does single DtoH instead of two transfers.
// __fsqrt_rn / __atan2f — fast GPU intrinsics (~4 ULP accuracy).
__launch_bounds__(256)
extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ fft_output,
    float2_t* __restrict__ mag_phase,   // interleaved: .x=mag, .y=phase
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = fft_output[gid];
    float2_t mp;
    mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);
    mp.y = atan2f(z.y, z.x);
    mag_phase[gid] = mp;
}

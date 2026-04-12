

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

struct float2_t {
    float x;
    float y;
};

__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ input,
    float2_t* __restrict__ mag_phase,
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    float2_t mp;
    mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);
    mp.y = atan2f(z.y, z.x);
    mag_phase[gid] = mp;
}

__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_magnitude(
    const float2_t* __restrict__ input,
    float* __restrict__ output,
    float inv_n,
    unsigned int total)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    output[gid] = __fsqrt_rn(z.x * z.x + z.y * z.y) * inv_n;
}


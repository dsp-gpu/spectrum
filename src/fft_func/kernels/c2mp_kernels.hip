

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

struct float2_t {
    float x;
    float y;
};

// ═══════════════════════════════════════════════════════════════
// Kernel: complex_to_mag_phase
// Converts complex data to magnitude + phase (interleaved output)
// One thread per element. 1D grid.
// ═══════════════════════════════════════════════════════════════
__launch_bounds__(BLOCK_SIZE)
extern "C" __global__ void complex_to_mag_phase(
    const float2_t* __restrict__ input,      // Complex input: {re, im}
    float2_t* __restrict__ mag_phase,         // Interleaved output: {mag, phase}
    unsigned int total)                       // Total elements (beam_count * n_point)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    float2_t z = input[gid];
    float2_t mp;
    mp.x = __fsqrt_rn(z.x * z.x + z.y * z.y);  // fast sqrt intrinsic
    mp.y = atan2f(z.y, z.x);                 // fast atan2 intrinsic (~4 ULP)
    mag_phase[gid] = mp;
}


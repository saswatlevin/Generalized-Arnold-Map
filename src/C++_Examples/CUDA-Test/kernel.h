#include <cuda_runtime_api.h>
#include <cuda.h>
extern "C" void run_saxpy(int n, float a, float *x, float *y,dim3 blocks, dim3 block_size);

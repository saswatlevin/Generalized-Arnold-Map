__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}
extern "C" void run_saxpy(int n, float a, float *x, float *y, dim3 blocks, dim3 block_size)
{
	saxpy <<< blocks, block_size >>> (n,2.0f,x,y);
}


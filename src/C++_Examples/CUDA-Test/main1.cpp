#include <iostream>
#include <cstdio>
#include "kernel.h"
#include <math.h>
#include <opencv2/opencv.hpp> 	/*For OpenCV*/
#include <cstdint>

using namespace std;
using namespace cv;

int main()
{
  
  int N = 1<<20;
  /*Device and haost pointers for x,y and z*/
  float *x,*y;
  uint8_t a=4;
  Mat image;
  image=imread("minray.png",0);
  printf("\nLoaded image");

  /*Allocate Unified Memory – accessible from CPU or GPU*/
  cudaMallocManaged((void**)&x, N*sizeof(float));
  cudaMallocManaged((void**)&y, N*sizeof(float));
  
  dim3 block(16, 16);
  dim3 grid((int)ceil(double((N+256.0) / 256.0)));
  
  for (int i = 0; i < N; i++) 
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  
  run_saxpy(N, 2.0f, x, y, grid, block);
  

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  
  
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(x);
  cudaFree(y);
  free(x);
  free(y); 
  
  return 0;
}

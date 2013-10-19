
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void dostuffKernel(int *ptr)
{
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int r = 0;
	int g = 0;
	int b = 0;
	int a = 255;
	r = (index_x / 799.0) * 255;
	ptr[index_y * 800 + index_x] = a << 24 | r << 16  | g << 8 | b;

}

cudaError_t dostuff(void *ptr)
{
	dim3 block_size;
	block_size.x = 4;
	block_size.y = 4;

	dim3 grid_size;
	grid_size.x = 800 / block_size.x;
	grid_size.y = 600 / block_size.y;

	dostuffKernel<<<grid_size, block_size>>>(static_cast<int*>(ptr));

	return cudaDeviceSynchronize();
}
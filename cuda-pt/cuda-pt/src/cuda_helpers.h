#ifndef CUDA_HELPERS_H_
#define CUDA_HELPERS_H_

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

#endif
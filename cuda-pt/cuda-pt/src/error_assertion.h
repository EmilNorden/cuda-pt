#ifndef ERROR_ASSERTION_H_
#define ERROR_ASSERTION_H_

#include "cuda.h"
#include "driver_types.h"
#include <SDL_opengl.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

const char *cudaErrorToString(cudaError_t error);

void reportErrorAndDie(cudaError_t error, const char *file, int line);
void reportCudaDriverErrorAndDie(CUresult error, const char *file, int line);
void reportGLErrorAndDie(GLenum error, const char *file, int line);

#define CUDA_CALL(x) if(x != cudaSuccess) { reportErrorAndDie(x, __FILE__, __LINE__);  }
#define CUDA_DRIVER_CALL(x) if(x != CUDA_SUCCESS) { reportCudaDriverErrorAndDie(x, __FILE__, __LINE__);  }

#define GL_CALL(x) if(x != GL_NO_ERROR) { reportGLErrorAndDie(x, __FILE__, __LINE__); } 

#endif
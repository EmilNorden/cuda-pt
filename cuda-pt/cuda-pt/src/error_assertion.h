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

const char *cuda_error_to_string(cudaError_t error);
void report_error_and_die(cudaError_t error, const char *file, int line);
void report_cuda_driver_error_and_die(CUresult error, const char *file, int line);
void report_gl_error_and_die(GLenum error, const char *file, int line);

#define CUDA_CALL(x) if(x != cudaSuccess) { report_error_and_die(x, __FILE__, __LINE__);  }
#define CUDA_DRIVER_CALL(x) if(x != CUDA_SUCCESS) { report_cuda_driver_error_and_die(x, __FILE__, __LINE__);  }

#define GL_CALL(x) if((GLenum)x != (GLenum)GL_NO_ERROR) { report_gl_error_and_die(x, __FILE__, __LINE__); } 

#endif
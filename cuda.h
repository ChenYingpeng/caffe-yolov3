/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04
 */

#ifndef __CUDA_H_
#define __CUDA_H_
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#define BLOCK 512

void check_error(cudaError_t status);

dim3 cuda_gridsize(size_t n);

float* cuda_make_array(float* x,size_t n);

void cuda_free(float* x_gpu);

void cuda_push_array(float *x_gpu,float* x,size_t n);

void cuda_pull_array(float *x_gpu,float* x,size_t n);


#endif

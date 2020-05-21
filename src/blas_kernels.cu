/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04
 */


#include <assert.h>

#include "cuda.h"
#include "blas.h"

__global__ void copy_kernel(int N,float* X,int OFFX,int INCX,float* Y,int OFFY,int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

__global__ void fill_kernel(int N, float ALPHA, float *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

void copy_gpu_offset(int N,float* X,int OFFX,int INCX,float* Y,int OFFY,int INCY)
{
    copy_kernel<<<cuda_gridsize(N),BLOCK>>>(N,X,OFFX,INCX,Y,OFFY,INCY);
    check_error(cudaPeekAtLastError());
}

void copy_gpu(int N,float* X,int INCX,float* Y,int INCY)
{
    copy_gpu_offset(N,X,0,INCX,Y,0,INCY);
}


void fill_gpu(int N, float ALPHA, float * X, int INCX)
{
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, ALPHA, X, INCX);
    check_error(cudaPeekAtLastError());
}

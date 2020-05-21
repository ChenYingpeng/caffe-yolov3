/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04
 */

#ifndef __BLAS_H_
#define __BLAS_H_

void copy_gpu(int N,float* X,int INCX,float* Y,int INCY);

void fill_gpu(int N, float ALPHA, float * X, int INCX);

#endif

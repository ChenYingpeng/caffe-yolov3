/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2019/01/24	
 */

#ifndef __MAX_POOL_1D_H__
#define __MAX_POOL_1D_H__

void max_pool_1d_gpu(float* input_data_gpu,int batch_size,int c,int h,int w,int size,int stride,int pad,float* output);

#endif

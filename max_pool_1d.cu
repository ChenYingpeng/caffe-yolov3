/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2019/01/24	
 */
#include "max_pool_1d.h"
#include "cuda.h"


__global__ void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, float *input, float *output)
{
    int h = (in_h + pad - size)/stride + 1;
    int w = (in_w + pad - size)/stride + 1;
    int c = in_c;

    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int j = id % w;
    id /= w;
    int i = id % h;
    id /= h;
    int k = id % c;
    id /= c;
    int b = id;

    int w_offset = -pad/2;
    int h_offset = -pad/2;

    int out_index = j + w*(i + h*(k + c*b));
    float max = -INFINITY;
    //int max_i = -1;
    int l, m;
    for(l = 0; l < size; ++l){
        for(m = 0; m < size; ++m){
            int cur_h = h_offset + i*stride + l;
            int cur_w = w_offset + j*stride + m;
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));
            int valid = (cur_h >= 0 && cur_h < in_h &&
                    cur_w >= 0 && cur_w < in_w);
            float val = (valid != 0) ? input[index] : -INFINITY;
            //max_i = (val > max) ? index : max_i;
            max   = (val > max) ? val   : max;
        }
    }
    output[out_index] = max;
    //indexes[out_index] = max_i;
}

void max_pool_1d_gpu(float* input_data_gpu,int batch_size,int c,int h,int w,int size,int stride,int pad,float* output_data_gpu)
{
    size_t n = h*w*c*batch_size;

    forward_maxpool_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, h, w, c, stride, size, pad, input_data_gpu, output_data_gpu);

    check_error(cudaPeekAtLastError());	
}

#include <cmath>
#include <vector>

#include "caffe/layers/mish_layer.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype tanh_activate_kernel(Dtype x){return (2/(1 + expf(-2*x)) - 1);}


template <typename Dtype>
__device__ Dtype softplus_kernel(Dtype x, float threshold = 20) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return expf(x);    // too small
    return logf(expf(x) + 1);
}

/*__device__ float tanh_activate_kernel(float x){return (2/(1 + expf(-2*x)) - 1);}

__device__ float softplus_kernel(float x, float threshold = 20) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return expf(x);    // too small
    return logf(expf(x) + 1);
}*/

template <typename Dtype>
__global__ void MishForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * tanh_activate_kernel(softplus_kernel(in[index]));
  }
}

template <typename Dtype>
void MishLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  MishForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void MishBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    //const Dtype sigmoid_x = out_data[index];
    //out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);

    const float MISH_THRESHOLD = 20.0f;
    // implementation from TensorFlow: https://github.com/tensorflow/addons/blob/093cdfa85d334cbe19a37624c33198f3140109ed/tensorflow_addons/custom_ops/activations/cc/kernels/mish_op.h#L66-L80
    // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
    const float inp = out_data[index];
    const float sp = softplus_kernel(inp, MISH_THRESHOLD);
    const float grad_sp = 1 - expf(-sp);
    const float tsp = tanh(sp);
    const float grad_tsp = (1 - tsp*tsp) * grad_sp;
    const float grad = inp * grad_tsp + tsp;

    out_diff[index] = in_diff[index] * grad;

  }
}

template <typename Dtype>
void MishLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MishBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MishLayer);


}  // namespace caffe

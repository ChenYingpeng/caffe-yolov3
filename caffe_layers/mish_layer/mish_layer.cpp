#include <cmath>
#include <vector>

#include "caffe/layers/mish_layer.hpp"

namespace caffe {


template <typename Dtype>
inline Dtype tanh_activate(Dtype x) { return (2 / (1 + expf(-2 * x)) - 1); }


template <typename Dtype>
inline Dtype softplus_activate(Dtype x, float threshold) {
    if (x > threshold) return x;                // too large
    else if (x < -threshold) return expf(x);    // too small
    return logf(expf(x) + 1);
}

template <typename Dtype>
void MishLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  const float MISH_THRESHOLD = 20;
  for (int i = 0; i < count; ++i) {
    float x_val = bottom_data[i];
    top_data[i] = x_val * tanh_activate(softplus_activate(x_val, MISH_THRESHOLD));
  }
}

template <typename Dtype>
void MishLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      // const Dtype sigmoid_x = top_data[i];
      // bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);

      const float MISH_THRESHOLD = 20.0f;
      // implementation from TensorFlow: https://github.com/tensorflow/addons/commit/093cdfa85d334cbe19a37624c33198f3140109ed
      // implementation from Pytorch: https://github.com/thomasbrandon/mish-cuda/blob/master/csrc/mish.h#L26-L31
      Dtype inp = top_data[i];
      const Dtype sp = softplus_activate(inp, MISH_THRESHOLD);
      const Dtype grad_sp = 1 - exp(-sp);
      const Dtype tsp = tanh(sp);
      const Dtype grad_tsp = (1 - tsp*tsp) * grad_sp;
      const Dtype grad = inp * grad_tsp + tsp;
      bottom_diff[i] = top_diff[i] * grad;

    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MishLayer);
#endif

INSTANTIATE_CLASS(MishLayer);
REGISTER_LAYER_CLASS(Mish);

}  // namespace caffe

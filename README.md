# caffe-yolov3
# Paltform
Have tested on Ubuntu16.04LTS with Jetson-TX2 and Ubuntu16.04LTS with gtx1060;

NOTE: You need change CMakeList.txt on Ubuntu16.04LTS with GTX1060.

# Install
git clone https://github.com/ChenYingpeng/caffe-yolov3

cd caffe-yolov3

mkdir build

cd build

cmake ..

make -j6

# Demo
First,download model and put it into dir caffemodel.

$ ./x86_64/bin/detectnet ../prototxt/yolov3.prototxt ../caffemodel/yolov3.caffemodel ../images/dog.jpg 

# Download Model

Baidu link [model](https://pan.baidu.com/s/1yiCrnmsOm0hbweJBiiUScQ)


# Note

1.Only inference on GPU platform,such as RTX2080, GTX1060,Jetson Tegra X1,TX2,nano,Xavier etc.

2.Support model such as yolov3、yolov3-spp、yolov3-tiny、mobilenet_v1_yolov3、mobilenet_v2_yolov3 etc and input network size 320x320,416x416,608x608 etc.

3.Mobilenet_v1 + yolov3 (test COCO,mAP = 0.3798,To be optimized)

4.Yolov3-tiny: Caffe can not duplicate the layer that maxpool layer (params:kernel_size = 2,stride = 1),so add these code into /caffe/layers/pooling_layer.cpp for recurrenting it.

```
 pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;

  /*added by chen for darknet yolov3 tiny maxpool layer stride=1,size =2*/
  ++if((kernel_h_ - stride_h_) % 2 == 1){
  ++  pooled_height_ += 1;
  ++  pooled_width_ += 1;
  ++}
```

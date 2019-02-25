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

# Test

Example 1: yolov3

$ ./x86_64/bin/detectnet 0 ../../data/yolov3/prototxt/yolov3.prototxt ../../data/yolov3/caffemodel/yolov3.caffemodel

Example 2: yolov3-spp

$ ./x86_64/bin/detectnet 0 ../../data/yolov3/prototxt/yolov3-spp.prototxt ../../data/yolov3/caffemodel/yolov3-spp.caffemodel 

Example 3: mobilenet_v1 + yolov3

$ ./x86_64/bin/detectnet 0 ../../data/yolov3/prototxt/mobilenet_v1_yolov3.prototxt ../../data/yolov3/caffemodel/mobilenet_v1_yolov3.caffemodel 

Example 4:yolov3-tiny

$ ./x86_64/bin/detectnet 1 ../../data/yolov3/prototxt/yolov3-tiny-1.prototxt ../../data/yolov3/prototxt/yolov3-tiny-2.prototxt ../../data/yolov3/caffemodel/yolov3-tiny.caffemodel 

# Download Model

Baidu link [model](https://pan.baidu.com/s/1yiCrnmsOm0hbweJBiiUScQ)


# Note

1.Only inference

2.Support model such as yolov3、yolov3-spp、yolov3-tiny、mobilenet_v1_yolov3 etc and input network size 320x320,416x416,608x608 etc.

3.Mobilenet_v1 + yolov3 (test COCO,mAP = 0.3798,To be optimized)

4.Yolov3-tiny: Caffe can not duplicate the layer that maxpool layer (params:kernel_size = 2,stride = 1),so rewrite max_pool_1d function for recurrenting it.

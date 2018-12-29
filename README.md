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


# Download Model

Baidu link [model](https://pan.baidu.com/s/1yiCrnmsOm0hbweJBiiUScQ)


# Note

Only inference

Support input res 320x320,416x416,608x608 etc.

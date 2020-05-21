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

$ ./x86_64/bin/demo ../prototxt/yolov4.prototxt ../caffemodel/yolov4.caffemodel ../images/dog.jpg 

# Eval
1. run
$ ./x86_64/bin/eval ../prototxt/yolov4.prototxt ../caffemodel/yolov4.caffemodel /path/to/coco/val2017/

generate `coco_results.json` on `results/`.

2. run
$ python coco_eval/coco_eval.py --gt-json path/to/coco/annotations/instances_val2017.json  --pred-json results/coco_results.json

# Download Model

Baidu link [model](https://pan.baidu.com/s/1yiCrnmsOm0hbweJBiiUScQ)


# Note

1.Only inference on GPU platform,such as RTX2080, GTX1060,Jetson Tegra X1,TX2,nano,Xavier etc.

2.Support model such as yolov4,yolov3,yolov3-spp,yolov3-tiny,mobilenet_v1_yolov3,mobilenet_v2_yolov3 etc.

3. darknet2caffe link [github](https://github.com/ChenYingpeng/darknet2caffe)


### References
Appreciate the great work from the following repositories:
- [official/Yolo](https://pjreddie.com/darknet/yolo/)
- [AlexeyAB/YoloV4](https://github.com/AlexeyAB/darknet)

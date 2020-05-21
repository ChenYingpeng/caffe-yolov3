/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2020/04/26	
 */
#include <string>
#include <vector>
#include <iostream>
#include <glog/logging.h>

#include <caffe/caffe.hpp>


#include "image_opencv.h"
#include "yolo_layer.h"

using namespace caffe;
// using namespace cv;


struct bbox_t{
    unsigned int x,y,w,h;   //(x,y) - top-left corner, (w,h) - width & height of bounded box
    float prob;             // confidence - probability that the object was found correctly
    unsigned int obj_id;    // class of object - from range [0,classes - 1]
};

class Detector{
public:
    Detector(std::string prototxt,std::string caffemodel,int gpu_id);
    ~Detector();

    std::vector<bbox_t> detect(std::string image_path,float thresh);
    std::vector<bbox_t> detect(cv::Mat mat,float thresh);

private:
    shared_ptr<Net<float> > m_net;
    Blob<float> * m_net_input_data_blobs;
    vector<Blob<float>*> m_blobs;

    float m_thresh = 0.001;
    int m_classes = 80; //coco classes
};

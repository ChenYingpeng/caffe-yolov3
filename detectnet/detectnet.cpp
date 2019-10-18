
/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04	
 */

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>

#include <string>
#include <vector>
#include <sys/time.h>
#include <glog/logging.h>

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

#include "image.h"
#include "yolo_layer.h"

using namespace caffe;
using namespace cv;


bool signal_recieved = false;

void sig_handler(int signo){
    if( signo == SIGINT ){
            printf("received SIGINT\n");
            signal_recieved = true;
    }
}

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

int main( int argc, char** argv )
{
    string model_file;
    string weights_file;
    string image_path;
    if(4 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        image_path = argv[3];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [image_path]";
        return -1;
    }	

    // Initialize the network.
    Caffe::set_mode(Caffe::GPU);

    image im,sized;
    vector<Blob<float>*> blobs;
    blobs.clear();

    int nboxes = 0;
    int size;
    detection *dets = NULL;
        
    /* load and init network. */
    shared_ptr<Net<float> > net;
    net.reset(new Net<float>(model_file, TEST));
    net->CopyTrainedLayersFrom(weights_file);
    LOG(INFO) << "net inputs numbers is " << net->num_inputs();
    LOG(INFO) << "net outputs numbers is " << net->num_outputs();

    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";

    Blob<float> *net_input_data_blobs = net->input_blobs()[0];
    LOG(INFO) << "input data layer channels is  " << net_input_data_blobs->channels();
    LOG(INFO) << "input data layer width is  " << net_input_data_blobs->width();
    LOG(INFO) << "input data layer height is  " << net_input_data_blobs->height();

    size = net_input_data_blobs->channels()*net_input_data_blobs->width()*net_input_data_blobs->height();


    uint64_t beginDataTime =  current_timestamp();
    //load image
    im = load_image_color((char*)image_path.c_str(),0,0);
    sized = letterbox_image(im,net_input_data_blobs->width(),net_input_data_blobs->height());
    cuda_push_array(net_input_data_blobs->mutable_gpu_data(),sized.data,size);

    uint64_t endDataTime =  current_timestamp();
    LOG(INFO) << "processing data operation avergae time is "
              << endDataTime - beginDataTime << " ms";

    uint64_t startDetectTime = current_timestamp();
    // forward
    net->Forward();
    for(int i =0;i<net->num_outputs();++i){
        blobs.push_back(net->output_blobs()[i]);
    }
    dets = get_detections(blobs,im.w,im.h,
        net_input_data_blobs->width(),net_input_data_blobs->height(),&nboxes);

    uint64_t endDetectTime = current_timestamp();
    LOG(INFO) << "caffe yolov3 : processing network yolov3 tiny avergae time is "
              << endDetectTime - startDetectTime << " ms";

    //show detection results
    Mat img = imread(image_path.c_str());
    int i,j;
    for(i=0;i< nboxes;++i){
        char labelstr[4096] = {0};
        int cls = -1;
        for(j=0;j<80;++j){
            if(dets[i].prob[j] > 0.5){
                if(cls < 0){
                    cls = j;
                }
                LOG(INFO) << "label = " << cls
                          << ", prob = " << dets[i].prob[j]*100;
            }
        }
        if(cls >= 0){
            box b = dets[i].bbox;
//            LOG(INFO) << "  x = " << b.x
//                      << ", y = " << b.y
//                      << ", w = " << b.w
//                      << ", h = " << b.h;
            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
            rectangle(img,Point(left,top),Point(right,bot),Scalar(0,0,255),3,8,0);
            LOG(INFO) << "  left = " << left
                      << ", right = " << right
                      << ", top = " << top
                      << ", bot = " << bot;

        }
    }

    namedWindow("show",CV_WINDOW_AUTOSIZE);
    imshow("show",img);
    waitKey(0);

    free_detections(dets,nboxes);
    free_image(im);
    free_image(sized);
        
    LOG(INFO) << "done.";
    return 0;
}


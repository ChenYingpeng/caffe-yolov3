
/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04	
 */

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "yolo_layer.h"
#include "image.h"
#include "cuda.h"

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <sys/time.h>

using namespace caffe;
using namespace cv;

const int timeIters = 10;

//YOLOV3
const string& model_file = "/home/chen/projects/caffe-yolov3/data/yolov3/yolov3.prototxt";//modify your model file path
const string& weights_file = "/home/chen/projects/caffe-yolov3/data/yolov3/yolov3.caffemodel";//modify your weights file path
const char* imgFilename = "/home/chen/projects/caffe-yolov3/data/images/dog.jpg"; //modify your images file path

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

bool signal_recieved = false;

void sig_handler(int signo)
{
    if( signo == SIGINT ){
	printf("received SIGINT\n");
	signal_recieved = true;
    }
}


int main( int argc, char** argv )
{
    printf("detectnet\n  args (%i):  ", argc);

    for( int i=0; i < argc; i++ )
	printf("%i [%s]  ", i, argv[i]);
	
    printf("\n\n");	

    if( signal(SIGINT, sig_handler) == SIG_ERR )
	printf("\ncan't catch SIGINT\n");

    // Initialize the network.
    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    shared_ptr<Net<float> > net;
    net.reset(new Net<float>(model_file, TEST));
    net->CopyTrainedLayersFrom(weights_file);

    printf("num_inputs is %d\n",net->num_inputs());
    printf("num_outputs is %d\n",net->num_outputs());
    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net->num_outputs(), 3) << "Network should have exactly three outputs.";	

    Blob<float> *input_data_blobs = net->input_blobs()[0];
    LOG(INFO) << "Input data layer channels is  " << input_data_blobs->channels();
    LOG(INFO) << "Input data layer width is  " << input_data_blobs->width();
    LOG(INFO) << "Input data layer height is  " << input_data_blobs->height();

    int size = input_data_blobs->channels()*input_data_blobs->width()*input_data_blobs->height();

    uint64_t dataTime = 0;
    uint64_t networkTime = 0;
    image im,sized;
    int nboxes = 0;
    detection *dets = NULL;
    for(int i=0;i<timeIters;++i){
    	uint64_t beginDataTime =  current_timestamp();
    	//load image
    	im = load_image_color((char*)imgFilename,0,0);
    	sized = letterbox_image(im,input_data_blobs->width(),input_data_blobs->height());
    	cuda_push_array(input_data_blobs->mutable_gpu_data(),sized.data,size);

    	uint64_t endDataTime =  current_timestamp();
        dataTime += (endDataTime - beginDataTime);

    	//YOLOV3 objection detection implementation with Caffe

    	net->Forward();

    	vector<Blob<float>*> blobs;
    	blobs.clear();
    	Blob<float>* out_blob1 = net->output_blobs()[1];
    	blobs.push_back(out_blob1);
    	Blob<float>* out_blob2 = net->output_blobs()[2];
    	blobs.push_back(out_blob2);
    	Blob<float>* out_blob3 = net->output_blobs()[0];
    	blobs.push_back(out_blob3);

    	//printf("output blob1 shape c= %d, h = %d, w = %d\n",out_blob1->channels(),out_blob1->height(),out_blob1->width());
    	//printf("output blob2 shape c= %d, h = %d, w = %d\n",out_blob2->channels(),out_blob2->height(),out_blob2->width());
    	//printf("output blob3 shape c= %d, h = %d, w = %d\n",out_blob3->channels(),out_blob3->height(),out_blob3->width());

    	//int nboxes = 0;
    	//printf("img width =%d, height = %d\n",im.w,im.h);
    	dets = get_detections(blobs,im.w,im.h,&nboxes);

    	uint64_t endDetectTime = current_timestamp();
        networkTime += (endDetectTime - endDataTime);
    }
    
    printf("object-detection: total iters = %d done, processing data operation avergae time is  (%zu)ms\n", timeIters,dataTime/timeIters);
    printf("object-detection: total iters = %d done, processing network yolov3 avergae time is (%zu)ms\n", timeIters,networkTime/timeIters);

    //show detection results
    Mat img = imread(imgFilename);

    int i,j;
    for(i=0;i< nboxes;++i){
        char labelstr[4096] = {0};
        int cls = -1;
        for(j=0;j<80;++j){
            if(dets[i].prob[j] > 0.5){
                if(cls < 0){
                    cls = j;
                }
                printf("%d: %.0f%%\n",cls,dets[i].prob[j]*100);
            }
        }
        if(cls >= 0){
            box b = dets[i].bbox;
            printf("x = %f,y =  %f,w = %f,h =  %f\n",b.x,b.y,b.w,b.h);

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;
            rectangle(img,Point(left,top),Point(right,bot),Scalar(0,0,255),3,8,0);
            printf("left = %d,right =  %d,top = %d,bot =  %d\n",left,right,top,bot);
        }
    }

    namedWindow("show",CV_WINDOW_AUTOSIZE);
    imshow("show",img);
    waitKey(0);

    free_detections(dets,nboxes);
    free_image(im);
    free_image(sized);
        
    printf("detectnet-camera:  video device has been un-initialized.\n");
    printf("detectnet-camera:  this concludes the test of the video device.\n");
    return 0;
}


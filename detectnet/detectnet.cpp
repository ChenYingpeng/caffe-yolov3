
/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04	
 */

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>

#include "yolo_layer.h"
#include "image.h"
#include "cuda.h"
#include "max_pool_1d.h"
#include "blas.h"

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <sys/time.h>

using namespace caffe;
using namespace cv;

const char* imgFilename = "/home/chen/projects/data/images/dog.jpg"; //modify your images file path

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

//! Note: Net的Blob是指，每个层的输出数据，即Feature Maps
unsigned int get_blob_index(boost::shared_ptr< Net<float> > & net, char *query_blob_name)
{
    std::string str_query(query_blob_name);    
    vector< string > const & blob_names = net->blob_names();
    for( unsigned int i = 0; i != blob_names.size(); ++i ) 
    { 
        //std::cout << "blob names " << i << " is " << blob_names[i] << std::endl; 
        if( str_query == blob_names[i] ) 
        { 
            return i;
        } 
    }
    LOG(FATAL) << "Unknown blob name: " << str_query;
}


int main( int argc, char** argv )
{
    printf("detectnet\n  args (%i):  ", argc);
//YOLOV3
    string model_file;
    string weights_file;

    //yolov3-tiny
    string model1_file;
    string model2_file;
    string tiny_weights_file;

    if(4 == argc){
        assert(0 == atoi(argv[1]));
        model_file = argv[2];
        weights_file = argv[3];
    }
    else if(5 == argc){
        assert(1 == atoi(argv[1]));
        model1_file = argv[2];
        model2_file = argv[3];
        tiny_weights_file = argv[4];
    }
    else{
        printf("Input error: please input ./xx YOLOV3 [model_path] [weights_path] or ./xx YOLOV3_TINY [model1_path] [model2_path] [weights_path]\n");
        return -1;
    }
	
    printf("\n\n");	

    // Initialize the network.
    Caffe::set_mode(Caffe::GPU);

    image im,sized;
    vector<Blob<float>*> blobs;
    blobs.clear();

    int nboxes = 0;
    int size;
    detection *dets = NULL;

    NetType type = (NetType)atoi(argv[1]);

    if(0 == type){
        /* Load the network. */
        shared_ptr<Net<float> > net;
        net.reset(new Net<float>(model_file, TEST));
        net->CopyTrainedLayersFrom(weights_file);

        printf("net num_inputs is %d\n",net->num_inputs());
        printf("net num_outputs is %d\n",net->num_outputs());
        CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
        CHECK_EQ(net->num_outputs(), 3) << "Network should have exactly three outputs.";

        Blob<float> *input_data_blobs = net->input_blobs()[0];
        LOG(INFO) << "Input data layer channels is  " << input_data_blobs->channels();
        LOG(INFO) << "Input data layer width is  " << input_data_blobs->width();
        LOG(INFO) << "Input data layer height is  " << input_data_blobs->height();

        size = input_data_blobs->channels()*input_data_blobs->width()*input_data_blobs->height();
        
        //load image
        uint64_t beginDataTime =  current_timestamp();
        im = load_image_color((char*)imgFilename,0,0);
        sized = letterbox_image(im,input_data_blobs->width(),input_data_blobs->height());
        cuda_push_array(input_data_blobs->mutable_gpu_data(),sized.data,size);
        uint64_t endDataTime =  current_timestamp();

        //YOLOV3 objection detection implementation with Caffe
        net->Forward();

        Blob<float>* out_blob1 = net->output_blobs()[1];
        blobs.push_back(out_blob1);
        Blob<float>* out_blob2 = net->output_blobs()[2];
        blobs.push_back(out_blob2);
        Blob<float>* out_blob3 = net->output_blobs()[0];
        blobs.push_back(out_blob3);

        dets = get_detections(blobs,im.w,im.h,input_data_blobs->width(),input_data_blobs->height(),&nboxes,type);
        uint64_t endDetectTime = current_timestamp();

        printf("object-detection:  processing data operation avergae time is  (%zu)ms\n", endDataTime - beginDataTime);
        printf("object-detection:  processing network yolov3 avergae time is (%zu)ms\n", endDetectTime - endDataTime);

    }

    if(1 == type){
        
        /* Load the network. */
        shared_ptr<Net<float> > net1,net2;
        net1.reset(new Net<float>(model1_file, TEST));
        net2.reset(new Net<float>(model2_file, TEST));
        net1->CopyTrainedLayersFrom(tiny_weights_file);
        net2->CopyTrainedLayersFrom(tiny_weights_file);

        printf("net1 num_inputs is %d\n",net1->num_inputs());
        printf("net1 num_outputs is %d\n",net1->num_outputs());
        printf("net2 num_inputs is %d\n",net2->num_inputs());
        printf("net2 num_outputs is %d\n",net2->num_outputs());

        CHECK_EQ(net1->num_inputs(), 1) << "Network should have exactly one input.";
        CHECK_EQ(net1->num_outputs(), 1) << "Network should have exactly three outputs.";

        CHECK_EQ(net2->num_inputs(), 2) << "Network should have exactly one input.";
        CHECK_EQ(net2->num_outputs(), 2) << "Network should have exactly three outputs.";

        Blob<float> *net1_input1_data_blobs = net1->input_blobs()[0];
        Blob<float> *net2_input1_data_blobs = net2->input_blobs()[0];
        Blob<float> *net2_input2_data_blobs = net2->input_blobs()[1];
        LOG(INFO) << "Input1 data layer channels is  " << net1_input1_data_blobs->channels();
        LOG(INFO) << "Input1 data layer width is  " << net1_input1_data_blobs->width();
        LOG(INFO) << "Input1 data layer height is  " << net1_input1_data_blobs->height();

        LOG(INFO) << "Input2 data1 layer channels is  " << net2_input1_data_blobs->channels();
        LOG(INFO) << "Input2 data1 layer width is  " << net2_input1_data_blobs->width();
        LOG(INFO) << "Input2 data1 layer height is  " << net2_input1_data_blobs->height();

        LOG(INFO) << "Input2 data2 layer channels is  " << net2_input2_data_blobs->channels();
        LOG(INFO) << "Input2 data2 layer width is  " << net2_input2_data_blobs->width();
        LOG(INFO) << "Input2 data2 layer height is  " << net2_input2_data_blobs->height();

        size = net1_input1_data_blobs->channels()*net1_input1_data_blobs->width()*net1_input1_data_blobs->height();

        //load image
        printf("start forward yolov3-tiny!\n");
        uint64_t beginDataTime =  current_timestamp();
        im = load_image_color((char*)imgFilename,0,0);
        sized = letterbox_image(im,net1_input1_data_blobs->width(),net1_input1_data_blobs->height());
        cuda_push_array(net1_input1_data_blobs->mutable_gpu_data(),sized.data,size);
        uint64_t endDataTime =  current_timestamp();

        net1->Forward();

        //temp output
        Blob<float>* out1_blob1 = net1->output_blobs()[0];
        LOG(INFO) << "temp output data layer channels is  " << out1_blob1->channels();
        LOG(INFO) << "temp outputdata layer width is  " << out1_blob1->width();
        LOG(INFO) << "temp output data layer height is  " << out1_blob1->height();

        char *query_blob_name = "layer9-conv";
        unsigned int blob_id = get_blob_index(net1, query_blob_name);
        boost::shared_ptr<Blob<float> > out1_blob2 = net1->blobs()[blob_id];

        //load input data1
        //Note: size = 2 stride = 1
        int kernel_size = 2;
        int stride = 1;
        int pad = kernel_size - stride;
        max_pool_1d_gpu(out1_blob1->mutable_gpu_data(),1,out1_blob1->channels(),out1_blob1->height(),out1_blob1->width(),kernel_size,stride,pad,net2_input1_data_blobs->mutable_gpu_data());
        
        //load input data2
        copy_gpu(out1_blob2->count(),(float*)out1_blob2->mutable_gpu_data(),1,net2_input2_data_blobs->mutable_gpu_data(),1);
        
        net2->Forward();

        Blob<float>* out2_blob1 = net2->output_blobs()[0];
        blobs.push_back(out2_blob1);
        Blob<float>* out2_blob2 = net2->output_blobs()[1];
        blobs.push_back(out2_blob2);
        dets = get_detections(blobs,im.w,im.h,net1_input1_data_blobs->width(),net1_input1_data_blobs->height(),&nboxes,type);

        uint64_t endDetectTime = current_timestamp();

        printf("object-detection:  processing data operation avergae time is  (%zu)ms\n", endDataTime - beginDataTime);
        printf("object-detection:  processing network yolov3 tiny avergae time is (%zu)ms\n", endDetectTime - endDataTime);
    }

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
        
    printf("done.\n");
    return 0;
}


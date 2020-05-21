
/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2018/06/04	
 */

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <sys/time.h>

#include "detector.h"

using namespace cv;


bool signal_recieved = false;

static int coco_ids[] = { 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 };


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
    std::string model_file;
    std::string weights_file;
    std::string file_path;
    if(4 == argc){
        model_file = argv[1];
        weights_file = argv[2];
        file_path = argv[3];
    }
    else{
        LOG(ERROR) << "Input error: please input ./xx [model_path] [weights_path] [file_path]";
        return -1;
    }	

    //init network
    Detector detector = Detector(model_file,weights_file,0);

    std::vector<String> files;
    file_path = file_path + "/*.jpg";
    LOG(INFO) << "images dir path is " << file_path;
    glob(file_path,files,false);

    char* prefix = "../results";
    char* outfile = "coco_results";
    FILE *fp = 0;

    char buff1[1024];
    snprintf(buff1, 1024, "%s/%s.json", prefix, outfile);
    fp = fopen(buff1, "w");
    fprintf(fp, "[\n");

    for(int i=0;i<files.size();i++){
        LOG(INFO) <<"The " << i << " image path is " << files[i];

        size_t pos = files[i].find_last_of('/');
        std::string name(files[i].substr(pos+1));

        size_t pos1 = name.find_last_of('.');
        std::string id(name.substr(0,pos1));

        int image_id = stoi(id);
        LOG(INFO) << "image id is " << image_id;

        //load image with opencv
        Mat img = imread(files[i]);
        
        //detect
        float thresh = 0.0;
        std::vector<bbox_t> bbox_vec = detector.detect(img,thresh);

        //show detection results
        for (int i=0;i<bbox_vec.size();++i){
            bbox_t b = bbox_vec[i];
  
            float bx = b.x;
            float by = b.y;
            float bw = b.w;
            float bh = b.h;

            char buff2[1024];
            sprintf(buff2, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", 
                image_id, coco_ids[b.obj_id], bx, by, bw, bh, b.prob);
            fprintf(fp, buff2);
            // LOG(INFO) << buff2;
            // int left  = b.x;
            // int right = b.x + b.w;
            // int top   = b.y;
            // int bot   = b.y + b.h;
            // rectangle(img,Point(left,top),Point(right,bot),Scalar(0,0,255),3,8,0);
        }

        //show with opencv
        // namedWindow("show",CV_WINDOW_AUTOSIZE);
        // imshow("show",img);
        // waitKey(1);
    }
    fseek(fp, -2, SEEK_CUR); //x64
    fprintf(fp, "\n]\n");
    if (fp) fclose(fp);

    LOG(INFO) << "done.";
    return 0;
}


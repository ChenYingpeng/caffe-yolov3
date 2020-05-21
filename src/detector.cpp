/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2020/04/26	
 */

 #include "detector.h"

int max_index(float *a, int n)
{
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

 Detector::Detector(std::string prototxt,std::string caffemodel,int gpu_id){
     // set device
    Caffe::set_mode(Caffe::GPU);

    if (gpu_id >= 0){
        Caffe::SetDevice(gpu_id);
        LOG(INFO) << "Using GPU #" << gpu_id;
    }
    else{
        LOG(ERROR) << "Not supported CPU!";
    }

    /* load and init network. */
    m_net.reset(new Net<float>(prototxt, TEST));
    m_net->CopyTrainedLayersFrom(caffemodel);
    LOG(INFO) << "net inputs numbers is " << m_net->num_inputs();
    LOG(INFO) << "net outputs numbers is " << m_net->num_outputs();

    CHECK_EQ(m_net->num_inputs(), 1) << "Network should have exactly one input.";

    m_net_input_data_blobs = m_net->input_blobs()[0];
    LOG(INFO) << "input data layer channels is  " << m_net_input_data_blobs->channels();
    LOG(INFO) << "input data layer width is  " << m_net_input_data_blobs->width();
    LOG(INFO) << "input data layer height is  " << m_net_input_data_blobs->height();

    

 }

Detector::~Detector(){

    //release memory
    // free_image(m_sized);
    // free_image(m_im);

}




std::vector<bbox_t> Detector::detect(std::string image_path,float thresh){
    //load image
    image im = load_image_color((char*)image_path.c_str(),0,0);
    image sized = letterbox_image(im,m_net_input_data_blobs->width(),m_net_input_data_blobs->height());

    //copy data from cpu to gpu
    int size = m_net_input_data_blobs->channels()*m_net_input_data_blobs->width()*m_net_input_data_blobs->height();
    cuda_push_array(m_net_input_data_blobs->mutable_gpu_data(),sized.data,size);

    //clean blobs
    m_blobs.clear();
        
    int nboxes = 0;
    detection *dets = NULL;

    // forward
    m_net->Forward();
    for(int i =0;i<m_net->num_outputs();++i){
        m_blobs.push_back(m_net->output_blobs()[i]);
    }

    dets = get_detections(m_blobs,im.w,im.h,
        m_net_input_data_blobs->width(),m_net_input_data_blobs->height(),m_thresh, m_classes, &nboxes);

    //deal with results
    std::vector<bbox_t> bbox_vec;
    for (int i = 0; i < nboxes; ++i) {
        box b = dets[i].bbox;
        int const obj_id = max_index(dets[i].prob, m_classes);
        float const prob = dets[i].prob[obj_id];

        if (prob > thresh)
        {
            bbox_t bbox;
            bbox.x = std::max((double)0, (b.x - b.w / 2.)*im.w);
            bbox.y = std::max((double)0, (b.y - b.h / 2.)*im.h);
            bbox.w = b.w*im.w;
            bbox.h = b.h*im.h;
            bbox.obj_id = obj_id;
            bbox.prob = prob;

            bbox_vec.push_back(bbox);
        }
    }

    free_detections(dets,nboxes);
    free_image(sized);
    free_image(im);
    return bbox_vec;
}



std::vector<bbox_t> Detector::detect(cv::Mat mat,float thresh){
    //convert mat to image
    if(mat.data == NULL)
        throw std::runtime_error("Mat is empty");
    image im = mat_to_image(mat);
    image sized = letterbox_image(im,m_net_input_data_blobs->width(),m_net_input_data_blobs->height());

    //copy data from cpu to gpu
    int size = m_net_input_data_blobs->channels()*m_net_input_data_blobs->width()*m_net_input_data_blobs->height();
    cuda_push_array(m_net_input_data_blobs->mutable_gpu_data(),sized.data,size);

    //clean blobs
    m_blobs.clear();
        
    int nboxes = 0;
    detection *dets = NULL;

    // forward
    m_net->Forward();
    for(int i =0;i<m_net->num_outputs();++i){
        m_blobs.push_back(m_net->output_blobs()[i]);
    }

    dets = get_detections(m_blobs,im.w,im.h,
        m_net_input_data_blobs->width(),m_net_input_data_blobs->height(),m_thresh, m_classes, &nboxes);

    //deal with results
    std::vector<bbox_t> bbox_vec;
    for (int i = 0; i < nboxes; ++i) {
        box b = dets[i].bbox;
        int const obj_id = max_index(dets[i].prob, m_classes);
        float const prob = dets[i].prob[obj_id];

        if (prob > thresh)
        {
            bbox_t bbox;
            bbox.x = std::max((double)0, (b.x - b.w / 2.)*im.w);
            bbox.y = std::max((double)0, (b.y - b.h / 2.)*im.h);
            bbox.w = b.w*im.w;
            bbox.h = b.h*im.h;
            bbox.obj_id = obj_id;
            bbox.prob = prob;

            bbox_vec.push_back(bbox);
        }
    }

    free_detections(dets,nboxes);
    free_image(sized);
    free_image(im);
    return bbox_vec;
}

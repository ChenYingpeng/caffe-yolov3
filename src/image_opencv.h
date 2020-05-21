/*
 * Company:	Synthesis
 * Author: 	Chen
 * Date:	2020/04/26	
 */

 #ifndef __IMAGE_OPENCV_H_
#define __IMAGE_OPENCV_H_

#include <opencv2/opencv.hpp>
#include "image.h"

image mat_to_image(cv::Mat mat);

cv::Mat image_to_mat(image im);

#endif
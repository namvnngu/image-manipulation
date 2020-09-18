#include "Sharp.h"
#include <iostream>
#include <string>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdint.h>

using namespace cv;
using namespace std;

Sharp::Sharp(){}

void Sharp::sharpen_img(Mat &input, Mat &output, double sharpen_force) {
    Mat image = input; 
    Mat processed_image[3] = {Mat::zeros(image.size(), CV_8UC1), Mat::zeros(image.size(), CV_8UC1), Mat::zeros(image.size(), CV_8UC1)} ;
    uint8_t t = 244;
    cout << t;

    // Create sharp kernel
    Mat kernel = create_kernel(sharpen_force);
    int offset = 1;

    // OpenCV
    // Mat sharpen_image_res;
    // filter2D(image, sharpen_image_res, image.depth(), kernel);

    // Split channels
    Mat split_channels[3];
    split(image, split_channels);

    // Carry out sharp operation
    for(int r = offset; r < image.rows - offset; r++) {
        for(int c = offset; c < image.cols - offset; c++) {
            double rgb[3] = {0, 0, 0};
            for(int x = 0; x < kernel.rows; x++) {
                for(int y = 0; y < kernel.cols; y++) {
                    int rn = r + x - offset;
                    int cn = c + y - offset;
                    rgb[0] += split_channels[0].at<uint8_t>(rn,cn) * kernel.at<double>(x,y);
                    rgb[1] += split_channels[1].at<uint8_t>(rn,cn) * kernel.at<double>(x,y);
                    rgb[2] += split_channels[2].at<uint8_t>(rn,cn) * kernel.at<double>(x,y);
                }
            }
            processed_image[0].at<uint8_t>(r, c) = clip((int)(rgb[0]));
            processed_image[1].at<uint8_t>(r, c) = clip((int)(rgb[1]));
            processed_image[2].at<uint8_t>(r, c) = clip((int)(rgb[2]));
        }
    }

    // Merge processed_image to form blur image
    merge(processed_image, 3, output);
}

Mat Sharp::create_kernel(double sharpen_force) {
    double x = -1 * sharpen_force;
    double y = (4 * sharpen_force) + 1;
    Mat kernel = Mat::zeros(Size(3,3), CV_64F);

    // Row 1
    kernel.at<double>(0,0) = 0.0;
    kernel.at<double>(0,1) = x;
    kernel.at<double>(0,2) = 0.0;
    // Row 2
    kernel.at<double>(1,0) = x;
    kernel.at<double>(1,1) = y;
    kernel.at<double>(1,2) = x;
    // Row 3
    kernel.at<double>(2,0) = 0.0;
    kernel.at<double>(2,1) = x;
    kernel.at<double>(2,2) = 0.0;

    return kernel;
    // For light edge
    // double sum = 4 * abs(x) + y;

    // for(int i = 0; i < 3; i++) 
    //     for(int j = 0; j < 3; j++)
    //         kernel[i][j] /= sum;
}

int Sharp::clip(int value) {
    if (value >= 255) 
        return 255;
    else if (value <= 0) 
        return 0;
    else
        return value;
}
#include "OptimizedBlur.h"
#include <iostream>
#include <string>
#include <math.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdint.h>
#include <omp.h>

using namespace cv;
using namespace std;

OptimizedBlur::OptimizedBlur(){}

void OptimizedBlur::blur_img(Mat &input, Mat &output, double std_dev) {
    Mat image = input;
    int size = 5;
    Mat processed_image[3] = {Mat::zeros(Size(image.cols + size * 2, image.rows + size * 2), CV_8UC1), Mat::zeros(Size(image.cols + size * 2, image.rows + size * 2), CV_8UC1), Mat::zeros(Size(image.cols + size * 2, image.rows + size * 2), CV_8UC1)} ;
    Mat extended_image;
    copyMakeBorder(image, extended_image, size, size, size, size, BORDER_REFLECT);

    // Create blur matrix
    Mat blur_kernel = create_blur_kernel(size, size, std_dev);
    int offset = blur_kernel.cols / 2;

    // Split channels
    Mat split_channels[3];
    split(extended_image, split_channels);

    // Carry out blur operation
    #pragma omp parallel for shared(extended_image) 
        for(int r = offset; r < extended_image.rows - offset; r++) {
            for(int c = offset; c < extended_image.cols - offset; c++) {
                double rgb[3] = {0, 0, 0};
                for(int x = 0; x < blur_kernel.rows; x++) {
                    for(int y = 0; y < blur_kernel.cols; y++) {
                        int rn = r + x - offset;
                        int cn = c + y - offset;
                        rgb[0] += split_channels[0].at<uint8_t>(rn,cn) * blur_kernel.at<double>(x,y);
                        rgb[1] += split_channels[1].at<uint8_t>(rn,cn) * blur_kernel.at<double>(x,y);
                        rgb[2] += split_channels[2].at<uint8_t>(rn,cn) * blur_kernel.at<double>(x,y);
                    }
                }
                processed_image[0].at<uint8_t>(r, c) = (int)(rgb[0]);
                processed_image[1].at<uint8_t>(r, c) = (int)(rgb[1]);
                processed_image[2].at<uint8_t>(r, c) = (int)(rgb[2]);
            }
        }

    // Merge processed_image to form blur image
    merge(processed_image, 3, output);
    
    input = output(Rect(size, size, image.cols, image.rows));
}

Mat OptimizedBlur::create_blur_kernel(int rows, int cols, double std_dev) {
    Mat blur_kernel = Mat::zeros(Size(rows,cols), CV_64F);
    int half_c = blur_kernel.cols / 2;
    int half_r = blur_kernel.rows / 2;
    double sum = 0.0;

    // Generate kernel
    for(int i = 0; i < blur_kernel.rows; i++)
        for(int j = 0; j < blur_kernel.cols; j++) {
            blur_kernel.at<double>(i,j) = calculate_blur(i-half_r, j-half_c, std_dev);
            sum += blur_kernel.at<double>(i,j);
        }
    
    // Normalize
    for(int i = 0; i < blur_kernel.rows; i++)
        for(int j = 0; j < blur_kernel.cols; j++) 
            blur_kernel.at<double>(i,j) /= sum;
    
    return blur_kernel;
}

double OptimizedBlur::calculate_blur(int x, int y, double std_dev) {
    double e = exp(-(x*x + y*y)/(2.0*std_dev*std_dev));
    double coff = 1/(2.0*M_PI*std_dev*std_dev);
    return coff * e;
}



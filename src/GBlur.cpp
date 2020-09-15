#include "GBlur.h"
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

GBlur::GBlur(string img_path) : img_path(img_path) {
    image = imread(img_path, CV_LOAD_IMAGE_COLOR);
    if(!image.data) {
        cout << "No image data\n";
        exit(-1);
    }
    // imshow("Image", image);
}

void GBlur::blur_img(double std_dev) {
    Mat output[3] = {Mat::zeros(image.size(), CV_8UC1), Mat::zeros(image.size(), CV_8UC1), Mat::zeros(image.size(), CV_8UC1)} ;

    // Create blur matrix
    this->std_dev = std_dev;
    Mat blur_kernel = create_blur_kernel(5,5);
    int offset = blur_kernel.cols / 2;

    // Split channels
    Mat split_channels[3];
    split(image, split_channels);

    // Carry out blur operation
    for(int r = offset; r < image.rows - offset; r++) {
        for(int c = offset; c < image.cols - offset; c++) {
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
            output[0].at<uint8_t>(r, c) = (int)(rgb[0]);
            output[1].at<uint8_t>(r, c) = (int)(rgb[1]);
            output[2].at<uint8_t>(r, c) = (int)(rgb[2]);
        }
    }

    // Merge output to form blur image
    Mat blur_image_res;
    merge(output, 3, blur_image_res);

    exported_image = blur_image_res;
    imshow("Blur",blur_image_res);
}

Mat GBlur::create_blur_kernel(int rows, int cols) {
    Mat blur_kernel = Mat::zeros(Size(rows,cols), CV_64F);
    int half_c = blur_kernel.cols / 2;
    int half_r = blur_kernel.rows / 2;
    double sum = 0.0;

    // Generate kernel
    for(int i = 0; i < blur_kernel.rows; i++)
        for(int j = 0; j < blur_kernel.cols; j++) {
            blur_kernel.at<double>(i,j) = calculate_blur(i-half_r, j-half_c);
            sum += blur_kernel.at<double>(i,j);
        }
    
    // Normalize
    for(int i = 0; i < blur_kernel.rows; i++)
        for(int j = 0; j < blur_kernel.cols; j++) 
            blur_kernel.at<double>(i,j) /= sum;
    
    return blur_kernel;
}

double GBlur::calculate_blur(int x, int y) {
    double e = exp(-(x*x + y*y)/(2.0*std_dev*std_dev));
    double coff = 1/(2.0*M_PI*std_dev*std_dev);
    return coff * e;
}

void GBlur::export_img(string file_name) {
    string path = "./output/" + file_name;
    imwrite(path, exported_image);
}


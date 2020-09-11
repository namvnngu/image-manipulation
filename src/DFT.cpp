#include "DFT.h"
#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

DFT::DFT(string image_path) : image_path(image_path) {
    original_image  = imread(image_path , CV_LOAD_IMAGE_GRAYSCALE);
    if(!original_image.data ) {
        cout <<  "Could not open or find the image" << endl ;
        exit(-1);
    }
}

void DFT::prepare_dft() {
    // Convert to float number
    Mat original_float;
    original_image.convertTo(original_float, CV_32FC1, 1.0 / 255.0);
    // Merge real and imaginary component
    Mat original_complex[2] = { original_float, Mat::zeros(original_float.size(), CV_32F)};
    merge(original_complex, 2, merged_dft);
}

void DFT::perform() {
    prepare_dft();
    dft(merged_dft, dft_of_original, DFT_COMPLEX_OUTPUT);
}

Mat DFT::invert_dft() {
    Mat inverted_mat;
    idft(dft_of_original, inverted_mat, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
    return inverted_mat;
}

void DFT::recenter_dft(Mat &mat) {
    int center_x = mat.cols / 2;
    int center_y = mat.rows / 2;

    Mat q1(mat, Rect(0, 0, center_x, center_y));
    Mat q2(mat, Rect(center_x, 0, center_x, center_y));
    Mat q3(mat, Rect(0, center_y, center_x, center_y));
    Mat q4(mat, Rect(center_x, center_y, center_x, center_y));

    Mat temp_mat;
    q1.copyTo(temp_mat);
    q4.copyTo(q1);
    temp_mat.copyTo(q4);

    q2.copyTo(temp_mat);
    q3.copyTo(q2);
    temp_mat.copyTo(q3);
}

void DFT::show_dft() {
    // Split channels
    Mat split_arr[2] = {Mat::zeros(dft_of_original.size(), CV_32F), Mat::zeros(dft_of_original.size(), CV_32F)};
    split(dft_of_original, split_arr);
    
    // Magnitude
    Mat dft_magnitude;
    magnitude(split_arr[0], split_arr[1], dft_magnitude);
    dft_magnitude += Scalar::all(1);
    log(dft_magnitude, dft_magnitude);
    normalize(dft_magnitude, dft_magnitude, 0, 1, NORM_MINMAX);

    // Recenter
    recenter_dft(dft_magnitude);

    // Show
    imshow("DFT", dft_magnitude);
    namedWindow("DFT", WINDOW_AUTOSIZE);
    waitKey();
}


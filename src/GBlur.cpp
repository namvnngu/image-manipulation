#include "GBlur.h"
#include <iostream>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

GBlur::GBlur(string img_path) : img_path(img_path) {
    Mat image = imread(img_path, CV_LOAD_IMAGE_UNCHANGED);
    if(!image.data) {
        cout << "No image data\n";
        exit(-1);
    }
    imshow("Image", image);
}
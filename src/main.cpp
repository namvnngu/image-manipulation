#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include "DFT.h"
#include "GBlur.h"
#include "Sharp.h"

using namespace cv;
using namespace std;
int main(int argc, char** argv) {
  if ( argc != 2 ) {
    cout << "Provide image path\n";
    return -1;
  }
  string image_path = argv[1];

  // Mat image = imread(image_path, CV_LOAD_IMAGE_UNCHANGED);
  // if(!image.data) {
  //   cout << "No image data\n";
  //   return -1;
  // }

  // Gaussian Blur
  GBlur g_blur(image_path);
  g_blur.blur_img(20.0);
  g_blur.export_img("blur.jpg");

  // Sharpen
  Sharp sharp(image_path);
  sharp.sharpen_img(1);
  sharp.export_img("sharp.jpg");

  waitKey();
  return 0;
}
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include "DFT.h"
#include "GBlur.h"

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
  GBlur g(image_path);
  g.blur_img(5);
  g.export_img("export.jpg");

  waitKey();
  return 0;
}
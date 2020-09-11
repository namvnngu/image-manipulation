#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
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

  GBlur g(image_path);
  waitKey();
  return 0;
}
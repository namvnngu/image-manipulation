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

  Mat image = imread(image_path, CV_LOAD_IMAGE_COLOR);
  if(!image.data) {
    cout << "No image data\n";
    return -1;
  }

  // Gaussian Blur
  Mat output;
  GBlur g_blur;
  g_blur.blur_img(image, output, 20.0);
  imwrite("./output/blur.jpg", output);

  // Sharpen
  Sharp sharp;
  sharp.sharpen_img(image, output, 2);
  imwrite("./output/sharp.jpg", output);

  waitKey();
  return 0;
}
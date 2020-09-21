#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "DFT.h"
#include "GBlur.h"
#include "Sharp.h"
#include "time.h"
#include "OptimizedSharp.h"
#include "OptimizedBlur.h"

using namespace cv;
using namespace std;

int divisor = 10;
void visualize_grid(Mat &image);

int main(int argc, char** argv) {
  // Load images
  string image_path = argv[1];
  if(argc <= 1) 
    cout << "Please provide more arguments";

  Mat image = imread(image_path, CV_LOAD_IMAGE_COLOR);
  if(!image.data) {
    cout << "No image data\n";
    return -1;
  }

  // Blur
  OptimizedBlur g_blur;
  OptimizedSharp sharp;
  Mat output;
  // g_blur.blur_img(image, output, 20);
  for(int y = 0; y < image.cols; y += image.cols / divisor) 
    for(int x = 0; x < image.rows; x += image.rows / divisor) {
        Mat block(image, Rect(y, x, (image.cols / divisor), (image.rows / divisor)));
        g_blur.blur_img(block, output, 20);
        block.copyTo(image(Rect(y, x, (image.cols / divisor), (image.rows / divisor))));
    }

  imshow("Image", image);
  waitKey();
  // // Gaussian Blur
  // Mat output;
  // OptimizedBlur g_blur;
  // g_blur.blur_img(image, output, 20.0);
  // imwrite("./output/blur.jpg", output);

  // // Sharpen
  // clock_t start = clock();

  // optimizedsharp sharp;
  // sharp.sharpen_img(image, output, 2);
  // imwrite("./output/sharp.jpg", output);

  // clock_t end = clock();
  // double execution_time = double(end-start) / (CLOCKS_PER_SEC / divisor00);
  // cout << "Time of gaining expected filtered image: " << execution_time << "ms\n";
  // waitKey();
  // return 0;
}
void visualize_grid(Mat &image) {
  Mat mask_image = image.clone();
  for(int y = 0; y < image.cols; y += image.cols / divisor) {
    for(int x = 0; x < image.rows; x += image.rows / divisor) {
      rectangle(mask_image, Point(y,x), Point(y + (image.cols / divisor), x + (image.rows / divisor)), CV_RGB(255, 0, 0), 1);
      imshow("Image", mask_image);
      waitKey(0);
    }
  }
}


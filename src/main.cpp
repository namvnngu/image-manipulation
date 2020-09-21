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

int divisor = 5;
void visualize_grid(Mat &image);
void extend_image(Mat &image);
void shrink_image(Mat &image, int cols, int rows);

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
  
  // Global variables
  int cols = image.cols, rows = image.rows;
  OptimizedBlur g_blur;
  OptimizedSharp sharp;
  Mat output;

  // Extend temporarily the image by reflecting border 
  // if the width and height is not divisible by divisor
  extend_image(image);

  // Split into blocks and apply blur operation on them
  for(int y = 0; y < image.cols; y += image.cols / divisor) 
    for(int x = 0; x < image.rows; x += image.rows / divisor) {
        Mat block(image, Rect(y, x, (image.cols / divisor), (image.rows / divisor)));
        g_blur.blur_img(block, output, 20);
        block.copyTo(image(Rect(y, x, (image.cols / divisor), (image.rows / divisor))));
    }

  // If the image was extended, shrink it back to original
  shrink_image(image, cols, rows);
  imshow("Image", image);
  waitKey();

  return 0;
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
void extend_image(Mat &image) {
  Mat output;
  int rows = image.rows;
  int cols = image.cols;
  int added_rows = rows, added_cols = cols;

  while(added_rows % divisor != 0)
    added_rows++;
  while(added_cols % divisor != 0)
    added_cols++;
  
  added_cols = added_cols - cols;
  added_rows = added_rows - rows;

  copyMakeBorder(image, output, 0, added_rows, 0, added_cols, BORDER_REFLECT);
  image = output;
}
void shrink_image(Mat &image, int cols, int rows) {
  Mat output = image(Rect(0,0,cols,rows));
  image = output;
}


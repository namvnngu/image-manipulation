#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>
#include <vector>

using namespace cv;
using namespace std;

void extend_image(Mat &image, int divisor) {
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

void visualize_grid(Mat &image, int divisor) {
  Mat mask_image = image.clone();
  for(int y = 0; y < image.cols; y += image.cols / divisor) {
    for(int x = 0; x < image.rows; x += image.rows / divisor) {
      rectangle(mask_image, Point(y,x), Point(y + (image.cols / divisor), x + (image.rows / divisor)), CV_RGB(255, 0, 0), 1);
      imshow("Image", mask_image);
      waitKey(0);
    }
  }
}

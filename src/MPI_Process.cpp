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
#include "DFT.h"
#include "GBlur.h"
#include "Sharp.h"
#include "time.h"
#include "OptimizedSharp.h"
#include "OptimizedBlur.h"

using namespace cv;
using namespace std;

void extend_image(Mat &image, int divisor);
void shrink_image(Mat &image, int cols, int rows);
double process_image(Mat &image, int divisor) {
  if(!image.data) {
    cout << "No image data\n";
    exit(-1);
  }
  
  // Global variables
  int cols = image.cols, rows = image.rows;
  OptimizedBlur g_blur;
  OptimizedSharp sharp;
  Mat output;

  // Extend temporarily the image by reflecting border 
  // if the width and height is not divisible by divisor
  extend_image(image, divisor);

  // Start measuring execution time of multiplication operation
  clock_t start = clock();

  // Split into blocks and apply blur operation on them
  #pragma omp single
  for(int y = 0; y < image.cols; y += image.cols / divisor)
    #pragma omp task
    for(int x = 0; x < image.rows; x += image.rows / divisor) {
        Mat block(image, Rect(y, x, (image.cols / divisor), (image.rows / divisor)));
        sharp.sharpen_img(block, output, 2);
        // g_blur.blur_img(block, output, 20);
        block.copyTo(image(Rect(y, x, (image.cols / divisor), (image.rows / divisor))));
    }

  // Stop measuring execution time of multiplication operation
  clock_t end = clock();
  // Print execution time
  double execution_time = double(end-start) / (CLOCKS_PER_SEC / 1000);
  cout << "Time of gaining the filtered image: " << execution_time << "ms\n"; 

  // If the image was extended, shrink it back to original
  shrink_image(image, cols, rows);
  return execution_time;
}

void process_video(VideoCapture &cap, int divisor) {
  int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH); 
  int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT); 
  VideoWriter video("./output/output_video.avi",CV_FOURCC('M','J','P','G'), 24, Size(frame_width, frame_height));
  double sum_processing_time = 0.0;
  unsigned int count_frame = 0;

  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    exit(-1);
  }
  
  while(true) {
    Mat frame;
    cap >> frame;

    // If the frame is empty, break immediately
    if(frame.empty())
      break;

    // Apply filter
    count_frame++;
    sum_processing_time += process_image(frame, divisor);

    // Write frame into video
    video.write(frame);

    // Show frame
    imshow("Frame", frame);

    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;
  }

  // Free all
  // When everything done, release the video capture object
  cap.release();
  // Closes all the frames
  destroyAllWindows();

  double avg = sum_processing_time / count_frame;
  printf("The average processing time on one frame is %f\n", avg);
}
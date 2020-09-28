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

const int divisor = 10;
string INPUT_FILE_NAME = "5.jpg";
string INPUT_PATH = "./img/" + INPUT_FILE_NAME;

void visualize_grid(Mat &image, int divisor);
void extend_image(Mat &image, int divisor);
void shrink_image(Mat &image, int cols, int rows);
double process_image(Mat &image, int divisor);
void process_video(VideoCapture &video, int divisor);
void hybrid_image_process(int &process_id, int &num_process, int &num_node_workers, string INPUT_PATH, int divisor);

int main(int argc, char* argv[]) {
  //////////////////////
  //// OpenMP Only ////
  ////////////////////
  /* 
    Image
  */
  // string image_path = argv[1];
  // if(argc <= 1) 
  //   cout << "Please provide more arguments";

  // Mat image = imread(image_path, CV_LOAD_IMAGE_COLOR);
  // process_image(image, divisor);
  // imshow("Filtered", image);
  // imwrite("./output/output_image.jpg", image);
  // waitKey();

  /* 
    Video
  */
  // string video_path = argv[1];
  // if(argc <= 1) 
  //   cout << "Please provide more arguments" << endl;
  
  // VideoCapture cap(video_path);
  // process_video(cap, divisor);

  
  /////////////////////// 
  //// MPI + OpenMP ////
  /////////////////////
  /* 
    Image
  */
  int num_process, process_id, num_node_workers;
  Mat image = imread(INPUT_PATH, CV_LOAD_IMAGE_COLOR);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_process);

  // Number of node workers
  num_node_workers = num_process - 1;
  if(num_node_workers <= 0) {
    printf("Please increase number of processes\n");
    return 0;
  }

  hybrid_image_process(process_id, num_process, num_node_workers, INPUT_PATH, divisor);

  return 0;
}






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

#define MASTER 0
#define MASTER_TAG 1
#define WORKER_TAG 2
const int divisor = 10;
const int MAX_BLOCKS = 1000000;
string INPUT_FILE_NAME = "5.jpg";
string INPUT_PATH = "./img/" + INPUT_FILE_NAME;
Mat blocks[MAX_BLOCKS];

void visualize_grid(Mat &image);
void extend_image(Mat &image);
void shrink_image(Mat &image, int cols, int rows);
double process_image(Mat &image);
void process_video(VideoCapture &video);

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
  // process_image(image);
  // imshow("Filtered", image);
  // imwrite("./output/output_image.jpg", image);

  /* 
    Video
  */
  // string video_path = argv[1];
  // if(argc <= 1) 
  //   cout << "Please provide more arguments" << endl;
  
  // VideoCapture cap(video_path);
  // process_video(cap);

  
  /////////////////////// 
  //// MPI + OpenMP ////
  /////////////////////

  int num_process, process_id, num_node_workers;
  int rows; // The number of rows sent from MASTER to Node Workers
  int row_number; // The row number which Node Worker starts working from
  int row_processed_by_worker, remainder_row;
  int block_width, block_height;
  double start, end;
  Mat output, block;
  MPI_Status status;
  OptimizedBlur g_blur;
  OptimizedSharp sharp;
  ostringstream s;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_process);

  // Number of node workers
  num_node_workers = num_process - 1;
  if(num_node_workers <= 0) {
    printf("Please increase number of processes\n");
    return 0;
  }

  /* 
   * MASTER Process
   */
  if(process_id == MASTER) {
    printf("\nI am from MASTER\n");
    printf("There are %d processes and %d node workers\n", num_process, num_node_workers);

    // Load image
    Mat image = imread(INPUT_PATH, CV_LOAD_IMAGE_COLOR);
    if(!image.data) {
      cout << "No image data\n";
      return -1;
    }
    // Set up variables
    int image_cols = image.cols, image_rows = image.rows;

    // Extend temporarily the image by reflecting border 
    // if the width and height is not divisible by divisor
    extend_image(image);
    int new_cols = image.cols;
    int new_rows = image.rows;
    
    // Start measuring execution time of multiplication operation
    start = MPI_Wtime();

    row_processed_by_worker = divisor / num_node_workers;
    remainder_row = divisor % num_node_workers;    
    row_number = 0;
    block_width = new_cols;
    block_height = new_rows / divisor;

    for(int dest = 1; dest <= num_node_workers; dest++) {
      rows = dest <= remainder_row ? row_processed_by_worker + 1 : row_processed_by_worker;
      printf("MASTER sent to Node worker %d with the starting row %d and the image heightxwidth %dx%d \n", dest, row_number, block_height * rows, block_width);

      // Send the number of rows
      MPI_Send(&rows, 1, MPI_INT, dest, MASTER_TAG, MPI_COMM_WORLD);
      // Send the row number
      MPI_Send(&row_number, 1, MPI_INT, dest, MASTER_TAG, MPI_COMM_WORLD);
      // Send the block_width
      MPI_Send(&block_width, 1, MPI_INT, dest, MASTER_TAG, MPI_COMM_WORLD);
      // Send the block_height
      MPI_Send(&block_height, 1, MPI_INT, dest, MASTER_TAG, MPI_COMM_WORLD);
      // Send blocks
      block = image(Rect(0, row_number * block_height, block_width, block_height * rows));
      MPI_Send(block.data, rows * block_height * block_width * 3, MPI_BYTE, dest, MASTER_TAG, MPI_COMM_WORLD);

      row_number += rows;
    }
    // Received filtered blocks from node workers
    for(int dest = 1; dest <= num_node_workers; dest++) {
      MPI_Recv(&rows, 1, MPI_INT, dest, WORKER_TAG, MPI_COMM_WORLD, &status);
      MPI_Recv(&row_number, 1, MPI_INT, dest, WORKER_TAG, MPI_COMM_WORLD, &status);
      printf("MASTER received from Node worker %d with the starting row %d and the image heightxwidth %dx%d \n", dest, row_number, block_height * rows, block_width);

      block = Mat(block_height * rows, block_width, CV_8UC3);
      MPI_Recv(block.data, rows * block_height * block_width * 3, MPI_BYTE, dest, WORKER_TAG, MPI_COMM_WORLD, &status);

      // Merge filtered block image into the original image
      ostringstream file_name;
      file_name << "./output/" << dest << ".jpeg";
      Mat m = imread(file_name.str(), CV_LOAD_IMAGE_COLOR);
      block.copyTo(image(Rect(0, row_number * block_height, block_width, block_height * rows)));
    }
    // Stop measuring execution time of multiplication operation
    end = MPI_Wtime();

    vector<Mat> ms;
    Mat new_image = Mat(new_rows, new_cols, CV_8UC3);
    row_number = 0;
    for(int dest = 1; dest <= num_node_workers; dest++) {
      ostringstream file_name;
      file_name << "./output/" << dest << ".jpeg";
      Mat m = imread(file_name.str(), CV_LOAD_IMAGE_COLOR);
      ms.push_back(m);


      rows = dest <= remainder_row ? row_processed_by_worker + 1 : row_processed_by_worker;
      ms[dest-1].copyTo(new_image(Rect(0, row_number * block_height, block_width, block_height * rows)));
      row_number += rows;
    }

    // If the image was extended, shrink it back to original
    shrink_image(new_image, image_cols, image_rows);
    imshow("Filtered Image", new_image);
    waitKey(0);

    // Print execution time
    double execution_time = double(end-start) * 1000;
    printf("\nTime of gaining the filtered image: %f ms\n", execution_time);
  }

  /* 
   * Node Worker
   */
  if(process_id != MASTER) {
    MPI_Recv(&rows, 1, MPI_INT, MASTER, MASTER_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&row_number, 1, MPI_INT, MASTER, MASTER_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&block_width, 1, MPI_INT, MASTER, MASTER_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&block_height, 1, MPI_INT, MASTER, MASTER_TAG, MPI_COMM_WORLD, &status);
    printf("Node worker %d received from MASTER with the starting row %d and the image heightxwidth %dx%d \n", process_id, row_number, block_height * rows, block_width);

    block = Mat(block_height * rows, block_width, CV_8UC3);
    MPI_Recv(block.data, rows * block_height * block_width * 3, MPI_BYTE, MASTER, MASTER_TAG, MPI_COMM_WORLD, &status);

    // Apply filter on received image
    g_blur.blur_img(block, output, 20);
    // sharp.sharpen_img(block, output, 2);

    printf("Node worker %d sent to MASTER with the starting row %d and the image heightxwidth %dx%d \n", process_id, row_number, block_height * rows, block_width);
    MPI_Send(&rows, 1, MPI_INT, MASTER, WORKER_TAG, MPI_COMM_WORLD);
    MPI_Send(&row_number, 1, MPI_INT, MASTER, WORKER_TAG, MPI_COMM_WORLD);
    MPI_Send(block.data, rows * block_height * block_width * 3, MPI_BYTE, MASTER, WORKER_TAG, MPI_COMM_WORLD);

    s << "./output/" << process_id << ".jpeg";
    imwrite(s.str(), block);
  }
  MPI_Finalize();

  return 0;
}

double process_image(Mat &image) {
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
  extend_image(image);

  // Start measuring execution time of multiplication operation
  clock_t start = clock();

  // Split into blocks and apply blur operation on them
  #pragma omp single
  for(int y = 0; y < image.cols; y += image.cols / divisor)
    #pragma omp task
    for(int x = 0; x < image.rows; x += image.rows / divisor) {
        Mat block(image, Rect(y, x, (image.cols / divisor), (image.rows / divisor)));
        sharp.sharpen_img(block, output, 2);
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
void process_video(VideoCapture &cap) {
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
    sum_processing_time += process_image(frame);

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


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

void extend_image(Mat &image, int divisor);
void shrink_image(Mat &image, int cols, int rows);

void hybrid_image_process(int &process_id, int &num_process, int &num_node_workers, string INPUT_PATH, int divisor) {
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
      exit(-1);
    }
    // Set up variables
    int image_cols = image.cols, image_rows = image.rows;

    // Extend temporarily the image by reflecting border 
    // if the width and height is not divisible by divisor
    extend_image(image, divisor);
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
    for(int dest = 1; dest <= num_node_workers; dest++) {
      ostringstream file_name;
      file_name << "./output/" << dest << ".jpeg";
      Mat m = imread(file_name.str(), CV_LOAD_IMAGE_COLOR);
      ms.push_back(m);

    }
    Mat new_image = Mat(new_rows, new_cols, CV_8UC3);
    row_number = 0;
    for(int dest = 1; dest <= num_node_workers; dest++) {
      rows = dest <= remainder_row ? row_processed_by_worker + 1 : row_processed_by_worker;
      ms[dest-1].copyTo(new_image(Rect(0, row_number * block_height, block_width, block_height * rows)));
      row_number += rows;
    }

    // If the image was extended, shrink it back to original
    shrink_image(new_image, image_cols, image_rows);
    imshow("Filtered Image", new_image);
    imwrite("./output/output.jpeg", new_image);
    waitKey(0);

    // print execution time
    double execution_time = double(end-start) * 1000;
    printf("\ntime of gaining the filtered image: %f ms\n", execution_time);
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
    // g_blur.blur_img(block, output, 20);
    sharp.sharpen_img(block, output, 2);

    printf("Node worker %d sent to MASTER with the starting row %d and the image heightxwidth %dx%d \n", process_id, row_number, block_height * rows, block_width);
    MPI_Send(&rows, 1, MPI_INT, MASTER, WORKER_TAG, MPI_COMM_WORLD);
    MPI_Send(&row_number, 1, MPI_INT, MASTER, WORKER_TAG, MPI_COMM_WORLD);
    MPI_Send(block.data, rows * block_height * block_width * 3, MPI_BYTE, MASTER, WORKER_TAG, MPI_COMM_WORLD);

    s << "./output/" << process_id << ".jpeg";
    imwrite(s.str(), block);
  }
  MPI_Finalize();
}

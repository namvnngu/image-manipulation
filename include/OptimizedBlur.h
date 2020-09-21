#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

#ifndef OPTIMIZEDBLUR_H
#define OPTIMIZEDBLUR_H
class OptimizedBlur {
    public:
        OptimizedBlur();
        void blur_img(Mat &input, Mat &output, double std_dev);

    private:

        Mat create_blur_kernel(int rows, int cols, double std_dev);
        double calculate_blur(int x, int y, double std_dev);
};
#endif 

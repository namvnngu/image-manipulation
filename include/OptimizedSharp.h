
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

#ifndef OPTIMIZEDSHARP_H 
#define OPTIMIZEDSHARP_H
class OptimizedSharp {
    public:
        OptimizedSharp();
        void sharpen_img(Mat &input, Mat &ouput, double sharpen_force);

    private:
        Mat create_kernel(double sharpen_force);
        int clip(int value);

};
#endif  

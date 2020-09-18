#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

#ifndef SHARP_H
#define SHARP_H
class Sharp {
    public:
        Sharp();
        void sharpen_img(Mat &input, Mat &ouput, double sharpen_force);

    private:
        Mat create_kernel(double sharpen_force);
        int clip(int value);

};
#endif // SHARP_H 

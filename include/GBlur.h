#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

#ifndef GBLUR_H
#define GBLUR_H
class GBlur {
    public:
        GBlur(string img_path);
        void blur_img(double std_dev);

    private:
        string img_path;
        Mat image;
        double std_dev;

        Mat create_blur_kernel(int rows, int cols);
        double calculate_blur(int x, int y);
};
#endif // GBLUR_H

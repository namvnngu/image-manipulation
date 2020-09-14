#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

#ifndef SHARP_H
#define SHARP_H
class GBlur {
    public:
        GBlur(string img_path);
        void blur_img(double std_dev);
        void export_img(string path);

    private:
        string img_path;
        Mat image, exported_image;
        double std_dev;

        Mat create_blur_kernel(int rows, int cols);
        double calculate_blur(int x, int y);
};
#endif // SHARP_H 

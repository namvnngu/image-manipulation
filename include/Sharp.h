#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

#ifndef SHARP_H
#define SHARP_H
class Sharp {
    public:
        Sharp(string img_path);
        void sharpen_img(double sharpen_force);
        void export_img(string file_name);

    private:
        string img_path;
        Mat image, exported_image;

        Mat create_kernel(double sharpen_force);
        int clip(int value);

};
#endif // SHARP_H 

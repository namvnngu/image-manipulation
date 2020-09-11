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
    private:
        string img_path;
        Mat image;
};
#endif // GBLUR_H

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace std;

#ifndef DFT_H
#define DFT_H
class DFT {
    public:
        DFT(string image_path);
        void perform();
        void show_dft();
        Mat invert_dft();
    private:
        Mat original_image;
        Mat merged_dft;
        Mat dft_of_original;
        string image_path;

        void prepare_dft();
        void recenter_dft(Mat &mat);
};
#endif // DFT_H
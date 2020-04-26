#include "opencv2/opencv.hpp"


#include <iostream>
#include <string>
#include <chrono>

using namespace std;
using namespace cv;

template <typename T>
double resizePerfEval(const Mat& frame, unsigned int n, T resizeFlag) {

    auto start = chrono::steady_clock::now();

    for (auto i = 0; i < n; i++) {
        Mat temp;
        resize(frame, temp, Size(960,540), 1, 1, resizeFlag); 
    }

    return chrono::duration <double, milli>(chrono::steady_clock::now() - start).count();
}

void runTest(const Mat& frame, unsigned int n) {

    cout << "INTER_LINEAR "     << resizePerfEval(frame, n, INTER_LINEAR) << "ms" << endl;
    cout << "INTER_NEAREST "    << resizePerfEval(frame, n, INTER_NEAREST) << "ms" << endl;
}

int main(int argc, char* argv[])
{

    Mat gsframe, frame = Mat::ones(Size(1920, 1080), CV_8UC3);
    randu(frame, Scalar::all(0), Scalar::all(255));
    cvtColor(frame, gsframe, COLOR_BGR2GRAY);
    auto n = 10000;

    cout << "Colour" << endl;
    runTest(frame, n);

    cout << endl << "Grayscale" << endl;
    runTest(gsframe, n);    

    return 0;
}
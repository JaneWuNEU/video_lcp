#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>

#include <unistd.h>

#define OPENCV

#include "../build/darknet/include/yolo_v2_class.hpp"    // imported functions from DLL

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names)
{
    for (auto &i : result_vec) {
        cv::rectangle(mat_img, cv::Point(i.x, i.y), cv::Point(i.x+i.w, i.y+i.h), cv::Scalar(255, 178, 50), 3);
        if (obj_names.size() > i.obj_id) {
            std::string label = cv::format("%.2f", i.prob);
			label = obj_names[i.obj_id] + ":" + label;
			
			int baseLine;
            cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1 , &baseLine);
            int top = std::max((int)i.y, labelSize.height);

            cv::rectangle(mat_img, cv::Point(i.x, i.y - round(1.5*labelSize.height)), cv::Point(i.x + round(1.5*labelSize.width), i.y + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
            putText(mat_img, label, cv::Point2f(i.x, i.y), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
        }
    }
}

void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id] << " - ";
        std::cout << "obj_id = " << i.obj_id << ",  x = " << i.x << ", y = " << i.y
            << ", w = " << i.w << ", h = " << i.h
            << std::setprecision(3) << ", prob = " << i.prob << std::endl;
    }
}

std::vector<std::string> objects_names_from_file(std::string const filename) {
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";
    return file_lines;
}


int main(int argc, char *argv[])
{
	std::string names_file = "darknet/data/coco.names";
    std::string cfg_file = "darknet/cfg/yolov3.cfg";
    std::string weights_file = "darknet/yolov3.weights";
    
	Detector detector(cfg_file, weights_file);
    auto obj_names = objects_names_from_file(names_file);
   	
	cv::Mat image, image2, blob;
	std::vector<uchar> vec;
	image = cv::imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
	
	cv::imencode(".jpg", image, vec);

	image2 = cv::imdecode (vec, 1);
	
	auto start = std::chrono::steady_clock::now();
	std::vector<bbox_t> result_vec = detector.detect(image2);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> spent = end - start;
	std::cout << " Time: " << spent.count() << " sec \n";

	draw_boxes(image2, result_vec, obj_names);
	cv::imshow("Result", image2);
	//show_console_result(result_vec, obj_names);
	cv::waitKey(0);
}
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>

#include <unistd.h>

#define OPENCV

#include "../build/darknet/include/yolo_v2_class.hpp"    // imported functions from DLL

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h"

#include <sys/types.h>
#include <dirent.h>

static std::unique_ptr<Detector> detector;
static std::unique_ptr<Detector> detector2;

Detector* detectors[2];

void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names, double multiplier)
{
    for (auto &i : result_vec) {
		int x = (int)(i.x/multiplier);
		int w = (int)(i.w/multiplier);
		int y = (int)(i.y/multiplier);
		int h = (int)(i.h/multiplier);
        cv::rectangle(mat_img, cv::Point(x, y), cv::Point(x+w, y+h), cv::Scalar(255, 178, 50), 3);
        if (obj_names.size() > i.obj_id) {
            std::string label = cv::format("%.2f", i.prob);
			label = obj_names[i.obj_id] + ":" + label;
			
			int baseLine;
            cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1 , &baseLine);
            int top = std::max((int)y, labelSize.height);

            cv::rectangle(mat_img, cv::Point(x, y - round(1.5*labelSize.height)), cv::Point(x + round(1.5*labelSize.width), y + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
            putText(mat_img, label, cv::Point2f(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
        }
    }
}

void show_console_result(std::vector<bbox_t> const result_vec, std::vector<std::string> const obj_names, int frame_id = -1) {
    if (frame_id >= 0) std::cout << " Frame: " << frame_id << std::endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) std::cout << obj_names[i.obj_id];
        printf(" %.3f | ",i.prob);
    }
	std::cout << std::endl;
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
	auto obj_names = objects_names_from_file(names_file);
	
	if(argc < 4){
		perror("Usage: ./detection darknet/cfg/yolov3_x_x.cfg resize_height resize_width path_to_img_folder\n");
		return 1;
	}

	
    std::string weights_file = "darknet/yolov3.weights";
	std::string cfg_file = argv[1];
	int height = atoi(argv[2]);
	int width = atoi(argv[3]);
	
	//std::vector<Detector> detectors;
	//Detector* detectors[2] = {new Detector(cfg_file1, weights_file1), new Detector(cfg_file1, weights_file1)};
    	
	detectors[0] = new Detector(cfg_file, weights_file);
	
	cv::Mat image, image2;
	std::vector<bbox_t> result_vec;
	
	DIR* dirp = opendir(argv[4]);
    struct dirent * dp;
    
	auto start = std::chrono::steady_clock::now();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> spent;
	int i = 0;	
	
	while ((dp = readdir(dirp)) != NULL) {
	    if( strcmp(dp->d_name, ".") != 0 && strcmp(dp->d_name, "..") != 0 ) {
			std::string name = dp->d_name;
			image = cv::imread(argv[4] + name, 1 );
			if ( !image.data ) {
				printf("No image data \n");
				return -1;
			}
			cv::resize(image, image2, cv::Size(width,height), 1, 1, cv::INTER_NEAREST);
			start = std::chrono::steady_clock::now();
			result_vec = detectors[0]->detect(image2);
			end = std::chrono::steady_clock::now();
			spent = end - start;
			std::cout << i << " | " << spent.count() << " | ";
			show_console_result(result_vec, obj_names);
			i++;
		}
	}
	
    closedir(dirp);
	
	//draw_boxes(image2, result_vec, obj_names, 1);
	//cv::imshow("Result", image2);
	//cv::waitKey(0);
}
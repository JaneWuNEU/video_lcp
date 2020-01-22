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
	auto obj_names = objects_names_from_file(names_file);
	
	
    std::string weights_file = "darknet/yolov3.weights";
	//std::string cfg_file = "darknet/cfg/yolov3_608.cfg";
	//std::string cfg_file1 = "darknet/cfg/yolov3_640_640.cfg";
    //std::string cfg_file2 = "darknet/cfg/yolov3_480_640.cfg";
	//std::string cfg_file3 = "darknet/cfg/yolov3_480_480.cfg";
	
	std::string cfg_file = "darknet/cfg/yolov3_480_480.cfg";
    std::string cfg_file1 = "darknet/cfg/yolov3_320_320.cfg";
	std::string cfg_file2 = "darknet/cfg/yolov3_320_320.cfg";
		
	//std::vector<Detector> detectors;
	//Detector* detectors[2] = {new Detector(cfg_file1, weights_file1), new Detector(cfg_file1, weights_file1)};
    	
	detectors[0] = new Detector(cfg_file, weights_file);
	
	cv::Mat image, image2, image3, image4, image5, image6;
	std::vector<uchar> vec;
	image = cv::imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
	
	cv::imencode(".jpg", image, vec);
	printf("size of normal vec: %zu\n",vec.size());
	image2 = cv::imdecode (vec, 1);
	cv::resize(image2, image2, cv::Size(480,480), 1, 1, cv::INTER_NEAREST);
	
	image4 = image2.clone();
	image5 = image2.clone();
	image6 = image2.clone();
	
/*	auto re_start = std::chrono::steady_clock::now();
	cv::resize(image2, image3, cv::Size(0,0), 0.6, 0.6, cv::INTER_NEAREST);
	auto re_end = std::chrono::steady_clock::now();
	std::chrono::duration<double> re_spent = re_end - re_start;
	std::cout << " Time: " << re_spent.count() << " sec \n";
	
	cv::imencode(".jpg", image3, vec);
	printf("size of reduced vec: %zu\n",vec.size());
*/

	std::vector<bbox_t> result_vec;
	//auto det_image = detector.mat_to_image_resize(image2);
	//std::vector<bbox_t> result_vec = detector.detect_resized(*det_image, image2.size().width, image2.size().height);
	
	auto start = std::chrono::steady_clock::now();
	detectors[0] = new Detector(cfg_file,weights_file);
    auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> spent = end - start;
	std::cout << " Time for loading 480_480 cfg : " << spent.count() << " sec \n";
	
	start = std::chrono::steady_clock::now();
	for(int i =0; i<100; i++){
		result_vec = detectors[0]->detect(image2);
	}
	end = std::chrono::steady_clock::now();
	spent = end - start;
	std::cout << " Time for detection 480_480 cfg : " << spent.count()/100 << " sec \n";

	show_console_result(result_vec, obj_names);

	draw_boxes(image4, result_vec, obj_names, 1);
	cv::imshow("Result", image4);
	cv::waitKey(0);
	
	delete detectors[0];
	
	start = std::chrono::steady_clock::now();
	detectors[0] = new Detector(cfg_file1,weights_file);
    end = std::chrono::steady_clock::now();
	spent = end - start;
	std::cout << " Time for loading 320_320 cfg : " << spent.count() << " sec \n";
    	
	start = std::chrono::steady_clock::now();
	for(int i =0; i<100; i++){
		result_vec = detectors[0]->detect(image2);
	}
	end = std::chrono::steady_clock::now();
	spent = end - start;
	std::cout << " Time for detection 320_320 cfg : " << spent.count()/100 << " sec \n";
	
	show_console_result(result_vec, obj_names);

	draw_boxes(image5, result_vec, obj_names,1);
	cv::imshow("Result", image5);
	cv::waitKey(0);

	delete detectors[0];

	cv::resize(image2, image3, cv::Size(0,0), (2.0/3.0), (2.0/3.0), cv::INTER_NEAREST);
	
	start = std::chrono::steady_clock::now();
	detectors[0] = new Detector(cfg_file2,weights_file);
    end = std::chrono::steady_clock::now();
	spent = end - start;
	std::cout << " Time for loading 320_320 after resize cfg : " << spent.count() << " sec \n";
    	
	start = std::chrono::steady_clock::now();
	for(int i =0; i<100; i++){
		result_vec = detectors[0]->detect(image3);
	}
	end = std::chrono::steady_clock::now();
	spent = end - start;
	std::cout << " Time for detection 320_320 after resizecfg : " << spent.count()/100 << " sec \n";

	show_console_result(result_vec, obj_names);
	
	draw_boxes(image3, result_vec, obj_names, 1);
	cv::imshow("Result", image3);
	cv::waitKey(0);

	
	draw_boxes(image6, result_vec, obj_names, (2.0/3.0));
	cv::imshow("Result", image6);
	cv::waitKey(0);

	delete detectors[0];
/*	
	start = std::chrono::steady_clock::now();
	detectors[0] = new Detector(cfg_file3,weights_file);
    end = std::chrono::steady_clock::now();
	spent = end - start;
	std::cout << " Time for loading 480_480 cfg : " << spent.count() << " sec \n";
    	
	start = std::chrono::steady_clock::now();
	for(int i =0; i<100; i++){
		result_vec = detectors[0]->detect(image2);
	}
	end = std::chrono::steady_clock::now();
	spent = end - start;
	std::cout << " Time for detection 480_480 cfg : " << spent.count()/100 << " sec \n";
    
	show_console_result(result_vec, obj_names);	
*/	

	draw_boxes(image2, result_vec, obj_names, 1);
	cv::imshow("Result", image2);
	cv::waitKey(0);
}
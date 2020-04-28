#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <pthread.h>
#include <chrono>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace std;

VideoCapture capture;

//receive result from server and process result
void *recvrend(void *fd) {
	int err;
	double spent;
	unsigned int used_model;
	
	vector<double> control_buffer;
	int pos = 0;
	int control_window = 50;
	
	int on_time_count = 0;
	int late_count = 0;
	int local_curr_model = 0;
	
	while(true) {
		
		bool on_time = (spent <= 0.40) ? true : false;
		double diff = ((spent - 0.40)*1000);
		diff = on_time ? pow(diff,1.5) : pow(diff,1.1);
		
		if (control_buffer.size() == control_window) { //control window full, start checking
			control_buffer[pos] = diff;
			pos = (pos + 1) % control_window;
		} else {	//control window not full yet, wait till its full
			control_buffer.push_back(diff);
		}

		//int total_on_time = count(control_buffer.begin(), control_buffer.end(), true);
		double sum = std::accumulate(control_buffer.begin(), control_buffer.end(), 0.0);
		if (sum <= 500) { 
			if (local_curr_model < 10) {
				local_curr_model++;
				control_buffer.clear();
				pos = 0;
				//printf("U | sum %f | new model %d\n", sum, local_curr_model);
				
			}
		} else if (sum >= 100) { 
			if(local_curr_model > 0) {
				local_curr_model--;
				control_buffer.clear();
				pos = 0;
				//printf("U | sum %f | new model %d\n", sum, local_curr_model);s
			}
		}
	} 
} 

//update detector model based on timing of frame results 
void *control2(void *) {
	int err;
	double spent;
	unsigned int used_model;
	
	vector<double> control_buffer;
	int pos = 0;
	int control_window = 50;
	
	int on_time_count = 0;
	int late_count = 0;
	int local_curr_model = 0;
	
	while(true) {
		
		bool on_time = (spent <= 0.40) ? true : false;
		double diff = ((spent - 0.40)*1000);
		diff = on_time ? pow(diff,1.5) : pow(diff,1.1);
		
		if (control_buffer.size() == control_window) { //control window full, start checking
			control_buffer[pos] = diff;
			pos = (pos + 1) % control_window;
		} else {	//control window not full yet, wait till its full
			control_buffer.push_back(diff);
		}

		//int total_on_time = count(control_buffer.begin(), control_buffer.end(), true);
		double sum = std::accumulate(control_buffer.begin(), control_buffer.end(), 0.0);
		if (sum <= 500) { 
			if (local_curr_model < 10) {
				local_curr_model++;
				control_buffer.clear();
				pos = 0;
				//printf("U | sum %f | new model %d\n", sum, local_curr_model);
				
			}
		} else if (sum >= 100) { 
			if(local_curr_model > 0) {
				local_curr_model--;
				control_buffer.clear();
				pos = 0;
				//printf("U | sum %f | new model %d\n", sum, local_curr_model);s
			}
		}
	} 
}


//update detector model based on timing of frame results 
void *control1(void *) {
	int err;
	double spent;
	unsigned int used_model;
	
	vector<double> control_buffer;
	int pos = 0;
	int control_window = 50;
	
	int on_time_count = 0;
	int late_count = 0;
	int local_curr_model = 0;
	
	while(true) {
		
		bool on_time = (spent <= 0.40) ? true : false;
		double diff = ((spent - 0.40)*1000);
		diff = on_time ? pow(diff,1.5) : pow(diff,1.1);
		
		if (control_buffer.size() == control_window) { //control window full, start checking
			control_buffer[pos] = diff;
			pos = (pos + 1) % control_window;
		} else {	//control window not full yet, wait till its full
			control_buffer.push_back(diff);
		}

		//int total_on_time = count(control_buffer.begin(), control_buffer.end(), true);
		double sum = std::accumulate(control_buffer.begin(), control_buffer.end(), 0.0);
		if (sum <= 500) { 
			if (local_curr_model < 10) {
				local_curr_model++;
				control_buffer.clear();
				pos = 0;
				//printf("U | sum %f | new model %d\n", sum, local_curr_model);
				
			}
		} else if (sum >= 100) { 
			if(local_curr_model > 0) {
				local_curr_model--;
				control_buffer.clear();
				pos = 0;
				//printf("U | sum %f | new model %d\n", sum, local_curr_model);s
			}
		}
	} 
}

//update detector model based on timing of frame results 
void *control(void *) {
	int err;
	double spent;
	unsigned int used_model;
	
	vector<double> control_buffer;
	int pos = 0;
	int control_window = 50;
	
	int on_time_count = 0;
	int late_count = 0;
	int local_curr_model = 0;
	
	while(true) {
		
		bool on_time = (spent <= 0.40) ? true : false;
		double diff = ((spent - 0.40)*1000);
		diff = on_time ? pow(diff,1.5) : pow(diff,1.1);
		
		if (control_buffer.size() == control_window) { //control window full, start checking
			control_buffer[pos] = diff;
			pos = (pos + 1) % control_window;
		} else {	//control window not full yet, wait till its full
			control_buffer.push_back(diff);
		}

		//int total_on_time = count(control_buffer.begin(), control_buffer.end(), true);
		double sum = std::accumulate(control_buffer.begin(), control_buffer.end(), 0.0);
		if (sum <= 500) { 
			if (local_curr_model < 10) {
				local_curr_model++;
				control_buffer.clear();
				pos = 0;
				//printf("U | sum %f | new model %d\n", sum, local_curr_model);
				
			}
		} else if (sum >= 100) { 
			if(local_curr_model > 0) {
				local_curr_model--;
				control_buffer.clear();
				pos = 0;
				//printf("U | sum %f | new model %d\n", sum, local_curr_model);s
			}
		}
	} 
}

void *input(void *) {
	pid_t pid = fork();
	if(pid < 0){
		perror("fork failed");
		exit(1);
	} else if (pid == 0) {
		printf("child spawns\n");
		if(0) {
			execl("/bin/bash", "bash", "../simulation/shape.sh");
		}
		printf("child done\n");
		exit(0);
	} else {
		int err;
		vector<uchar> vec;
		unsigned int frame_counter = 0;
		size_t buffer_size;
		Mat frame;
		
		while(true) {
			capture.read(frame);
			if (frame.empty()) {
				perror("ERROR no frame\n");
				break;
			}
			auto start = std::chrono::system_clock::now();
			frame_counter++;
						
			auto e1 = getTickCount();

			cvtColor(frame, frame, COLOR_BGR2GRAY);
			//imshow("Result",frame);
			//waitKey(1);
			resize(frame, frame, Size(400,400), 1, 1, INTER_NEAREST);
			
			auto e2 = getTickCount();
			auto time1 = (e2 - e1)/ getTickFrequency();
			
			imencode(".jpg", frame, vec);
			size_t n = vec.size();
			
			auto e3 = getTickCount();
			auto time2 = (e3 - e2)/ getTickFrequency();
			
			cout << time1 << " | " << time2 << " | " << vec.size() << "\n";
		}
	}
}

int main(int argc, char *argv[]) {
	cout << getBuildInformation();
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);
	capture.open(argv[1], CAP_GSTREAMER);
	if (!capture.isOpened()) {
		perror("ERROR opening video\n");
		return 1;
	}		
		
	int capture_frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int capture_frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
	printf("input frame size : height: %d, width: %d\n", capture_frame_height, capture_frame_width);
	
	pthread_t thread1, thread2, thread3, thread4, thread5;
	pthread_create(&thread1, NULL, input, NULL);
	pthread_create(&thread2, NULL, recvrend, NULL);
	pthread_create(&thread3, NULL, control, NULL);
	pthread_create(&thread4, NULL, control1, NULL);
	pthread_create(&thread5, NULL, control2, NULL);
	
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);
	pthread_join(thread4, NULL);
	pthread_join(thread5, NULL);
	
	return 0;
}


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

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h"

using namespace cv;
using namespace std;

VideoCapture capture;
frame_obj global_frame_obj;
pthread_mutex_t frameMutex;
pthread_mutex_t modelMutex;
pthread_cond_t frameCond;
vector<string> obj_names;

unsigned int curr_model;
int capture_frame_height;
int capture_frame_width;

string shaping_input;
bool shaping = true;

int controlPipe[2]; //pipe for communicating if frame results are received on time
bool detector_update;


// make a connection to the server and open two sockets one for sending data, one for receiving data
void connect_to_server(int &sockfd1, int &sockfd2, char *argv[]) {
	int err;
	struct sockaddr_in serv_addr;
	struct hostent *server;
	struct in_addr *addr;
	socklen_t addrlen = sizeof(struct sockaddr_in);

	sockfd1 = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd1 < 0){
		perror("failed to open socket.\n");
		exit(1);
	}

	sockfd2 = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd2 < 0){
		perror("failed to open socket.\n");
		exit(1);
	}
	
	server = gethostbyname(argv[1]);
	if (server==NULL) {
		perror("Address not found for\n");
		close(sockfd1);
		close(sockfd2);
		exit(1);
	} else {
		addr = (struct in_addr*) server->h_addr_list[0];
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(atoi(argv[2]));
	serv_addr.sin_addr.s_addr = inet_addr(inet_ntoa(*addr));
	
	err = connect(sockfd1,(struct sockaddr *)&serv_addr,addrlen);
	if(err < 0){
		perror("Connecting to server failed\n");
		close(sockfd1);
		close(sockfd2);
		exit(1);
	}
	err = connect(sockfd2,(struct sockaddr *)&serv_addr,addrlen);
	if(err < 0){
		perror("Connecting to server failed\n");
		close(sockfd1);
		close(sockfd2);
		exit(1);
	}
}

//function to read all object names from a file so the object id of an object can be matched with a name
vector<string> objects_names_from_file(string const filename) {
    ifstream file(filename);
    vector<string> file_lines;
    if (!file.is_open()) return file_lines;
    for(string line; getline(file, line);) file_lines.push_back(line);
    cout << "object names loaded \n";
    return file_lines;
}

//render the bounding boxes of objects stored in the result vec, which is returned by the server, in the current frame 
void drawBoxes(frame_obj local_frame_obj, vector<result_obj> result_vec, unsigned int curr_frame_id) {
	// for each located object 
	for (auto &i : result_vec) { 
		int x = (int)(i.x/(n_width[local_frame_obj.correct_model]*1.0/capture_frame_width*1.0));
		int w = (int)(i.w/(n_width[local_frame_obj.correct_model]*1.0/capture_frame_width*1.0));
		int y = (int)(i.y/(n_height[local_frame_obj.correct_model]*1.0/capture_frame_height*1.0));
		int h = (int)(i.h/(n_height[local_frame_obj.correct_model]*1.0/capture_frame_height*1.0));
		
		rectangle(local_frame_obj.frame, Point(x, y), Point(x+w, y+h), Scalar(255, 178, 50), 3);
        if (obj_names.size() > i.obj_id) {
            string label = format("%.2f", i.prob);
			label = obj_names[i.obj_id] + ":" + label;
			
			int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1 , &baseLine);
            int top = max((int)y, labelSize.height);

            rectangle(local_frame_obj.frame, Point(x, top - round(1.5*labelSize.height)), Point(x + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
            putText(local_frame_obj.frame, label, Point(x, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
        }
    } 
	// add latency of the frame on which detection is performed and the difference between that frame and the current frame  
	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> spent = end - local_frame_obj.start;
	//printf("result frame %d is now %f sec old\n",local_frame_obj.frame_id, spent.count());
	string label = format("Curr: %d | Inf: %d | Time: %f", curr_frame_id, local_frame_obj.frame_id, spent.count());
	int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1 , &baseLine);
	int top = max(20, labelSize.height);
    rectangle(local_frame_obj.frame, Point(0,0), Point(round(1.5*labelSize.width), top+baseLine), Scalar(255, 255, 255), FILLED);
    putText(local_frame_obj.frame, label, Point(0, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

//print console output for a frame instead of rendering and showing an image 
void consoleOutput(frame_obj local_frame_obj, vector<result_obj> result_vec, unsigned int curr_frame_id) {
	//print latency and frame difference between current frame and frame on which detection is performed
    auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> spent = end - local_frame_obj.start;
	printf("------------------------------------------------------------\n");
	printf("Received inference result from frame %d\n",local_frame_obj.frame_id);
	printf("Frame %d was captured %f seconds ago\n",local_frame_obj.frame_id,spent.count());
	printf("Currently captured frame %d is %d frames newer\n",curr_frame_id, curr_frame_id - local_frame_obj.frame_id);
	printf("A total of %zu objects have been recognized\n", result_vec.size());
	
	//print names and confidence of located objects
	for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) {
            string label = format("%.2f", i.prob);
			label = obj_names[i.obj_id] + ":" + label;
			cout << label << "\n";
		}
	}
	printf("\n");
}

//receive result from server and process result
void *recvrend(void *fd) {
	int sockfd = *(int*)fd;
	int err;
	bool local_detector_update;
	
	//wait until first frame is captured so this can be copied for rendering
	pthread_mutex_lock(&frameMutex);
	while(global_frame_obj.frame.empty()){
		pthread_cond_wait(&frameCond, &frameMutex);
	}
	pthread_mutex_unlock(&frameMutex);
	
	while(waitKey(1) < 0) {
		frame_obj local_frame_obj;
		vector<result_obj> result_vec;
		
		//receive frame id on which the server performed object detection
		err = read(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR reading socket");
			close(sockfd);
			exit(1);
		} 
		//printf("read new frame id : %d\n",local_frame_obj.frame_id);
		
		//receive capture time of frame on which server performed detection
		err = read(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR reading socket");
			close(sockfd);
			exit(1);
		} 
		//printf("read capture time of frame id %d\n",local_frame_obj.frame_id);

		//receive correct model value
		err = read(sockfd, &local_frame_obj.correct_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR reading socket");
			close(sockfd);
			exit(1);
		} 
		//printf("read correct model of frame id %d : %d\n",local_frame_obj.frame_id, local_frame_obj.correct_model);

		
		//receive used model value
		err = read(sockfd, &local_frame_obj.used_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR reading socket");
			close(sockfd);
			exit(1);
		} 
		//printf("read used model of frame id %d : %d\n",local_frame_obj.frame_id, local_frame_obj.used_model);
				
		//receive amount of located objects to know how many result_obj should be received
		size_t n;
		err = read(sockfd,&n,sizeof(size_t));
		if (err < 0){ 
			perror("ERROR reading from socket");
			close(sockfd);
			exit(1);
		}
		//printf(" %zu objects found\n",n);
		
		//for each located object, receive one result_obj and store this in the result vector
		for (size_t i = 0; i < n; ++i) {
			result_obj obj;
			err = read(sockfd, &obj.x, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = read(sockfd, &obj.y, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = read(sockfd, &obj.w, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = read(sockfd, &obj.h, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = read(sockfd, &obj.prob, sizeof(float)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = read(sockfd, &obj.obj_id, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			
		/*	err = read(sockfd,&obj,sizeof(result_obj));
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				exit(1);
		  	} */
 			result_vec.push_back(obj); 
		}
		
		//printf("read all objects done \n");
		//copy the frame from the global frame object so the last captured frame can be used for rendering
		pthread_mutex_lock(&frameMutex);
		local_frame_obj.frame = global_frame_obj.frame.clone();
		unsigned int curr_frame_id = global_frame_obj.frame_id;
		pthread_mutex_unlock(&frameMutex);
		
		//check how old frame is to see if it is past the deadline or not and write to pipe
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> spent = end - local_frame_obj.start;
		double time_spent = spent.count();
		/*bool onTime = (spent.count() <= FRAME_DEADLINE) ? true : false;
		
		if (onTime) {
			//printf("frame is %f old while deadline is %f, so it is on time\n",spent.count(), FRAME_DEADLINE);
		} else {
			//printf("frame is %f old while deadline is %f, so it is not on time\n",spent.count(), FRAME_DEADLINE);
		} */
		
		printf("R | id %d | correct model %d | used model %d | objects %zu | time %f\n", local_frame_obj.frame_id, local_frame_obj.correct_model, local_frame_obj.used_model, n, spent.count());
		
		err = write(controlPipe[1], &local_frame_obj.used_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR reading from pipe");
			close(sockfd);
			exit(1);
		}
		err = write(controlPipe[1], &time_spent, sizeof(double));
		if (err < 0){
			perror("ERROR reading from pipe");
			close(sockfd);
			exit(1);
		}
		
		/*if(local_frame_obj.used_model == local_frame_obj.correct_model){ //correct model is being used, so server is not updating model
			//printf("detector is not updating writing to pipe\n");
			err = write(controlPipe[1], &onTime, sizeof(bool));
			if (err < 0){
				perror("ERROR reading from pipe");
				close(sockfd);
				exit(1);
			} 
		} else {
			//printf("detector is updating, do not write to pipe\n");
		} */
			

/*		pthread_mutex_lock(&updateMutex);
		local_detector_update = detector_update; 
		pthread_mutex_unlock(&updateMutex);

		if(!local_detector_update){
			printf("detector is not updating writing to pipe\n");
			err = write(controlPipe[1], &onTime, sizeof(bool));
			if (err < 0){
				perror("ERROR reading from pipe");
				close(sockfd);
				exit(1);
			}
		} else {
			printf("detector is updating, do not write to pipe\n");
		} */

		//enable next line to use console output
		//consoleOutput(local_frame_obj, result_vec, curr_frame_id);
		
		//enable next two lines to use image output and show the rendered frame with bounding boxes
		drawBoxes(local_frame_obj, result_vec, curr_frame_id);
		imshow("Result", local_frame_obj.frame);
	}
} 

//capture and send a frame to the server for object detection
void *capsend(void *fd) {
	int sockfd = *(int*)fd;
	int err;
	vector<uchar> vec;
	unsigned int frame_counter = 0;
	frame_obj local_frame_obj;
	size_t buffer_size;
	
	//system("../simulation/shape.sh ../simulation/sample.txt &");
	
	pid_t pid = fork();
	if(pid < 0){
		perror("fork failed");
		close(sockfd);
		exit(1);
	} else if (pid == 0) {
		printf("child spawns\n");
		execl("/bin/bash", "bash", "../simulation/shape.sh", "../simulation/sample.txt");
		//sleep(1);
		printf("child execl done\n");
	} else {
		printf("parent continues\n");
		while(true) {
			//capture frame into local frame object so capturing is not done within mutex
			capture.read(local_frame_obj.frame);
			if (local_frame_obj.frame.empty()) {
				perror("ERROR no frame\n");
				break;
			}
			local_frame_obj.start = std::chrono::system_clock::now();
			local_frame_obj.frame_id = frame_counter;
			frame_counter++;
			
			//copy local frame into the global frame variable so it can be used for rendering of an image
			pthread_mutex_lock(&frameMutex);
			global_frame_obj = local_frame_obj;
			if (frame_counter == 1) {
				pthread_cond_signal(&frameCond);
			}
			pthread_mutex_unlock(&frameMutex);
			
			//send frame id 
			//printf("S: %d\n",  local_frame_obj.frame_id);
			err = write(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			} 
			
			//send capture time of frame
			//printf("S: %d capture time\n",  local_frame_obj.frame_id);
			err = write(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}

			//read and send correct model value
			pthread_mutex_lock(&modelMutex);
			local_frame_obj.correct_model = curr_model;
			pthread_mutex_unlock(&modelMutex);
			
			//printf("S: %d correct model %d\n",  local_frame_obj.frame_id, local_frame_obj.correct_model);
			err = write(sockfd, &local_frame_obj.correct_model, sizeof(unsigned int));
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			} 

			//resize and encode frame, send the size of the encoded frame so the server knows how much to read, and then send the data vector 
			resize(local_frame_obj.frame, local_frame_obj.frame, cv::Size(n_width[local_frame_obj.correct_model],n_height[local_frame_obj.correct_model]), 1, 1, cv::INTER_NEAREST);
			imencode(".jpg", local_frame_obj.frame, vec);
			size_t n = vec.size();
			//printf("S: %d vec size : %zu\n",  local_frame_obj.frame_id, n);

			err = write(sockfd, &n, sizeof(size_t));
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			} 
			
			err = write(sockfd, vec.data(), vec.size());
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			} 
			//printf("S: image %d written\n", local_frame_obj.frame_id);
			printf("S | id %d | correct model %d | vec size %zu\n", local_frame_obj.frame_id, local_frame_obj.correct_model, n);
			//imshow("Live", frame);
			err = read(sockfd, &buffer_size, sizeof(size_t));
			if (err < 0){
				perror("ERROR reading ack from socket");
				close(sockfd);
				exit(1);
			} 
			printf("S | ack for id %d\n", local_frame_obj.frame_id);
		}
	}
}

//update detector model based on timing of frame results 
void *control(void *) {
	int err;
	double spent;
	unsigned int used_model;
	unsigned int local_curr_model = STARTING_MODEL;
	
	vector<bool> control_buffer;
	int pos = 0;
	int control_window = 50;
	
	int update_after = 50;
	int on_time_count = 0;
	int late_count = 0;
	
	while(true) {
		err = read(controlPipe[0], &used_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR reading from pipe");
			exit(1);
		}
		err = read(controlPipe[0], &spent, sizeof(double));
		if (err < 0){
			perror("ERROR reading from pipe");
			exit(1);
		}
		
		if (used_model != local_curr_model) {
			continue;
		}
		
		bool on_time = (spent <= FRAME_DEADLINE) ? true : false;
		
		if (control_buffer.size() == control_window) { //control window full
			control_buffer[pos] = on_time;
			pos = (pos + 1) % control_window;
				
			int total_on_time = count(control_buffer.begin(), control_buffer.end(), true);
			if (total_on_time >= 45) { //90% of frames on time
				if (local_curr_model < MAX_MODEL) {
					local_curr_model++;
					control_buffer.clear();
					pos = 0;
					printf("U | on time %d | new model %d\n", total_on_time, local_curr_model);
					
					pthread_mutex_lock(&modelMutex);
					curr_model = local_curr_model; 
					pthread_mutex_unlock(&modelMutex);
				}
			} else if (total_on_time <= 20) { //40% of frames on time
				if(local_curr_model > MIN_MODEL) {
					local_curr_model--;
					control_buffer.clear();
					pos = 0;
					printf("U | on time %d | new model %d\n", total_on_time, local_curr_model);
					
					pthread_mutex_lock(&modelMutex);
					curr_model = local_curr_model; 
					pthread_mutex_unlock(&modelMutex);
				}
			}
		} else {
			control_buffer.push_back(on_time);
		}

/*		
		if (on_time) {
			late_count = 0;
			on_time_count++;
			//printf("frame on time, now %d on time\n", on_time_count);
			if (local_curr_model < MAX_MODEL) {
				if (on_time_count >= update_after) { 
					local_curr_model++;
					on_time_count = 0;
					printf("U | current model %d\n", local_curr_model);
					
					pthread_mutex_lock(&modelMutex);
					curr_model = local_curr_model; 
					pthread_mutex_unlock(&modelMutex);
					//printf("update completed, now using model %d\n", local_curr_model);
				}
			}
		} else { //frame too late 
			on_time_count = 0;
			late_count++;
			//printf("frame not time, now %d on late\n", late_count);
			if (local_curr_model > MIN_MODEL) { 
				if (late_count >= update_after) { 
					local_curr_model--;
					late_count = 0;
					printf("U | current model %d\n", local_curr_model);

					pthread_mutex_lock(&modelMutex);
					curr_model = local_curr_model; 
					pthread_mutex_unlock(&modelMutex);
					//printf("update completed, now using model %d\n", local_curr_model);
				}
			}
		} */
	}
}

int main(int argc, char *argv[]) {
	if(argc < 3){
		perror("Usage: ./client [server_hostname] [server_port_number] [network_shaping_file (0 to run without shaping)] [(optional)path_to_video].\n");
		return 1;
	}
	
	int sockfd1, sockfd2, err;
	
	connect_to_server(sockfd1, sockfd2, argv);
	
	if(argv[3] == 0){
		printf("no shaping\n");
		shaping = false;
	} else {
		shaping_input = argv[3];
		printf("shaping file : %s\n", shaping_input.c_str());
	}
	
	if(argc == 5){	//use video file input, gstreamer to enforce realtime reading of frames
		capture.open(argv[4],CAP_GSTREAMER);
		if (!capture.isOpened()) {
			perror("ERROR opening video\n");
			return 1;
		}		
	} else { //use webcam as input
		capture.open(0);
		if (!capture.isOpened()) {
			perror("ERROR opening camera\n");
			return 1;
		}
	}
	err = pipe(controlPipe);
	if (err < 0){
		perror("ERROR creating pipe");
		close(sockfd1);
		close(sockfd2);
		exit(1);
	}
	
	capture_frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
	capture_frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
	printf("height: %d, width: %d\n", capture_frame_height, capture_frame_width);
	
	curr_model = STARTING_MODEL;
	string names_file = "darknet/data/coco.names";
	obj_names = objects_names_from_file(names_file);
	
	pthread_mutex_init(&frameMutex, NULL);
	pthread_mutex_init(&modelMutex, NULL);
	pthread_cond_init(&frameCond, NULL);
	
	pthread_t thread1, thread2, thread3;
	pthread_create(&thread1, NULL, capsend, (void*) &sockfd1);
	pthread_create(&thread2, NULL, recvrend, (void*) &sockfd2);
	pthread_create(&thread3, NULL, control, NULL);
	
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);

	close(sockfd1);
	close(sockfd2);
	
	return 0;
}
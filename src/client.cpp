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

#define BUFF_SIZE 2048
#define MAX_FRAME_BUFFER_SIZE 30

using namespace cv;
using namespace std;

// object that is returned by the server in which information on a detected object is stored
struct result_obj {					
    unsigned int x, y, w, h;      
    float prob;                    
    unsigned int obj_id; 
};

// frame object that stores all the information about a frame
struct frame_obj {
	unsigned int frame_id;
	std::chrono::system_clock::time_point start;
	double multiplier;
	Mat frame;
};

VideoCapture capture;
frame_obj global_frame_obj;
double global_multiplier;
pthread_mutex_t frameMutex;
pthread_mutex_t multiplierMutex;
pthread_cond_t frameCond;
vector<string> obj_names;

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

// make a connection to the monitor
void connect_to_monitor(int &sockfd, int port) {
	int err;
	struct sockaddr_in serv_addr;
	struct hostent *server;
	struct in_addr *addr;
	socklen_t addrlen = sizeof(struct sockaddr_in);

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd < 0){
		perror("failed to open socket.\n");
		exit(1);
	}
	
	server = gethostbyname("localhost");
	if (server==NULL) {
		perror("Address not found for\n");
		close(sockfd);
		exit(1);
	} else {
		addr = (struct in_addr*) server->h_addr_list[0];
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(port);
	serv_addr.sin_addr.s_addr = inet_addr(inet_ntoa(*addr));
	
	err = connect(sockfd,(struct sockaddr *)&serv_addr,addrlen);
	if(err < 0){
		perror("Connecting to server failed\n");
		close(sockfd);
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
		int x = (int)(i.x/local_frame_obj.multiplier);
		int w = (int)(i.w/local_frame_obj.multiplier);
		int y = (int)(i.y/local_frame_obj.multiplier);
		int h = (int)(i.h/local_frame_obj.multiplier);
		
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
	printf("result frame %d is now %f sec old\n",local_frame_obj.frame_id, spent.count());
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
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//receive capture time of frame on which server performed detection
		err = read(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//receive multiplier value
		err = read(sockfd, &local_frame_obj.multiplier, sizeof(double));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
				
		//receive amount of located objects to know how many result_obj should be received
		size_t n;
		err = read(sockfd,&n,sizeof(size_t));
		if (err < 0){ 
			perror("ERROR reading from socket");
			close(sockfd);
			exit(1);
		}
		//printf("2: %zu objects found\n",n);
		
		//for each located object, receive one result_obj and store this in the result vector
		for (size_t i = 0; i < n; ++i) {
			result_obj obj;
			err = read(sockfd,&obj,sizeof(result_obj));
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				exit(1);
			}
			result_vec.push_back(obj);
		}
		
		//copy the frame from the global frame object so the last captured frame can be used for rendering
		pthread_mutex_lock(&frameMutex);
		local_frame_obj.frame = global_frame_obj.frame.clone();
		unsigned int curr_frame_id = global_frame_obj.frame_id;
		pthread_mutex_unlock(&frameMutex);
				
		//enable next line to use console outpu
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
	int frame_counter = 0;
	frame_obj local_frame_obj;
	
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
		pthread_cond_signal(&frameCond);
		pthread_mutex_unlock(&frameMutex);
		
		pthread_mutex_lock(&multiplierMutex);
		local_frame_obj.multiplier = global_multiplier;
		pthread_mutex_unlock(&multiplierMutex);

		if (local_frame_obj.multiplier!=1){
			resize(local_frame_obj.frame, local_frame_obj.frame, cv::Size(0,0), local_frame_obj.multiplier, local_frame_obj.multiplier, cv::INTER_NEAREST);
		}
 		
		//send frame id 
		err = write(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//send capture time of frame
		err = write(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}

		//send multiplier value
		err = write(sockfd, &local_frame_obj.multiplier, sizeof(double));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 

				
		//encode frame, send the size of the encoded frame so the server knows how much to read, and then send the data vector 
		imencode(".jpg", local_frame_obj.frame, vec);
		size_t n = vec.size();
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
		//imshow("Live", frame);
	}
}


//receive bandwidth from monitor
void *monitor(void *fd) {
	int sockfd = *(int*)fd;
	int err;
	unsigned int speed;
	
	while(true) {
		err = read(sockfd, &speed, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 

		printf("Received speed from monitor, updated speed: %d \n", speed);
		
		double local_multiplier;
		
		if (speed >= 30) {
			local_multiplier = 1;
		} else if (speed >= 20) {
			local_multiplier = 0.75;
		} else {
			local_multiplier = 0.5;
		}
		
		pthread_mutex_lock(&multiplierMutex);
		global_multiplier = local_multiplier;
		pthread_mutex_unlock(&multiplierMutex);
 	}
} 


int main(int argc, char *argv[]) {
	if(argc < 3){
		perror("Usage: ./client [server_hostname] [server_port_number] [monitor_port_number] [(optional)path to video].\n");
		return 1;
	}
	
	int sockfd1, sockfd2, sockfd3, err;
	
	connect_to_server(sockfd1, sockfd2, argv);
	
	if (atoi(argv[3]) != 0) {
		connect_to_monitor(sockfd3, atoi(argv[3]));
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
	
	string names_file = "darknet/data/coco.names";
	obj_names = objects_names_from_file(names_file);
	global_multiplier = 1;
	
	pthread_mutex_init(&frameMutex, NULL);
	pthread_mutex_init(&multiplierMutex, NULL);
	pthread_cond_init(&frameCond, NULL);
	
	pthread_t thread1, thread2, thread3;
	pthread_create(&thread1, NULL, capsend, (void*) &sockfd1);
	pthread_create(&thread2, NULL, recvrend, (void*) &sockfd2);
	if (atoi(argv[3]) != 0) {
		pthread_create(&thread3, NULL, monitor, (void*) &sockfd3);
	}
	
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);
	
	close(sockfd1);
	close(sockfd2);
	return 0;
}
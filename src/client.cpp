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

struct result_obj {
    unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;           // class of object - from range [0, classes-1]
};

struct frame_obj {
	unsigned int frame_id;
	std::chrono::system_clock::time_point start;
	Mat frame;
};

VideoCapture capture;
frame_obj global_frame_obj;
pthread_mutex_t frameMutex;
pthread_cond_t frameCond;
vector<string> obj_names;
	
void connect_to_server(int &sockfd1, int &sockfd2, char *argv[]){
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

vector<string> objects_names_from_file(string const filename) {
    ifstream file(filename);
    vector<string> file_lines;
    if (!file.is_open()) return file_lines;
    for(string line; getline(file, line);) file_lines.push_back(line);
    cout << "object names loaded \n";
    return file_lines;
}

void drawBoxes(frame_obj local_frame_obj, vector<result_obj> result_vec, unsigned int curr_frame_id)
{
    for (auto &i : result_vec) {
        rectangle(local_frame_obj.frame, Point(i.x, i.y), Point(i.x+i.w, i.y+i.h), Scalar(255, 178, 50), 3);
        if (obj_names.size() > i.obj_id) {
            string label = format("%.2f", i.prob);
			label = obj_names[i.obj_id] + ":" + label;
			
			int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1 , &baseLine);
            int top = max((int)i.y, labelSize.height);

            rectangle(local_frame_obj.frame, Point(i.x, top - round(1.5*labelSize.height)), Point(i.x + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
            putText(local_frame_obj.frame, label, Point(i.x, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
        }
    }
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

void consoleOutput(frame_obj local_frame_obj, vector<result_obj> result_vec, unsigned int curr_frame_id)
{
    auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> spent = end - local_frame_obj.start;
	printf("------------------------------------------------------------\n");
	printf("Received inference result from frame %d\n",local_frame_obj.frame_id);
	printf("Frame %d was captured %f seconds ago\n",local_frame_obj.frame_id,spent.count());
	printf("Currently captured frame %d is %d frames newer\n",curr_frame_id, curr_frame_id - local_frame_obj.frame_id);
	printf("A total of %zu objects have been recognized\n", result_vec.size());
	
	for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) {
            string label = format("%.2f", i.prob);
			label = obj_names[i.obj_id] + ":" + label;
			cout << label << "\n";
		}
	}
	printf("\n");
}

void *recvrend(void *fd){
	int sockfd = *(int*)fd;
	int err;
	
	pthread_mutex_lock(&frameMutex);
	while(global_frame_obj.frame.empty()){
		//printf("2: waiting for captured frame\n");
		pthread_cond_wait(&frameCond, &frameMutex);
	}
	pthread_mutex_unlock(&frameMutex);
	
	while(waitKey(1) < 0){
		frame_obj local_frame_obj;
		vector<result_obj> result_vec;
		
		err = read(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		err = read(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		size_t n;
		err = read(sockfd,&n,sizeof(size_t));
		if (err < 0){ 
			perror("ERROR reading from socket");
			close(sockfd);
			exit(1);
		}
		//printf("2: %zu objects found\n",n);
		
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
		
		//printf("2: waiting for frame mutex\n");
		pthread_mutex_lock(&frameMutex);
		local_frame_obj.frame = global_frame_obj.frame.clone();
		unsigned int curr_frame_id = global_frame_obj.frame_id;
		pthread_mutex_unlock(&frameMutex);
		//printf("2: frame mutex unlocked\n"); 
		
		//consoleOutput(local_frame_obj, result_vec, curr_frame_id);
		drawBoxes(local_frame_obj, result_vec, curr_frame_id);
		imshow("Result", local_frame_obj.frame);
	}
} 

void *capsend(void *fd){
	int sockfd = *(int*)fd;
	int err;
	vector<uchar> vec;
	int frame_counter = 0;
	frame_obj local_frame_obj;
	
	while(true){
		capture.read(local_frame_obj.frame);
		if (local_frame_obj.frame.empty()) {
			perror("ERROR no frame\n");
			continue;
		}
		local_frame_obj.start = std::chrono::system_clock::now();
		local_frame_obj.frame_id = frame_counter;
		frame_counter++;
				
		//printf("1: waiting for frame mutex\n");
		pthread_mutex_lock(&frameMutex);
		global_frame_obj = local_frame_obj;
		pthread_cond_signal(&frameCond);
		pthread_mutex_unlock(&frameMutex);
		//printf("1: frame mutex unlocked\n");
		
		imencode(".jpg", local_frame_obj.frame, vec);
		
		err = write(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		err = write(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
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

int main(int argc, char *argv[]) {
	if(argc < 3){
		perror("Usage: ./client [hostname] [port number] [(optional)path to video].\n");
		return 1;
	}
	
	int sockfd1, sockfd2, err;
	
	connect_to_server(sockfd1, sockfd2, argv);
	
	if(argc == 4){
		capture.open(argv[3]);
		if (!capture.isOpened()) {
			perror("ERROR opening video\n");
			return 1;
		}		
	} else { 
		capture.open(0);
		if (!capture.isOpened()) {
			perror("ERROR opening camera\n");
			return 1;
		}
	}
	
	string names_file = "darknet/data/coco.names";
	obj_names = objects_names_from_file(names_file);
	
	pthread_mutex_init(&frameMutex, NULL);
	pthread_cond_init(&frameCond, NULL);
	
	pthread_t thread1, thread2;
	pthread_create(&thread1, NULL, capsend, (void*) &sockfd1);
	pthread_create(&thread2, NULL, recvrend, (void*) &sockfd2);
	
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	close(sockfd1);
	close(sockfd2);
	return 0;
}
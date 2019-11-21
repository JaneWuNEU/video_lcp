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

#define OPENCV
#include "../build/darknet/include/yolo_v2_class.hpp"

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
	Mat frame;
};

vector<frame_obj> frame_buffer;
vector<string> obj_names;
vector<bbox_t> result_vec;

pthread_mutex_t bufferMutex;
pthread_cond_t bufferCond;
pthread_barrier_t initBarrier;

// make a connection to the client and open two sockets one for sending data, one for receiving data
void connect_to_client(int &sockfd, int &newsockfd1, int &newsockfd2, char *argv[]) {
	int err;
	struct sockaddr_in servAddr, clientAddr;
	socklen_t addrlen = sizeof(struct sockaddr_in);

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0){
		perror("failed to open socket.\n");
		close(sockfd);
		exit(1);
	}

	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(atoi(argv[1]));
	servAddr.sin_addr.s_addr = htonl(INADDR_ANY);

	err = bind(sockfd, (struct sockaddr *)&servAddr, addrlen);
	if (err < 0){
		perror("failed to bind address to socket.\n");
		close(sockfd);
		exit(1);
	}

	err = listen(sockfd, 5);
	if (err < 0){
		perror("listen failed.\n");
		close(sockfd);
		exit(1);
	}

	newsockfd1 = accept(sockfd, (struct sockaddr *)&clientAddr, &addrlen);
	newsockfd2 = accept(sockfd, (struct sockaddr *)&clientAddr, &addrlen);
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

//perform object detection on a received frame and send the result vector to the client
void *getSendResult(void *fd) {
	int sockfd = *(int *)fd;
	int err;
	
	string names_file = "darknet/data/coco.names";
    string cfg_file = "darknet/cfg/yolov3.cfg";
    string weights_file = "darknet/yolov3.weights";
    
	Detector detector(cfg_file, weights_file);
    obj_names = objects_names_from_file(names_file);
	
	frame_obj local_frame_obj;
	vector<bbox_t> local_result_vec;
	result_obj curr_result_obj;
	
	//wait until yolo detector is initialized
	pthread_barrier_wait(&initBarrier);
	
	while (true) {
		pthread_mutex_lock(&bufferMutex);
		//wait until there is a frame object in the buffer
		while (frame_buffer.size() <= 0) {
			pthread_cond_wait(&bufferCond, &bufferMutex);
		}
		//copy the first frame in the frame object buffer and remove it from the buffer so it can only be used once for object detection
		local_frame_obj = frame_buffer.front();
		frame_buffer.erase(frame_buffer.begin());
		pthread_mutex_unlock(&bufferMutex);
		
		//perform object detection on the copied frame		
		local_result_vec = detector.detect(local_frame_obj.frame);
		
		//temporary timing to see how old the frame is after object detection
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> spent = end - local_frame_obj.start;
		printf("Detected frame %d is now %f sec old\n",local_frame_obj.frame_id, spent.count());
		
		//send the frame id of the frame on which object detection is performed
		err = write(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//send the capture time of the frame on which object detection is performed 
		err = write(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//send the amount of objects that are found so client knows how many result vectors to read.
		size_t n = local_result_vec.size();
		//printf("2: %zu objects found\n",n);
		err = write(sockfd, &n, sizeof(size_t));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}
		
		//for each detected object copy the data into a result_obj and send this to the client
		for (auto &i : local_result_vec) {
			curr_result_obj.x = i.x; 
			curr_result_obj.y = i.y;
			curr_result_obj.w = i.w;
			curr_result_obj.h = i.h;
			curr_result_obj.prob = i.prob;
			curr_result_obj.obj_id = i.obj_id;
			
			err = write(sockfd, &curr_result_obj, sizeof(result_obj));
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
		}
	}
}

//receive a frame and store it in a buffer
void *recvFrame(void *fd) {
	int sockfd = *(int *)fd;
	int err;
	size_t n;
	frame_obj local_frame_obj;
	
	//wait until yolo detector is initialized
	pthread_barrier_wait(&initBarrier);

	while (true) {
		vector<uchar> vec;
		err = BUFF_SIZE;
		
		//read frame id of received frame
		err = read(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//read capture time of received frame
		err = read(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//read the size of the vector containing the frame data 
		err = read(sockfd,&n,sizeof(size_t));
		if (err < 0){ 
			perror("ERROR reading from socket");
			close(sockfd);
			exit(1);
		}
		
		//read until frame is fully received and add this to the vector
		size_t curr = 0;
		while (curr < n) {
			uchar buffer[BUFF_SIZE];
			err = read(sockfd, buffer, min((int)(n-curr),BUFF_SIZE));
			if (err < 0)
			{
				perror("ERROR reading from socket");
				close(sockfd);
				exit(1);
			}
			vec.insert(vec.end(), buffer, buffer + err);
			curr += err;
		}
		
		//decode frame
		local_frame_obj.frame = imdecode(vec, 1);
		
		//temporary timing to see how old the frame is after receiving
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> spent = end - local_frame_obj.start;
		printf("received frame %d is now %f sec old\n",local_frame_obj.frame_id, spent.count());
		
		if (!local_frame_obj.frame.empty()) {
			//frame is not empty
			pthread_mutex_lock(&bufferMutex);
			//If the buffer is not full, push to back of the buffer, else drop the frame
			if (frame_buffer.size() < MAX_FRAME_BUFFER_SIZE){
				frame_buffer.push_back(local_frame_obj);
				pthread_cond_signal(&bufferCond);
			}	
			printf("there are now %zu frames in buffer\n", frame_buffer.size());
			pthread_mutex_unlock(&bufferMutex);
		} 
	}
}

int main(int argc, char *argv[]) {
	if (argc != 2){
		perror("Usage: ./serv [port number].\n");
		return 1;
	}

	int sockfd, newsockfd1, newsockfd2, err, n;
	connect_to_client(sockfd, newsockfd1, newsockfd2, argv);

	pthread_mutex_init(&bufferMutex, NULL);
	pthread_cond_init(&bufferCond, NULL);
	pthread_barrier_init(&initBarrier,NULL,2);
	
	pthread_t thread1, thread2;
	pthread_create(&thread1, NULL, recvFrame, (void *)&newsockfd1);
	pthread_create(&thread2, NULL, getSendResult, (void *)&newsockfd2);

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);

	close(sockfd);
	close(newsockfd1);
	close(newsockfd2);
	return 0;
}
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

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common.h" 

#define OPENCV
#include "../build/darknet/include/yolo_v2_class.hpp"

using namespace cv;
using namespace std;

Detector* detectors[MAX_MODEL+1];
bool detector_ready[19] = {false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false};
	

vector<frame_obj> frame_buffer;
vector<string> obj_names;
vector<bbox_t> result_vec;

unsigned int curr_model;
int modelPipe[2]; //pipe for communicating model ids

pthread_mutex_t bufferMutex;
pthread_mutex_t detectorMutex;
pthread_cond_t bufferCond;

// make a connection to the client and open two sockets one for sending data, one for receiving data
void create_listen_socket(int &sockfd, char *argv[]) {
	int err;
	struct sockaddr_in servAddr;
	socklen_t addrlen = sizeof(struct sockaddr_in);

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0){
		perror("failed to open socket.\n");
		close(sockfd);
		exit(1);
	}
	
	if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &(int){1}, sizeof(int)) < 0)
    error("setsockopt(SO_REUSEADDR) failed");

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

void *updateDetectionModel(void *) {
	int err;
	unsigned int new_model;
	//local bool used since this is the only thread that modifies the global version, which allows for reading without lock
	
	pthread_mutex_lock(&detectorMutex);
	int local_detector = curr_model;
	pthread_mutex_unlock(&detectorMutex);
	

	while (true) {
		//read from sock to receive message from client
		err = read(modelPipe[0], &new_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR reading from pipe");
			exit(1);
		}
		//printf("U | received %d\n",new_model);
		
		if(new_model == UINT_MAX){
			printf("update done\n");
			break;
		}
				
		if(detector_ready[new_model]) { //model should always be loaded in previous iteration already
			pthread_mutex_lock(&detectorMutex);
			curr_model = new_model;
			pthread_mutex_unlock(&detectorMutex);
		}

		if(new_model > MIN_MODEL+1 && detector_ready[new_model-2]) { // delete old (non neighbour) model
			detector_ready[new_model-2] = false;
			delete detectors[new_model-2];
		} 
		if(new_model < MAX_MODEL-1 && detector_ready[new_model+2]) { // delete old (non neighbour) model 
			detector_ready[new_model+2] = false;
			delete detectors[new_model+2];
		}	
		if(new_model > MIN_MODEL && !detector_ready[new_model-1]) { // if neighbour model not ready, load it.
			detectors[new_model-1] = new Detector(cfg_files[new_model-1],weights_file);
			detector_ready[new_model-1] = true;
		}
		if(new_model < MAX_MODEL && !detector_ready[new_model+1]) { // if neighbour model not ready, load it.
			detectors[new_model+1] = new Detector(cfg_files[new_model+1],weights_file);
			detector_ready[new_model+1] = true;
		}
	}	
}

//perform object detection on a received frame and send the result vector to the client
void *getSendResult(void *fd) {
	int sockfd = *(int *)fd;
	int err;
	
	frame_obj local_frame_obj;
	vector<bbox_t> local_result_vec;
	result_obj curr_result_obj;
	bool localUseDetector0;
	int local_detector;
	
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
	
		if(local_frame_obj.frame_id == UINT_MAX){
			err = write(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			} 
			printf("get and send done\n");
			break;
		}
		
		//check which detector to use
		pthread_mutex_lock(&detectorMutex);
		local_frame_obj.used_model = curr_model;
		pthread_mutex_unlock(&detectorMutex);
		
		//auto start = std::chrono::system_clock::now();
		
		local_result_vec = detectors[local_frame_obj.used_model]->detect(local_frame_obj.frame);

		//temporary timing to see how old the frame is after object detection
		//auto end = std::chrono::system_clock::now();
		//std::chrono::duration<double> spent = end - start;
		
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

		//send detection time 
		/*
		err = write(sockfd, &spent, sizeof(std::chrono::duration<double>));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}
		*/
		
		//send correct model value 
		err = write(sockfd, &local_frame_obj.correct_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//send used model value 
		err = write(sockfd, &local_frame_obj.used_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
				
		//send the amount of objects that are found so client knows how many result vectors to read.
		size_t n = local_result_vec.size();
		//printf("%d: %zu objects found\n",local_frame_obj.frame_id,n);
		err = write(sockfd, &n, sizeof(size_t));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}
		
		//for each detected object copy the data into a result_obj and send this to the client
		for (auto &i : local_result_vec) {
/*			err = write(sockfd, &i.x, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = write(sockfd, &i.y, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = write(sockfd, &i.w, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = write(sockfd, &i.h, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = write(sockfd, &i.prob, sizeof(float)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
			err = write(sockfd, &i.obj_id, sizeof(unsigned int)); 
			if(err<0) { perror("ERROR writing to socket"); close(sockfd); exit(1); }
*/			
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
		
		//printf("S | id %d | correct model %d | used model %d | detection time %f | objects %zu \n", local_frame_obj.frame_id, local_frame_obj.correct_model, local_frame_obj.used_model, spent.count(), n);
		//printf("%d : written all objects\n", local_frame_obj.frame_id);
	}
}

//receive a frame and store it in a buffer
void *recvFrame(void *fd) {
	int sockfd = *(int *)fd;
	int err;
	size_t n;
	frame_obj local_frame_obj;
	unsigned int local_curr_model = STARTING_MODEL;
	
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
		if(local_frame_obj.frame_id == UINT_MAX){
			pthread_mutex_lock(&bufferMutex);
			frame_buffer.push_back(local_frame_obj);
			pthread_cond_signal(&bufferCond);
			pthread_mutex_unlock(&bufferMutex);
			
			err = write(modelPipe[1], &local_frame_obj.frame_id, sizeof(unsigned int));
			if (err < 0){
				perror("ERROR reading from pipe");
				close(sockfd);
				exit(1);
			} 
			printf("recv_frame done\n");
			break;
		}
		//printf("R: %d\n", local_frame_obj.frame_id);
		
		//read capture time of received frame
		err = read(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		//printf("R: %d time read\n", local_frame_obj.frame_id);
	
		//read correct model value
		err = read(sockfd, &local_frame_obj.correct_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		//printf("R: %d correct model %d, local model %d\n", local_frame_obj.frame_id, local_frame_obj.correct_model, local_curr_model);
		
		if(local_frame_obj.correct_model != local_curr_model){
			//update model
			//printf("R: writing to update model to %d\n",local_frame_obj.correct_model);  
			//printf("U | frame %d | writing %d | local %d \n", local_frame_obj.frame_id, local_frame_obj.correct_model, local_curr_model);
			err = write(modelPipe[1], &local_frame_obj.correct_model, sizeof(unsigned int));
			if (err < 0){
				perror("ERROR reading from pipe");
				close(sockfd);
				exit(1);
			} 
			local_curr_model = local_frame_obj.correct_model;
		}
				
		//read the size of the vector containing the frame data 
		err = read(sockfd,&n,sizeof(size_t));
		if (err < 0){ 
			perror("ERROR reading from socket");
			close(sockfd);
			exit(1);
		}
		//printf("R: %d vec size %zu\n", local_frame_obj.frame_id, n);
		
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
		
		size_t buffer_size = frame_buffer.size();
		err = write(sockfd, &buffer_size, sizeof(size_t));
		if (err < 0){
			perror("ERROR writing ack to socket");
			close(sockfd);
			exit(1);
		} 
		//printf("R | ack for id %d\n", local_frame_obj.frame_id);
		
		// communication done, decode frame
		local_frame_obj.frame = imdecode(vec, 1);
		
		if (!local_frame_obj.frame.empty()) {
			//frame is not empty
			pthread_mutex_lock(&bufferMutex);
			//If the buffer is not full, push to back of the buffer, else drop the frame
			if (frame_buffer.size() < MAX_FRAME_BUFFER_SIZE){
				frame_buffer.push_back(local_frame_obj);
				pthread_cond_signal(&bufferCond);
			}	
			//printf("there are now %zu frames in buffer\n", frame_buffer.size());
			pthread_mutex_unlock(&bufferMutex);
		} 
		
		//printf("R | id %d | correct model %d | local model %d | vec size %zu | buff size %zu\n", local_frame_obj.frame_id, local_frame_obj.correct_model, local_curr_model, n, frame_buffer.size());
	}
}

int main(int argc, char *argv[]) {
	if (argc < 2){
		perror("Usage: ./serv [port number].\n");
		return 1;
	}
	
	int sockfd;
	create_listen_socket(sockfd, argv);
	printf("server created\n");
	
	//while(true){
		if(argc == 3){
			curr_model = stoi(argv[2]);
			printf("%d starting model\n",curr_model);
		} else {
			curr_model = STARTING_MODEL;
		}
		
		if(curr_model!=MIN_MODEL){
			detectors[curr_model-1] = new Detector(cfg_files[curr_model-1],weights_file);
			detector_ready[curr_model-1] = true;
		}
		detectors[curr_model] = new Detector(cfg_files[curr_model],weights_file);
		detector_ready[curr_model] = true;
		if(curr_model!=MAX_MODEL){
			detectors[curr_model+1] = new Detector(cfg_files[curr_model+1],weights_file);
			detector_ready[curr_model+1] = true;
		}
		obj_names = objects_names_from_file(names_file);
		
		int err;
		err = pipe(modelPipe);
		if (err < 0){
			perror("ERROR creating pipe");
			exit(1);
		}

		pthread_mutex_init(&bufferMutex, NULL);
		pthread_mutex_init(&detectorMutex, NULL);
		pthread_cond_init(&bufferCond, NULL);
		
		//connect to client
		int newsockfd1, newsockfd2;
		struct sockaddr_in clientAddr;
		socklen_t addrlen = sizeof(struct sockaddr_in);
		newsockfd1 = accept(sockfd, (struct sockaddr *)&clientAddr, &addrlen);
		newsockfd2 = accept(sockfd, (struct sockaddr *)&clientAddr, &addrlen);
		
		pthread_t thread1, thread2, thread3;
		pthread_create(&thread1, NULL, recvFrame, (void *)&newsockfd1);
		pthread_create(&thread2, NULL, getSendResult, (void *)&newsockfd2);
		pthread_create(&thread3, NULL, updateDetectionModel, NULL);

		pthread_join(thread1, NULL);
		pthread_join(thread2, NULL);
		pthread_join(thread3, NULL);
		
		printf("all threads done\n");


		for(int i=MIN_MODEL; i<=MAX_MODEL; i++){
			if(detector_ready[i]){
				delete detectors[i];
				detector_ready[i]=false;
			}
		}

		/*if(curr_model!=MIN_MODEL){
			delete detectors[curr_model-1];
		}
		delete detectors[curr_model];
		if(curr_model!=MAX_MODEL){
			delete detectors[curr_model+1];
		} */
		

		close(newsockfd1);
		close(newsockfd2);
		close(modelPipe[0]);
		close(modelPipe[1]);
		
		pthread_mutex_destroy(&bufferMutex);
		pthread_mutex_destroy(&detectorMutex);
		pthread_cond_destroy(&bufferCond);
	//}	
	
	close(sockfd);
	return 0;
}
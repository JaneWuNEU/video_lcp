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

Detector* detectors[2];
bool useDetector0;

vector<frame_obj> frame_buffer;
vector<string> obj_names;
vector<bbox_t> result_vec;

unsigned int curr_model;
int modelPipe[2]; //pipe for communicating model ids

pthread_mutex_t bufferMutex;
pthread_mutex_t detectorMutex;
pthread_mutex_t detector0Mutex;
pthread_mutex_t detector1Mutex;
pthread_cond_t bufferCond;

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

void *updateDetectionModel(void *) {
	int err;
	unsigned int new_model;
	//local bool used since this is the only thread that modifies the global version, which allows for reading without lock
	bool localUseDetector0 = true;
	
	//string weights_file0 = "darknet/yolov3-tiny.weights";
	//string cfg_file0 = "darknet/cfg/yolov3-tiny.cfg";
    
	string weights_file = "darknet/yolov3.weights";
	string cfg_file0 = "darknet/cfg/yolov3_64_96.cfg";
    string cfg_file1 = "darknet/cfg/yolov3_128_192.cfg";
    string cfg_file2 = "darknet/cfg/yolov3_192_288.cfg";
    string cfg_file3 = "darknet/cfg/yolov3_256_384.cfg";
    string cfg_file4 = "darknet/cfg/yolov3_320_480.cfg";
    string cfg_file5 = "darknet/cfg/yolov3_384_576.cfg";
    string cfg_file6 = "darknet/cfg/yolov3_448_672.cfg";
    string cfg_file7 = "darknet/cfg/yolov3_512_768.cfg";
    
	while (true) {
		//read from sock to receive message from client
		err = read(modelPipe[0], &new_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR reading from pipe");
			exit(1);
		}
		printf("U | received %d\n",new_model);
		
		
		auto start = std::chrono::system_clock::now();
		
		//if detector0 is being used (by detection thread), update detector 1, else update detector 0
		if(localUseDetector0){
			//update detector 1
			//lock is required to prevent update of detector model while detection thread is still using the detector
			pthread_mutex_lock(&detector1Mutex);
		
			//case currModel 
			switch(new_model){
				case 0: 
					delete detectors[1];
					detectors[1] = new Detector(cfg_file0,weights_file);
					break;
				case 1: 
					delete detectors[1];
					detectors[1] = new Detector(cfg_file1,weights_file);
					break;
				case 2: 
					delete detectors[1];
					detectors[1] = new Detector(cfg_file2,weights_file);
					break;
				case 3: 
					delete detectors[1];
					detectors[1] = new Detector(cfg_file3,weights_file);
					break;
				case 4: 
					delete detectors[1];
					detectors[1] = new Detector(cfg_file4,weights_file);
					break;
				case 5: 
					delete detectors[1];
					detectors[1] = new Detector(cfg_file5,weights_file);
					break;
				case 6: 
					delete detectors[1];
					detectors[1] = new Detector(cfg_file6,weights_file);
					break;
				case 7: 
					delete detectors[1];
					detectors[1] = new Detector(cfg_file7,weights_file);
					break;
			}
			
			pthread_mutex_unlock(&detector1Mutex);
			
			localUseDetector0 = false; 
			pthread_mutex_lock(&detectorMutex);
			useDetector0 = false;
			curr_model = new_model;
			
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> spent = end - start;
		
			printf("U | detector 1 | update time %f | new model %d \n", spent.count(), curr_model);
			pthread_mutex_unlock(&detectorMutex);
			
		} else {
			//update detector 0
			pthread_mutex_lock(&detector0Mutex);
			
			//case currModel 
			switch(new_model){
				case 0: 
					delete detectors[0];
					detectors[0] = new Detector(cfg_file0,weights_file);
					break;
				case 1: 
					delete detectors[0];
					detectors[0] = new Detector(cfg_file1,weights_file);
					break;
				case 2: 
					delete detectors[0];
					detectors[0] = new Detector(cfg_file2,weights_file);
					break;
				case 3: 
					delete detectors[0];
					detectors[0] = new Detector(cfg_file3,weights_file);
					break;
				case 4: 
					delete detectors[0];
					detectors[0] = new Detector(cfg_file4,weights_file);
					break;
				case 5: 
					delete detectors[0];
					detectors[0] = new Detector(cfg_file5,weights_file);
					break;
				case 6: 
					delete detectors[0];
					detectors[0] = new Detector(cfg_file6,weights_file);
					break;
				case 7: 
					delete detectors[0];
					detectors[0] = new Detector(cfg_file7,weights_file);
					break;
			}
			
			pthread_mutex_unlock(&detector0Mutex);
			
			localUseDetector0 = true; 
			pthread_mutex_lock(&detectorMutex);
			useDetector0 = true;
			curr_model = new_model;

			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> spent = end - start;

			printf("U | detector 0 | update time %f | new model %d \n", spent.count(), curr_model);
			pthread_mutex_unlock(&detectorMutex);
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
				
		//check which detector to use
		pthread_mutex_lock(&detectorMutex);
		localUseDetector0 = useDetector0;
		local_frame_obj.used_model = curr_model;
		pthread_mutex_unlock(&detectorMutex);
		
		
		auto start = std::chrono::system_clock::now();
		
		//perform object detection on the copied frame using detector 0 or 1
		if(localUseDetector0){
			pthread_mutex_lock(&detector0Mutex);
			local_result_vec = detectors[0]->detect(local_frame_obj.frame);
			pthread_mutex_unlock(&detector0Mutex);
		} else {
			pthread_mutex_lock(&detector1Mutex);
			local_result_vec = detectors[1]->detect(local_frame_obj.frame);
			pthread_mutex_unlock(&detector1Mutex);
		}

		//temporary timing to see how old the frame is after object detection
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> spent = end - start;
		//printf("Detected frame %d is now %f sec old\n",local_frame_obj.frame_id, spent.count());
		
		//printf("%d : write frame id \n", local_frame_obj.frame_id);
		//send the frame id of the frame on which object detection is performed
		err = write(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//printf("%d : write capture time\n", local_frame_obj.frame_id);
		//send the capture time of the frame on which object detection is performed 
		err = write(sockfd, &local_frame_obj.start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//printf("%d : write correct model : %d \n", local_frame_obj.frame_id, local_frame_obj.correct_model);
		//send correct model value 
		err = write(sockfd, &local_frame_obj.correct_model, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//printf("%d : write used model : %d \n", local_frame_obj.frame_id, local_frame_obj.used_model);
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
			err = write(sockfd, &i.x, sizeof(unsigned int)); 
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
			
			/*curr_result_obj.x = i.x; 
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
			}*/
		}
		printf("S | id %d | correct model %d | used model %d | detection time %f | objects %zu \n", local_frame_obj.frame_id, local_frame_obj.correct_model, local_frame_obj.used_model, spent.count(), n);
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
			printf("U | frame %d | writing %d | local %d \n", local_frame_obj.frame_id, local_frame_obj.correct_model, local_curr_model);
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
		
		//decode frame
		local_frame_obj.frame = imdecode(vec, 1);
		
		//temporary timing to see how old the frame is after receiving
		//auto end = std::chrono::system_clock::now();
		//std::chrono::duration<double> spent = end - local_frame_obj.start;
		//printf("received frame %d is now %f sec old\n",local_frame_obj.frame_id, spent.count());
		
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
		
		printf("R | id %d | correct model %d | local model %d | vec size %zu | buff size %zu\n", local_frame_obj.frame_id, local_frame_obj.correct_model, local_curr_model, n, frame_buffer.size());
		
		size_t buffer_size = frame_buffer.size();
		err = write(sockfd, &buffer_size, sizeof(size_t));
		if (err < 0){
			perror("ERROR writing ack to socket");
			close(sockfd);
			exit(1);
		} 
		printf("R | ack for id %d\n", local_frame_obj.frame_id);
	}
}

int main(int argc, char *argv[]) {
	if (argc != 2){
		perror("Usage: ./serv [port number].\n");
		return 1;
	}

	string names_file = "darknet/data/coco.names";
	string cfg_file0 = "darknet/cfg/yolov3_64_96.cfg";
    string cfg_file1 = "darknet/cfg/yolov3_128_192.cfg";
    string cfg_file2 = "darknet/cfg/yolov3_192_288.cfg";
    string cfg_file3 = "darknet/cfg/yolov3_256_384.cfg";
    string cfg_file4 = "darknet/cfg/yolov3_320_480.cfg";
    string cfg_file5 = "darknet/cfg/yolov3_384_576.cfg";
    string cfg_file6 = "darknet/cfg/yolov3_448_672.cfg";
    string cfg_file7 = "darknet/cfg/yolov3_512_768.cfg";
    string weights_file = "darknet/yolov3.weights";
	
	switch(STARTING_MODEL){
		case 0: 
			detectors[0] = new Detector(cfg_file0,weights_file);
			break;
		case 1: 
			detectors[0] = new Detector(cfg_file1,weights_file);
			break;
		case 2: 
			detectors[0] = new Detector(cfg_file2,weights_file);
			break;
		case 3: 
			detectors[0] = new Detector(cfg_file3,weights_file);
			break;
		case 4: 
			detectors[0] = new Detector(cfg_file4,weights_file);
			break;
		case 5: 
			detectors[0] = new Detector(cfg_file5,weights_file);
			break;
		case 6: 
			detectors[0] = new Detector(cfg_file6,weights_file);
			break;
		case 7: 
			detectors[0] = new Detector(cfg_file7,weights_file);
			break;
	}
	
	obj_names = objects_names_from_file(names_file);
	
	useDetector0 = true;
	curr_model = STARTING_MODEL;
	
	int err;
	err = pipe(modelPipe);
	if (err < 0){
		perror("ERROR creating pipe");
		exit(1);
	}

	pthread_mutex_init(&bufferMutex, NULL);
	pthread_mutex_init(&detectorMutex, NULL);
	pthread_mutex_init(&detector0Mutex, NULL);
	pthread_mutex_init(&detector1Mutex, NULL);
	pthread_cond_init(&bufferCond, NULL);
	
	//connect to client
	int sockfd, newsockfd1, newsockfd2;
	connect_to_client(sockfd, newsockfd1, newsockfd2, argv);
	
	pthread_t thread1, thread2, thread3;
	pthread_create(&thread1, NULL, recvFrame, (void *)&newsockfd1);
	pthread_create(&thread2, NULL, getSendResult, (void *)&newsockfd2);
	pthread_create(&thread3, NULL, updateDetectionModel, NULL);

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	pthread_join(thread3, NULL);

	close(sockfd);
	close(newsockfd1);
	close(newsockfd2);
	return 0;
}
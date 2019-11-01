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
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#define BUFF_SIZE 2048
#define MAX_FRAME_BUFFER_SIZE 5

using namespace cv;
using namespace dnn;
using namespace std;

std::vector<string> classes;
std::vector<Mat> bufferFrames;
pthread_mutex_t bufferMutex;
pthread_cond_t bufferCond;

float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.3;  // Non-maximum suppression threshold

struct detObjects {
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> indices;
};

detObjects* postprocess(Mat& frame, const vector<Mat>& outs)
{
	detObjects* detected_objects = new detObjects();
	
	for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                detected_objects->classIds.push_back(classIdPoint.x);
                detected_objects->confidences.push_back((float)confidence);
                detected_objects->boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    NMSBoxes(detected_objects->boxes, detected_objects->confidences, confThreshold, nmsThreshold, detected_objects->indices);
    
	return detected_objects;
}

detObjects* object_detection(Mat &frame){
	int inpWidth = 416;        // Width of network's input image
	int inpHeight = 416;       // Height of network's input image
	confThreshold = 0.5; // Confidence threshold
	nmsThreshold = 0.3;  // Non-maximum suppression threshold
	
	// Load names of classes
	string classesFile = "darknet/data/coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);
	 
	// Give the configuration and weight files for the model
	string configPath = "darknet/cfg/yolov3.cfg";
	string modelPath = "darknet/yolov3.weights";
	 
	// Load the network
	//Net net = readNet(modelPath, configPath, "");
	Net net = readNetFromDarknet(configPath, modelPath);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	vector<string> outNames = net.getUnconnectedOutLayersNames();
	
	Mat blob;
	blobFromImage(frame, blob, 1/255.0, Size(inpWidth,inpHeight),Scalar(0,0,0), true, false);
	net.setInput(blob);
	
	vector<Mat> outs;
	net.forward(outs, outNames);
		 
	return postprocess(frame, outs);
}

void connect_to_client(int &sockfd, int &newsockfd1, int &newsockfd2, char *argv[]) {
	int err;
	struct sockaddr_in servAddr, clientAddr;
	socklen_t addrlen = sizeof(struct sockaddr_in);

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd < 0){
		printf("failed to open socket.\n");
		close(sockfd);
		exit(1);
	}

	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(atoi(argv[1]));
	servAddr.sin_addr.s_addr = htonl(INADDR_ANY);

	err = bind(sockfd, (struct sockaddr *) &servAddr, addrlen);
	if(err < 0){
		printf("failed to bind address to socket.\n");
		close(sockfd);
		exit(1);
	}

	err = listen(sockfd, 5);
	if(err < 0){
		printf("listen failed.\n");
		close(sockfd);
		exit(1);
	}

	newsockfd1 = accept(sockfd, (struct sockaddr *) &clientAddr, &addrlen);
	newsockfd2 = accept(sockfd, (struct sockaddr *) &clientAddr, &addrlen);
}

void *getsendResult(void *fd){
	int sockfd = *(int*)fd;
	int err;
	
	while(waitKey(1) < 0){
		pthread_mutex_lock(&bufferMutex);
		while(bufferFrames.size() <= 0){
			printf("waiting for frame in buffer\n");
			pthread_cond_wait(&bufferCond, &bufferMutex);
		}
		printf("there is a frame in buffer\n");
		Mat frame = bufferFrames.back().clone();
		pthread_mutex_unlock(&bufferMutex);
	
		detObjects* result = object_detection(frame);
	
		size_t n = result->indices.size();
		
		printf("%zu objects found\n",n);
		err = write(sockfd,&n,sizeof(size_t));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}
		for (size_t i = 0; i < n; ++i) {
			int idx = result->indices[i];
			string className = classes[result->classIds[idx]];
			size_t len = className.length();
			float conf = result->confidences[idx];
			Rect box = result->boxes[idx];
			err = write(sockfd,&len,sizeof(size_t));
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
			err = write(sockfd,className.data(),className.length());
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
			err = write(sockfd,&conf,sizeof(float));
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
			err = write(sockfd,&box,sizeof(Rect));
			if (err < 0){
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
		}
	}
}

void *recvFrame(void *fd){
	int sockfd = *(int*)fd;
	int err;
	sleep(1);
	while(waitKey(1) < 0){
		Mat frame;
		std::vector<uchar> vec;
		err = BUFF_SIZE;
		printf("start of frame reading\n" );
	
		while(err == BUFF_SIZE){
			uchar buffer[BUFF_SIZE];
			err = read(sockfd,buffer,BUFF_SIZE);
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				exit(1);
			}
			vec.insert(vec.end(), buffer, buffer+err);
			//printf("part of frame read\n" );
		}
		
		frame = imdecode (vec, 1);
		printf("image received and decoded\n" );
		pthread_mutex_lock(&bufferMutex);
		bufferFrames.push_back(frame);
		//printf("there are now %zu frames in buffer\n",bufferFrames.size());
		if(bufferFrames.size() > MAX_FRAME_BUFFER_SIZE){
			bufferFrames.erase(bufferFrames.begin());
		}
		pthread_cond_signal(&bufferCond);
		pthread_mutex_unlock(&bufferMutex);
	}
}

int main(int argc, char *argv[]) {
	if(argc != 2){
		perror("Usage: ./serv [port number].\n");
		return 1;
	}

	int sockfd, newsockfd1, newsockfd2, err, n;
	connect_to_client(sockfd, newsockfd1, newsockfd2, argv);
	
	printf("connected\n");
	pthread_mutex_init(&bufferMutex, NULL);
	pthread_cond_init(&bufferCond, NULL);
	
	pthread_t thread1, thread2;
	pthread_create(&thread1, NULL, recvFrame, (void*) &newsockfd1);
	pthread_create(&thread2, NULL, getsendResult, (void*) &newsockfd2);
	
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	close(sockfd);
	close(newsockfd1);
	close(newsockfd2);
	return 0;
}
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

using namespace cv;
using namespace std;

VideoCapture capture;
Mat frame;
pthread_mutex_t frameMutex;
pthread_cond_t frameCond;
vector<string> obj_names;

struct bbox_t {
    unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;           // class of object - from range [0, classes-1]
    unsigned int track_id;         // tracking id for video (0 - untracked, 1 - inf - tracked object)
    unsigned int frames_counter;   // counter of frames on which the object was detected
    float x_3d, y_3d, z_3d;        // center of object (in Meters) if ZED 3D Camera is used
};

struct result_obj {
    unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;           // class of object - from range [0, classes-1]
};
	
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

void drawPred(string className, float prob, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", prob);
    label = className + ":" + label;
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

void drawBoxes(Mat mat_img, vector<result_obj> result_vec)
{
    for (auto &i : result_vec) {
        rectangle(mat_img, Point(i.x, i.y), Point(i.x+i.w, i.y+i.h), Scalar(255, 178, 50), 3);
        if (obj_names.size() > i.obj_id) {
            string label = format("%.2f", i.prob);
			label = obj_names[i.obj_id] + ":" + label;
			
			int baseLine;
            Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1 , &baseLine);
            int top = max((int)i.y, labelSize.height);

            rectangle(mat_img, Point(i.x, i.y - round(1.5*labelSize.height)), Point(i.x + round(1.5*labelSize.width), i.y + baseLine), Scalar(255, 255, 255), FILLED);
            putText(mat_img, label, Point2f(i.x, i.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
        }
    }
}

void *recvrend(void *fd){
	int sockfd = *(int*)fd;
	int err;
	
	pthread_mutex_lock(&frameMutex);
	while(frame.empty()){
		printf("2: waiting for captured frame\n");
		pthread_cond_wait(&frameCond, &frameMutex);
	}
	pthread_mutex_unlock(&frameMutex);
	
	while(waitKey(1) < 0){
		auto start = std::chrono::steady_clock::now();
	
		printf("2: waiting for frame mutex\n");
		pthread_mutex_lock(&frameMutex);
		Mat resultFrame = frame.clone();
		pthread_mutex_unlock(&frameMutex);
		printf("2: frame mutex unlocked\n"); 
		
		//vector<bbox_t> result_vec;
		vector<result_obj> result_vec;
		size_t n;
		err = read(sockfd,&n,sizeof(size_t));
		if (err < 0){ 
			perror("ERROR reading from socket");
			close(sockfd);
			exit(1);
		}
		printf("2: %zu objects found\n",n);
		
		for (size_t i = 0; i < n; ++i) {
			//bbox_t obj;
			//err = read(sockfd,&obj,sizeof(bbox_t));
			
			result_obj obj;
			err = read(sockfd,&obj,sizeof(result_obj));
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				exit(1);
			}
			result_vec.push_back(obj);
		}
		
		drawBoxes(resultFrame, result_vec);
		
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> spent = end - start;
		std::cout << " Time: " << spent.count() << " sec \n";
	
		imshow("Result", resultFrame);
		//waitKey(5);
	}
} 

/* each object separately
void *recvrend(void *fd){
	int sockfd = *(int*)fd;
	int err;
	
	while(waitKey(1) < 0){
		//auto start = std::chrono::steady_clock::now();
	
		printf("2: waiting for frame mutex\n");
		pthread_mutex_lock(&frameMutex);
		while(frame.empty()){
			printf("2: waiting for captured frame\n");
			pthread_cond_wait(&frameCond, &frameMutex);
		}
		printf("2: there is a frame captured\n");
		Mat resultFrame = frame.clone();
		pthread_mutex_unlock(&frameMutex);
		
		printf("2: frame mutex unlocked\n"); 
		size_t n;
		err = read(sockfd,&n,sizeof(size_t));
		if (err < 0){ 
			perror("ERROR reading from socket");
			close(sockfd);
			exit(1);
		}
		printf("2: %zu objects found\n",n);
		
		for (size_t i = 0; i < n; ++i) {
			size_t len;
			string className;
			float conf;
			Rect box;
			char *buf;
			err = read(sockfd,&len,sizeof(size_t));
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				exit(1);
			}
			buf = new char[len];
			err = read(sockfd,buf,len);
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				exit(1);
			}
			className.assign(buf,len);
			delete []buf;
			err = read(sockfd,&conf,sizeof(float));
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				exit(1);
			}
			err = read(sockfd,&box,sizeof(Rect));
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				exit(1);
			}
			
			
			drawPred(className, conf, box.x, box.y, box.x + box.width, box.y + box.height, resultFrame);
		}
		
		////auto end = std::chrono::steady_clock::now();
		//std::chrono::duration<double> spent = end - start;
		//std::cout << " Time: " << spent.count() << " sec \n";
	
		imshow("Result", resultFrame);
		//waitKey(5);
	}
} */

void *capsend(void *fd){
	printf("capsend thread\n");
	int sockfd = *(int*)fd;
	int err;
	vector<uchar> vec;
	
	while(true){//waitKey(1) < 0){
		printf("1: waiting for frame mutex\n");
		pthread_mutex_lock(&frameMutex);
		printf("1: lock aquired\n");
		capture.read(frame);
		printf("1: frame captured\n");
	
		if (frame.empty()) {
			perror("ERROR no frame\n");
			pthread_mutex_unlock(&frameMutex);
			continue;
		}
		pthread_cond_signal(&frameCond);
		printf("1: frame exisits, signal sent\n");
		pthread_mutex_unlock(&frameMutex);
		
		printf("1: frame mutex unlocked\n");
		imencode(".jpg", frame, vec);
		
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
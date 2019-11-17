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

vector<string> obj_names;
vector<Mat> bufferFrames;
pthread_mutex_t bufferMutex;
pthread_mutex_t resultMutex;
pthread_cond_t bufferCond;
pthread_cond_t resultCond;
pthread_barrier_t initBarrier;

vector<bbox_t> result_vec;
bool resultReady;

struct result_obj {
    unsigned int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    unsigned int obj_id;           // class of object - from range [0, classes-1]
};

struct frame_obj {
	unsigned int frame_id;
	std::chrono::steady_clock::time_point start;
	Mat frame;
};
vector<frame_obj> frame_buffer;

void connect_to_client(int &sockfd, int &newsockfd1, int &newsockfd2, char *argv[])
{
	int err;
	struct sockaddr_in servAddr, clientAddr;
	socklen_t addrlen = sizeof(struct sockaddr_in);

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if (sockfd < 0)
	{
		perror("failed to open socket.\n");
		close(sockfd);
		exit(1);
	}

	servAddr.sin_family = AF_INET;
	servAddr.sin_port = htons(atoi(argv[1]));
	servAddr.sin_addr.s_addr = htonl(INADDR_ANY);

	err = ::bind(sockfd, (struct sockaddr *)&servAddr, addrlen);
	if (err < 0)
	{
		perror("failed to bind address to socket.\n");
		close(sockfd);
		exit(1);
	}

	err = listen(sockfd, 5);
	if (err < 0)
	{
		perror("listen failed.\n");
		close(sockfd);
		exit(1);
	}

	newsockfd1 = accept(sockfd, (struct sockaddr *)&clientAddr, &addrlen);
	newsockfd2 = accept(sockfd, (struct sockaddr *)&clientAddr, &addrlen);
}

/*
void *sendResult(void *fd)
{
	//printf("thread 3 started\n");
	int sockfd = *(int *)fd;
	int err;
	vector<bbox_t> local_result_vec;
	result_obj curr_result_obj;
	
	pthread_barrier_wait(&initBarrier);
	
	while (true)
	{
		//printf("3: waiting for result mutex\n");
		auto start = std::chrono::steady_clock::now();
		pthread_mutex_lock(&resultMutex);
		while (!resultReady)
		{
			//printf("3: waiting until result is ready\n");
			pthread_cond_wait(&resultCond, &resultMutex);
		}
		//printf("3: copying result\n");
		local_result_vec = result_vec;
		resultReady = false;
		pthread_mutex_unlock(&resultMutex);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> spent = end - start;
		std::cout << "1: Time: " << spent.count() << " sec \n";
		//printf("3: result mutex unlocked\n");
		
		size_t n = local_result_vec.size();

		//printf("3: %zu objects found\n",n);
		err = write(sockfd, &n, sizeof(size_t));
		if (err < 0)
		{
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}
		for (auto &i : local_result_vec)
		{
			curr_result_obj.x = i.x; 
			curr_result_obj.y = i.y;
			curr_result_obj.w = i.w;
			curr_result_obj.h = i.h;
			curr_result_obj.prob = i.prob;
			curr_result_obj.obj_id = i.obj_id;
			
			err = write(sockfd, &curr_result_obj, sizeof(result_obj));
			//err = write(sockfd, &i, sizeof(bbox_t));
			if (err < 0)
			{
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
		}
		end = std::chrono::steady_clock::now();
		spent = end - start;
		std::cout << "2: Time: " << spent.count() << " sec \n";
		
		//printf("3: Result sent\n");
	}
}  */

/*
void *sendResult(void *fd)
{
	//printf("thread 3 started\n");
	int sockfd = *(int *)fd;
	int err;
	vector<bbox_t> local_result_vec;
	
	pthread_barrier_wait(&initBarrier);
	
	while (true)
	{
		//printf("3: waiting for result mutex\n");
		pthread_mutex_lock(&resultMutex);
		while (!resultReady)
		{
			//printf("3: waiting until result is ready\n");
			pthread_cond_wait(&resultCond, &resultMutex);
		}
		//printf("3: copying result\n");
		local_result_vec = result_vec;
		resultReady = false;
		pthread_mutex_unlock(&resultMutex);
		//printf("3: result mutex unlocked\n");
		
		size_t n = local_result_vec.size();

		//printf("3: %zu objects found\n",n);
		err = write(sockfd, &n, sizeof(size_t));
		if (err < 0)
		{
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}
		for (auto &i : local_result_vec)
		{
			string className = obj_names[i.obj_id];
			size_t len = className.length();
			//float conf = localResult->confidences[idx];
			Rect box (i.x, i.y, i.w, i.h);
			err = write(sockfd, &len, sizeof(size_t));
			if (err < 0)
			{
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
			err = write(sockfd, className.data(), className.length());
			if (err < 0)
			{
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
			err = write(sockfd, &i.prob, sizeof(float));
			if (err < 0)
			{
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
			err = write(sockfd, &box, sizeof(Rect));
			if (err < 0)
			{
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
		}
		//printf("3: Result sent\n");
	}
}  */

vector<string> objects_names_from_file(string const filename) {
    ifstream file(filename);
    vector<string> file_lines;
    if (!file.is_open()) return file_lines;
    for(string line; getline(file, line);) file_lines.push_back(line);
    cout << "object names loaded \n";
    return file_lines;
}
/*
void *getResult(void *dummy)
{
	//printf("thread 2 started\n");
	int err;
	
	string names_file = "darknet/data/coco.names";
    string cfg_file = "darknet/cfg/yolov3.cfg";
    string weights_file = "darknet/yolov3.weights";
    
	Detector detector(cfg_file, weights_file);
    obj_names = objects_names_from_file(names_file);
	
	Mat frame;
	vector<bbox_t> local_result_vec;
	
	pthread_barrier_wait(&initBarrier);
	
	while (true)
	{
		//printf("2: waiting for buffer mutex\n");
		pthread_mutex_lock(&bufferMutex);
		while (bufferFrames.size() <= 0)
		{
			//printf("2: waiting for frame in buffer\n");
			pthread_cond_wait(&bufferCond, &bufferMutex);
		}
		//printf("2: there is a frame in buffer\n");
		frame = bufferFrames.front().clone();
		bufferFrames.erase(bufferFrames.begin());
		pthread_mutex_unlock(&bufferMutex);
		//printf("2: buffer mutex unlocked\n");
		
		if (!frame.empty())
		{
			local_result_vec = detector.detect(frame);
		} else {
			//printf("2: frame empty, no detection\n");
		}
		//printf("2: detection completed\n");

		//printf("2: waiting for result mutex\n");
		pthread_mutex_lock(&resultMutex);
		result_vec = local_result_vec;
		resultReady = true;
		pthread_cond_signal(&resultCond);
		//printf("2: result ready, signal sent\n");
		pthread_mutex_unlock(&resultMutex);
		//printf("2: result mutex unlocked\n");
	}
}*/

void *getSendResult(void *fd)
{
	//printf("thread 2 started\n");
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
	
	pthread_barrier_wait(&initBarrier);
	
	while (true)
	{
		//printf("2: waiting for buffer mutex\n");
		auto start = std::chrono::steady_clock::now();
		pthread_mutex_lock(&bufferMutex);
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> spent = end - start;
		std::cout << "2: locked Time: " << spent.count() << " sec \n";
		while (frame_buffer.size() <= 0)
		{
			//printf("2: waiting for frame in buffer\n");
			pthread_cond_wait(&bufferCond, &bufferMutex);
		}
		//printf("2: there is a frame in buffer\n");
		local_frame_obj = frame_buffer.front();
		frame_buffer.erase(frame_buffer.begin());
		pthread_mutex_unlock(&bufferMutex);
		//printf("2: buffer mutex unlocked\n");
		end = std::chrono::steady_clock::now();
		spent = end - start;
		std::cout << "2: unlocked Time: " << spent.count() << " sec \n";
		
		
		if (!local_frame_obj.frame.empty())
		{
			local_result_vec = detector.detect(local_frame_obj.frame);
		} else {
			//printf("2: frame empty, no detection\n");
		}
		//printf("2: detection completed\n");
		
		end = std::chrono::steady_clock::now();
		spent = end - start;
		std::cout << "2: detected Time: " << spent.count() << " sec \n";
		
		end = std::chrono::steady_clock::now();
		spent = end - local_frame_obj.start;
		std::cout << "2: frame is this old :::::: " << spent.count() << " sec \n";

		size_t n = local_result_vec.size();

		//printf("3: %zu objects found\n",n);
		err = write(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		err = write(sockfd, &local_frame_obj.start, sizeof(std::chrono::steady_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		err = write(sockfd, &n, sizeof(size_t));
		if (err < 0)
		{
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}
		for (auto &i : local_result_vec)
		{
			curr_result_obj.x = i.x; 
			curr_result_obj.y = i.y;
			curr_result_obj.w = i.w;
			curr_result_obj.h = i.h;
			curr_result_obj.prob = i.prob;
			curr_result_obj.obj_id = i.obj_id;
			
			err = write(sockfd, &curr_result_obj, sizeof(result_obj));
			//err = write(sockfd, &i, sizeof(bbox_t));
			if (err < 0)
			{
				perror("ERROR writing to socket");
				close(sockfd);
				exit(1);
			}
		}
		end = std::chrono::steady_clock::now();
		spent = end - start;
		std::cout << "2: result send Time: " << spent.count() << " sec \n";
	}
}

void *recvFrame(void *fd)
{
	//printf("thread 1 started\n");
	int sockfd = *(int *)fd;
	int err;
	size_t n;
	
	pthread_barrier_wait(&initBarrier);

	while (true)
	{
		auto start = std::chrono::steady_clock::now();
		frame_obj local_frame_obj;
		vector<uchar> vec;
		err = BUFF_SIZE;
		//printf("1: start of frame reading\n" );
		
		err = read(sockfd, &local_frame_obj.frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		err = read(sockfd, &local_frame_obj.start, sizeof(std::chrono::steady_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		err = read(sockfd,&n,sizeof(size_t));
		if (err < 0){ 
			perror("ERROR reading from socket");
			close(sockfd);
			exit(1);
		}
		
		size_t curr = 0;
		while (curr < n)
		{
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
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> spent = end - start;
		std::cout << "1: frame read Time: " << spent.count() << " sec \n";

		local_frame_obj.frame = imdecode(vec, 1);
		end = std::chrono::steady_clock::now();
		spent = end - start;
		std::cout << "1: frame decoded Time: " << spent.count() << " sec \n";
		end = std::chrono::steady_clock::now();
		spent = end - local_frame_obj.start;
		std::cout << "1: frame is this old :::::: " << spent.count() << " sec \n";
		
		if (!local_frame_obj.frame.empty())
		{
			//printf("1: frame received and decoded\n");
			pthread_mutex_lock(&bufferMutex);
			end = std::chrono::steady_clock::now();
		spent = end - start;
		std::cout << "1: locked Time: " << spent.count() << " sec \n";
			frame_buffer.push_back(local_frame_obj);
			//bufferFrames.push_back(frame);
			//printf("1: there are now %zu frames in buffer\n",bufferFrames.size());
			if (frame_buffer.size() > MAX_FRAME_BUFFER_SIZE)
			{
				frame_buffer.erase(frame_buffer.end());
			}
			pthread_cond_signal(&bufferCond);
			pthread_mutex_unlock(&bufferMutex);
			end = std::chrono::steady_clock::now();
		spent = end - start;
		std::cout << "1: unlocked Time: " << spent.count() << " sec \n";
			//printf("1: buffer mutex unlocked\n");
		} else {
			//printf("1: frame empty\n");
		}
	}
}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		perror("Usage: ./serv [port number].\n");
		return 1;
	}

	int sockfd, newsockfd1, newsockfd2, err, n;
	connect_to_client(sockfd, newsockfd1, newsockfd2, argv);

	pthread_mutex_init(&bufferMutex, NULL);
	pthread_mutex_init(&resultMutex, NULL);
	pthread_cond_init(&bufferCond, NULL);
	pthread_cond_init(&resultCond, NULL);
	
	resultReady = false;

/*	pthread_barrier_init(&initBarrier,NULL,3);
	pthread_t thread1, thread2, thread3;
	pthread_create(&thread1, NULL, recvFrame, (void *)&newsockfd1);
	pthread_create(&thread2, NULL, getResult, NULL);
	pthread_create(&thread3, NULL, sendResult, (void *)&newsockfd2);
*/	
	pthread_barrier_init(&initBarrier,NULL,2);
	pthread_t thread1, thread2;
	pthread_create(&thread1, NULL, recvFrame, (void *)&newsockfd1);
	pthread_create(&thread2, NULL, getSendResult, (void *)&newsockfd2);

	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	//pthread_join(thread3, NULL);
	close(sockfd);
	close(newsockfd1);
	close(newsockfd2);
	return 0;
}
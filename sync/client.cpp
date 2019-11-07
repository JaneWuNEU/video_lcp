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

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#define BUFF_SIZE 2048

using namespace cv;
using namespace std;

int connect_to_server(int &sockfd, char *argv[]){
	int err;
	struct sockaddr_in serv_addr;
	struct hostent *server;
	struct in_addr *addr;
	socklen_t addrlen = sizeof(struct sockaddr_in);

	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd < 0){
		perror("failed to open socket.\n");
		return 1;
	}

	server = gethostbyname(argv[1]);
	if (server==NULL) {
		perror("Address not found for\n");
		close(sockfd);
		return 1;
	} else {
		addr = (struct in_addr*) server->h_addr_list[0];
	}

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(atoi(argv[2]));
	serv_addr.sin_addr.s_addr = inet_addr(inet_ntoa(*addr));
	
	err = connect(sockfd,(struct sockaddr *)&serv_addr,addrlen);
	if(err < 0){
		perror("Connecting to server failed\n");
		close(sockfd);
		return 1;
	}
	return 0;
}

void drawPred(string className, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    label = className + ":" + label;
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

int main(int argc, char *argv[]) {
	if(argc < 3){
		perror("Usage: ./client [hostname] [port number] [(optional)path to video].\n");
		return 1;
	}
	
	int sockfd, err;
	
	if (connect_to_server(sockfd, argv) < 0) return 1;
	
	Mat frame;
	VideoCapture capture;
	std::vector<uchar> vec;
	
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
	
	while(waitKey(1) < 0){
		capture.read(frame);
		//capture >> frame;
		if (frame.empty()) {
			 perror("ERROR no frame\n");
			 continue;
		}
		imshow("Live", frame);
		waitKey(1);
		
		imencode(".jpg", frame, vec);
		
		err = write(sockfd, vec.data(), vec.size());
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			return 1;
		} 
		
		size_t n;
		err = read(sockfd,&n,sizeof(size_t));
		if (err < 0){ 
			perror("ERROR reading from socket");
			close(sockfd);
			return 1;
		}
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
				return 1;
			}
			buf = new char[len];
			err = read(sockfd,buf,len);
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				return 1;
			}
			className.assign(buf,len);
			delete []buf;
			err = read(sockfd,&conf,sizeof(float));
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				return 1;
			}
			err = read(sockfd,&box,sizeof(Rect));
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(sockfd);
				return 1;
			}
		
			drawPred(className, conf, box.x, box.y, box.x + box.width, box.y + box.height, frame);
		}
		imshow("Result",frame);
		waitKey(5);
	}
	
	close(sockfd);
	return 0;
}
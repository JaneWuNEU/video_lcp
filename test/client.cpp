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

using namespace cv;

//int sockfd; 

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



	void *capsend(void *fd){
		int sockfd = *(int*)fd;
		int err;
		char buffer[256];
		
		while(waitKey(1) < 0){
			printf("Please enter the message: ");
			bzero(buffer,256);
			fgets(buffer,255,stdin);
			err = write(sockfd, buffer, strlen(buffer));
			if (err < 0) 
				perror("ERROR writing to socket\n");
		}
	}

	void *recvrend(void *fd){
		int sockfd = *(int*)fd;
		int err;
		char buffer[256];
		
		while(waitKey(1) < 0){
			bzero(buffer,256);
			err = read(sockfd, buffer, 255);
			if (err < 0) 
				perror("ERROR reading from socket\n");
			printf("%s\n", buffer);
		}
	}


int main(int argc, char *argv[]) {
	if(argc < 3){
		perror("Usage: ./client [hostname] [port number] [(optional)path to video].\n");
		return 1;
	}
	
	int sockfd1, sockfd2, err;
	
	connect_to_server(sockfd1, sockfd2, argv);
	
	pthread_t thread1, thread2;
	pthread_create(&thread1, NULL, capsend, (void*) &sockfd1);
	pthread_create(&thread2, NULL, recvrend, (void*) &sockfd2);
	
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);

	/*
	char buffer[256];
	
	while(waitKey(1) < 0){
		printf("Please enter the message: ");
		bzero(buffer,256);
		fgets(buffer,255,stdin);
		err = write(sockfd, buffer, strlen(buffer));
		if (err < 0) 
			perror("ERROR writing to socket\n");
		bzero(buffer,256);
		err = read(sockfd, buffer, 255);
		if (err < 0) 
			perror("ERROR reading from socket\n");
		printf("%s\n", buffer);
	}
	 */
	
	close(sockfd1);
	close(sockfd2);
	return 0;
}
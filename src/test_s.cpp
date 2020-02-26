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

using namespace std;

// make a connection to the client and open two sockets one for sending data, one for receiving data
void connect_to_client(int &sockfd, int &newsockfd1, char *argv[]) {
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
}

//receive a frame and store it in a buffer
void *recvFrame(void *fd) {
	int sockfd = *(int *)fd;
	int err;
	size_t n;
	int frame_id;
	std::chrono::system_clock::time_point start;
	
	while (true) {
		//read frame id of received frame
		err = read(sockfd, &frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		//printf("R: %d\n", local_frame_obj.frame_id);
		
		//read capture time of received frame
		err = read(sockfd, &start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		//printf("R: %d time read\n", local_frame_obj.frame_id);
		
		size_t buffer_size = 1;
		err = write(sockfd, &buffer_size, sizeof(size_t));
		if (err < 0){
			perror("ERROR writing ack to socket");
			close(sockfd);
			exit(1);
		} 
		printf("R | ack for id %d\n", frame_id);
	}
}

int main(int argc, char *argv[]) {
	if (argc != 2){
		perror("Usage: ./serv [port number].\n");
		return 1;
	}
	//connect to client
	int sockfd, newsockfd1;
	connect_to_client(sockfd, newsockfd1, argv);
	
	pthread_t thread1;
	pthread_create(&thread1, NULL, recvFrame, (void *)&newsockfd1);

	pthread_join(thread1, NULL);

	close(sockfd);
	close(newsockfd1);
	return 0;
}
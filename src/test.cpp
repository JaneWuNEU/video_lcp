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

// make a connection to the server and open two sockets one for sending data, one for receiving data
void connect_to_server(int &sockfd1, char *argv[]) {
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
	
	server = gethostbyname(argv[1]);
	if (server==NULL) {
		perror("Address not found for\n");
		close(sockfd1);
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
		exit(1);
	}
}

//capture and send a frame to the server for object detection
void *capsend(void *fd) {
	int sockfd = *(int*)fd;
	int err;
	unsigned int frame_counter = 0;
	size_t buffer_size;
	
	while(true) {
		//capture frame into local frame object so capturing is not done within mutex
		std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
		int frame_id = frame_counter;
		frame_counter++;
		
		//send frame id 
		err = write(sockfd, &frame_id, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		} 
		
		//send capture time of frame
		err = write(sockfd, &start, sizeof(std::chrono::system_clock::time_point));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}

		//wait for ack of server that frame is received
		err = read(sockfd, &buffer_size, sizeof(size_t));
		if (err < 0){
			perror("ERROR reading ack from socket");
			close(sockfd);
			exit(1);
		} 
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> spent = end - start;
		double time_spent = spent.count();
		printf("S | ack for id %d | it is %f old\n", frame_id, time_spent);
	}
}

int main(int argc, char *argv[]) {
	if(argc < 3){
		perror("Usage: ./client [server_hostname] [server_port_number]\n");
		return 1;
	}
	
	int sockfd1, err;
	
	connect_to_server(sockfd1, argv);
		
	pthread_t thread1;

	pthread_create(&thread1, NULL, capsend, (void*) &sockfd1);
	
	pthread_join(thread1, NULL);

	close(sockfd1);
	
	return 0;
}
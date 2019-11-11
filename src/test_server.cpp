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

pthread_mutex_t mutex;
pthread_cond_t cond;
int nreplies;

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

	void *sendobj(void *fd){
		int sockfd = *(int*)fd;
		int err;
		char buffer[256];
		
		
		printf("1: started\n");
		
		
		while(true){
			printf("1: wait for mutex \n");
			pthread_mutex_lock(&mutex);
			while (nreplies <= 0){
				printf("1: wait for signal\n");
				pthread_cond_wait(&cond, &mutex);
				//printf("1: received signal\n");
			}
			printf("1: %d replies to be sent\n",nreplies);
			nreplies--;
			pthread_mutex_unlock(&mutex);
			printf("1: mutex unlocked \n");
			
			bzero(buffer,256);
			strcpy(buffer, "this is a reply "); 
			err = write(sockfd,buffer,255);
			if (err < 0) perror("ERROR writing to socket\n");
			printf("1: reply sent \n\n");
			//sleep(1);
		}
	}

	void *recvdet(void *fd){
		int sockfd = *(int*)fd;
		int err;
		char buffer[256];
		
		printf("2: started\n");
		
		while(true){
			printf("2: waiting for message\n");
			bzero(buffer,256);
			err = read(sockfd,buffer,255);
			if (err < 0) perror("ERROR reading from socket\n");
			printf("2: Here is the message: %s",buffer);
			
			printf("2: wait for mutex \n");
			pthread_mutex_lock(&mutex);
			nreplies++;
			printf("2: %d replies ready \n", nreplies);
			pthread_cond_signal(&cond);
			printf("2: signal sent \n");
			pthread_mutex_unlock(&mutex);
			printf("2: mutex unlocked \n");
   		}
	}

int main(int argc, char *argv[]) {
	if(argc != 2){
		perror("Usage: ./serv [port number].\n");
		return 1;
	}
	nreplies=0;
	int sockfd, newsockfd1, newsockfd2, err, n;
	connect_to_client(sockfd, newsockfd1, newsockfd2, argv);
	
	printf("connected\n");
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);
	
	pthread_t thread1, thread2;
	pthread_create(&thread1, NULL, recvdet, (void*) &newsockfd1);
	pthread_create(&thread2, NULL, sendobj, (void*) &newsockfd2);
	
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
/*	char buffer[256];
	n=0;
	while(waitKey(1) < 0){
		bzero(buffer,256);
	
		err = read(newsockfd,buffer,255);
		if (err < 0) perror("ERROR reading from socket\n");
		printf("Here is the message: %s\n",buffer);
		
		bzero(buffer,256);
		strcpy(buffer, "this is a reply "); 
		err = write(newsockfd,buffer,255);
		if (err < 0) perror("ERROR writing to socket\n");
		
		n++;
	} */

	close(sockfd);
	close(newsockfd1);
	close(newsockfd2);
	return 0;
}
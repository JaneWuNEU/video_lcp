#include <unistd.h>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>

// make a connection to the client and open two sockets one for sending data, one for receiving data
void connect_to_client(int &sockfd, int &newsockfd, int port) {
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
	servAddr.sin_port = htons(port);
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

	newsockfd = accept(sockfd, (struct sockaddr *)&clientAddr, &addrlen);
}

//read current bandwidth from user input (instead of monitoring) and send this data to the client 
//maybe also include latency
void monitor_speed(int sockfd){
	int err;
	unsigned int speed;
	while (std::cin >> speed) {
		printf("Speed is now %d\n", speed);
		err = write(sockfd, &speed, sizeof(unsigned int));
		if (err < 0){
			perror("ERROR writing to socket");
			close(sockfd);
			exit(1);
		}
	}	
}

int main(int argc, char *argv[]) {
	if (argc != 2){
		perror("Usage: ./monitor [port number].\n");
		return 1;
	}

	int sockfd, newsockfd, err;
	connect_to_client(sockfd, newsockfd, atoi(argv[1]));
	
	monitor_speed(newsockfd);

	return 0;
}	
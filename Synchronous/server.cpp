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

using namespace std;
using namespace cv;
using namespace dnn;

vector<string> classes;
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.3;  // Non-maximum suppression threshold

struct detObjects {
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> indices;
};

/*  Drawing is done at client side
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}
*/

detObjects* postprocess(Mat& frame, const vector<Mat>& outs)
{
    //vector<int> classIds;
    //vector<float> confidences;
    //vector<Rect> boxes;
	//vector<int> indices;
    
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
	/* Drawing is done at client side
	for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    } */
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

	/*
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time: %.2f ms", t);
    putText(image2, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	*/
}


void connect_to_client(int &sockfd, int &newsockfd, char *argv[]) {
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

	newsockfd = accept(sockfd, (struct sockaddr *) &clientAddr, &addrlen);
}

int main(int argc, char *argv[]) {
	if(argc != 2){
		perror("Usage: ./serv [port number].\n");
		return 1;
	}

	int sockfd, newsockfd, err;
	connect_to_client(sockfd, newsockfd, argv);
	
	printf("connected\n");
	
	while(waitKey(1) < 0){
		Mat frame;
		std::vector<uchar> vec;
		err = BUFF_SIZE;
	
		while(err == BUFF_SIZE){
			uchar buffer[BUFF_SIZE];
			//memset(buffer, 0, BUFF_SIZE);	
			err = read(newsockfd,buffer,BUFF_SIZE);
			//printf("%d\n", err);
			if (err < 0){ 
				perror("ERROR reading from socket");
				close(newsockfd);
				return 1;
			}
			vec.insert(vec.end(), buffer, buffer+err);
		}
		
		frame = imdecode (vec, 1);
		
		detObjects* result = object_detection(frame);
		
		// send indices.size()
		// for each 
		// send classIds[idx]
		// send confidences[idx]
		// send boxes[idx]
		
/*		for (size_t i = 0; i < result->indices.size(); ++i)
		{
        int idx = result->indices[i];
		Rect box = result->boxes[idx];
        drawPred(result->classIds[idx], result->confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
		} */
		size_t n = result->indices.size();
		//printf("%zu objects found\n",n);
		err = write(newsockfd,&n,sizeof(size_t));
		if (err < 0){
			perror("ERROR writing to socket");
			close(newsockfd);
			return 1;
		}
		for (size_t i = 0; i < n; ++i) {
			int idx = result->indices[i];
			string className = classes[result->classIds[idx]];
			size_t len = className.length();
			float conf = result->confidences[idx];
			Rect box = result->boxes[idx];
			err = write(newsockfd,&len,sizeof(size_t));
			if (err < 0){
				perror("ERROR writing to socket");
				close(newsockfd);
				return 1;
			}
			err = write(newsockfd,className.data(),className.length());
			if (err < 0){
				perror("ERROR writing to socket");
				close(newsockfd);
				return 1;
			}
			err = write(newsockfd,&conf,sizeof(float));
			if (err < 0){
				perror("ERROR writing to socket");
				close(newsockfd);
				return 1;
			}
			err = write(newsockfd,&box,sizeof(Rect));
			if (err < 0){
				perror("ERROR writing to socket");
				close(newsockfd);
				return 1;
			}
		}

		//imshow("Live", frame);	
		//waitKey(100);		
		
		/*
		char reply[BUFF_SIZE] = "message received\n";
		err = write(newsockfd,reply,BUFF_SIZE);
		if (err < 0){
			perror("ERROR writing to socket");
			close(newsockfd);
			return 1;
		}
		*/
	}
	
	close(sockfd);
	close(newsockfd);
	return 0;
}
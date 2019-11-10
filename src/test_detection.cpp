#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

vector<string> classes;
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold

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
	printf("%s \n",label.c_str());
}

void postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
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
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

int main(int argc, char** argv )
{
	//float scale = 0.005;
	//Scalar mean = 0;
	//bool swapRB = true;
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


    Mat image, image2, blob;
	std::vector<uchar> vec;
    image = imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    
	imencode(".jpg", image, vec);
	
	image2 = imdecode (vec, 1);
	//imshow("Live", image2);		
	
	//preprocess(image2, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
	blobFromImage(image2, blob, 1/255.0, Size(inpWidth,inpHeight),Scalar(0,0,0), true, false);
	net.setInput(blob);
	
	vector<Mat> outs;
	net.forward(outs, outNames);
		 
	postprocess(image2, outs);

    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time: %.2f ms", t);
    putText(image2, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	
	//imshow("test", image2);
		 
	//waitKey(0);
    return 0;
}
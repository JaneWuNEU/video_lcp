# video_lcp
A latency control protocol for video analytics pipeline

# versions
Simple detection model: used to capture or load image, perform object detection and render results

Synchronous communication model: 
Previous model split up in a server and client. Client captures a frame from video or webcam, sends this to the server and waits for the result. The server waits until it receives a frame, performs object detection and returns the location of the detected objects to the client. The client then renders the result and goes back to capture a new frame.

ASychronous communication model: 
the tasks of the client and server are both split up in several threads. The client uses has one thread that performs capturing of a frame and sending it to the server, the other thread receives and then renders the result. The server has a thread that receives a frame and stores it in a buffer, a thread that reads the last received frame from the buffer and performs object detection and store the result, a last thread is used to copy and send the stored result back to the client. In the "old" server model the tasks of performing object detection and sending the result are performed by a single thread.


# usage
If no filename is given, the standard webcam will be used to capture video
Server : ./A_Server {port_no}
Client : ./A_Client {serer_ip} {port_no} {optional: filename}

To use real time input from video file :
in v4l2loopback folder :
download v4l2loopback  : https://github.com/umlaeute/v4l2loopback
make								// to get kernel module
insmod v4l2loopback.ko		// to launch virtual webcam
ls /dev/ 							// to check virtual webcam video id (/dev/video*)
start Server
start Client with filename = /dev/video*
ffmpeg -re -i videos/crossroad360.mp4 -f v4l2 /dev/video2 	// to start streaming video to virtual webcam


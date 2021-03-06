#define BUFF_SIZE 8192 //4096 2048
#define MAX_FRAME_BUFFER_SIZE 30
#define FRAME_DEADLINE 1.0/30
#define MAX_MODEL 18  //15 for 16model server, 18 for others
#define MIN_MODEL 0
#define STARTING_MODEL 0

#define CONTROL_WINDOW 50
#define LOW_ON_TIME 35 // switch down if less or equal to X are on time
#define HIGH_ON_TIME 50 // switch up if more or equal to X are on time 

#define DOWN_SUM 2.5 //switch down when average latency score is 5ms late 
#define UP_SUM 5 //switch up when average latency score is 10ms early 
#define LATE_EXP 1.25 //score exponent for late latency
#define ON_TIME_EXP 1.0 //score exponent for early latency 

#define HISTORY_WEIGHT 0.0 //1.0/4 // weight for the history  		Normal weight = (1-History weight)


// object that is returned by the server in which information on a detected object is stored
struct result_obj {					
    unsigned int x, y, w, h;      
    float prob;                    
    unsigned int obj_id; 
};

// frame object that stores all the information about a frame
struct frame_obj {
	unsigned int frame_id;
	std::chrono::system_clock::time_point start;
	std::chrono::duration<double> detection_time;
	unsigned int correct_model;
	unsigned int used_model;
	cv::Mat frame;
	double time_till_send;
	double time_after_send;
	
};

const unsigned int n_height[19] = {64,96,128,160,192,224,256,288,320,352,384,416,448,480,512,544,576,608,640};
const unsigned int n_width[19] = {64,96,128,160,192,224,256,288,320,352,384,416,448,480,512,544,576,608,640};

const std::string names_file = "darknet/data/coco.names";
const std::string weights_file = "darknet/yolov3.weights";

const std::string cfg_file0 = "darknet/cfg/yolov3_64.cfg";
const std::string cfg_file1 = "darknet/cfg/yolov3_96.cfg";
const std::string cfg_file2 = "darknet/cfg/yolov3_128.cfg";
const std::string cfg_file3 = "darknet/cfg/yolov3_160.cfg";
const std::string cfg_file4 = "darknet/cfg/yolov3_192.cfg";
const std::string cfg_file5 = "darknet/cfg/yolov3_224.cfg";
const std::string cfg_file6 = "darknet/cfg/yolov3_256.cfg";
const std::string cfg_file7 = "darknet/cfg/yolov3_288.cfg";
const std::string cfg_file8 = "darknet/cfg/yolov3_320.cfg";
const std::string cfg_file9 = "darknet/cfg/yolov3_352.cfg";
const std::string cfg_file10 = "darknet/cfg/yolov3_384.cfg";
const std::string cfg_file11 = "darknet/cfg/yolov3_416.cfg";
const std::string cfg_file12 = "darknet/cfg/yolov3_448.cfg";
const std::string cfg_file13 = "darknet/cfg/yolov3_480.cfg";
const std::string cfg_file14 = "darknet/cfg/yolov3_512.cfg";
const std::string cfg_file15 = "darknet/cfg/yolov3_544.cfg";
const std::string cfg_file16 = "darknet/cfg/yolov3_576.cfg";
const std::string cfg_file17 = "darknet/cfg/yolov3_608.cfg";
const std::string cfg_file18 = "darknet/cfg/yolov3_640.cfg";

const std::string cfg_files[19] = { "darknet/cfg/yolov3_64.cfg", 
"darknet/cfg/yolov3_96.cfg", "darknet/cfg/yolov3_128.cfg", "darknet/cfg/yolov3_160.cfg", 
"darknet/cfg/yolov3_192.cfg", "darknet/cfg/yolov3_224.cfg", "darknet/cfg/yolov3_256.cfg", 
"darknet/cfg/yolov3_288.cfg", "darknet/cfg/yolov3_320.cfg", "darknet/cfg/yolov3_352.cfg", 
"darknet/cfg/yolov3_384.cfg", "darknet/cfg/yolov3_416.cfg", "darknet/cfg/yolov3_448.cfg", 
"darknet/cfg/yolov3_480.cfg", "darknet/cfg/yolov3_512.cfg", "darknet/cfg/yolov3_544.cfg", 
"darknet/cfg/yolov3_576.cfg", "darknet/cfg/yolov3_608.cfg", "darknet/cfg/yolov3_640.cfg" };

/*string cfg_file0 = "darknet/cfg/yolov3_64_96.cfg";
string cfg_file1 = "darknet/cfg/yolov3_128_192.cfg";
string cfg_file2 = "darknet/cfg/yolov3_192_288.cfg";
string cfg_file3 = "darknet/cfg/yolov3_256_384.cfg";
string cfg_file4 = "darknet/cfg/yolov3_320_480.cfg";
string cfg_file5 = "darknet/cfg/yolov3_384_576.cfg";
string cfg_file6 = "darknet/cfg/yolov3_448_672.cfg";
string cfg_file7 = "darknet/cfg/yolov3_512_768.cfg";
*/
//const unsigned int n_height[8] = {64,128,192,256,320,384,448,512};
//const unsigned int n_width[8] = {96,192,288,384,480,576,672,768};

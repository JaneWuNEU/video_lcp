#define BUFF_SIZE 2048
#define MAX_FRAME_BUFFER_SIZE 30
#define FRAME_DEADLINE 0.04
#define MAX_MODEL 18
#define MIN_MODEL 0
#define STARTING_MODEL 3

#define CONTROL_WINDOW 50
#define LOW_ON_TIME 35
#define HIGH_ON_TIME 50

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

#define BUFF_SIZE 2048
#define MAX_FRAME_BUFFER_SIZE 30
#define FRAME_DEADLINE 0.04
#define MAX_MODEL 7
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

const unsigned int n_height[8] = {64,128,192,256,320,384,448,512};
const unsigned int n_width[8] = {96,192,288,384,480,576,672,768};

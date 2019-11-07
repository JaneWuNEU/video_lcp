cmake_minimum_required(VERSION 2.8)
project( Thesis )
find_package( OpenCV 4 REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")

include_directories( 
	${CMAKE_CURRENT_SOURCE_DIR}
	${OpenCV_INCLUDE_DIRS} 
)

add_executable( detection simple_detection/detection.cpp)
add_executable( detection_short simple_detection/detectionShort.cpp)

add_executable( server src/server.cpp )
add_executable( client src/client.cpp )

target_link_libraries( detection ${OpenCV_LIBS} )
target_link_libraries( detection_short ${OpenCV_LIBS} )

target_link_libraries( server ${OpenCV_LIBS} )
target_link_libraries( client ${OpenCV_LIBS} )

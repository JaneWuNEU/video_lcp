cmake_minimum_required(VERSION 2.8)
project( Thesis )
find_package( OpenCV 4 REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")

include_directories( 
	${CMAKE_CURRENT_SOURCE_DIR}
	${OpenCV_INCLUDE_DIRS} 
)

add_executable( Detection detection.cpp)
add_executable( DetectionShort detectionShort.cpp)
add_executable( Server server.cpp )
add_executable( Client client.cpp )

target_link_libraries( Detection ${OpenCV_LIBS} )
target_link_libraries( DetectionShort ${OpenCV_LIBS} )
target_link_libraries( Server ${OpenCV_LIBS} )
target_link_libraries( Client ${OpenCV_LIBS} )


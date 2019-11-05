cmake_minimum_required(VERSION 2.8)
project( Thesis )
find_package( OpenCV 4 REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")

include_directories( 
	${CMAKE_CURRENT_SOURCE_DIR}
	${OpenCV_INCLUDE_DIRS} 
)

add_executable( Detection SimpleDetection/detection.cpp)
add_executable( DetectionShort SimpleDetection/detectionShort.cpp)

add_executable( S_Server Synchronous/server.cpp )
add_executable( S_Client Synchronous/client.cpp )

add_executable( A_Server ASynchronous/server.cpp )
add_executable( A_Server_old ASynchronous/server_old.cpp )
add_executable( A_Client ASynchronous/client.cpp )


target_link_libraries( Detection ${OpenCV_LIBS} )
target_link_libraries( DetectionShort ${OpenCV_LIBS} )

target_link_libraries( S_Server ${OpenCV_LIBS} )
target_link_libraries( S_Client ${OpenCV_LIBS} )

target_link_libraries( A_Server ${OpenCV_LIBS} )
target_link_libraries( A_Server_old ${OpenCV_LIBS} )
target_link_libraries( A_Client ${OpenCV_LIBS} )

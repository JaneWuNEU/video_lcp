cmake_minimum_required(VERSION 2.8)
project( Thesis )
find_package( OpenCV 4 REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")

include_directories( 
	${CMAKE_CURRENT_SOURCE_DIR}
	${OpenCV_INCLUDE_DIRS} 
)

add_executable( server src/server.cpp )
add_executable( client src/client.cpp )

add_executable( test_client src/test_client.cpp ) 
add_executable( test_server src/test_server.cpp )
add_executable( test_detection src/test_detection.cpp )

target_link_libraries( server ${OpenCV_LIBS} )
target_link_libraries( client ${OpenCV_LIBS} )

target_link_libraries( test_server ${OpenCV_LIBS} )
target_link_libraries( test_client ${OpenCV_LIBS} )
target_link_libraries( test_detection ${OpenCV_LIBS} )
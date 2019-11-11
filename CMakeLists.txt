cmake_minimum_required(VERSION 2.8)
project( Thesis )
find_package( OpenCV 3 REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")
set(DARKNET_PATH darknet)
set(DARKNET_LIB_NAME libdarknet)

include_directories( 
	${CMAKE_CURRENT_SOURCE_DIR}
	${OpenCV_INCLUDE_DIRS} 
	${DARKNET_PATH}/include
)

option(ENABLE_OPENCV "Enable OpenCV integration" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_CUDNN "Enable CUDNN" ON)
option(ENABLE_CUDNN_HALF "Enable CUDNN Half precision" ON)

add_library(${DARKNET_LIB_NAME} SHARED IMPORTED)
set_property(TARGET ${DARKNET_LIB_NAME} PROPERTY IMPORTED_LOCATION "${DARKNET_PATH}/libdarknet.so")

#add_executable( server src/server.cpp )
#add_executable( client src/client.cpp )

add_executable( test_client src/test_client.cpp ) 
add_executable( test_server src/test_server.cpp )
#add_executable( test_detection src/test_detection.cpp )
add_executable( test_detection_darknet src/test_detection_darknet.cpp )
add_executable( test_detection_darknet2 src/test_detection_darknet2.cpp )

#target_link_libraries( server ${OpenCV_LIBS} )
#target_link_libraries( client ${OpenCV_LIBS} )

target_link_libraries( test_server ${OpenCV_LIBS} )
target_link_libraries( test_client ${OpenCV_LIBS} )
#target_link_libraries( test_detection ${OpenCV_LIBS} )
target_link_libraries( test_detection_darknet ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( test_detection_darknet2 ${OpenCV_LIBS} ${DARKNET_LIB_NAME} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
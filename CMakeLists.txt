cmake_minimum_required(VERSION 2.8)
project( Thesis )
find_package( OpenCV 4 REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")
set(DARKNET_PATH darknet)
set(DARKNET_LIB_NAME libdarknet)
#set(IPERF_PATH iperf)
#set(IPERF_LIB_NAME libiperf)

include_directories( 
	${CMAKE_CURRENT_SOURCE_DIR}
	${OpenCV_INCLUDE_DIRS} 
	${DARKNET_PATH}/include
)

#add_library(${IPERF_LIB_NAME} SHARED IMPORTED)
#set_property(TARGET ${IPERF_LIB_NAME} PROPERTY IMPORTED_LOCATION "/usr/local/lib/libiperf.so")

add_library(${DARKNET_LIB_NAME} SHARED IMPORTED)
set_property(TARGET ${DARKNET_LIB_NAME} PROPERTY IMPORTED_LOCATION "${DARKNET_PATH}/libdarknet.so")

add_executable( server src/server.cpp )
add_executable( client_base src/client_base.cpp )
add_executable( client_thres src/client_thres.cpp )
add_executable( client_diff src/client_diff.cpp )
add_executable( detection src/detection.cpp )

target_link_libraries( server ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( client_base ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( client_thres ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( client_diff ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( detection ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )


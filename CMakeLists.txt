cmake_minimum_required(VERSION 2.8)
project( Thesis )
find_package( OpenCV 4 REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")
set(DARKNET_PATH darknet)
set(DARKNET_LIB_NAME libdark)
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
set_property(TARGET ${DARKNET_LIB_NAME} PROPERTY IMPORTED_LOCATION "${DARKNET_PATH}/libdark.so")

add_executable( server src/server.cpp )
add_executable( client src/client.cpp )
add_executable( server_ack src/server_ack.cpp )
add_executable( server_ack_3models src/server_ack_3models.cpp )
add_executable( client_ack src/client_ack.cpp )
add_executable( client_ack2 src/client_ack2.cpp )
add_executable( detection src/detection.cpp )
add_executable( test_client src/test.cpp)
add_executable( test_server src/test_s.cpp)


target_link_libraries( server ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( client ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( server_ack ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( server_ack_3models ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( client_ack ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( client_ack2 ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( detection ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )

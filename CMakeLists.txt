cmake_minimum_required(VERSION 2.8)
project( Thesis )
find_package( OpenCV 4 REQUIRED )
set(CMAKE_CXX_FLAGS "-std=c++11 -pthread")
set(DARKNET_PATH darknet)
set(DARKNET_LIB_NAME libdarknet)

include_directories( 
	${CMAKE_CURRENT_SOURCE_DIR}
	${OpenCV_INCLUDE_DIRS} 
	${DARKNET_PATH}/include
)

add_library(${DARKNET_LIB_NAME} SHARED IMPORTED)
set_property(TARGET ${DARKNET_LIB_NAME} PROPERTY IMPORTED_LOCATION "${DARKNET_PATH}/libdark.so")

add_executable( server src/server.cpp )
add_executable( client src/client.cpp )

target_link_libraries( server ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )
target_link_libraries( client ${OpenCV_LIBS} ${DARKNET_LIB_NAME} )

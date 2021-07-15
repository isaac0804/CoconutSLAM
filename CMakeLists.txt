cmake_minimum_required( VERSION 3.16.3 )
project( SLAM_C++ )

set( CMAKE_CXX_FLAGS "-std=c++11" )

find_package( OpenCV 3.4.14 REQUIRED ) 
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories( "/usr/include/eigen3/" )

add_executable( readImage readImage.cpp )
add_executable( stereoVision stereoVision.cpp )

target_link_libraries( readImage ${OpenCV_LIBS} )
target_link_libraries( stereoVision ${OpenCV_LIBS} )

cmake_minimum_required( VERSION 3.16.3 )
project( CoconutSLAM )

set( CMAKE_CXX_FLAGS "-std=c++14" )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package( OpenCV 3.4.14 REQUIRED ) 
find_package( Pangolin 0.4 REQUIRED )
find_package( G2O REQUIRED )
find_package( Eigen3 REQUIRED ) 
find_package( Cholmod )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Pangolin_INCLUDE_DIRS} )
include_directories( ${EIGEN3_INCLUDE_DIR} )
include_directories( ${CHOLMOD_INCLUDE_DIRS} )

add_executable( slam slam.cpp )
# add_executable( readImage readImage.cpp )
# add_executable( stereoVision stereoVision.cpp )


target_link_libraries( slam 
  ${OpenCV_LIBS} 
  ${Pangolin_LIBRARIES}
  g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
  ${CHOLMOD_LIBRARIES}
  )
# target_link_libraries( readImage ${OpenCV_LIBS} )
# target_link_libraries( stereoVision ${OpenCV_LIBS} )


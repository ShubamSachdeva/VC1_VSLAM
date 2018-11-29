cmake_minimum_required(VERSION 2.8.3)

project(simple_visual_odometry)

set( CMAKE_CXX_COMPILER "g++" )
#set( CMAKE_BUILD_TYPE "RelWithDebInfo" )
set( CMAKE_BUILD_TYPE "Debug" )
#set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -o2 -fopenmp" )
#set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 -fopenmp -Wall" )


find_package(OpenCV 2 REQUIRED)


#file(GLOB_RECURSE FIVE_POINT_SOURCE five-point-nister/*.cpp)
#add_library(five-point-nister SHARED ${FIVE_POINT_SOURCE})

#build the tracking node
file(GLOB_RECURSE VO_SOURCE 
          src/backend/*.cpp src/frontend/*.cpp src/logic/*.cpp)


add_executable(featuredetector test/featuredetector.cpp  )
add_executable(klt test/KLT.cpp  )



target_link_libraries(featuredetector ${OpenCV_LIBS} )
target_link_libraries(klt ${OpenCV_LIBS} )

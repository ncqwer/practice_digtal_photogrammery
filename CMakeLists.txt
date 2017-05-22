cmake_minimum_required(VERSION 2.8)

project(practice_digtal_photogrammery)


set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -pthread -std=c++11 -O0 -g -ggdb")

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV 
	     PATHS /usr/local/alternative/OpenCV/OpenCV2.0)


# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(dlib)
include_directories(${dlib_INCLUDE_DIRS})

include_directories(/usr/local/include/eigen3)

add_subdirectory(practice_digtal_photogrammery)
add_subdirectory(test)
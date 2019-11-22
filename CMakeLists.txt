cmake_minimum_required(VERSION 3.5)
project( img_location)
set(CMAKE_CXX_STANDARD 11)
# OpenCV
find_package( OpenCV 3.4 REQUIRED )
find_package(Ceres 1.14.0 REQUIRED)

#
include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
)
include_directories(${CERES_INCLUDE_DIRS})


add_executable( img_location main.cpp )
target_link_libraries( img_location ${OpenCV_LIBS} )

cmake_minimum_required(VERSION 3.24)
project(Charuco)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


add_executable( Charuco main.cpp Calibrator.cpp Calibrator.h Utils.cpp Utils.h)
target_link_libraries( Charuco ${OpenCV_LIBS} )


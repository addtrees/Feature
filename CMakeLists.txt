cmake_minimum_required(VERSION 3.5.1)
project(Feature)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 14)

add_executable(Feature main.cpp)
target_link_libraries(Feature ${OpenCV_LIBS})
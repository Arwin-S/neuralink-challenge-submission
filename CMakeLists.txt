cmake_minimum_required(VERSION 3.10)
project(NeuralinkCompressionChallege)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/libs/onnxruntime-linux-x64-1.18.0)

add_definitions(-DMODEL_DIR="${CMAKE_SOURCE_DIR}/models")
add_definitions(-DPROJECT_DIR="${CMAKE_SOURCE_DIR}")

include_directories(${CMAKE_SOURCE_DIR}/include) 
include_directories(${CMAKE_SOURCE_DIR}/libs)
include_directories(${ONNXRUNTIME_DIR}/include)

link_directories(${ONNXRUNTIME_DIR}/lib)


add_executable(encode encode.cpp)
add_executable(decode decode.cpp)

target_link_libraries(encode onnxruntime)
target_link_libraries(decode onnxruntime)




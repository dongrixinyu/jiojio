# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.0)

project(jiojio_cpp)  # 项目信息

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

message("Python_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("Python_LIBRARIES: ${PYTHON_LIBRARIES}")

include_directories(${PYTHON_INCLUDE_DIRS})

add_library(tagWordsConverter SHARED ${PROJECT_SOURCE_DIR}/tagWordsConverter.c)
add_library(featureExtractor SHARED ${PROJECT_SOURCE_DIR}/featureExtractor.c)

target_link_libraries(featureExtractor ${PYTHON_LIBRARIES})
target_link_libraries(tagWordsConverter ${PYTHON_LIBRARIES})

message("Successfully build `*.so` file.")

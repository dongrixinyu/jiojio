# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.0)

project(jiojio_cpp)  # 项目信息

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

message("Python_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("Python_LIBRARIES: ${PYTHON_LIBRARIES}")
message("numpy_INCLUDE_DIRS: ${NUMPY_INCLUDE_DIRS}")
message("numpy_LIBRARIES: ${NUMPY_LIBRARIES}")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS})

add_library(tagWordsConverter SHARED ${PROJECT_SOURCE_DIR}/tagWordsConverter.c)
add_library(cwsFeatureExtractor SHARED ${PROJECT_SOURCE_DIR}/cwsFeatureExtractor.c)
add_library(cwsFeatureToIndex SHARED ${PROJECT_SOURCE_DIR}/cwsFeatureToIndex.c)
add_library(posFeatureExtractor SHARED ${PROJECT_SOURCE_DIR}/posFeatureExtractor.c)
add_library(cwsCut SHARED ${PROJECT_SOURCE_DIR}/cwsCut.c ${PROJECT_SOURCE_DIR}/cwsFeatureToIndex.c ${PROJECT_SOURCE_DIR}/cwsFeatureExtractor.c)

target_link_libraries(tagWordsConverter ${PYTHON_LIBRARIES})
target_link_libraries(cwsFeatureExtractor ${PYTHON_LIBRARIES})
target_link_libraries(cwsFeatureToIndex ${PYTHON_LIBRARIES})
target_link_libraries(posFeatureExtractor ${PYTHON_LIBRARIES})
# target_link_libraries(cwsCut ${PYTHON_LIBRARIES} ${NUMPY_INCLUDE_DIRS})
target_link_libraries(cwsCut ${PYTHON_LIBRARIES})

message("Successfully build `*.so` file.")

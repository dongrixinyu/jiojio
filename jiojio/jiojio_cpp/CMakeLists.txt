# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.0)

project(jiojio_cpp)  # 项目信息

add_compile_options(-O3)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

message("Python_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message("Python_LIBRARIES: ${PYTHON_LIBRARIES}")
message("Numpy_INCLUDE_DIRS: ${NUMPY_INCLUDE_DIRS}")
message("Numpy_LIBRARIES_DIR: ${NUMPY_LIBRARIES_DIR}")
message("Numpy_LIBRARIES_FILENAME: ${NUMPY_LIBRARIES_FILENAME}")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIRS})
include_directories(/usr/include)

LINK_DIRECTORIES(${NUMPY_LIBRARIES_DIR})

add_library(tagWordsConverter SHARED ${PROJECT_SOURCE_DIR}/tagWordsConverter.c)
add_library(cwsFeatureExtractor SHARED ${PROJECT_SOURCE_DIR}/cwsFeatureExtractor.c)
add_library(cwsFeatureToIndex SHARED ${PROJECT_SOURCE_DIR}/cwsFeatureToIndex.c)
add_library(posFeatureExtractor SHARED ${PROJECT_SOURCE_DIR}/posFeatureExtractor.c)
add_library(cwsInterface SHARED
            ${PROJECT_SOURCE_DIR}/cwsInterface.c
            ${PROJECT_SOURCE_DIR}/cwsPrediction.c
            ${PROJECT_SOURCE_DIR}/wchar_t_hash_set.c
            ${PROJECT_SOURCE_DIR}/wchar_t_hash_dict.c)

target_link_libraries(tagWordsConverter ${PYTHON_LIBRARIES})
target_link_libraries(cwsFeatureExtractor ${PYTHON_LIBRARIES})
target_link_libraries(cwsFeatureToIndex ${PYTHON_LIBRARIES})
target_link_libraries(posFeatureExtractor ${PYTHON_LIBRARIES})
target_link_libraries(cwsInterface ${PYTHON_LIBRARIES} m -l:${NUMPY_LIBRARIES_FILENAME})

message("Successfully build `*.so` file.")

# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.0)

project(jiojio_cpp)  # 项目信息

set(CMAKE_BUILD_TYPE Release)

find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)
if(Python3_FOUND AND Python3_NumPy_FOUND)
    message("Python libs found: ${Python3_FOUND}, ${Python3_NumPy_FOUND}")
    message("PythonLibs_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}, ${Python3_NumPy_INCLUDE_DIRS}")
    include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
    # 指定生成可执行文件
    link_libraries(${Python3_LIBRARIES})
    # add_executable(inference inference.c)
    # add_executable(tagWordsConverter tagWordsConverter.c)
    add_library(tagWordsConverter SHARED ${PROJECT_SOURCE_DIR}/tagWordsConverter.c)
    add_library(featureExtractor SHARED ${PROJECT_SOURCE_DIR}/featureExtractor.c)
    # target_link_libraries(tagWordsConverter ${PYTHON_LIBRARIES})
else(Python3_FOUND AND Python3_NumPy_FOUND)
    message(FATAL_ERROR "Python3 NOT found")
endif(Python3_FOUND)

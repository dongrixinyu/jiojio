# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.0)

project(jiojio_cpp)  # 项目信息

# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_BUILD_TYPE Debug)

find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)
if(Python3_FOUND AND Python3_NumPy_FOUND)
    message("Python libs found: ${Python3_FOUND}") #, ${Python3_NumPy_FOUND}")
    message("Python_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}") #, ${Python3_NumPy_INCLUDE_DIRS}")
    message("Python_LIBRARIES: ${Python3_LIBRARIES}")
    # include_directories(${Python3_INCLUDE_DIRS}) # ${Python3_NumPy_INCLUDE_DIRS})
    set(Python3_INCLUDE_DIRS /usr/local/include/python3.11d)
    include_directories(${Python3_INCLUDE_DIRS})
    # 指定生成可执行文件
    # set(Python3_LIBRARIES /usr/local/lib/libpython3.11d.so.1.0)
    set(Python3_LIBRARIES /home/ubuntu/github/cpython/libpython3.11d.so)
    # link_directories(/home/ubuntu/anaconda3/lib/)
    # link_libraries(${Python3_LIBRARIES})
    # add_executable(featureExtractor featureExtractor.c)
    # add_library(featureEx SHARED ${PROJECT_SOURCE_DIR}/featureEx.c)
    add_library(tagWordsConverter SHARED ${PROJECT_SOURCE_DIR}/tagWordsConverter.c)
    add_library(featureExtractor SHARED ${PROJECT_SOURCE_DIR}/featureExtractor.c)
    target_link_libraries(featureExtractor ${Python3_LIBRARIES})
    # target_sources(featureExtractor ${Python3_INCLUDE_DIRS})
else(Python3_FOUND AND Python3_NumPy_FOUND)
    message(FATAL_ERROR "Python3 NOT found")
endif(Python3_FOUND AND Python3_NumPy_FOUND)

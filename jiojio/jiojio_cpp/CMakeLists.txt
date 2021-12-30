# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.0)

project(jiojio_cpp)  # 项目信息

set(CMAKE_BUILD_TYPE Release)

find_package(PythonLibs 3 REQUIRED)
if(PythonLibs_FOUND)
    message("PythonLibs_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
    include_directories(${PYTHON_INCLUDE_DIRS})
    # 指定生成可执行文件
    link_libraries(${PYTHON_LIBRARIES})
    # add_executable(inference inference.c)
    # add_executable(tagWordsConverter tagWordsConverter.c)
    add_library(tagWordsConverter SHARED ${PROJECT_SOURCE_DIR}/tagWordsConverter.c)
    # target_link_libraries(tagWordsConverter ${PYTHON_LIBRARIES})
else(PythonLibs_FOUND)
    message(FATAL_ERROR "PythonLibs NOT found")
endif(PythonLibs_FOUND)

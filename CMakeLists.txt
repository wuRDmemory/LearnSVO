cmake_minimum_required(VERSION 2.8.11)
project(LEARN_SVO)
set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/Ceres")

include_directories(/usr/include/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/ImageAlign)
include_directories(${PROJECT_SOURCE_DIR}/Utils)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message("-- Found Opencv ${OpenCV_VERSION}")
endif()

find_package(Ceres REQUIRED)
if (CERES_FOUND)
    include_directories(${CERES_INCLUDE_DIRS})
    message("-- Found CERES ${CERES_VERSION}")
endif()

find_package(Glog REQUIRED)
if (GLOG_FOUND)
    include_directories(${GLOG_INCLUDE_DIRS})
    message("-- Found Glog ${GLOG_VERSION}")
endif()

file(GLOB SOURCES 
        ${PROJECT_SOURCE_DIR}/ImageAlign/*.cc
        ${PROJECT_SOURCE_DIR}/Utils/*.cc)

add_executable(${PROJECT_NAME} main.cc ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CERES_LIBRARIES} ${GLOG_LIBRARIES})

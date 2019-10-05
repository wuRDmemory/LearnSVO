cmake_minimum_required(VERSION 2.8.11)
project(LEARN_SVO)
set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CmakeModules")

include_directories(/usr/include/eigen3)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/camera)

find_package(OpenCV REQUIRED)
find_package(Ceres  REQUIRED)
find_package(Glog   REQUIRED)
find_package(Sophus REQUIRED)

file(GLOB SOURCES 
        ${PROJECT_SOURCE_DIR}/src/*.cc
        ${PROJECT_SOURCE_DIR}/camera/*.cc)

list(APPEND LINK_LIBS
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES} 
    ${GLOG_LIBRARIES} 
    ${Sophus_LIBRARIES})

add_executable(${PROJECT_NAME} main.cc ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${LINK_LIBS})

add_executable(test_feature_align test/test_feature_align.cc src/alignment.cc)
target_link_libraries(test_feature_align ${LINK_LIBS})

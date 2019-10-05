#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

template <typename T>
Eigen::Matrix<T, 3, 3> symmetricMatrix(Eigen::Matrix<T, 3, 1> vec) {
    Eigen::Matrix<T, 3, 3> matrix;
    matrix <<       0, -vec(2),  vec(1), \
               vec(2),       0, -vec(0), \
              -vec(1),  vec(0),      0;
    return matrix;
}

template<typename T>
Eigen::Quaternion<T> deltaQuaternion(Eigen::Matrix<T, 3, 1> vec) {
    return Eigen::Quaternion<T>(1, vec[0]/2, vec[1]/2, vec[2]/2);
}
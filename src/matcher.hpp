#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

namespace mSVO {
    using namespace std;
    using namespace Eigen;
    using namespace cv;

    float calcuDepth(Matrix3f& Rcw, Vector3f& tcw, Vector3f& f1, Vector3f& f2);
    float calcuProjError(Matrix3f& R, Vector3f& t, Vector3f& point, Vector2f& xy, Vector2f& multi);
}
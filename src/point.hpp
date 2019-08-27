#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <glog/logging.h>

namespace mSVO {
    using namespace std;
    using namespace Eigen;
    using namespace cv;

    class LandMark{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        static float calcuDepth(Matrix3f& Rcw, Vector3f& tcw, Vector3f& f1, Vector3f& f2);
    }
}

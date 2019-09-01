#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <glog/logging.h>
#include "feature.hpp"

namespace mSVO {
    using namespace std;
    using namespace Eigen;
    using namespace cv;

    class LandMark{
    private:
        static int id;
        int mId;
        Vector3f mXYZ;
        vector<FeaturePtr> mFeatures;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        LandMark(Vector3f xyz);
        LandMark(Vector3f xyz, FeaturePtr feature);

        void addFeature(FeaturePtr feature);

        inline static void jacobian_xyz2uv(
            const Vector3f& bear,
            const Matrix3f& Rcw,
            Matrix<float, 2, 3>& point_jac) {
            const float z_inv = 1.0/bear[2];
            const float z_inv_sq = z_inv*z_inv;
            point_jac(0, 0) = z_inv;
            point_jac(0, 1) = 0.0;
            point_jac(0, 2) = -bear[0] * z_inv_sq;
            point_jac(1, 0) = 0.0;
            point_jac(1, 1) = z_inv;
            point_jac(1, 2) = -bear[1] * z_inv_sq;
            point_jac = -point_jac * Rcw;
        }  
    };

    typedef LandMark* LandMarkPtr;
}

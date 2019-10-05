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

        enum LANDMARK_TYPR {
            DELETE = 0,
            CANDIDATE,
            UNKNOWN,
            GOOD,
        };

        LandMark(Vector3f xyz);
        LandMark(Vector3f xyz, FeaturePtr feature);

        void addFeature(FeaturePtr feature);
        void findClosestObs(Vector3f& framePose, FeaturePtr feature) const;
        bool optimize(int nIter);

        inline Vector3f& xyz() { return mXYZ; }

        inline static void jacobian_uv2xyz(
            const Vector3f& bear, const Matrix3f& Rcw, Matrix<float, 2, 3>& point_jac) {
            const float z_inv = 1.0/bear[2];
            const float z_inv_sq = z_inv*z_inv;
            point_jac << z_inv, 0.0, -bear[0]*z_inv_sq, 0.0, z_inv, -bear[1]*z_inv_sq;
            point_jac = -point_jac * Rcw;
        }  

    public:
        int nOptimizeFrameId;
        int nProjectFrameFailed;
        int nProjectFrameSuccess;
        LANDMARK_TYPR type;
    };

    typedef LandMark* LandMarkPtr;
}

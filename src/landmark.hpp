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
    };

    typedef LandMark* LandMarkPtr;
}

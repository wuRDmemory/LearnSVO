#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "frame.hpp"
#include "map.hpp"
#include "feature.hpp"


namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    class PoseOptimize {
    private:
        static Quaternionf  mRcw, mOldRcw;
        static Vector3f     mtcw, mOldtcw;

    public:
        static bool optimize(FramePtr frame, int nIter, float projError, float& scale, float& initChi2, float& endChi2, int& obsnum);

    private:
        static float TukeyWeightFunction(float x);
        static float middleScaleEstimate(vector<float>& errors);
        static bool  updateModel(Matrix<float, 6, 1>& dx);
    };
}

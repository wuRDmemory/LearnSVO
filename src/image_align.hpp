#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <ceres/ceres.h>

#include "frame.hpp"

namespace mSVO {
    using namespace std;
    using namespace Eigen;
    using namespace cv;

    class ImageAlign {
    private:
        int mIterCnt;
        static const int halfPatchSize = 2;
        static const int patchSize = halfPatchSize*2;
        

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        ImageAlign(int minlevel, int maxlevel, int iterCnt);
        ~ImageAlign();

        void run(FramePtr refFrame, FramePtr curFrame);
    };
}


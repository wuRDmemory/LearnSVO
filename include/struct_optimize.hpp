#pragma once

#include <deque>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "landmark.hpp"
#include "frame.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    class StructOptimize {
    public:
        static bool optimize(FramePtr frame, const int nIter, const int maxPoints);
    };
}

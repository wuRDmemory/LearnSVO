#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <vector>
#include <algorithm>

#include "map.hpp"
#include "frame.hpp"

namespace mSVO {
    class FeatureAlign {
    private:
        static int halfPatchSize = 4;
        static int patchSize     = 8;
        static int patchArea     = 64;

        MapPtr mMap;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        FeatureAlign(MapPtr map);
        ~FeatureAlign();

        bool reproject(FramePtr curFrame);
    };
}


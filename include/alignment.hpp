#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    class Alignment {
    public:
        static float interpolateU8(cv::Mat& image, Vector2f& px);
        static bool align2D(cv::Mat& image, Vector2f& px, int iterCnt, int patchSize,
                            uint8_t* patch, uint8_t* patchWithBorder);
    };
}

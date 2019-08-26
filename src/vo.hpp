#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>
#include "frame.hpp"
#include "initialization.hpp"
#include "utils.hpp"
#include "camera.hpp"
#include "pinhole.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;
    using namespace mvk;

    class VO {
    enum UPDATE_LEVEL{
        UPDATE_FIRST  = 0, 
        UPDATE_SECOND = 1, 
        UPDATE_FRAME  = 2,
        UPDATE_NO_FRAME = 3,
    };

    private:
        FramePtr mNewFrame, mRefFrame;
        CameraModelPtr mCameraModel;
        KltHomographyInit mInitialor;
        UPDATE_LEVEL updateLevel;

    public:
        VO(const string config_file);
        ~VO();

        UPDATE_LEVEL addNewFrame(const cv::Mat& image, const double timestamp);
        UPDATE_LEVEL processFirstFrame();
        UPDATE_LEVEL processSencondFrame();
    };
}



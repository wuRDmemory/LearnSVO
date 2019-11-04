#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <glog/logging.h>

#include "frame.hpp"
#include "initialization.hpp"
#include "config.hpp"
#include "camera.hpp"
#include "pinhole.hpp"
#include "map.hpp"
#include "image_align.hpp"
#include "feature_align.hpp"
#include "pose_optimize.hpp"
#include "struct_optimize.hpp"
#include "depth_filter.hpp"
#include "bundle_adjustment.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;
    using namespace mvk;

    enum UPDATE_LEVEL{
        UPDATE_FIRST  = 0, 
        UPDATE_SECOND = 1, 
        UPDATE_FRAME  = 2,
        UPDATE_NO_KEYFRAME = 3,
        UPDATE_KEYFRAME = 4,
        UPDATE_RELOCAL = 5,
    };

    enum PROCESS_STATE {
        PROCESS_SUCCESS = 0,
        PROCESS_FAIL    = 1,
    };

    class VO {
    private:
        MapPtr          mLocalMap;
        FramePtr        mNewFrame, mRefFrame;
        CameraModelPtr  mCameraModel;
        DepthFilterPtr  mDepthFilter;
        FeatureAlignPtr mFeatureAlign;
        KltHomographyInitPtr mInitialor;
        BundleAdjustmentPtr  mBundleAdjust;

        UPDATE_LEVEL updateLevel;
        int mOldTrackCnt;

    public:
        VO(const string config_file);
        ~VO();

        void setup();
        void addNewFrame(const cv::Mat& image, const double timestamp);
        
    private:
        PROCESS_STATE processFirstFrame();
        PROCESS_STATE processSencondFrame();
        PROCESS_STATE processFrame();
        
        bool needKeyFrame(int trackCnt, FramePtr frame);
        void finishProcess(int frameid, PROCESS_STATE res);

    };
}



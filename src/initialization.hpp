#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
// #include <opencv/cxeigen.hpp>
#include <glog/logging.h>
#include "point.hpp"
#include "frame.hpp"
#include "feature.hpp"
#include "utils.hpp"
#include "matcher.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    enum InitResult {
        FAILURE, 
        NO_KEYFRAME, 
        SUCCESS 
    };

    class KltHomographyInit {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        KltHomographyInit();
        ~KltHomographyInit();
        InitResult addFirstFrame(FramePtr frameRef);
        InitResult addSecondFrame(FramePtr frameRef);
        void reset();

    protected:
        bool detectCorner(FramePtr frame, vector<cv::Point2f>& points);
        int computeInliers(Matrix3f& R21, Vector3f& t21, mvk::CameraModel* camera, vector<Vector3f>& pts1, vector<Vector3f>& pts2, \
                        vector<uchar>& inliers, vector<float>& depth, float th);

    private:
        int mGridCell, mImWidth, mImHeight, mCellFtrNumber, mFtrNumber;
        vector<cv::Rect> mGridCellRoi;
        vector<cv::Point2f> mFirstCorners;
        FramePtr mRefFrame;
    };
}


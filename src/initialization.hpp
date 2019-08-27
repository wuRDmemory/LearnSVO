#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv/cxeigen.hpp>
#include <glog/logging.h>
#include "point.hpp"
#include "frame.hpp"

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
        int computeInliers(cv::Mat& R, cv::Mat& t, cv::Mat& K, vector<cv::Point2f>& pts1, vector<cv::Point2f>& pts2, 
                        vector<uchar>& inliers, vector<float>& depth, float th);

    private:
        int mGridCell, mImWidth, mImHeight, mCellFtrNumber, mFtrNumber;
        vector<cv::Rect> mGridCellRoi;
        vector<cv::Point2f> mFirstCorners;
        FramePtr mRefFrame;
    };
}


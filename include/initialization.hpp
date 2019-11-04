#pragma once

#include <iostream>
#include <algorithm>
#include <set>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv/cxeigen.hpp>
#include <glog/logging.h>

#include "landmark.hpp"
#include "frame.hpp"
#include "feature.hpp"
#include "config.hpp"
#include "matcher.hpp"
#include "detector.hpp"

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
    private:
        int mImWidth, mImHeight, mCellFtrNumber, mFtrNumber;
        DetectorPtr mCornerDetector;
        vector<Point2f> mFirstCorners;
        vector<int>     mFirstCornersLevel;
        FramePtr mRefFrame;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        KltHomographyInit();
        ~KltHomographyInit();
        InitResult addFirstFrame(FramePtr frameRef);
        InitResult addSecondFrame(FramePtr frameRef);
        void reset();

    protected:
        bool detectCorner(FramePtr frame, vector<cv::Point2f>& points);
        float findFundamental(vector<Point2f> &refPoints, vector<Point2f> &curPoints, vector<bool>& inliers, cv::Mat &F21);
        float checkFundamental(const Mat &F21, vector<Point2f> &refPoints, vector<Point2f> &curPoints, vector<bool> &vbMatchesInliers, float sigma);
        Mat computeF21(const vector<Point2f> &vP1, const vector<Point2f> &vP2);
        void normalize(vector<Point2f> &vKeys, vector<Point2f> &vNormalizedPoints, Mat &T);
        void decomposeE(Mat &E, Mat &R1, Mat &R2, Mat &t);
        int computeInliers(Mat& R21, Mat& t21, const Mat& K, vector<Point2f>& pts1, vector<Point2f>& pts2, \
                        vector<uchar>& inliers, vector<float>& depth, float th);
        int testMatch(Mat& refImage, Mat& curImage, vector<cv::Point2f>& refPoints, vector<cv::Point2f>& curPoints);
        int testDepth(FramePtr curFrame);
    };

    typedef KltHomographyInit* KltHomographyInitPtr;
}


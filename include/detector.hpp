#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include "frame.hpp"
#include "feature.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace mSVO {
    struct CellElem {
        CellElem(): score(-1), level(0) {}
        Point2f uv;
        int   level;
        float score;
    };

    class Detector {
    private:
        int mStep, mRows, mCols, mWidth, mHeight, mLevels;
        float mThreshold;
        Mat mMask;
        Ptr<FastFeatureDetector> mDetector;
        vector<bool>     mOccupied;
        vector<CellElem> mGridCell;
        vector<Rect>     mGridCellRoi;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Detector(int width, int height, int step, int levels, float threshold);
        bool detect(Frame* frame, vector<Point2f>& keyPoints, vector<int>& pointsLevel);
        bool setMask(const Vector2f& uv);
        bool setMask(const Features& obs);
        bool reset();

    private:
        float calculateScore(const Mat& img, int u, int v);
    };

    typedef Detector* DetectorPtr;
}

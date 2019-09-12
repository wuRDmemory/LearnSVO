#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <glog/logging.h>

#include "time.h"
#include "frame.hpp"
#include "feature.hpp"
#include "landmark.hpp"

namespace mSVO {
    using namespace std;
    using namespace Eigen;
    using namespace cv;

    class ImageAlign {
    private:
        int mInliersCnt;
        int mIterCnt, mMinLevel, mMaxLevel, mMaxFailThr;
        static const int halfPatchSize = 2;
        static const int patchSize = halfPatchSize*2;
        static const int patchArea = patchSize*patchSize;
        float meps, chi2;

        vector<bool> mVisables;
        MatrixXf mRefPatchCache;
        MatrixXf mJacobianCache;

        Matrix<float, 6, 6> mH;
        Matrix<float, 6, 1> mb;
        Matrix<float, 6, 1> mDeltaX;

        Sophus::SE3 mTc_r_new;
        Sophus::SE3 mTc_r_old;

        cv::Mat mRefImage;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        ImageAlign(int minlevel, int maxlevel, int iterCnt);
        ~ImageAlign();

        void run(FramePtr refFrame, FramePtr curFrame);
    
    protected:
        void prepareData(FramePtr refFrame, int level);
        void optimize(FramePtr refFrame, FramePtr curFrame, int level);
        
        bool solve();
        bool reset();
        bool update(Sophus::SE3& new_model, Sophus::SE3& old_model);
        float maxLimit(Matrix<float, 6, 1>& x);
        float computeError(FramePtr refFrame, FramePtr curFrame, int level, bool linearSystem, bool useWeight);
        float weightFunction(float res);
    };
}


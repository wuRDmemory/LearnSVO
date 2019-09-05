#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <ceres/ceres.h>

#include "frame.hpp"
#include "feature.hpp"

namespace mSVO {
    using namespace std;
    using namespace Eigen;
    using namespace cv;

    class ImageAlign {
    private:
        int mIterCnt, mMinLevel, mMaxLevel;
        static const int halfPatchSize = 2;
        static const int patchSize = halfPatchSize*2;
        static const int patchArea = patchSize*patchSize;
        
        vector<bool> mVisables;
        MatrixXf mRefPatchCache;
        MatrixXf mJacobianCache;

        Matrix<float, 6, 6> mH;
        Matrix<float, 6, 1> mb;

        Sophus::SE3 mTc_r_new;
        Sophus::SE3 mTc_r_old;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        ImageAlign(int minlevel, int maxlevel, int iterCnt);
        ~ImageAlign();

        void run(FramePtr refFrame, FramePtr curFrame);
    
    protected:
        void prepareData(FramePtr refFrame, int level);
        void optimize(FramePtr refFrame, FramePtr curFrame, int level);
        
        bool solve(Sophus::SE3& new_model, Sophus::SE3& old_model);
        float computeError(FramePtr refFrame, FramePtr curFrame, int level, bool linearSystem, bool useWeight);
        float weightFunction(float res);
    };
}


#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <glog/logging.h>

#include "config.hpp"
#include "frame.hpp"
#include "landmark.hpp"
#include "alignment.hpp"

namespace mSVO {
    using namespace std;
    using namespace Eigen;
    using namespace cv;

    float calcuDepth(Matrix3f& Rcw, Vector3f& tcw, Vector3f& f1, Vector3f& f2);
    float calcuProjError(Matrix3f& R, Vector3f& t, Vector3f& point, Vector3f& xyz);

    class Matcher {
    private:
        static int patchSize;
        static int patchArea;
        static int halfPatchSize;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Matcher() {};
        ~Matcher() {};
        
        // findDirectMatch
        int  getBestSearchLevel(const Matrix2f& Acr, const int max_level);
        bool findDirectMatch(FramePtr curFrame, LandMarkPtr landmark, Vector2f& px);
        bool findEpipolarMatch(FramePtr curFrame, FeaturePtr feature, Quaternionf& Rcr, Vector3f& tcr, float minDepth, float maxDepth, float z);

    private:
        bool getWarpAffineMatrix(FramePtr& refFrame, FramePtr& curFrame, Quaternionf& Rcr, Vector3f& tcr, Vector3f& refDirect, float depth, int level, Matrix2f& warpMatrix);
        bool warpAffine(cv::Mat& image, Vector2f& px, Matrix2f& Acr, int level, int searchLevel, int halfPatchSize, uint8_t* patchPtr);
        bool createPatchFromBorderPatch(uint8_t* patch, uint8_t* patchWithBorder, int patchSize);

    };

    typedef Matcher* MatcherPtr;
}
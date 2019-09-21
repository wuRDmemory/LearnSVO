#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <glog/logging.h>

#include "utils.hpp"
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
        static int halfPatchSize = 4;
        static int patchSize     = 8;
        static int patchArea     = 64;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Matcher() {};
        ~Matcher() {};
        
        // findDirectMatch
        int  getBestSearchLevel(const Matrix2d& Acr, const int max_level);
        bool findDirectMatch(FramePtr curFrame, LandMarkPtr landmark, Vector2f& px);
        bool getWarpAffineMatrix(FramePtr refFrame, FramePtr curFrame, Vector3f& refDirect, float depth, Matrix2f& warpMatrix);
        bool warpAffine(cv::Mat& image, Vector2f& px, Matrix2f& Acr, int level, int searchLevel, int halfPatchSize, uint8_t* patchPtr);
        bool createPatchFromBorderPatch(uint8_t* patch, const uint8_t* const patchWithBorder, const int patchSize);

    };

    typedef Matcher* MatcherPtr;
}
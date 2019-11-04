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

    bool  calcu3DPoint(Mat& Rcw, Mat& tcw, Mat& p1, Mat& p2, Mat& point);
    float calcuDepth(Matrix3f& Rcw, Vector3f& tcw, Vector3f& f1, Vector3f& f2);
    float calcuProjError(Mat& R, Mat& t, Mat& K, Point3f& point, Point2f uv);

    class Matcher {
    private:
        static int patchSize;
        static int patchArea;
        static int halfPatchSize;

        Vector2f mCuv;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        Matcher() {};
        ~Matcher() {};
        
        Vector2f& getUV() { return mCuv; }
        // findDirectMatch
        int  getBestSearchLevel(const Matrix2f& Acr, const int max_level);
        bool findDirectMatch(Frame* curFrame, LandMarkPtr landmark, Vector2f& px);
        bool findEpipolarMatch(Frame* curFrame, FeaturePtr feature, Quaternionf& Rcr, Vector3f& tcr, float minDepth, float depth, float maxDepth, float& z);

    private:
        bool getWarpAffineMatrix(Frame* refFrame, Frame* curFrame, Quaternionf& Rcr, Vector3f& tcr, Vector3f& refDirect, float depth, int level, Matrix2f& warpMatrix);
        bool warpAffine(cv::Mat& image, Vector2f& px, Matrix2f& Acr, int level, int searchLevel, int halfPatchSize, uint8_t* patchPtr);
        bool createPatchFromBorderPatch(uint8_t* patch, uint8_t* patchWithBorder, int patchSize);
        bool createPatch(cv::Mat& image, Vector2i& uv, uint8_t* patchSize, int patchArea);
        float NCCScore(uint8_t* refPatch,  uint8_t* curPatch, int patchArea) const;

    };

    typedef Matcher* MatcherPtr;
}
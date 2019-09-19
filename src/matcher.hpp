#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "frame.hpp"
#include "landmark.hpp"

namespace mSVO {
    using namespace std;
    using namespace Eigen;
    using namespace cv;

    float calcuDepth(Matrix3f& Rcw, Vector3f& tcw, Vector3f& f1, Vector3f& f2);
    float calcuProjError(Matrix3f& R, Vector3f& t, Vector3f& point, Vector3f& xyz);

    class Matcher {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        Matcher() {};
        ~Matcher() {};
        
        // findDirectMatch
        bool findDirectMatch(FramePtr curFrame, LandMarkPtr landmark, Vector2f& px);
    };

    typedef Matcher* MatcherPtr;
}
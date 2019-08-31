#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;
    
    enum FeatureType {
        CORNER,
        EDGELET
    };
    class LandMark;
    class Frame;

    class Feature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        int mLevel;            //!< Image pyramid level where feature was extracted.
        Frame* mFrame;         //!< Pointer to frame in which the feature was detected.
        FeatureType mType;     //!< Type can be corner or edgelet.
        Vector2f mPx;          //!< Coordinates in pixels on pyramid level 0.
        Vector3f mDirect;      //!< Unit-bearing vector of the feature.
        LandMark* mLandmark;   //!< Pointer to 3D point which corresponds to the feature.

        Feature(Frame* frame, const Vector2f& px, int level=1) :
            mType(CORNER), mFrame(frame), mPx(px),
            mDirect(frame->camera()->cam2world(px)),
            mLevel(level), mLandmark(NULL)
        {}

        Feature(Frame* frame, const Vector2f& px, const Vector3f& direct, int level) :
            mType(CORNER), mFrame(frame), mPx(px),
            mDirect(direct), mLevel(level), mLandmark(NULL)
        {}

        Feature(Frame* frame, Landmark* ldmk, const Vector2f& px, 
                const Vector3f& direct, int level) :
            mType(CORNER), mFrame(frame), mPx(px),
            mDirect(direct), mLevel(level), mLandmark(ldmk)
        {}
    };

    typedef Feature* FeaturePtr;
}

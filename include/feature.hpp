#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "frame.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    class LandMark;

    class Feature {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        enum FeatureType {
            CORNER,
            EDGELET,
        };

        int mLevel;            //!< Image pyramid level where feature was extracted.
        Frame* mFrame;         //!< Pointer to frame in which the feature was detected.
        FeatureType mType;     //!< Type can be corner or edgelet.
        Vector2f mPx;          //!< Coordinates in pixels on pyramid level 0.
        Vector3f mDirect;      //!< Unit-bearing vector of the feature.
        LandMark* mLandmark;   //!< Pointer to 3D point which corresponds to the feature.

        Feature(Frame* frame, const Vector2f& px, int level=0) :
            mType(CORNER), mFrame(frame), mPx(px),
            mDirect(frame->camera()->cam2world(px)),
            mLevel(level), mLandmark(NULL)
        {
            mDirect.normalize();
        }

        Feature(Frame* frame, const Vector2f& px, const Vector3f& direct, int level) :
            mType(CORNER), mFrame(frame), mPx(px),
            mDirect(direct), mLevel(level), mLandmark(NULL)
        {
            mDirect.normalize();
        }

        Feature(Frame* frame, LandMark* ldmk, const Vector2f& px, 
                const Vector3f& direct, int level) :
            mType(CORNER), mFrame(frame), mPx(px),
            mDirect(direct), mLevel(level), mLandmark(ldmk)
        {
            mDirect.normalize();
        }
    };

    typedef Feature* FeaturePtr;
    typedef list<FeaturePtr> Features;
}

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "Frame.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;
    
    class Feature {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        enum FeatureType {
            CORNER,
            EDGELET
        };

        FeatureType type;     //!< Type can be corner or edgelet.
        Frame* frame;         //!< Pointer to frame in which the feature was detected.
        Vector2d px;          //!< Coordinates in pixels on pyramid level 0.
        Vector3d f;           //!< Unit-bearing vector of the feature.
        int level;            //!< Image pyramid level where feature was extracted.
        Point* point;         //!< Pointer to 3D point which corresponds to the feature.
        Vector2d grad;        //!< Dominant gradient direction for edglets, normalized.

        Feature(Frame* _frame, const Vector2d& _px, int _level) :
            type(CORNER),
            frame(_frame),
            px(_px),
            f(frame->cam_->cam2world(px)),
            level(_level),
            point(NULL),
            grad(1.0,0.0)
        {}

        Feature(Frame* _frame, const Vector2d& _px, const Vector3d& _f, int _level) :
            type(CORNER),
            frame(_frame),
            px(_px),
            f(_f),
            level(_level),
            point(NULL),
            grad(1.0,0.0)
        {}

        Feature(Frame* _frame, Point* _point, const Vector2d& _px, const Vector3d& _f, int _level) :
            type(CORNER),
            frame(_frame),
            px(_px),
            f(_f),
            level(_level),
            point(_point),
            grad(1.0,0.0)
        {}
    }
}

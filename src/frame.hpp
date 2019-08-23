#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include "sophus/se3.h"
#include "sophus/so3.h"
#include "camera.hpp"

using namespace std;
using namespace mvk;

namespace mSVO {
    class Feature;
    typedef list<Feature*> Features;
    typedef vector<cv::Mat> ImagePyr;

    class Frame {
    public:
        static int fID;
        int mID;
        bool isKeyFrame;
        double mTimestamp;

        CameraModel* mCammera;
        Sophus::SE3 mTwc;

        Features mObs;
        ImagePyr mImagePyr;
        

    public:
        Frame(double timestamp, CameraModel* camera, cv::Mat& img);
        ~Frame();

        void initFrame(const cv::Mat& img);
        void addFeature(Feature* feature);

        inline void setKeyFrame() { isKeyFrame = true; }
        inline bool getKeyFrame() { return isKeyFrame; }
        inline int getObs() { return mObs.size(); }
        inline Eigen::Vector3f getPose() { return mTwc.translation().cast<float>(); }

        inline static void jacobian_xyz2uv(Vector3d& xyz_in_f, Matrix<double,2,6>& J) {
            const double x = xyz_in_f[0];
            const double y = xyz_in_f[1];
            const double z_inv = 1./xyz_in_f[2];
            const double z_inv_2 = z_inv*z_inv;

            J(0,0) = -z_inv;              // -1/z
            J(0,1) = 0.0;                 // 0
            J(0,2) = x*z_inv_2;           // x/z^2
            J(0,3) = y*J(0,2);            // x*y/z^2
            J(0,4) = -(1.0 + x*J(0,2));   // -(1.0 + x^2/z^2)
            J(0,5) = y*z_inv;             // y/z

            J(1,0) = 0.0;                 // 0
            J(1,1) = -z_inv;              // -1/z
            J(1,2) = y*z_inv_2;           // y/z^2
            J(1,3) = 1.0 + y*J(1,2);      // 1.0 + y^2/z^2
            J(1,4) = -J(0,3);             // -x*y/z^2
            J(1,5) = -x*z_inv;            // x/z
        }
    };

    typedef FramePtr std::shared_ptr<Frame*>;
}

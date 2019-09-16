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
    private:
        static int id;
        int mID;
        bool mIsKeyFrame;
        double mTimestamp;

        CameraModel* mCamera;
        Sophus::SE3  mTwc, mTcw;

        Features mObs;
        ImagePyr mImagePyr;

    public:
        Frame(double timestamp, CameraModel* camera, const cv::Mat& img);
        ~Frame();

        void initFrame(const cv::Mat& img);
        void addFeature(Feature* feature);
        bool isVisible(const Vector3f& xyz);
        bool isVisible(const Vector2f& uv, int border);

        Vector2f world2camera(const Vector3d& XYZ);

        inline void setKeyFrame() { mIsKeyFrame = true; }
        inline bool isKeyFrame() { return mIsKeyFrame; }
        
        inline double       timestamp() { return mTimestamp; }
        inline Features&    obs()       { return mObs; }
        inline Sophus::SE3& pose()      { return mTwc; }
        inline CameraModel* camera()    { return mCamera; }
        inline ImagePyr&    imagePyr()  { return mImagePyr; }

        inline static void jacobian_uv2se3(Vector3f& xyz_in_f, Matrix<float,2,6>& J) {
            const float x = xyz_in_f[0];
            const float y = xyz_in_f[1];
            const float z_inv = 1.0/xyz_in_f[2];
            const float z_inv_2 = z_inv*z_inv;

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

    typedef Frame* FramePtr;
}

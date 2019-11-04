#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include "camera.hpp"
#include "math_utils.hpp"

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
        Quaternionf  mRwc;
        Vector3f     mtwc;

        Features mObs;
        ImagePyr mImagePyr;

        Matrix<float, 6, 6> mCovariance;

    public:
        Frame(double timestamp, CameraModel* camera, const cv::Mat& img);
        ~Frame();

        void initFrame(const cv::Mat& img);
        void addFeature(Feature* feature);
        bool isVisible(const Vector3f& xyz);
        bool isVisible(const Vector2f& uv, int border, int level=0);

        Vector2f world2uv(const Vector3f& XYZ);
        Vector3f world2camera(const Vector3f& XYZ);

        inline void setKeyFrame() { mIsKeyFrame = true; }
        inline bool isKeyFrame() { return mIsKeyFrame; }
        
        inline int          ID()        { return mID;  }
        inline Features&    obs()       { return mObs; }
        inline Quaternionf& Rwc()       { return mRwc; }
        inline Vector3f&    twc()       { return mtwc; }
        inline double       timestamp() { return mTimestamp; }
        inline CameraModel* camera()    { return mCamera; }
        inline ImagePyr&    imagePyr()  { return mImagePyr; }
        inline Matrix<float, 6, 6>&   covariance()  { return mCovariance; }

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

        inline static void jacobian_uv2se3New(Vector3f& xyz_c, Vector3f rxyz, Matrix<float,2,6>& J) {
            const float x = xyz_c(0);
            const float y = xyz_c(1);
            const float z_inv  = 1.0/xyz_c(2);
            const float z_inv2 = z_inv*z_inv;

            Matrix<float, 2, 3> reduce;
            reduce << -z_inv, 0, x*z_inv2, 0, -z_inv, y*z_inv2;

            Matrix<float, 3, 6> j_pc_se3;
            j_pc_se3.leftCols(3).setIdentity();
            j_pc_se3.rightCols(3) = -symmetricMatrix(rxyz);

            J = reduce*j_pc_se3;
        }
    };

    typedef shared_ptr<Frame> FramePtr;
}

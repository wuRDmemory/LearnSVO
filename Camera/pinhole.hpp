#pragma once

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "camera.hpp"

namespace mvk {
    using namespace std;
    using namespace Eigen;

    class PinholeCamera: public CameraModel {
        protected:
            float mfx, mfy, mcx, mcy;
            float md0, md1, md2, md3, md4;

            Eigen::Matrix3f mK, mKInv;
            Eigen::Matrix<float, 1, 5> mD;

            cv::Mat mCVK, mCVD;

            bool mUseDistort;        
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            PinholeCamera(int width, int height, float fx, float fy, float cx, float cy, 
                            float d0=0, float d1=0, float d2=0, float d3=0, float d4=0);
            virtual ~PinholeCamera();
            virtual Eigen::Vector3f cam2world(const float& x, const float& y) const;
            virtual Eigen::Vector3f cam2world(const Eigen::Vector2f& px) const;
            virtual Eigen::Vector2f world2cam(const Eigen::Vector3f& xyz_c) const;
            virtual Eigen::Vector2f world2cam(const Eigen::Vector2f& uv) const;

            const Eigen::Vector2f focal_length() const { return Vector2f(mfx, mfy);}
            virtual float errorMultiplier2() const { return fabs(mfx);}
            virtual float errorMultiplier() const { return fabs(4.0*mfx*mfy); }
            inline const Eigen::Matrix3f& K() { return mK; }
            inline const Eigen::Matrix3f& K_inv() { return mKInv; }
            inline float fx() { return mfx; };
            inline float fy() { return mfy; };
            inline float cx() { return mcx; };
            inline float cy() { return mcy; };
            inline float d0() { return md0; };
            inline float d1() { return md1; };
            inline float d2() { return md2; };
            inline float d3() { return md3; };
            inline float d4() { return md4; };

            int initUnistortionMap();
            int undistortImage(const cv::Mat& raw, cv::Mat& rectified);
    };

}

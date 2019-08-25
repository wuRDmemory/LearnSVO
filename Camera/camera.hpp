#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

namespace mvk {
    using namespace std;
    using namespace Eigen;

    class CameraModel {
        protected:
            int mWidth, mHeight;
        
        public:
            CameraModel() {}
            CameraModel(int width, int height): mWidth(width), mHeight(height) {}

            virtual ~CameraModel() {}

            virtual Vector3f
            cam2world(const float& x, const float& y) const = 0;

            /// Project from pixels to world coordiantes. Returns a bearing vector of unit length.
            virtual Vector3f
            cam2world(const Vector2f& px) const = 0;

            virtual Vector2f
            world2cam(const Vector3f& xyz_c) const = 0;

            /// projects unit plane coordinates to camera coordinates
            virtual Vector2f
            world2cam(const Vector2f& uv) const = 0;

            virtual float
            errorMultiplier2() const = 0;

            virtual float
            errorMultiplier() const = 0;

            virtual Vector2f
            project2d(const Vector3f& xyz) const {
                return xyz.head<2>()/xyz(2);
            }

            inline int width() const { return mWidth; }
            inline int height() const { return mHeight; }
    };


    typedef CameraModel* CameraModelPtr;
}

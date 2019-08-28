#include "pinhole.hpp"

namespace mvk {
    PinholeCamera::PinholeCamera(int width, int height, 
                    float fx, float fy, float cx, float cy, 
                    float d0, float d1, float d2, float d3, float d4): 
                    CameraModel(width, height), 
                    mfx(fx), mfy(fy), mcx(cx), mcy(cy),
                    md0(d0), md1(d1), md2(d2), md3(d3), md4(d4), 
                    mUseDistort(d0>1e-5) {
        mCVK = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        mCVD = (cv::Mat_<float>(1, 5) << d0, d1, d2, d3, d4);

        mK << fx, 0, cx, 0, fy, cy, 0, 0, 1;
        mKInv = mK.inverse();
        mD << d0, d1, d2, d3, d4;
    }

    PinholeCamera::PinholeCamera(const string config_file) {
        cv::FileStorage file(config_file, cv::FileStorage::READ);
        if (!file.isOpened()) {
            throw std::io_errc();
        }

        int width  = static_cast<float>(file["image_width"]);
        int height = static_cast<float>(file["image_height"]);
        this->setSize(width, height);

        cv::FileNode node = file["instrinsc"];
        { // instrinsc
            float fx = static_cast<float>(node["fx"]);
            float fy = static_cast<float>(node["fy"]);
            float cx = static_cast<float>(node["cx"]);
            float cy = static_cast<float>(node["cy"]);

            float d0 = static_cast<float>(node["d0"]);
            float d1 = static_cast<float>(node["d1"]);
            float d2 = static_cast<float>(node["d2"]);
            float d3 = static_cast<float>(node["d3"]);
            float d4 = static_cast<float>(node["d4"]);

            mCVK = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
            mCVD = (cv::Mat_<float>(1, 5) << d0, d1, d2, d3, d4);

            mK << fx, 0, cx, 0, fy, cy, 0, 0, 1;
            mD << d0, d1, d2, d3, d4;

            mUseDistort = d0>1e-5;
        }

        mKInv = mK.inverse();
    }

    PinholeCamera::~PinholeCamera() {
        ;
    }

    Eigen::Vector3f PinholeCamera::cam2world(const float& x, const float& y) const { 
        Eigen::Vector3f xyz;
        if(!mUseDistort) {
            xyz[0] = (x - mcx)/mfx;
            xyz[1] = (y - mcy)/mfy;
            xyz[2] = 1.0;
        } else {
            cv::Point2f uv(x, y), px;
            const cv::Mat src_pt(1, 1, CV_32FC2, &uv.x);
            cv::Mat dst_pt(1, 1, CV_32FC2, &px.x);
            cv::undistortPoints(src_pt, dst_pt, mCVK, mCVD);
            xyz[0] = px.x;
            xyz[1] = px.y;
            xyz[2] = 1.0;
        }
        return xyz.normalized();
    }

    Vector3f PinholeCamera::
    cam2world (const Vector2f& uv) const {
        return cam2world(uv[0], uv[1]);
    }

    Vector2f PinholeCamera::
    world2cam(const Vector2f& uv) const {
        Vector2f px;
        if(!mUseDistort) {
            px[0] = mfx*uv[0] + mcx;
            px[1] = mfy*uv[1] + mcy;
        } else {
            double x, y, r2, r4, r6, a1, a2, a3, cdist, xd, yd;
            x = uv[0];
            y = uv[1];
            r2 = x*x + y*y;
            r4 = r2*r2;
            r6 = r4*r2;
            a1 = 2*x*y;
            a2 = r2 + 2*x*x;
            a3 = r2 + 2*y*y;
            cdist = 1 + md0*r2 + md1*r4 + md4*r6;
            xd = x*cdist + md2*a1 + md3*a2;
            yd = y*cdist + md2*a3 + md3*a1;
            px[0] = xd*mfx + mcx;
            px[1] = yd*mfy + mcy;
        }
        return px;
    }

    Eigen::Vector2f PinholeCamera::
    world2cam(const Eigen::Vector3f& xyz_c) const { 
        Eigen::Vector2f uv = project2d(xyz_c);
        return world2cam(uv);
    }
}

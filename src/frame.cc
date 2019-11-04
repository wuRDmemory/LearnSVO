#include "frame.hpp"
#include "config.hpp"
#include "feature.hpp"

namespace mSVO {
    int Frame::id = 0;
    
    Frame::Frame(double timestamp, CameraModel* camera, const cv::Mat& img):
        mTimestamp(timestamp), mCamera(camera), mID(id++), mIsKeyFrame(false) {
        initFrame(img);
    }

    Frame::~Frame() {
        std::for_each(mObs.begin(), mObs.end(), [](Feature* ftr) {
            delete(ftr);
            ftr = NULL;
        });
    }

    void Frame::initFrame(const cv::Mat& img) {
        if (img.empty()) {
            throw std::runtime_error(">>> [ERROR] feed image error");
        }
        std::for_each(mObs.begin(), mObs.end(), [](Feature* ftr) {
            ftr = NULL;
        });

        float scale = 1.0f/Config::pyramidFactor();
        mImagePyr.resize(Config::pyramidNumber());
        mImagePyr[0] = img.clone();
        for (int i=1, N = Config::pyramidNumber(); i<N; i++) {
            cv::pyrDown(mImagePyr[i-1], mImagePyr[i]);
        }
    }

    void Frame::addFeature(Feature* feature) {
        mObs.push_back(feature);
    }

    bool Frame::isVisible(const Vector3f& xyz) {
        Vector2f cxy = world2uv(xyz);
        return cxy(0) >= 0 && cxy(0) < mCamera->width() && cxy(1) >= 0 && cxy(1) < mCamera->height();
    }

    bool Frame::isVisible(const Vector2f& uv, int border, int level) {
        const float scale = 1.0f/(1<<level);
        const int   width = mCamera->width() *scale;
        const int  height = mCamera->height()*scale;
        return uv(0) >= border && uv(0) <  width-border && uv(1) >= border && uv(1) < height-border;
    }

    Vector2f Frame::world2uv(const Vector3f& XYZ) {
        Vector3f cxyz = mRwc.inverse()*(XYZ - mtwc);
        return mCamera->world2cam(cxyz);
    }

    Vector3f Frame::world2camera(const Vector3f& XYZ) {
        return mRwc.inverse()*(XYZ - mtwc);
    }
}

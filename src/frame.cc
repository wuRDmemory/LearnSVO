#include "frame.hpp"
#include "utils.hpp"

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
            cv::resize(mImagePyr[i-1], mImagePyr[i], cv::Size(), scale, scale);
        }
    }

    void Frame::addFeature(Feature* feature) {
        mObs.push_back(feature);
    }

    bool Frame::isVisible(Vector3f& xyz) {
        Vector2f cxy = mCamera->world2cam(xyz);
        return cxy(0) >= 0 && cxyz(0) < Config::width() && cxy(1) >= 0 && cxyz(1) < Config::height();
    }
}

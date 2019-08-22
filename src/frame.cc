#include "frame.hpp"
#include "utils.hpp"

namespace mSVO {
    int Frame::fID = 0;
    Frame::Frame(double timestamp, CameraModel* camera, cv::Mat& img):
        mTimestamp(timestamp), mCammera(camera), mID(fID++), isKeyFrame(false), 
        mObs(5) {
        initFrame(img);
    }

    void Frame::initFrame(const cv::Mat& img) {
        if (img.empty()) {
            throw std::runtime_error(">>> [ERROR] feed image error");
        }
        std::for_each(mObs.begin(), mObs.end(), [](const Feature* ftr) {
            ftr = NULL;
        });

        float scale = 1.0f/Config::getPyramidFactor();
        mImagePyr.resize(Config::getPyramidNumber());
        mImagePyr[0] = img.clone();
        for (int i=1, N = Config::getPyramidNumber(); i<N; i++) {
            cv::resize(mImagePyr[i-1], mImagePyr[i], cv::Size(), scale, scale);
        }
    }

    void Frame::addFeature(Feature* feature) {
        mObs.push_back(feature);
    }
}

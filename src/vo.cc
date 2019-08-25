#include "vo.hpp"


namespace mSVO {
    VO::VO(const string config_file): mNewFrame(NULL), mCameraModel(NULL), updateLevel(UPDATE_FIRST) {
        Config::initInstance(config_file);
        mCameraModel = new PinholeCamera(config_file);
        mInitialor   = new KltHomographyInit();
    }

    VO::~VO() { 
        if (mNewFrame) {
            delete mNewFrame;
            mNewFrame = NULL;
        }

        if (mCameraModel) {
            delete mCameraModel;
            mCameraModel = NULL;
        }

        updateLevel = UPDATE_FIRST;
    }

    bool VO::addNewFrame(const cv::Mat& image, const double timestamp) {
        if (timestamp < -1) {
            throw std::runtime_error("timestamp is invalid");
        }
        if (image.empty()) {
            throw std::runtime_error("image is invalid");
        }
        mNewFrame = new Frame(timestamp, mCameraModel, image);
        
        return true;
    }
}

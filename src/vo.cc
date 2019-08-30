#include "vo.hpp"


namespace mSVO {
    VO::VO(const string config_file): mNewFrame(NULL), mCameraModel(NULL), updateLevel(UPDATE_FIRST) {
        Config::initInstance(config_file);
        mCameraModel = new PinholeCamera(Config::width(), Config::height(), 
                                         Config::fx(), Config::fy(), Config::cx(), Config::cy(), 
                                         Config::d0(), Config::d1(), Config::d2(), Config::d3(), Config::d4());
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
        
        if (updateLevel == UPDATE_FIRST) {
            updateLevel = processFirstFrame();
        } else if (updateLevel == UPDATE_SECOND) {
            updateLevel = processSencondFrame();
        } else {
            updateLevel = processFrame();
        }

        return true;
    }

    UPDATE_LEVEL VO::processFirstFrame() { 
        mNewFrame->mTwc = Sophus::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        if (FAILURE == mInitialor->addFirstFrame(mNewFrame)) {
            LOG(ERROR) << ">>> Faild to create first key frame!";
            return UPDATE_NO_FRAME;
        }
        // TODO: add the frame to map
        
        return UPDATE_SECOND;
    }

    
}

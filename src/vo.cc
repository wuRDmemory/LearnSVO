#include "vo.hpp"


namespace mSVO {
    VO::VO(const string config_file): mNewFrame(NULL), mCameraModel(NULL), updateLevel(UPDATE_FIRST) {
        Config::initInstance(config_file);
        mCameraModel = new PinholeCamera(Config::width(), Config::height(), 
                                         Config::fx(), Config::fy(), Config::cx(), Config::cy(), 
                                         Config::d0(), Config::d1(), Config::d2(), Config::d3(), Config::d4());
        mInitialor   = new KltHomographyInit();
        mLocalMap    = new Map(Config::keyFrameNum());
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

    UPDATE_LEVEL VO::addNewFrame(const cv::Mat& image, const double timestamp) {
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

        return updateLevel;
  
    }

    UPDATE_LEVEL VO::processFirstFrame() { 
        mNewFrame->pose() = Sophus::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        if (FAILURE == mInitialor->addFirstFrame(mNewFrame)) {
            LOG(ERROR) << ">>> [first frame] Faild to create first key frame!";
            return UPDATE_NO_FRAME;
        }
        // TODO: add the frame to map
        mLocalMap->addKeyFrame(mNewFrame);
        mRefFrame = mNewFrame;
        return UPDATE_SECOND;
    }

    UPDATE_LEVEL VO::processSencondFrame() {
        mNewFrame->pose() = mRefFrame->pose();
        if (FAILURE == mInitialor->addSecondFrame(mNewFrame)) {
            LOG(ERROR) << ">>> [second frame] Faild to create first key frame!";
            return UPDATE_NO_FRAME;
        }
        // TODO: add the frame to map, update refer frame
        mLocalMap->addKeyFrame(mNewFrame);
        mRefFrame = mNewFrame;

        // TODO: reset the mInitialor
        mInitialor->reset();
        return UPDATE_FRAME;
    }

    UPDATE_LEVEL VO::processFrame() {
        mNewFrame->pose() = mRefFrame->pose();
        // image align
        ImageAlign imageAlign(0, Config::pyramidNumber(), 10);
        imageAlign.run(mRefFrame, mNewFrame);
        // feature align
        

        mRefFrame = mNewFrame;
        return UPDATE_FRAME;
    }
    
}
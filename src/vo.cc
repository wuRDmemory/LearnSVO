#include "vo.hpp"

namespace mSVO {
    VO::VO(const string config_file): mNewFrame(NULL), mCameraModel(NULL), updateLevel(UPDATE_FIRST) {
        Config::initInstance(config_file);
        mCameraModel = new PinholeCamera(Config::width(), Config::height(), 
                                         Config::fx(), Config::fy(), Config::cx(), Config::cy(), 
                                         Config::d0(), Config::d1(), Config::d2(), Config::d3(), Config::d4());
        mInitialor   = new KltHomographyInit();
        mLocalMap    = new Map(Config::keyFrameNum());
        mFeatureAlign = new FeatureAlign(mLocalMap);
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

    void VO::addNewFrame(const cv::Mat& image, const double timestamp) {
        if (timestamp < -1) {
            throw std::runtime_error("timestamp is invalid");
        }
        if (image.empty()) {
            throw std::runtime_error("image is invalid");
        }
        mNewFrame = new Frame(timestamp, mCameraModel, image);
        
        PROCESS_STATE res;
        if (updateLevel == UPDATE_FIRST) {
            res = processFirstFrame();
        } else if (updateLevel == UPDATE_SECOND) {
            res = processSencondFrame();
        } else if (updateLevel == UPDATE_FRAME) {
            res = processFrame();
        } else {
            // TODO: relocal
        }

        finishProcess(mNewFrame->ID(), res);
    }

    PROCESS_STATE VO::processFirstFrame() { 
        mNewFrame->Rwc() = Eigen::Quaternionf::Identity();
        mNewFrame->twc() = Vector3f::Zero();
        if (FAILURE == mInitialor->addFirstFrame(mNewFrame)) {
            LOG(ERROR) << ">>> [first frame] Faild to create first key frame!";
            return PROCESS_FAIL;
        }
        // TODO: add the frame to map
        mLocalMap->addKeyFrame(mNewFrame);
        mRefFrame = mNewFrame;
        updateLevel = UPDATE_SECOND;
        return PROCESS_SUCCESS;
    }

    PROCESS_STATE VO::processSencondFrame() {
        mNewFrame->Rwc() = mRefFrame->Rwc();
        mNewFrame->twc() = mRefFrame->twc();
        if (FAILURE == mInitialor->addSecondFrame(mNewFrame)) {
            LOG(ERROR) << ">>> [second frame] Faild to create first key frame!";
            return PROCESS_FAIL;
        }
        // TODO: add the frame to map, update refer frame
        mLocalMap->addKeyFrame(mNewFrame);
        mRefFrame = mNewFrame;

        // TODO: reset the mInitialor
        mInitialor->reset();
        updateLevel = UPDATE_FRAME;
        return PROCESS_SUCCESS;
    }

    PROCESS_STATE VO::processFrame() {
        mNewFrame->Rwc() = mRefFrame->Rwc();
        mNewFrame->twc() = mRefFrame->twc();
        // image align
        ImageAlign imageAlign(0, Config::pyramidNumber(), 10);
        imageAlign.run(mRefFrame, mNewFrame);
        
        // feature align
        mFeatureAlign->reproject(mNewFrame);
        LOG(INFO) << ">>> [process frame] feature trails:  " << mFeatureAlign->trails();
        LOG(INFO) << ">>> [process frame] feature matches: " << mFeatureAlign->matches();
        if (mFeatureAlign->matches() < Config::featureMatchMinThr()) {
            LOG(INFO) << ">>> [process frame] too few feature track, start relocal";
            return PROCESS_FAIL;
        }

        // optimize the frame's pose
        float estimateScale, initChi2, endChi2; 
        int   inlierCnt;
        PoseOptimize::optimize(mNewFrame, Config::poseOptimizeIterCnt(), Config::minProjError(), estimateScale, initChi2, endChi2, inlierCnt);
        LOG(INFO) << ">>> [process frame] Pose optimize chi2 update: " << initChi2 << "-->" << endChi2;
        LOG(INFO) << ">>> [process frame] Pose optimize inlier count: " << inlierCnt;
        if (inlierCnt < Config::poseOptimizeInlierThr()) {
            // TODO： return false
            LOG(INFO) << "[trackFrame] pose optimize failed! inlier: " << inlierCnt;
            return PROCESS_FAIL;
        }

        // TODO: optimize the point seen
        StructOptimize::optimize(mNewFrame, Config::structOptimizeIterCnt(), Config::structOptimizePointCnt());

        // TODO: check wether need new key frame
        if (!needKeyFrame(inlierCnt)) {
            mRefFrame = mNewFrame;
            // TODO: add to depth filter

            return UPDATE_NO_KEYFRAME;
        }

        // TODO: ALL BA
        

        mRefFrame = mNewFrame;
        return UPDATE_KEYFRAME;
    }
    
    void VO::finishProcess(int frameId, PROCESS_STATE res) {
        LOG(INFO) << "[finishProcess] Process frame " << frameId << "finish";
        if (res == PROCESS_FAIL && (updateLevel == UPDATE_FRAME || updateLevel == UPDATE_RELOCAL)) {
            updateLevel = UPDATE_RELOCAL;
        }
    }

    bool VO::needKeyFrame(int trackCnt, FramePtr frame) {
        // check the track feature count
        if (trackCnt < Config::minTrackFeatureCnt()) {
            return true;
        }

        Vector3f& curPos = frame->twc();
        auto& keyFrames = mLocalMap->keyFrames();
        float min_distance = FLT_MAX;
        for (auto begin = keyFrames.begin(), end = keyFrames.end(); begin != end; ++begin) {
            Vector3f& kfPos = (*begin)->twc();
            float dist = (curPos-kfPos).norm();
            min_distance = min(min_distance, dist);
        }

        return min_distance > Config::trackMoveDistance();
    }

}
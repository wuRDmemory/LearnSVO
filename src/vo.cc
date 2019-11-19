#include "vo.hpp"

namespace mSVO {
    VO::VO(const string config_file): mNewFrame(NULL), mCameraModel(NULL), updateLevel(UPDATE_FIRST) {
        Config::initInstance(config_file);
        mCameraModel = new PinholeCamera(Config::width(), Config::height(), 
                                         Config::fx(), Config::fy(), Config::cx(), Config::cy(), 
                                         Config::d0(), Config::d1(), Config::d2(), Config::d3(), Config::d4());
        mInitialor   = new KltHomographyInit();
        mLocalMap    = new Map(Config::keyFrameNum());
        mDepthFilter = new DepthFilter(mLocalMap);
        mFeatureAlign = new FeatureAlign(mLocalMap);
        mBundleAdjust = new BundleAdjustment(5, mLocalMap);
    }

    VO::~VO() { 
        if (mNewFrame) {
            mNewFrame.reset();
        }

        if (mCameraModel) {
            delete mCameraModel;
            mCameraModel = NULL;
        }

        updateLevel = UPDATE_FIRST;
    }

    void VO::setup() { 
        LOG(INFO) << ">>> [VO] setup now";
        updateLevel = UPDATE_FIRST;
        mDepthFilter->startThread();
    }

    void VO::addNewFrame(const cv::Mat& image, const double timestamp) {
        if (timestamp < -1) {
            throw std::runtime_error("timestamp is invalid");
        }
        if (image.empty()) {
            throw std::runtime_error("image is invalid");
        }
        mNewFrame.reset(new Frame(timestamp, mCameraModel, image));
        
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
        LOG(INFO) << ">>> [first frame] Oc: " << mNewFrame->twc().transpose();
        return PROCESS_SUCCESS;
    }

    PROCESS_STATE VO::processSencondFrame() {
        mNewFrame->Rwc() = mRefFrame->Rwc();
        mNewFrame->twc() = mRefFrame->twc();
        if (FAILURE == mInitialor->addSecondFrame(mNewFrame)) {
            LOG(ERROR) << ">>> [second frame] Faild to create first key frame!";
            return PROCESS_FAIL;
        }

        mOldTrackCnt = mNewFrame->obs().size();
        // add the frame to map, update refer frame
        mNewFrame->setKeyFrame();
        mLocalMap->addKeyFrame(mNewFrame);

        float minDepth, meanDepth;
        mNewFrame->getDepth(minDepth, meanDepth);

        mDepthFilter->addNewKeyFrame(mNewFrame, minDepth, meanDepth);
        mRefFrame = mNewFrame;

        // TODO: reset the mInitialor
        mInitialor->reset();
        updateLevel = UPDATE_FRAME;
        LOG(INFO) << ">>> [second frame] Oc: " << mNewFrame->twc().transpose();
        return PROCESS_SUCCESS;
    }

    PROCESS_STATE VO::processFrame() {
        LOG(INFO) << ">>> [track] id: " << mRefFrame->ID() << " --> " << mNewFrame->ID();
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
            mNewFrame->Rwc() = mRefFrame->Rwc();
            mNewFrame->twc() = mRefFrame->twc();
            LOG(INFO) << ">>> [process frame] too few feature track (" \
                      << mFeatureAlign->matches() << "/" << Config::featureMatchMinThr() << ")";
            return PROCESS_FAIL;
        }

        // optimize the frame's pose
        float estimateScale, initChi2, endChi2; 
        int   inlierCnt;
        PoseOptimize::optimize(mNewFrame, Config::poseOptimizeIterCnt(), Config::minProjError(), estimateScale, initChi2, endChi2, inlierCnt);
        LOG(INFO) << ">>> [process frame] Pose optimize chi2 update: "  << initChi2 << "-->" << endChi2;
        LOG(INFO) << ">>> [process frame] Pose optimize inlier count: " << inlierCnt;
        LOG(INFO) << ">>> [process frame] Pose optimize Oc: " << mNewFrame->twc().transpose(); 

        // optimize the point seen, there maybe some error
        StructOptimize::optimize(mNewFrame, Config::structOptimizeIterCnt(), Config::structOptimizePointCnt());

        // check wether need new key frame
        float minDepth, meanDepth;
        mNewFrame->getDepth(minDepth, meanDepth);
        if (!needKeyFrame(inlierCnt, mNewFrame)) {
            mDepthFilter->addNewFrame(mNewFrame);
            mRefFrame = mNewFrame;
            updateLevel = UPDATE_FRAME;
            return PROCESS_SUCCESS;
        }
        
        mNewFrame->setKeyFrame();
        LOG(INFO) << ">>> [process frame] add a key frame! frame Id: " << mNewFrame->ID() << "; map size: " << mLocalMap->getKeyframeSize();

        // add the candidate landmark into the key frame.
        // the landmark become UNKNOWN,  and can be GOOD.
        // CANDIDATE only though this way to become UNKOWN
        auto& obs = mNewFrame->obs();
        for (auto it = obs.begin(); it != obs.end(); it++) {
            FeaturePtr feature = *it;
            feature->mLandmark->addFeature(feature);
        }
        mLocalMap->candidatePointManager().addLandmarkToFrame(mNewFrame);
        
        // ALL BA
        mBundleAdjust->run();
        LOG(INFO) << ">>> [process frame] Oc: " << mNewFrame->twc().transpose();

        // remove most far keyframe in localmap
        if (mLocalMap->getKeyframeSize() > Config::keyFrameNum()) {
            LOG(INFO) << ">>> [process frame] begin remove key frame! Size: " << mLocalMap->getKeyframeSize();
            FramePtr removeKeyFrame;
            mLocalMap->getFarestFrame(mNewFrame, removeKeyFrame);
            mLocalMap->removeKeyFrame(removeKeyFrame);
            mDepthFilter->removeKeyFrame(removeKeyFrame);
            LOG(INFO) << ">>> [process frame] remove key frame " << removeKeyFrame->ID() << " done! Size: " << mLocalMap->getKeyframeSize();
        }

        // add key frame into depth update
        mDepthFilter->addNewKeyFrame(mNewFrame, minDepth, meanDepth);

        // add frame to the map
        mLocalMap->addKeyFrame(mNewFrame);

        mRefFrame = mNewFrame;
        updateLevel = UPDATE_FRAME;
        return PROCESS_SUCCESS;
    }
    
    void VO::finishProcess(int frameId, PROCESS_STATE res) {
        LOG(INFO) << ">>> [finishProcess] Process frame " << frameId << " finish";
        if (res == PROCESS_FAIL && updateLevel == UPDATE_FRAME) {
            // TODO: add relocal condition
            updateLevel = UPDATE_FRAME;
        }
        LOG(INFO) << ">>> ";
        // usleep(500000);
    }

    bool VO::needKeyFrame(int trackCnt, FramePtr frame) {
        // check the track feature count
        if (trackCnt < Config::trackMinFeatureCnt()) {
            LOG(INFO) << ">>> [needKeyFrame] Track count: " << trackCnt << ", add new key frame";
            return true;
        }

        // add drop check
        int dropCnt  = mOldTrackCnt - trackCnt;
        mOldTrackCnt = trackCnt;
        if (dropCnt > 10) {
            LOG(INFO) << ">>> [needKeyFrame] drop feature count: " << dropCnt << "(" << mOldTrackCnt << "-->" << trackCnt << "), add new key frame";
            return true;
        }

        // check translation
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
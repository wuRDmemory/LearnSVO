#include "map.hpp"
#include "config.hpp"
#include "feature.hpp"

namespace mSVO {
    bool CandidateLandmark::reset() { 
        unique_lock<mutex> lock(mMutex);
        mCandidatePoints.clear();
    }

    bool CandidateLandmark::addCandidateLandmark(LandMarkPtr& point, FramePtr& frame) {
        unique_lock<mutex> lock(mMutex);
        point->type = LandMark::LANDMARK_TYPR::CANDIDATE;
        mCandidatePoints.push_back(make_pair(point, frame.get()));
        return true;
    }

    bool CandidateLandmark::addLandmarkToFrame(FramePtr& frame) {
        int  addN = 0;
        auto it = mCandidatePoints.begin();
        while (it != mCandidatePoints.end()) {
            LandMarkPtr landmark = it->first;
            if (landmark->obs().front()->mFrame == frame.get()) {
                // change to the unkonw state
                addN++;
                landmark->type = LandMark::LANDMARK_TYPR::UNKNOWN;
                landmark->nProjectFrameFailed = 0;
                landmark->nProjectFrameSuccess = 0;
                landmark->nOptimizeFrameId = frame->ID();
                frame->addFeature(landmark->obs().front());
                {
                    unique_lock<mutex> lock(mMutex);
                    it = mCandidatePoints.erase(it);
                }
            } else {
                it++;
            }
        }
        LOG(INFO) << ">>> [addLandmarkToFrame] Add candidates: " << addN;
        return true;
    }

    bool CandidateLandmark::removeCandidateLandmark(Frame* frame) {
        auto it = mCandidatePoints.begin();
        while (it != mCandidatePoints.end()) {
            LandMarkPtr landmark = it->first;
            Frame* originFrame = it->second;
            if (originFrame == frame) {
                {
                    unique_lock<mutex> lock(mMutex);
                    mTrashPoints->push_back(it->first);
                }
                it = mCandidatePoints.erase(it);
            } else {
                it++;
            }
        }
        return 1;
    }

    Map::Map(int mapSize): mMapSize(mapSize) {
        mKeyFrames.clear();
        mCandidatePointsManager.setTrash(&mTrashPoints);
    }

    Map::~Map() {
        LOG(INFO) << ">>> map destruct" << endl;
        reset();
    }

    void Map::reset() {
        mKeyFrames.clear();
    }

    void Map::addKeyFrame(FramePtr& currFrame) {
        mKeyFrames.push_back(currFrame);
    }

    bool Map::getCloseFrame(FramePtr& frame, vector<pair<FramePtr, double> >& keyframes) {
        keyframes.clear();
        for (FramePtr& keyframe : mKeyFrames) {
            auto& obs = keyframe->obs();
            int thrN  = obs.size()*Config::projectRatioThr();
            int cnt   = 0;
            for (auto it = obs.begin(); it != obs.end(); ++it) {
                if (!(*it)->mLandmark or (*it)->mLandmark->type == LandMark::LANDMARK_TYPR::DELETE) {
                    continue;
                }
                if (frame->isVisible((*it)->mLandmark->xyz())) {
                    cnt++;
                    if (cnt > thrN) {
                        keyframes.emplace_back(keyframe, keyframe->timestamp());
                        break;
                    }
                }
            }
        }
        return true;
    }

    bool Map::getClosestFrame(FramePtr& frame, FramePtr& keyframe) {
        float minDis = FLT_MAX;
        Vector3f& pose = frame->twc();
        for (FramePtr& kframe : mKeyFrames) {
            Vector3f& xyz = kframe->twc();
            float dis = (xyz - pose).norm();
            if (dis < minDis) {
                minDis = dis;
                keyframe.reset(kframe.get());
            }
        }
        return true;
    }

    bool Map::getFarestFrame(FramePtr& frame, FramePtr& keyframe) {
        float maxDis = FLT_MIN;
        Vector3f& pose = frame->twc();
        for (FramePtr& kframe : mKeyFrames) {
            Vector3f& xyz = kframe->twc();
            float dis = (xyz - pose).norm();
            if (dis > maxDis) {
                maxDis = dis;
                keyframe = kframe;
            }
        }
        return true;
    }

    bool Map::removeKeyFrame(Frame* keyframe) {
        // find the key frame 
        auto iter = mKeyFrames.begin();
        while (iter != mKeyFrames.end()) {
            Frame* kframe  = iter->get();
            if (kframe == keyframe) {
                // remove all landmark in this key frame.
                auto& features = kframe->obs();
                for (auto it = features.begin(); it != features.end(); ) {
                    LandMarkPtr landmark = (*it)->mLandmark;
                    if (!landmark) {
                        it++;
                        continue;
                    }
                    landmark->safeRemoveFeature(*it);
                    if (landmark->type == LandMark::DELETE) {
                        mTrashPoints.push_back(landmark);
                    }
                    delete (*it);
                    it = features.erase(it);
                }
                mKeyFrames.erase(iter);
                break;
            }
            iter++;
        }
        
        // remove the CANDIDATE point in candidate list
        mCandidatePointsManager.removeCandidateLandmark(keyframe);
        return true;
    }

    bool Map::addLandmarkToTrash(LandMarkPtr ldmk) { 
        mTrashPoints.push_back(ldmk);
        auto& features = ldmk->obs();
        for (auto it = features.begin(); it != features.end();it++) {
            FeaturePtr feature = *it;
            feature->mLandmark = NULL;
        }
        return true;
    }
}

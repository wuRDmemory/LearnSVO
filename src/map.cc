#include "map.hpp"
#include "utils.hpp"
#include "feature.hpp"

namespace mSVO {
    Map::Map(int mapSize): mMapSize(mapSize) {
        mKeyFrames.clear();
    }

    Map::~Map() {
        LOG(INFO) << ">>> map destruct" << endl;
        reset();
    }

    void Map::reset() {
        mKeyFrames.clear();
    }

    void Map::addKeyFrame(FramePtr currFrame) {
        currFrame->setKeyFrame();
        mKeyFrames.push_back(currFrame);
    }

    bool Map::getCloseFrame(FramePtr frame, vector<pair<FramePtr, double> >& keyframes) {
        keyframes.clear();
        for (FramePtr& keyframe : mKeyFrames) {
            auto& obs = keyframe->obs();
            int N     = obs.size();
            for (FeaturePtr& feature : obs) {
                int cnt = 0;
                if (frame->isVisible(feature->mLandmark->xyz())) {
                    cnt++;
                    if (cnt/N > Config::projectThreshold()) {
                        keyframes.emplace_back(keyframe, keyframe->timestamp());
                        break;
                    }
                }
            }
        }
        return true;
    }
}

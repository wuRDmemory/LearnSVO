#include "map.hpp"

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
}

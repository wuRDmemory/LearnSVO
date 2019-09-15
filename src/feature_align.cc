#include "feature_align.hpp"

namespace mSVO {
    FeatureAlign::FeatureAlign(MapPtr map): mMap(map) {
        ;
    }

    FeatureAlign::~FeatureAlign() {
        ;
    }

    bool FeatureAlign::reproject(FramePtr curFrame) {
        // TODO: 1. fetch the close key frames
        vector<pair<FramePtr, double> > keyframes;
        mMap->getCloseFrame(curFrame, keyframes);

        // sort key frame by time
        sort(keyframes.begin(), keyframes.end(), [](const pair<FramePtr, double>& a, const pair<FramePtr, double>& b) {
            return a.second > b.second;
        });

        // TODO: 2. project key frame's key points into current frame
        //          assign them to grid cell
        keyframe.resize(Config::closeKeyFrameCnt());
        

        // TODO: 3. 
    }
}

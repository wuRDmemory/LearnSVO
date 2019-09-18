#include "feature_align.hpp"

namespace mSVO {
    Grid::Grid(int imWidth, int imHeight, int cellSize) {
        rows = imHeight/cellSize;
        cols = imWidth /cellSize;
        step = cellSize;
        cells.resize(rows*cols);
    }
};

namespace mSVO {
    FeatureAlign::FeatureAlign(MapPtr map): mMap(map) {
        if (mGrid) {
            delete mGrid;
            mGrid = NULL;
        }
        mGrid = new Grid(Config::width(), Config::height(), 30);
    }

    FeatureAlign::~FeatureAlign() {
        if (mGrid) {
            delete mGrid;
            mGrid = NULL;
        }
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
        for (int i = 0; i < keyframe.size(); ++i) {
            auto& obs = keyframe[i].first->obs();
            for (FeaturePtr& feature : obs) {
                if (!feature->mLandmark) {
                    continue;
                }
                // 
                projectToCurFrame(curFrame, feature);
            }
        }

        // TODO: 3. align features in each cells
        // for each cell
        for (int i = 0; i < mGrid->cells.size(); i++) {
            vector<CandidateCell>& cell = mGrid->cells[i];
            for (int j = 0; j < cell.size(); j++) {
                CandidateCell& candidate = cell[j];
                alignGridCell()
            }
        }
    }

    bool FeatureAlign::projectToCurFrame(FramePtr curFrame, FeaturePtr feature) {
        Vector2f cuv = curFrame->world2camera(feature->mLandmark->xyz());
        if (curFrame->isVisible(cuv, patchSize)) {
            mGrid->setCell(cuv, feature->mLandmark);
            return true;
        }
        return false;
    }

    bool FeatureAlign::alignGridCell(FramePtr curFrame, CandidateCell& candidate) {
        auto feature = candidate.begin();

        while (feature != candidate.end()) {
            
            if (feature->xyz->type == LandMark::DELETE) {
                candidate.erase(feature);
                continue;
            }

            candidate.erase(feature);
            continue;
        }
        
        for (auto begin = candidate.begin(), end = candidate.end(); begin != end; ++begin) {
            CandidateFeature* feature = begin;
            if (feature->mLandmark->type == LandMark::DELETE)
                continue;
            

        }
        return true;
    }
}

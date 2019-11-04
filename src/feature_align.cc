#include "feature_align.hpp"

#define SHOW_PROJECT 1

namespace mSVO {
    Grid::Grid(int imWidth, int imHeight, int cellSize) {
        rows = imHeight/cellSize;
        cols = imWidth /cellSize;
        step = cellSize;
        cells.resize(rows*cols);
    }

    Grid::~Grid() {
        for (int i = 0; i < cells.size(); i++) {
            cells[i].clear();
        }
    }

    FeatureAlign::FeatureAlign(MapPtr map): mMap(map), mMatches(0), mTrails(0) {
        mGrid = new Grid(Config::width(), Config::height(), 30);
    }

    FeatureAlign::~FeatureAlign() {
        if (mGrid) {
            delete mGrid;
            mGrid = NULL;
        }
    }

    bool FeatureAlign::resetGridCell() {
        mMatches = 0; 
        mTrails  = 0;
        mProjectNum = 0;
        for (auto begin = mGrid->cells.begin(), end = mGrid->cells.end(); begin != end; ++begin) {
            (*begin).clear();
        }
        return true;
    }

    bool FeatureAlign::reproject(FramePtr curFrame) {
        resetGridCell();
        // TODO: 1. fetch the close key frames
        vector<pair<FramePtr, double> > keyframes;
        mMap->getCloseFrame(curFrame, keyframes);

        // sort key frame by time
        sort(keyframes.begin(), keyframes.end(), [](const pair<FramePtr, double>& a, const pair<FramePtr, double>& b) {
            return a.second < b.second;
        });

        // TODO: 2. project key frame's key points into current frame
        //          assign them to grid cell
        keyframes.resize(min((int)keyframes.size(), 5));
        for (int i = 0; i < keyframes.size(); ++i) {
            FramePtr& frame = keyframes[i].first;
            auto& obs = frame->obs();
            for (auto it = obs.begin(); it != obs.end(); it++) {
                LandMarkPtr landmark = (*it)->mLandmark;
                if (!landmark || landmark->type == LandMark::LANDMARK_TYPR::DELETE) {
                    continue;
                }
                projectToCurFrame(curFrame, landmark);
            }
        }

        // TODO: 3. project all candidate landmark into current frame
        //          assign them to grid cell
        auto& candidates = mMap->candidatePointManager().candidateLandmark();
        for (auto it = candidates.begin(); it != candidates.end();) {
            if (it->first->type == LandMark::LANDMARK_TYPR::DELETE) {
                continue;
            }
            projectToCurFrame(curFrame, it->first);
        }

        LOG(INFO) << ">>> [reproject] All project point: " << mProjectNum;
        // TODO: 4. align features in each cells
        // for each cell
        for (int i = 0; i < mGrid->cells.size(); i++) {
            CandidateCell& candidates = mGrid->cells[i];
            const int candidatesN = candidates.size();
            if (alignGridCell(curFrame, candidates)) {
                mMatches++;
            }
            if (mMatches > Config::featureNumber()) {
                break;
            }
        }
        LOG(INFO) << "reproject done!!! align :" << mMatches;
        return true;
    }

    bool FeatureAlign::projectToCurFrame(FramePtr curFrame, LandMarkPtr landmark) {
        Vector3f XYZ = landmark->xyz();
        Vector2f cuv = curFrame->world2uv(XYZ);
        if (curFrame->isVisible(cuv, 8)) {
            mProjectNum++;
            mGrid->setCell(cuv, landmark);
            return true;
        }
        return false;
    }

    bool FeatureAlign::alignGridCell(FramePtr curFrame, CandidateCell& candidate) {
        auto it = candidate.begin();
        while (it != candidate.end()) {
            mTrails ++;
            LandMarkPtr& ldmk = it->xyz;
            // check the landmark type
            if (ldmk->type == LandMark::DELETE) {
                it = candidate.erase(it);
                continue;
            }

            // feature alignment
            Vector2f pix = it->px;
            if (!mMatcher->findDirectMatch(curFrame.get(), ldmk, pix)) {
                // if project failed, add the failed cnt
                ldmk->nProjectFrameFailed += 1;
                if (ldmk->type == LandMark::UNKNOWN and ldmk->nProjectFrameFailed > 5) {
                    // TODO: delete the landmark
                    ldmk->type = LandMark::LANDMARK_TYPR::DELETE;
                    mMap->addLandmarkToTrash(ldmk);
                } else if (ldmk->type == LandMark::CANDIDATE and ldmk->nProjectFrameFailed > 8) {
                    // TODO: delete the landmark from candidate list
                    ldmk->type = LandMark::LANDMARK_TYPR::DELETE;
                    mMap->addLandmarkToTrash(ldmk);
                }
                it = candidate.erase(it);
                continue;
            }

            ldmk->nProjectFrameSuccess++;
            if (ldmk->type == LandMark::UNKNOWN and ldmk->nProjectFrameSuccess > 10) {
                ldmk->type = LandMark::GOOD;
            }

            FeaturePtr newFeature = new Feature(curFrame.get(), pix);
            newFeature->mLandmark = ldmk;
            curFrame->addFeature(newFeature);
            it = candidate.erase(it);
            return true;
        }
        return false;
    }

    bool FeatureAlign::testProject(cv::Mat& img) {
        Mat image;
        cvtColor(img, image, COLOR_GRAY2BGR);
        for (int i = 0; i < mGrid->cells.size(); i++) {
            CandidateCell& candidates = mGrid->cells[i];
            auto it = candidates.begin();
            while (it != candidates.end()) {
                Vector2f pix = it->px;
                Scalar color(theRNG().uniform(0, 255), theRNG().uniform(0, 255), theRNG().uniform(0, 255));
                cv::circle(image, cv::Point(pix(0), pix(1)), 2, color, 1);
                it++;
            }
        }
        imshow("[feature align] image", image);
        waitKey();
        destroyWindow("[feature align] image");
        return true;
    }
}

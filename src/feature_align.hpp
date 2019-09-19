#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <vector>
#include <algorithm>

#include "map.hpp"
#include "frame.hpp"
#include "feature.hpp"
#include "landmark.hpp"
#include "matcher.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    struct CandidateFeature{
        Vector2f px;
        LandMarkPtr xyz;

        CandidateFeature(Vector2f& px_, LandMarkPtr landmark_): px(px_), xyz(landmark_) {}
    };

    typedef list<CandidateFeature>  CandidateCell; 

    struct Grid {
        int step;
        int rows;
        int cols;
        vector<CandidateCell> cells;

        Grid(int imWidth, int imHeight, int cellSize);
        ~Grid() {};

        void setCell(Vector2f& uv, LandMarkPtr landmark) {
            const int x = uv(0) / step;
            const int y = uv(1) / step;
            cells[y*step + x].push_back(CandidateFeature(uv, landmark));
        }
    };
    
    class FeatureAlign {
    private:
        static int halfPatchSize = 4;
        static int patchSize     = 8;
        static int patchArea     = 64;

        MapPtr     mMap;
        MatcherPtr mMatcher;
        Grid*      mGrid;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        FeatureAlign(MapPtr map);
        ~FeatureAlign();

        bool reproject(FramePtr curFrame);

    private:
        bool projectToCurFrame(FramePtr curFrame, FeaturePtr feature);
        bool alignGridCell(FramePtr curFrame, CandidateCell& candidate);

    };
}


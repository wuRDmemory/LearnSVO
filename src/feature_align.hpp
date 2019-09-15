#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <vector>
#include <algorithm>

#include "map.hpp"
#include "frame.hpp"
#include "landmark.hpp"

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
    };
    
    class FeatureAlign {
    private:
        static int halfPatchSize = 4;
        static int patchSize     = 8;
        static int patchArea     = 64;

        MapPtr mMap;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        FeatureAlign(MapPtr map);
        ~FeatureAlign();

        bool reproject(FramePtr curFrame);
    };
}


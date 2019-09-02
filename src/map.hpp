#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "frame.hpp"
#include "landmark.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    class Map {
    private:
        int mMapSize;
        list<FramePtr> mKeyFrames;

    public:
        Map(int mapSize);
        ~Map();
        
        void reset();   //!< when lost, reset the map
        void addKeyFrame(FramePtr currFrame);

        list<FramePtr>& keyFrames() { return mKeyFrames; }
    };

    typedef Map* MapPtr;
}

#pragma once

#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "frame.hpp"
#include "feature.hpp"
#include "landmark.hpp"
#include "matcher.hpp"
#include "map.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    struct Seed {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
        static int ID;
        int seenFrameID;
        int id;
        int a, b;
        float mu, sigma2;
        float zRange;
        FeaturePtr feature;

        Seed(FeaturePtr feature, float depthMin, float depthMean);
    };

    typedef Seed* SeedPtr;

    class DepthFilter {
    private:
        thread* mThread1;
        MapPtr  mMap;
        list<SeedPtr> mSeedList;

    public:
        DepthFilter(MapPtr map);
        ~DepthFilter();

        bool addNewFrame(FramePtr frame);
        bool addNewKeyFrame(FramePtr frame);
    
    private:
        bool runFilter(FramePtr frame);
        bool initialKeyFrame(FramePtr keyframe);
    };
}

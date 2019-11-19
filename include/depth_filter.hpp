#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <random>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "frame.hpp"
#include "feature.hpp"
#include "landmark.hpp"
#include "detector.hpp"
#include "matcher.hpp"
#include "map.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    struct Seed {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
        static int batchID;
        static int ID;
        int seenFrameID;
        int id;
        int a, b;
        float mu, sigma2;
        float zRange;
        FeaturePtr feature;

        Seed(FeaturePtr ftr, float depthMin, float depthMean);
    };

    typedef Seed* SeedPtr;

    class DepthFilter {
    private:
        thread* mThread;
        MapPtr  mMap;
        Matcher mMatcher;
        FramePtr mNewKeyFrame;
        DetectorPtr mDetector;

        list<SeedPtr> mSeedList;
        queue<FramePtr> mAddFrames;

        mutex mAddFrameLock, mAddSeedLock;
        condition_variable mConditionVariable;

        int mNFailNum;
        int mNMatched;
        float mDepthMean, mDepthMin;
        bool mStop;
        bool mNewKeyFrameFlag;
        bool mSeedUpdateHalt;
        
    public:
        DepthFilter(MapPtr map);
        ~DepthFilter();

        bool startThread(bool createThread = false);
        bool addNewFrame(FramePtr& frame);
        bool addNewKeyFrame(FramePtr& frame, float minDepth, float meanDepth);
        bool removeKeyFrame(FramePtr& frame);

        int failedNum()  const { return mNFailNum; }
        int matchedNum() const { return mNMatched; }

    private:
        bool mainloop();
        bool clearFrameList();
        bool runFilter(FramePtr& frame);
        bool initialKeyFrame(FramePtr& keyframe);
        bool updateSeed(const float x, const float tau2, Seed* seed);
        float computeTau(Quaternionf& Rcr, Vector3f& tcr, Vector3f& direct, float z, float noiseAngle);

        float normalPDF(float x, float mean, float sigma);
        
    };

    typedef DepthFilter* DepthFilterPtr;
}

#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <mutex>

#include "frame.hpp"
#include "landmark.hpp"

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    typedef pair<LandMarkPtr, Frame*> CandidateStruct;

    class CandidateLandmark {
    public:
        void setTrash(Landmarks* trashPoints) { mTrashPoints = trashPoints; }

        bool reset();
        bool addCandidateLandmark(LandMarkPtr& point, FramePtr& frame);
        bool addLandmarkToFrame(FramePtr& frame);
        bool removeCandidateLandmark(FramePtr& frame);

        list<CandidateStruct>& candidateLandmark() { return mCandidatePoints; }

    private:
        mutex mMutex;
        list<CandidateStruct> mCandidatePoints;
        Landmarks* mTrashPoints;
    };

    class Map {
    private:
        int mMapSize;
        list<FramePtr>    mKeyFrames;
        Landmarks         mTrashPoints;
        CandidateLandmark mCandidatePointsManager;

    public:
        Map(int mapSize);
        ~Map();
        
        void reset();   //!< when lost, reset the map
        void addKeyFrame(FramePtr& currFrame);
        bool getCloseFrame(FramePtr& frame, vector<pair<FramePtr, double> >& keyframes);
        bool getClosestFrame(FramePtr& frame, FramePtr& keyframe);
        bool getFarestFrame(FramePtr& frame,  FramePtr& keyframe);
        bool removeKeyFrame(FramePtr& keyframe);
        bool addLandmarkToTrash(LandMarkPtr ldmk);

        int  getKeyframeSize() { return mKeyFrames.size(); }
        list<FramePtr>&  keyFrames() { return mKeyFrames; }
        CandidateLandmark& candidatePointManager() { return mCandidatePointsManager; }

    };

    typedef Map* MapPtr;
}

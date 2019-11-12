#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "frame.hpp"
#include "feature.hpp"
#include "map.hpp"
#include "pangolin/pangolin.h"

namespace mSVO {
    class Viewer {
    private:
        bool mStop;

        float mCameraSize;
        float mCameraLineWidth;
        float mPointSize;
        mutex mAddFrameMutex;
        condition_variable mConditionVariable;

        thread* mThread;
        Frame* mCurFrame;
        MapPtr mMap;

    public:
        Viewer(MapPtr map);
        ~Viewer();

        bool setup();
        bool addCurrentFrame(Frame* curFrame);

        void run();
        bool stop();

    private:
        bool convertMatrix(Quaternionf& Qwc, Vector3f& twc, pangolin::OpenGlMatrix& M);
        bool drawFrame(pangolin::OpenGlMatrix &Twc);
        bool drawPoint(Features& features, bool active);
    };
}

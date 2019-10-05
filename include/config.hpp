#pragma once

#include <vector>
#include <iostream>
#include <fstream>  
#include <dirent.h>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;

class Config {
private:
    int mVerbose;
    int mPyramidNum;
    int mFeatureNum;
    int mAlignIterCnt;
    int mGridCell;
    int mImageWidth; 
    int mImageHeight;
    int mMinTrackThr;
    int mMinInlierThr;
    int mMinDispartyThr; 
    int mMinCornerThr;
    int mKeyFrameNum; 
    int mCloseKeyFrameCnt; 
    int mFeatureMatchMinThr;
    int mPoseOptimizeIterCnt;
    int mPoseOptimizeInlierThr;
    int mStructOptimizeIterCnt;
    int mStructOptimizePointCnt;
    int mTrackExminFeatureThr;
    int mTrackMinFeatureThr;
    int mTrackMoveDistance;
    int mDepthFilterTrackMaxGap;
    int mDepthFilterIterCnt;
    float mProjectRatioThr;
    float mPyramidFactor;
    float mMinProjError;
    float mEPS;
    float mfx, mfy, mcx, mcy, md0, md1, md2, md3, md4; 
    
    static Config* mInstance;

private:
    Config(string filePath);

public:
    static void initInstance(string configFile) { 
        if (mInstance) {
            delete mInstance;
        }
        mInstance = new Config(configFile);
    }

    static Config* getInstance() { return mInstance; }

    // verbose
    inline static int   verbose()   { return mInstance->mVerbose; }
    inline static float EPS()       { return mInstance->mEPS;     }

    // pyraimd parameters
    inline static int   pyramidNumber()  { return mInstance->mPyramidNum;   }
    inline static float pyramidFactor()  { return mInstance->mPyramidFactor; }

    // feature parameters
    inline static int   width()  { return mInstance->mImageWidth;  }
    inline static int   height() { return mInstance->mImageHeight; }
    inline static int   featureNumber()  { return mInstance->mFeatureNum;   }
    inline static int   gridCellNumber() { return mInstance->mGridCell;     }

    // threshold
    inline static int   minTrackThr()    { return mInstance->mMinTrackThr;  }
    inline static int   minInlierThr()   { return mInstance->mMinInlierThr; }
    inline static int   minDispartyThr() { return mInstance->mMinDispartyThr; }
    inline static int   minCornerThr()   { return mInstance->mMinCornerThr; }
    inline static float minProjError()  { return mInstance->mMinProjError;  }

    // map parameters
    inline static int   keyFrameNum()      { return mInstance->mKeyFrameNum;      }
    inline static float projectRatioThr()  { return mInstance->mProjectRatioThr;  }
    inline static int   closeKeyFrameCnt() { return mInstance->mCloseKeyFrameCnt; }

    // feature alignment
    inline static int   alignIterCnt()       { return mInstance->mAlignIterCnt;       }
    inline static int   featureMatchMinThr() { return mInstance->mFeatureMatchMinThr; }

    // pose optimize
    inline static int   poseOptimizeIterCnt()   { return mInstance->mPoseOptimizeIterCnt;   }
    inline static int   poseOptimizeInlierThr() { return mInstance->mPoseOptimizeInlierThr; }

    // struct optimize
    inline static int   structOptimizeIterCnt()  { return mInstance->mStructOptimizeIterCnt;  }
    inline static int   structOptimizePointCnt() { return mInstance->mStructOptimizePointCnt; }

    // track parameter
    inline static int   trackExminFeatureThr()   { return mInstance->mTrackExminFeatureThr; }
    inline static int   trackMinFeatureCnt()     { return mInstance->mTrackMinFeatureThr;   }
    inline static int   trackMoveDistance()      { return mInstance->mTrackMoveDistance;    }

    // depth filter
    inline static int   depthFilterTrackMaxGap() { return mInstance->mDepthFilterTrackMaxGap; }
    inline static int   depthFilterIterCnt()     { return mInstance->mDepthFilterIterCnt;     }

    // instrinsc
    inline static float fx() { return mInstance->mfx; }
    inline static float fy() { return mInstance->mfy; }
    inline static float cx() { return mInstance->mcx; }
    inline static float cy() { return mInstance->mcy; }
    inline static float d0() { return mInstance->md0; }
    inline static float d1() { return mInstance->md1; }
    inline static float d2() { return mInstance->md2; }
    inline static float d3() { return mInstance->md3; }
    inline static float d4() { return mInstance->md4; }
};

int loadDirectory(string dir_path, std::vector<std::string>& file_list, string pattern);
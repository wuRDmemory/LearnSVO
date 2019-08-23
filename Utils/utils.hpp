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
    int mPyramidNum, mFeatureNum;
    int mImageWidth, mImageHeight;
    int mGridCell;
    float mPyramidFactor;
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

    inline static int   getPyramidNumber() { return mInstance->mPyramidNum; }
    inline static int   getFeatureNumber() { return mInstance->mFeatureNum; }
    inline static int   getImageWidth()  { return mInstance->mImageWidth;  }
    inline static int   getImageHeight() { return mInstance->mImageHeight; }
    inline static int   getGridCellNumber() { return mInstance->mGridCell; }
    inline static float getPyramidFactor() { return mInstance->mPyramidFactor; }
};

int loadDirectory(string dir_path, std::vector<std::string>& file_list, string pattern);

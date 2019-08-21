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
    int pyramidNum;
    float pyramidFactor;
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

    inline static int   getPyramidNumber() { return mInstance->pyramidNum; }
    inline static float getPyramidFactor() { return mInstance->pyramidFactor; }
};

int loadDirectory(string dir_path, std::vector<std::string>& file_list, string pattern);

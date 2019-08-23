#include "utils.hpp"

Config* Config::mInstance = NULL;

Config::Config(string configFile) {
    cv::FileStorage file(configFile, cv::FileStorage::READ);
    if (!file.isOpened()) {
        throw std::io_errc();
    }

    mGridCell    = static_cast<int>(file["grid_cell"]);
    mImageWidth  = static_cast<int>(file["image_width"]);
    mImageHeight = static_cast<int>(file["image_height"]);
    mPyramidNum  = static_cast<int>(file["pyr_num"]);
    mFeatureNum  = static_cast<int>(file["ftr_num"]);
    mPyramidFactor = static_cast<float>(file["pyr_factor"]);
}

int loadDirectory(std::string dir_path, std::vector<std::string>& file_list, string pattern) {
    int pattern_len = pattern.size();
    DIR * dir;
    struct dirent * ptr;
    string x, dirPath;
    dir = opendir(dir_path.c_str()); //打开一个目录
    while((ptr = readdir(dir)) != NULL) {
        x = ptr->d_name;
        if (x == "." or x == "..")
            continue;
        if (x.find(pattern) == string::npos) 
            continue;
        dirPath = dir_path + "/" + x;
        // printf(">>> file_path : %s\n", dirPath.c_str()); //输出文件绝对路径
        file_list.emplace_back(x.substr(0, x.size()-pattern_len));
    }
    closedir(dir);

    sort(file_list.begin(), file_list.end(), [](const string& a, const string& b) {
        return atof(a.c_str()) < atof(b.c_str());
    });

    int N = file_list.size();
    file_list.resize(N);
    return N;
}

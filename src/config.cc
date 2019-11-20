#include "config.hpp"

Config* Config::mInstance = NULL;

Config::Config(string configFile) {
    cv::FileStorage file(configFile, cv::FileStorage::READ);
    if (!file.isOpened()) {
        throw std::io_errc();
    }

    mVerbose     = static_cast<int>(file["verbose"]);
    mEPS         = static_cast<float>(file["eps"]);

    mGridCell    = static_cast<int>(file["grid_cell"]);
    mImageWidth  = static_cast<int>(file["image_width"]);
    mImageHeight = static_cast<int>(file["image_height"]);
    mPyramidNum  = static_cast<int>(file["pyr_num"]);
    mFeatureNum  = static_cast<int>(file["ftr_num"]);

    mKeyFrameNum      = static_cast<int>(file["key_frame_size"]);
    mProjectRatioThr  = static_cast<float>(file["project_ratio_thr"]);
    mCloseKeyFrameCnt = static_cast<int>(file["close_key_frame_cnt"]);
    mFeatureMatchMinThr = static_cast<int>(file["feature_track_min_thr"]);

    mMinTrackThr  = static_cast<int>(file["min_track_thr"]);
    mMinInlierThr = static_cast<int>(file["min_inlier_thr"]);
    mMinCornerThr = static_cast<int>(file["min_corner_thr"]);
    mMinDispartyThr = static_cast<int>(file["min_disparty_thr"]);

    mPyramidFactor = static_cast<float>(file["pyr_factor"]);
    mMinProjError  = static_cast<float>(file["min_proj_error"]);

    mAlignIterCnt  = static_cast<int>(file["align_iter_cnt"]);

    mPoseOptimizeIterCnt   = static_cast<int>(file["pose_optimize_iter_cnt"]);
    mPoseOptimizeInlierThr = static_cast<int>(file["pose_optimize_inlier_thr"]);

    mStructOptimizeIterCnt   = static_cast<int>(file["struct_optimize_iter_cnt"]);
    mStructOptimizePointCnt  = static_cast<int>(file["struct_optimize_point_cnt"]);

    mTrackExminFeatureThr  = static_cast<int>(file["track_exmin_feature_thr"]);
    mTrackMinFeatureThr    = static_cast<int>(file["track_min_feature_thr"]);
    mTrackMoveDistance     = static_cast<int>(file["track_move_distance"]);

    mDepthFilterTrackMaxGap = static_cast<int>(file["depth_filter_track_max_gap"]);
    mDepthFilterIterCnt     = static_cast<int>(file["depth_filter_iter_cnt"]);
    mDepthFilterNCCScore    = static_cast<float>(file["depth_filter_ncc_score"]);
    mDepthFilterSigmaThr    = static_cast<float>(file["depth_filter_sigma_thr"]);

    mWaitSeconds            = static_cast<float>(file["waitSeconds"]);
    
    cv::FileNode node = file["instrinsc"];
    { // instrinsc
        mfx = static_cast<float>(node["fx"]);
        mfy = static_cast<float>(node["fy"]);
        mcx = static_cast<float>(node["cx"]);
        mcy = static_cast<float>(node["cy"]);

        md0 = static_cast<float>(node["d0"]);
        md1 = static_cast<float>(node["d1"]);
        md2 = static_cast<float>(node["d2"]);
        md3 = static_cast<float>(node["d3"]);
        md4 = static_cast<float>(node["d4"]);
    }

    node = file["viewer"];
    { // instrinsc
        mViewKeyFrameSize      = static_cast<float>(node["KeyFrameSize"]);
        mViewKeyFrameLineWidth = static_cast<float>(node["KeyFrameLineWidth"]);
        mViewGraphLineWidth    = static_cast<float>(node["GraphLineWidth"]);
        mViewGraphLineWidth    = static_cast<float>(node["GraphLineWidth"]);

        mViewPointSize         = static_cast<float>(node["PointSize"]);
        mViewCameraSize        = static_cast<float>(node["CameraSize"]);
        mViewCameraLineWidth   = static_cast<float>(node["CameraLineWidth"]);
        mViewViewpointX        = static_cast<float>(node["ViewpointX"]);
        mViewViewpointY        = static_cast<float>(node["ViewpointY"]);
        mViewViewpointZ        = static_cast<float>(node["ViewpointZ"]);
        mViewViewpointF        = static_cast<float>(node["ViewpointF"]);
    }
}

int loadDirectory(std::string dir_path, std::vector<std::string>& file_list, string pattern) {
    int pattern_len = pattern.size();
    DIR * dir;
    struct dirent * ptr;
    string x, dirPath;
    dir = opendir(dir_path.c_str()); //打开一个目录
    
    if (!dir) {
        LOG(INFO) << ">>> [loadDirectory] Can not open directory";
        return -1;
    }

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

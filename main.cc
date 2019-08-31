#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "utils.hpp"
#include "vo.hpp"

using namespace std;

const string config_file = "/home/ubuntu/Projects/SLAM/LearnSVO/config.yml";
const string dirPath = "/home/ubuntu/Datum/datum/SLAM/rgbd_dataset_freiburg1_xyz/rgb/";
const string pattern = ".png";

int main(int argc, char* argv[]) {
    // glog config
    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    // load all image
    vector<string> image_list;
    { // TODO: load the dir files
        LOG(INFO) << ">>> load " << loadDirectory(dirPath, image_list, pattern) << " images";
    }

    mSVO::VO vo(config_file);
    // main loop
    double last_timestamp = -1.0;
    for (string& image_name: image_list) {
        double timestamp = timestamp = atof(image_name.c_str());
        if (timestamp < last_timestamp + 0.1f) { // control frequency
            continue;
        }
        last_timestamp = timestamp;
        cv::Mat image; {
            string image_path = dirPath + image_name + pattern;
            LOG(INFO) << ">>> image path: " << image_path;
            image = cv::imread(image_path, 0);
        }

        vo.addNewFrame(image, timestamp);
    }
    
    google::ShutdownGoogleLogging();
    return 1;
}

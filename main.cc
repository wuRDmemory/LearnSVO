#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include <image_align.hpp>
#include <utils.hpp>

using namespace std;

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

    // main loop
    for (string& image_name: image_list) {
        double timestamp = 0.0;
        cv::Mat image;
        {
            timestamp = atof(image_name.c_str());
            string image_path = dirPath + "/" + image_name;
            image = cv::imread(image_path, 0);
        }

        
    }
    
    google::ShutdownGoogleLogging();
    return 1;
}

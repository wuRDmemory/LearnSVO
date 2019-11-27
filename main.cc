#include <opencv2/opencv.hpp>
#include <glog/logging.h>

#include "viewer.hpp"
#include "config.hpp"
#include "vo.hpp"

using namespace std;
using namespace mSVO;

bool VIEWER = false;

const string config_file = "/home/ubuntu/Projects/SLAM/LearnSVO/config.yml";
const string pattern     = ".png";
string dirPath           = "/home/ubuntu/Documents/DATA/SLAM/rgbd_dataset_freiburg1_xyz/rgb/";


const char* keys = {
    "{h help   | false | use camera or not}"
    "{s source | ""    | source directory }"
    "{v viewer | false | show the viewer  }"
};

int argparse(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);

    if (parser.get<bool>("h")) {
        parser.printMessage();
        return 0;
    }

    dirPath = parser.get<string>("s");
    VIEWER  = parser.get<bool>("v");
    
    if (dirPath == "") {
        parser.printMessage();
        return 0;
    }

    LOG(INFO) << ">>> [argparse] Directory: " << dirPath;
    LOG(INFO) << ">>> [argparse] Viewer   : " << (VIEWER ? "TRUE" : "FALSE");
    return 1;
}

int main(int argc, char* argv[]) {
    // glog config
    // FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    
    // argue parse
    if (0 == argparse(argc, argv)) {
        return -1;
    }
    // load all image
    vector<string> image_list;
    { // TODO: load the dir files
        int N = loadDirectory(dirPath, image_list, pattern);
        LOG(INFO) << ">>> load " << N << " images";
        if (N < 0) return -1;
    }

    // vo class
    VO vo(config_file);
    vo.setup();

    // main loop
    double last_timestamp = -1.0;
    for (string& image_name: image_list) {
        double timestamp = timestamp = atof(image_name.c_str());
        if (timestamp < last_timestamp + 0.05f) { // control frequency
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

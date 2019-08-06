#include <opencv2/opencv.hpp>
#include <image_align.hpp>
#include <utils.hpp>

using namespace std;

const string dirPath = "/home/ubuntu/Datum/datum/SLAM/rgbd_dataset_freiburg1_xyz/rgb/";
const string pattern = ".png";

int main(int argc, char* argv[]) {
    vector<string> image_list;
    { // TODO: load the dir files
        printf(">>> load %d images\n", loadDirectory(dirPath, image_list, pattern));
        // for (const string& file_name : image_list) {
        //     string abs_path = dirPath + file_name + pattern;
        //     printf(">>> file_path: %s\n", abs_path.c_str());
        // } 
    }

    
    return 1;
}

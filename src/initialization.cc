#include "initialization.hpp"
#include "utils.hpp"

namespace mSVO {
    KltHomographyInit::KltHomographyInit() { 
        mDetector = cv::FastFeatureDetector::create();
    }

    InitResult KltHomographyInit::addFirstFrame(FramePtr frameRef) {
        reset();

    }


    bool KltHomographyInit::detectCorner(FramePtr frame, 
                      vector<cv::Point>& points, 
                      vector<Eigen::Vector3f>& ftrs) {
        
    }
}

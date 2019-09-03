#include "image_align.hpp"

namespace mSVO {
    ImageAlign::ImageAlign(int minlevel, int maxlevel, int iterCnt): mMinLevel(minlevel), mMaxLevel(maxlevel), mIterCnt(iterCnt){
        ;
    }

    ImageAlign::~ImageAlign() {
        ;
    }

    void ImageAlign::prepareData(FramePtr refFrame, int level) {
        const int border = halfPatchSize+1;
        const cv::Mat& ref_img = refFrame->imagePyr()[level];
        const int stride  = ref_img.cols;
        const float scale = 1.0f/(1<<level);
        

    }
}

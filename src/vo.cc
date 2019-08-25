#include "vo.hpp"
#include "utils.hpp"

namespace mSVO {
    VO::VO(const string config_file): FramePtr(NULL), mCameraModel(NULL), updateLevel(UPDATE_FIRST) {
        Config::initInstance(config_file);
    }
}

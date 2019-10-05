#include "struct_optimize.hpp"

namespace mSVO {
    bool StructOptimize::optimize(FramePtr frame, const int nIter, const int maxPoints) {
        deque<LandMarkPtr> pts;
        auto& features = frame->obs();
        for (auto begin = features.begin(), end = features.end(); begin != end; begin++) {
            if ((*begin)->mLandmark) {
                continue;
            }
            pts.push_back((*begin)->mLandmark);
        }
        int cnt = min(maxPoints, (int)pts.size());
        std::nth_element(pts.begin(), pts.begin() + cnt, pts.end(), [](LandMarkPtr& a, LandMarkPtr& b) {
            return a->nOptimizeFrameId > b->nOptimizeFrameId;
        });

        for (auto begin = pts.begin(), end = pts.end(); begin != end; ++begin) {
            if ((*begin)->optimize(nIter)) {
                (*begin)->nOptimizeFrameId = frame->ID();
            }
        }
        return true;
    }
}

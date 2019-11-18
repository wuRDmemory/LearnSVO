#include "struct_optimize.hpp"

namespace mSVO {
    bool StructOptimize::optimize(FramePtr frame, const int nIter, const int maxPoints) {
        deque<LandMarkPtr> pts;
        auto& features = frame->obs();
        for (auto begin = features.begin(), end = features.end(); begin != end; begin++) {
            LandMarkPtr landmark = (*begin)->mLandmark;
            if (!landmark or landmark->type != LandMark::LANDMARK_TYPR::GOOD) {
                continue;
            }
            pts.push_back(landmark);
        }
        
        int cnt = min(maxPoints, (int)pts.size());
        std::nth_element(pts.begin(), pts.begin() + cnt, pts.end(), [](LandMarkPtr& a, LandMarkPtr& b) {
            return a->nOptimizeFrameId > b->nOptimizeFrameId;
        });

        int success = 0;
        for (auto begin = pts.begin(), end = pts.end(); begin != end; ++begin) {
            if ((*begin)->optimize(nIter)) {
                (*begin)->nOptimizeFrameId = frame->ID();
                success++;
            }
        }
        LOG(INFO) << ">>> [StructOptimize] Optimzed success: " << success;
        return true;
    }
}

#include "landmark.hpp"

namespace mSVO {
    int LandMark::id = 0;

    LandMark::LandMark(Vector3f xyz): mId(id++), mXYZ(xyz),
        nProjectFrameFailed(0), nProjectFrameSuccess(0) {
        std::for_each(mFeatures.begin(), mFeatures.end(), [](FeaturePtr& feature) {
            if (feature) delete feature;
            feature = NULL;
        });
    }

    LandMark::LandMark(Vector3f xyz, FeaturePtr feature): mId(id++), mXYZ(xyz),
        nProjectFrameFailed(0), nProjectFrameSuccess(0) {
        std::for_each(mFeatures.begin(), mFeatures.end(), [](FeaturePtr& feature) {
            if (feature) delete feature;
            feature = NULL;
        });

        mFeatures.push_back(feature);
    }

    void LandMark::addFeature(FeaturePtr feature) {
        mFeatures.push_back(feature);
    }
}

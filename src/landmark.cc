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

    void LandMark::findClosestObs(Vector3f& framePose, FeaturePtr feature) const {
        Vector3f Vlf = mXYZ - framePose;
        Vlf.normalize();
        float minCosAngle = FLT_MAX;
        for (int i = 0, N = mFeatures.size; i < N; ++i) {
            FeaturePtr& f = mFeatures[i];
            Vector3f refPose = f->mFrame->pose().translation().cast<float>();
            Vector3f Vlr  = mXYZ - refPose;
            Vlr.normalize();
            float cosAngle = Vlf.dot(Vlr);

            if (cosAngle < minCosAngle) {
                minCosAngle = cosAngle;
                feature = f;
            }
        }

        if (minCosAngle < 0.5) {
            feature = NULL;
        }
    }
}

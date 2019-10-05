#include "landmark.hpp"
#include "config.hpp"

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
        for (int i = 0, N = mFeatures.size(); i < N; ++i) {
            FeaturePtr f = mFeatures[i];
            Vector3f refPose = f->mFrame->twc();
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

    bool LandMark::optimize(int nIter) {
        Vector3f oldXYZ = mXYZ;
        Matrix<float, 3, 3> A;
        Matrix<float, 3, 1> b;

        float chi2 = 0, oldChi2 = 0;
        int   cnt  = 0;
        for (int i = 0; i < nIter; i++) {
            chi2 = 0;
            for (int  j = 0, N = mFeatures.size(); i < N; i++) {
                FeaturePtr feature = mFeatures[i];
                if (!feature->mFrame) {
                    continue;
                }

                Matrix3f Rcw = feature->mFrame->Rwc().inverse().toRotationMatrix();
                Vector3f xyz = feature->mFrame->world2camera(mXYZ);
                Vector2f xy  = xyz.head(2) / xyz(2);
                Vector2f error = feature->mDirect.head(2) - xy;

                Matrix<float, 2, 3> J;
                LandMark::jacobian_uv2xyz(xyz, Rcw, J);

                A.noalias() += J.transpose()*J;
                b.noalias() -= J.transpose()*error;

                chi2 += error.squaredNorm();
                cnt ++;
            }

            Vector3f dx = A.ldlt().solve(b);
            if ((i != 0 && chi2 < oldChi2) || isnan(dx[0]) || isinf(dx[0])) {
                if (Config::verbose()) {
                    LOG(INFO) << ">>> [ldmk optimize] " << i << " failed, chi2 update: " << oldChi2 << "->" << chi2;
                }
                mXYZ = oldXYZ;
                break;
            }

            if (Config::verbose()) {
                LOG(INFO) << ">>> [ldmk optimize] " << i << " success, chi2 update: " << oldChi2 << "->" << chi2 << ": " << dx.norm();
            }
            mXYZ    = oldXYZ + dx;
            oldXYZ  = mXYZ;
            oldChi2 = chi2;

            if (dx.norm() < Config::EPS()) {
                break;
            }
        }
        return true;
    }
}

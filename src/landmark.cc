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

        mFeatures.push_front(feature);
    }

    void LandMark::addFeature(FeaturePtr feature) {
        mFeatures.push_front(feature);
    }

    void LandMark::findClosestObs(Vector3f& framePose, FeaturePtr& feature) const {
        Vector3f Vlf = framePose - mXYZ; Vlf.normalize();
        float minCosAngle = 0;
        for (auto it = mFeatures.begin(); it != mFeatures.end(); it++) {
            FeaturePtr f = *it;
            Vector3f refPose = f->mFrame->twc();
            Vector3f Vlr  = refPose - mXYZ; Vlr.normalize();
            float cosAngle = Vlf.dot(Vlr);

            if (cosAngle > minCosAngle) {
                minCosAngle = cosAngle;
                feature = f;
            }
        }

        if (minCosAngle < 0.5) {
            feature = NULL;
        }
    }

    void LandMark::safeRemoveLandmark() {
        auto it = mFeatures.begin();
        while (it != mFeatures.end()) {
            (*it)->mLandmark = NULL;
            it++;
        }
    }

    void LandMark::safeRemoveFeature(FeaturePtr feature) {
        bool clearAll = mFeatures.size() < 2;
        auto it = mFeatures.begin();
        while (it != mFeatures.end()) {
            if (clearAll) {
                (*it)->mLandmark = NULL;
                it = mFeatures.erase(it);
            } else if ((*it) == feature) {
                (*it)->mLandmark = NULL;
                it = mFeatures.erase(it);
                break;
            }
        }
        if (clearAll) this->type = DELETE;
    }

    bool LandMark::optimize(int nIter) {
        Vector3f oldXYZ = mXYZ;
        Matrix<float, 3, 3> A;
        Matrix<float, 3, 1> b;

        bool  opti = true;
        float chi2 = 0, oldChi2 = 0;
        int   cnt  = 0;
        for (int i = 0; i < nIter; i++) {
            A.setZero(); b.setZero();
            chi2 = 0; cnt = 0;
            for (auto it = mFeatures.begin(); it != mFeatures.end(); it++) {
                FeaturePtr feature = (*it);
                if (!feature->mFrame) {
                    continue;
                }

                Matrix3f Rcw  = feature->mFrame->Rwc().inverse().toRotationMatrix();
                Vector3f Pc   = feature->mFrame->world2camera(mXYZ);
                Vector2f Pcxy = Pc.head(2) / Pc(2);
                Vector3f Pt   = feature->mDirect;
                Vector2f Ptxy = Pt.head(2) / Pt(2);
                Vector2f error = Ptxy - Pcxy;
                error /= (1<<feature->mLevel);

                Matrix<float, 2, 3> J;
                LandMark::jacobian_uv2xyz(Pc, Rcw, J);

                A.noalias() += J.transpose()*J;
                b.noalias() -= J.transpose()*error;

                chi2 += error.squaredNorm();
                cnt ++;
            }
            chi2 /= cnt;

            Vector3f dx = A.ldlt().solve(b);
            if ((i != 0 && chi2 < oldChi2) || isnan(dx[0]) || isinf(dx[0])) {
                if (Config::verbose()) {
                    LOG(INFO) << ">>> [ldmk optimize] " << i << " failed, chi2 update: " << oldChi2 << "->" << chi2;
                }
                mXYZ = oldXYZ;
                opti = false;
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
        return opti;
    }
}

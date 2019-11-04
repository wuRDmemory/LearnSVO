#include "pose_optimize.hpp"
#include "config.hpp"

namespace mSVO {

    Quaternionf  PoseOptimize::mRcw    = Quaternionf::Identity(); 
    Quaternionf  PoseOptimize::mOldRcw = Quaternionf::Identity();
    Vector3f     PoseOptimize::mtcw    = Vector3f::Zero();
    Vector3f     PoseOptimize::mOldtcw = Vector3f::Zero();

    bool PoseOptimize::optimize(FramePtr frame, int nIter, float projError, float& estscale, float& initChi2, float& endChi2, int& obsnum) {
        const float EPS = Config::EPS();
        // set the model
        mRcw = frame->Rwc().inverse();
        mtcw = mRcw*frame->twc()*-1;

        mOldRcw = mRcw;
        mOldtcw = mtcw;
        
        float chi2 = 0, oldChi2 = 0;
        int   cnt  = 0;
        Matrix<float, 6, 6> A;
        Matrix<float, 6, 1> b;
        A.setZero();
        b.setZero();

        auto& features = frame->obs();
        for (auto it = features.begin(), end = features.end(); it != end; it++) {
            if (!(*it)->mLandmark) {
                continue;
            }
            Vector3f Pw   = (*it)->mLandmark->xyz();
            Vector3f Pc   = mRcw*Pw + mtcw;
            Vector2f Pcxy = Pc.head<2>() / Pc(2);
            Vector3f Pf   = (*it)->mDirect;
            Vector2f Ptxy = Pf.head<2>() / Pf(2);
            Vector2f error = Ptxy - Pcxy;
            chi2 += error.squaredNorm();
            cnt ++;
        }

        estscale = chi2 / cnt;
        initChi2 = estscale;
        // TODO: set the old model as model
        for (int i = 0; i < nIter; i++) {
            A.setZero();
            b.setZero();
            cnt  = 0;
            chi2 = 0;
            // compute the H and b matrix
            for (auto it = features.begin(), end = features.end(); it != end; it++) {
                if (!(*it)->mLandmark) {
                    continue;
                }
                Vector3f Pw   = (*it)->mLandmark->xyz();
                Vector3f Pc   = mRcw*Pw + mtcw;
                Vector2f Pcxy = Pc.head<2>() / Pc(2);
                Vector3f Pf   = (*it)->mDirect;
                Vector2f Ptxy = Pf.head<2>() / Pf(2);
                Vector2f error = Ptxy - Pcxy;
                error /= (1<<(*it)->mLevel);

                Matrix<float, 2, 6> J;
                Frame::jacobian_uv2se3New(Pc, mRcw*Pw, J);
                float weight = 1.0f; //TukeyWeightFunction(error.squaredNorm()/estscale);
                A.noalias() += J.transpose()*J*weight;
                b.noalias() -= J.transpose()*error*weight;

                chi2 += error.squaredNorm()*weight;
                cnt ++;
            }
            chi2 /= cnt;
            
            Matrix<float, 6, 1> dx = A.ldlt().solve(b);
            if ((i != 0 && chi2 > oldChi2) || isnan(dx[0]) || isinf(dx[0])) {
                if (Config::verbose()) {
                    LOG(INFO) << ">>> [pose optimize] failed!!! Iter "<< i << ", chi2 update: " << oldChi2 << "->" << chi2;
                }
                mRcw = mOldRcw;
                mtcw = mOldtcw;
                break;
            }

            updateModel(dx);
            if (Config::verbose())
                LOG(INFO) << ">>> [pose optimize] success!!! Iter "<< i << ", chi2 update: " << oldChi2 << "->" << chi2;
            oldChi2 = chi2;
            mOldRcw = mRcw;
            mOldtcw = mtcw;

            if (dx.norm() < EPS) {
                break;
            }
        }

        // change the covariance to camera ordination
        frame->Rwc() = mRcw.inverse();
        frame->twc() = mRcw.inverse()*mtcw*-1;
        frame->covariance() = (A*frame->camera()->errorMultiplier2()).inverse();
        
        chi2 = 0; cnt = 0; obsnum = 0;
        for (auto it = features.begin(), end = features.end(); it != end; it++) {
            if (!(*it)->mLandmark) {
                continue;
            }
            Vector3f Pw   = (*it)->mLandmark->xyz();
            Vector3f Pc   = mRcw*Pw + mtcw;
            Vector2f Pcxy = Pc.head<2>() / Pc(2);
            Vector3f Pf   = (*it)->mDirect;
            Vector2f Ptxy = Pf.head<2>() / Pf(2);
            Vector2f error = Ptxy - Pcxy;
            chi2 += error.squaredNorm();
            cnt ++;

            error /= (1<<(*it)->mLevel);
            if (error.norm() <= projError) {
                obsnum++;
            }
        }
        endChi2 = chi2 / cnt;
        return true;
    }

    bool PoseOptimize::updateModel(Matrix<float, 6, 1>& dx) {
        Vector3f dq = dx.bottomRows(3);
        Vector3f dt = dx.topRows(3);

        mRcw = deltaQuaternion(dq)*mOldRcw;
        mtcw = mOldtcw + dt;
        mRcw.normalize();

        return true;
    }

    float PoseOptimize::TukeyWeightFunction(float x) {
        const float x_square = x * x;
        if(x_square <= 25) {
            const float tmp = 1.0f - x_square / 25;
            return tmp * tmp;
        } else {
            return 0.0f;
        }
    }

    float PoseOptimize::middleScaleEstimate(vector<float>& errors) {
        int N = errors.size();
        std::nth_element(errors.begin(), errors.begin() + N/2, errors.end());
        return errors[N/2];
    }
}


#include "pose_optimize.hpp"
#include "config.hpp"

namespace mSVO {

    Quaternionf  PoseOptimize::mRcw    = Quaternionf::Identity(); 
    Quaternionf  PoseOptimize::mOldRcw = Quaternionf::Identity();
    Vector3f     PoseOptimize::mtcw    = Vector3f::Zero();
    Vector3f     PoseOptimize::mOldtcw = Vector3f::Zero();

    bool PoseOptimize::optimize(FramePtr frame, int nIter, float projError, float& estscale, float& initChi2, float& endChi2, int& obsnum) {
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
        vector<float> initErrors; initErrors.reserve(features.size());
        for (auto it = features.begin(), end = features.end(); it != end; it++) {
            if (!(*it)->mLandmark) {
                continue;
            }
            Vector3f point = (*it)->mLandmark->xyz();
            Vector3f pc = mRcw*point + mtcw;
            Vector2f px = pc.head<2>() / pc(2);
            Vector2f pt = ((*it)->mDirect).head<2>();
            Vector2f error = pt - px;
            initErrors.emplace_back(error.squaredNorm());
        }

        if (initErrors.empty()) {
            return false;
        }

        estscale = middleScaleEstimate(initErrors);
        initChi2 = estscale;
        // TODO: set the old model as model
        for (int i = 0; i < nIter; i++) {
            A.setZero();
            b.setZero();
            // compute the H and b matrix
            for (auto it = features.begin(), end = features.end(); it != end; it++) {
                if (!(*it)->mLandmark) {
                    continue;
                }
                Vector3f point = (*it)->mLandmark->xyz();
                Vector3f pc = mRcw*point + mtcw;
                Vector2f px = pc.head<2>() / pc(2);
                Vector2f pt = ((*it)->mDirect).head<2>();
                Vector2f error = pt - px;
                error /= (1<<(*it)->mLevel);

                Matrix<float, 2, 6> J;
                Frame::jacobian_uv2se3New(pc, mRcw*point, J);
                float weight = TukeyWeightFunction(error.squaredNorm()/estscale);
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

            if (dx.norm() < 1e-7f) {
                break;
            }
        }

        // change the covariance to camera ordination
        frame->covariance() = (A*frame->camera()->errorMultiplier2()).inverse();
        
        obsnum = 0;
        initErrors.clear(); initErrors.reserve(features.size());
        for (auto it = features.begin(), end = features.end(); it != end; it++) {
            if (!(*it)->mLandmark) {
                continue;
            }
            Vector3f point = (*it)->mLandmark->xyz();
            Vector3f pc = mRcw*point + mtcw;
            Vector2f px = pc.head<2>() / pc(2);
            Vector2f pt = ((*it)->mDirect).head<2>();
            Vector2f error = pt - px;
            initErrors.emplace_back(error.squaredNorm());

            error /= (1<<(*it)->mLevel);
            if (error.norm() <= projError) {
                obsnum++;
            }
        }
        endChi2 = middleScaleEstimate(initErrors);
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


#include "bundle_adjustment.hpp"
#include "config.hpp"

namespace mSVO {
    BundleAdjustment::BundleAdjustment(int maxIter, MapPtr map): mMaxIter(maxIter), mMap(map) {
        ;
    }

    bool BundleAdjustment::run() { 
        // initial ceres
        const float focal = Config::fx();
        ceres::Problem problem;
        ceres::LossFunction*       lossfunction = new ceres::HuberLoss(5.0/focal);

        vector<double*> keyFramePoses;
        vector<double*> keyPointXYZ;
        
        list<FramePtr>& keyFrames = mMap->keyFrames();
        keyFramePoses.reserve(keyFrames.size());
        keyPointXYZ.reserve(keyFrames.size()*100);
        
        int i = 0, j = 0;
        auto it = keyFrames.begin();
        while (it != keyFrames.end()) {
            Frame* frame = it->get();
            Quaternionf Rcw = frame->Rwc().inverse();
            Vector3f    tcw = Rcw*frame->twc()*-1.0f;

            double* tmpPose = new double[7];
            tmpPose[0] = tcw.x(); tmpPose[0] = tcw.y(); tmpPose[0] = tcw.z(); 
            tmpPose[3] = Rcw.x(); tmpPose[4] = Rcw.y(); tmpPose[5] = Rcw.z(); tmpPose[6] = Rcw.w();
            keyFramePoses.push_back(tmpPose);

            ceres::LocalParameterization* poseLocal = new PoseLocalParameterization();
            problem.AddParameterBlock(tmpPose, 7, poseLocal);
            if (i == 0 || i == 1) {
                problem.SetParameterBlockConstant(tmpPose);
            }

            auto& obs = frame->obs();
            auto iter = obs.begin();
            while (iter != obs.end()) {
                FeaturePtr  feature  = *iter;
                LandMarkPtr landmark = feature->mLandmark;
                if (!landmark || landmark->type == LandMark::LANDMARK_TYPR::DELETE) {
                    iter++;
                    continue;
                }
                
                Vector3f xyz = landmark->xyz();
                Vector3f uvz = feature->mDirect;
                Vector2f uv  = uvz.head(2)/uvz(2);

                double* tmpXYZ = new double[3];
                tmpXYZ[0] = xyz(0); tmpXYZ[1] = xyz(1); tmpXYZ[2] = xyz(2);
                keyPointXYZ.push_back(tmpXYZ);

                problem.AddParameterBlock(tmpXYZ, 3);
                ceres::CostFunction* cost = new BACostFunction(uv);
                vector<double*> params = {tmpPose, tmpXYZ};
                problem.AddResidualBlock(cost, lossfunction, params);
                iter++;
                j++;
            }
            it++;
            i++;
        }
        keyPointXYZ.resize(j);
        LOG(INFO) << ">>> problem scale: vertex: " << i << " edge: " << j;

        ceres::Solver::Options options;
        options.max_num_iterations = mMaxIter;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type   = ceres::DOGLEG;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        i = 0, j = 0;
        it = keyFrames.begin();
        while (it != keyFrames.end()) {
            Frame* frame = it->get();
            double* &tmpPose = keyFramePoses[i];
            Vector3f    tcw(tmpPose[0], tmpPose[1], tmpPose[2]); 
            Quaternionf Rcw(tmpPose[6], tmpPose[3], tmpPose[4], tmpPose[5]);

            frame->Rwc() = Rcw.inverse();
            frame->twc() = Rcw.inverse()*tcw*-1.0f;

            auto& obs = frame->obs();
            auto iter = obs.begin();
            while (iter != obs.end()) {
                FeaturePtr feature   = *iter;
                LandMarkPtr landmark = feature->mLandmark;
                if (!landmark || landmark->type == LandMark::LANDMARK_TYPR::DELETE) {
                    iter++;
                    continue;
                }
                
                double* &tmpXYZ = keyPointXYZ[j];
                Vector3f xyz(tmpXYZ[0], tmpXYZ[1], tmpXYZ[2]);
                feature->mLandmark->xyz() = xyz;
                iter++;
                j++;
            }
            it++;
            i++;
        }
        assert(i == keyFrames.size());
        assert(j == keyPointXYZ.size());

        for (auto& pose : keyFramePoses)
            delete(pose);
        
        for (auto& xyz : keyPointXYZ)
            delete(xyz);
        
        return true;
    }
}

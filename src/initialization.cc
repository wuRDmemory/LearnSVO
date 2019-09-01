#include "initialization.hpp"


namespace mSVO {
    KltHomographyInit::KltHomographyInit() { 
        mGridCell  = Config::gridCellNumber();
        mImWidth   = Config::width();
        mImHeight  = Config::height();
        mFtrNumber = Config::featureNumber();
        mCellFtrNumber = mFtrNumber/(mGridCell*mGridCell);
        // initial grid cell
        int row_step = mImHeight / mGridCell;
        int col_step = mImWidth  / mGridCell;
        mGridCellRoi.resize(mGridCell*mGridCell);
        for (int i=0; i<mGridCell; i++) {
            for (int j=0; j<mGridCell; j++) {
                mGridCellRoi[i*mGridCell + j] = \
                    cv::Rect(cv::Point(i*col_step, j*row_step), 
                             cv::Point((i+1)*col_step-1, (j+1)*row_step-1));
            }
        }
    }

    InitResult KltHomographyInit::addFirstFrame(FramePtr currFrame) {
        reset();
        mFirstCorners.clear();
        detectCorner(currFrame, mFirstCorners);
        
        LOG(INFO) << ">>> [first frame]corner detected " << mFirstCorners.size();
        if (mFirstCorners.size() < Config::minCornerThr()) {
            LOG(ERROR) << ">>> [first frame] Too few corner detected " << mFirstCorners.size();
            return FAILURE;
        }
        mRefFrame = currFrame;
        return SUCCESS;
    }

    InitResult KltHomographyInit::addSecondFrame(FramePtr currFrame) {
        vector<cv::Point2f> nextCorners;
        vector<uchar> status;
        vector<float> error;
        cv::calcOpticalFlowPyrLK(mRefFrame->imagePyr()[0], currFrame->imagePyr()[0], mFirstCorners, nextCorners, status, error);

        int j = 0;
        mvk::CameraModel* camera = currFrame->camera();
        Features& ref_features = mRefFrame->obs();
        vector<Vector3f> ref_obs(status.size());
        vector<Vector3f> cur_obs(status.size());
        for (int i = 0; i < status.size(); i++) 
        if (status[i]) {
            Vector2f pref(mFirstCorners[i].x, mFirstCorners[i].y);
            Vector2f pcur(nextCorners[i].x,    nextCorners[i].y);
            mFirstCorners[j] = mFirstCorners[i];
            nextCorners[j]    = nextCorners[i];

            ref_obs[j] = camera->cam2world(pref);
            cur_obs[j] = camera->cam2world(pcur);
            j++;
        }
        mFirstCorners.resize(j);
        nextCorners.resize(j);
        ref_obs.resize(j);
        cur_obs.resize(j);
        LOG(INFO) << ">>> [second frame] track key point : " << j;

        if (j <= Config::minTrackThr()) {
            LOG(ERROR) << ">>> [second frame] too few track points!!!";
            return FAILURE;
        }
        // use fundamental matrix to shift some outlier
        const cv::Mat& K = camera->cvK();
        cv::Mat  mask;
        cv::Mat  F = cv::findFundamentalMat(mFirstCorners, nextPoints, mask, cv::FM_RANSAC, 3.0, 0.99);
        F.convertTo(F, CV_32F);
        cv::Mat  E = K.t()*F*K;
        cv::Mat  cvR1, cvR2, cvt;
        LOG(INFO) << ">>> [second frame] find fundamental done!!! inliers " << cv::sum(mask)[0];
        cv::decomposeEssentialMat(E, cvR1, cvR2, cvt);
        // LOG(INFO) << ">>> [second frame] find essential done!!!";
        Matrix3f ER1, ER2;
        Vector3f Et1, Et2;
        cv2eigen(cvR1, ER1); cv2eigen(cvR2, ER2);
        cv2eigen(cvt,  Et1); Et2 = -Et1;
        
        int best_inlier_cnt = INT_MIN, best_index = 0;
        vector<uchar> best_inliers;
        vector<float> best_depths;
        Matrix3f list_R[] = {ER1, ER1, ER2, ER2};
        Vector3f list_t[] = {Et1, Et2, Et1, Et2};
        for (int i = 0; i <4; i++) {
            vector<uchar> inliers;
            vector<float> depths;
            int good = computeInliers(list_R[i], list_t[i], camera, ref_obs, cur_obs, inliers, depths, Config::minProjError());
            LOG(INFO) << ">>> [second frame] good match is " << good;
            if (good > best_inlier_cnt) {
                best_index = i;
                best_inlier_cnt = good;
                best_inliers = std::move(inliers);
                best_depths  = std::move(depths);
            }
        }

        LOG(INFO) << ">>> [second frame] good inlier number: " << best_inlier_cnt;
        if (best_inlier_cnt < Config::minInlierThr()) {
            LOG(INFO) << ">>> [second frame] too few inliers!!!";
            return FAILURE;
        }

        auto* middle = best_depths.begin() + best_depths.size() / 2;
        std::nth_element(best_depths.begin(), middle, best_depths.end());
        float scale  = *middle;

        Matrix3f R21 = list_R[best_index];
        Vector3f t21 = list_t[best_index];
        t21 = t21*scale/t21.norm();

        Matrix3d R12d = +R21.transpose().cast<double>();
        Vector3d t12d = -R12d.dot(t21.cast<double>());
        currFrame->pose() = Sophus::SE3(R12d, t12d);
        // clean the ref and cur frame's features
        mRefFrame->obs().clear(); currFrame->obs().clear();
        for (int i = 0; i < best_inliers.size(); i++) 
        if (best_inliers[i]) {
            float depth = calcuDepth(R21, t21, ref_obs[i], cur_obs[i]);
            if (depth <= 0) {
                continue;
            }
            Vector3f xyz = ref_obs[i]*depth;
            LandMarkPtr ldmk = new LandMark(xyz);

            Vector2f px1(ref_obs[i].x, ref_obs[i].y), px2(cur_obs[i].x, cur_obs[i].y);
            FeaturePtr ref_feature = new Feature(mRefFrame, ldmk, px1, ref_obs[i], 1);
            FeaturePtr cur_feature = new Feature(currFrame, ldmk, px2, cur_obs[i], 1);

            mRefFrame->addFeature(ref_feature);
            currFrame->addFeature(cur_feature);

            ldmk->addFeature(ref_feature);
            ldmk->addFeature(cur_feature);
        }
        return SUCCESS;
    }

    int KltHomographyInit::computeInliers(Matrix3f& R21, Vector3f& t21, mvk::CameraModel* camera, vector<Vector3f>& pts1, vector<Vector3f>& pts2, \
                        vector<uchar>& inliers, vector<float>& depthes, float th) {
        int N = pts1.size();
        inliers.resize(N, 0);
        depthes.resize(N, 0);
        

        Vector3f O1 = Vector3f::Zero();
        Vector3f O2 = -R21.transpose()*t21;

        Matrix3f R11 = Matrix3f::Identity();
        Vector3f t11 = Vector3f::Zero();

        int good = 0;
        float errorMulti = camera->errorMultiplier2();
        for (int i = 0; i < N; i++) {
            Vector3f p1 = pts1[i];
            Vector3f p2 = pts2[i];

            float depth = calcuDepth(R21, t21, p1, p2);
            if (depth <= 0 or std::isnan(depth) or std::isinf(depth)) {
                continue;
            }

            Vector3f Pw = p1*depth;
            Vector3f line1 = O1 - Pw;
            Vector3f line2 = O2 - Pw;

            float theta = line1.dot(line2)/(line1.norm()*line2.norm());
            if (theta > 0.98) {
                continue;
            }

            float error1 = errorMulti * calcuProjError(R11, t11, Pw, p1);
            if (error1 > th) {
                continue;
            }
            float error2 = errorMulti * calcuProjError(R21, t21, Pw, p2);
            if (error2 > th) {
                continue;
            }
            inliers[i] = 1;
            depthes[i] = depth;
            good += 1;
        }
        return good;
    }

    bool KltHomographyInit::detectCorner(FramePtr frame, vector<cv::Point2f>& points) {
        cv::Mat& image = frame->imagePyr()[0];
        for (int i=0; i<mGridCell; i++) {
            for (int j=0; j<mGridCell; j++) {
                cv::Rect rroi = mGridCellRoi[i*mGridCell + j];
                cv::Point tl = rroi.tl();
                cv::Point br = rroi.br();
                cv::Mat roi  = image.rowRange(tl.y, br.y).colRange(tl.x, br.x);
                // find fast corner
                vector<cv::Point2f> corner;
                cv::goodFeaturesToTrack(roi, corner, mCellFtrNumber, 0.1, 10);
                for (int i=0; i<corner.size(); i++) {
                    Vector2f px(corner[i].x + tl.x, corner[i].y+tl.y);
                    points.emplace_back(corner[i].x + tl.x, corner[i].y + tl.y);
                    frame->addFeature(new Feature(frame, px, 1));
                }
            }
        }
        LOG(INFO) << ">>> detect corner done!!! all feature " << frame->obs().size();
        #if 0
        cv::Mat show = image.clone();
        cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
        for (int i = 0; i < points.size(); i++) {
            cv::circle(show, cv::Point(int(points[i].x), int(points[i].y)), 2, cv::Scalar(0, 255, 0), 1);
        }
        cv::imshow("show image", show);
        cv::waitKey();
        #endif
        return true;
    }

    void KltHomographyInit::reset() { 
        mRefFrame = NULL;
        mFirstCorners.clear();
    }
}

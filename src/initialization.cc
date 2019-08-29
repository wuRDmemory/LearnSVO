#include "initialization.hpp"


namespace mSVO {
    KltHomographyInit::KltHomographyInit() { 
        mGridCell  = Config::getGridCellNumber();
        mImWidth   = Config::getImageWidth();
        mImHeight  = Config::getImageHeight();
        mFtrNumber = Config::getFeatureNumber();
        mCellFtrNumber = mFtrNumber/mGridCell;
        // initial grid cell
        int row_step = mImHeight / mGridCell;
        int col_step = mImWidth  / mGridCell;
        mGridCellRoi.resize(mGridCell*mGridCell);
        for (int i=0; i<mGridCell; i++) {
            for (int j=0; j<mGridCell; j++) {
                mGridCellRoi[i*mGridCell + j] = \
                    cv::Rect(cv::Point(i*col_step, j*row_step), 
                             cv::Point(i*col_step, (j+1)*row_step));
            }
        }
    }

    InitResult KltHomographyInit::addFirstFrame(FramePtr frameRef) {
        reset();
        mFirstCorners.clear();
        detectCorner(frameRef, mFirstCorners);
        
        if (mFirstCorners.size() < 200) {
            LOG(ERROR) << ">>> Too few corner detected " << mFirstCorners.size();
            return FAILURE;
        }
        mRefFrame = frameRef;
        return SUCCESS;
    }

    InitResult KltHomographyInit::addSecondFrame(FramePtr frameRef) {
        vector<cv::Point2f> nextPoints;
        vector<uchar> status;
        vector<float> error;
        cv::calcOpticalFlowPyrLK(mRefFrame->mImagePyr[0], frameRef->mImagePyr[0], mFirstCorners, nextPoints, status, error);

        int j = 0;
        mvk::CameraModel* camera = frameRef->mCamera;
        Features& ref_features = mRefFrame->mObs;
        vector<Vector3f> ref_obs(status.size());
        vector<Vector3f> cur_obs(status.size());
        for (int i = 0; i < status.size(); i++) 
        if (status[i]) {
            Vector2f pref(mFirstCorners[i].x, mFirstCorners[i].y);
            Vector2f pcur(nextPoints[i].x,    nextPoints[i].y);
            mFirstCorners[j] = mFirstCorners[i];
            nextPoints[j]    = nextPoints[i];

            ref_obs[j] = camera->cam2world(pref);
            cur_obs[j] = camera->cam2world(pcur);
            j++;
        }
        ref_obs.resize(j);
        cur_obs.resize(j);

        if (j <= 8) {
            return FAILURE;
        }
        // use fundamental matrix to shift some outlier
        cv::Mat& K = camera->cvK();
        cv::Mat  F = cv::findFundamentalMat(mFirstCorners, nextPoints, cv::Mat(), cv::FM_RANSAC, 3.0, 0.99);
        cv::Mat  E = K.t()*F*K;
        cv::Mat  R1, R2, t;
        cv::decomposeEssentialMat(E, R1, R2, t);
        // Matrix3f ER1, ER2;
        // Vector3f Et1, Et2;
        // cv2eigen(R1, ER1);
        // cv2eigen(R2, ER2);
        // cv2eigen(t, Et1);
        // Et2 = -Et1;
        

        return SUCCESS;
    }

    int KltHomographyInit::computeInliers(Matrix3f& R21, Vector3f& t21, mvk::CameraModel* camera, vector<Vector3f>& pts1, vector<Vector3f>& pts2, \
                        vector<uchar>& inliers, vector<float>& depth, float th) {
        inliers.clear();
        inliers.resize(pts1.size(), 0);
        int N = inliers.size();

        Vector3f O1 = Vector3f::Zero();
        Vector3f O2 = -R21.transpose()*t21;

        Matrix3f R11 = Matrix3f::Identity();
        Vector3f t11 = Vector3f::Zero();

        float errorMulti = camera->errorMultiplier2();
        for (int i = 0; i < N; i++) {
            Vector3f p1 = pts1[i];
            Vector3f p2 = pts2[i];

            float depth = calcuDepth(R21, t21, p1, p2);
            if (std::isnan(depth) or std::isinf(depth)) {
                continue;
            }

            Vector3f Pw = p1*depth;
            Vector3f line1 = O1 - Pw;
            Vector3f line2 = O2 - Pw;

            float theta = line1.dot(line2)/(line1.norm()*line2.norm());
            if (theta > 0.98) {
                continue;
            }       

            float error1 = errorMulti * calculateError(R11, t11, Pw, p1);
            if (error1 > th) {
                continue;
            }
            float error2 = errorMulti * calculateError(R21, t21, Pw, p2);
            if (error2 > th) {
                continue;
            }
            inliers[i] = 1;
        }
    }

    bool KltHomographyInit::detectCorner(FramePtr frame, vector<cv::Point2f>& points) {
        cv::Mat& image = frame->mImagePyr[0];
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
                    points.emplace_back(corner[i]);
                    const Vector2f px(corner[i].x, corner[i].y);
                    frame->mObs.push_back(new Feature(frame, px, 1));
                }
            }
        }
        return true;
    }

    void KltHomographyInit::reset() { 
        mRefFrame = NULL;
        mFirstCorners.clear();
    }
}

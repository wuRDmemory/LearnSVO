#include "initialization.hpp"

#define SHOW_MATCH 0

namespace mSVO {
    KltHomographyInit::KltHomographyInit() { 
        mCornerDetector = new Detector(Config::width(), Config::height(), Config::gridCellNumber(), Config::pyramidNumber(), 10.0f);
    }

    InitResult KltHomographyInit::addFirstFrame(FramePtr currFrame) {
        reset();
        mFirstCorners.clear();
        mCornerDetector->detect(currFrame.get(), mFirstCorners, mFirstCornersLevel);
        
        LOG(INFO) << ">>> [first frame] corner detected " << mFirstCorners.size();
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
        float distarties = 0.0f;
        mvk::CameraModel* camera = currFrame->camera();
        Features& ref_features = mRefFrame->obs();
        vector<Vector3f> ref_obs(status.size());
        vector<Vector3f> cur_obs(status.size());
        for (int i = 0; i < status.size(); i++) 
        if (status[i]) {
            Vector2f pref(mFirstCorners[i].x, mFirstCorners[i].y);
            Vector2f pcur(nextCorners[i].x,    nextCorners[i].y);
            mFirstCorners[j]      = mFirstCorners[i];
            nextCorners[j]        = nextCorners[i];
            mFirstCornersLevel[j] = mFirstCornersLevel[i];

            ref_obs[j] = camera->cam2world(pref);
            cur_obs[j] = camera->cam2world(pcur);

            ref_obs[j].normalize(); 
            cur_obs[j].normalize();

            distarties += (pcur - pref).norm();
            j++;
        }
        mFirstCorners.resize(j);
        mFirstCornersLevel.resize(j);
        nextCorners.resize(j);
        ref_obs.resize(j);
        cur_obs.resize(j);
        LOG(INFO) << ">>> [second frame] track key point : " << j;

        if (distarties / (j+1) < Config::minDispartyThr()) {
            LOG(INFO) << ">>> [second frame] too few disparty: " << distarties / (j+1);
        }

        if (j <= Config::minTrackThr()) {
            LOG(ERROR) << ">>> [second frame] too few track points!!!";
            return FAILURE;
        }

        #if SHOW_MATCH
        testMatch(mRefFrame->imagePyr()[0], currFrame->imagePyr()[0], mFirstCorners, nextCorners);
        #endif

        // use fundamental matrix to shift some outlier
        Mat F21;
        const Mat& K = camera->cvK();
        vector<bool> inliers;
        findFundamental(mFirstCorners, nextCorners, inliers, F21);
        F21.convertTo(F21, CV_32F);
        cv::Mat  E = K.t()*F21*K;
        cv::Mat  R1, R2, t1, t2;
        LOG(INFO) << ">>> [second frame] find fundamental done!!! inliers " << std::accumulate(inliers.begin(), inliers.end(), 0);
        decomposeE(E, R1, R2, t1); t2 = -t1;
        // LOG(INFO) << ">>> [second frame] find essential done!!!";
        
        int best_inlier_cnt = INT_MIN, best_index = 0;
        vector<uchar> best_inliers;
        vector<float> best_depths;
        vector<Mat>   list_R = { R1, R1, R2, R2 };
        vector<Mat>   list_t = { t1, t2, t1, t2 };
        for (int i = 0; i <4; i++) {
            vector<uchar> inliers;
            vector<float> depths;
            int good = computeInliers(list_R[i], list_t[i], K, mFirstCorners, nextCorners, inliers, depths, Config::minProjError());
            LOG(INFO) << ">>> [second frame] good match is " << good;
            if (good > best_inlier_cnt) {
                best_index = i;
                best_inlier_cnt = good;
                best_inliers = std::move(inliers);
                best_depths  = std::move(depths );
            }
        }

        LOG(INFO) << ">>> [second frame] good inlier number: " << best_inlier_cnt;
        if (best_inlier_cnt < Config::minInlierThr()) {
            LOG(INFO) << ">>> [second frame] too few inliers!!!";
            return FAILURE;
        }

        vector<float>::iterator middle = best_depths.begin() + (int)best_depths.size() / 2;
        std::nth_element(best_depths.begin(), middle, best_depths.end());
        const float fixscale = 1.5f;
        const float scale    = fixscale/(*middle);

        Mat R21 = list_R[best_index];
        Mat t21 = list_t[best_index];
        Matrix3f Rcr; Vector3f tcr;
        cv2eigen(R21, Rcr); cv2eigen(t21, tcr);
        
        Matrix3f Rcw = Rcr;
        Vector3f tcw;
        Vector3f twc = -Rcw.transpose()*tcr;
        currFrame->Rwc() = Eigen::Quaternionf(Rcw.transpose());
        currFrame->twc() = twc*scale;
        tcw = -Rcw*twc*scale;
        // clean the ref and cur frame's features
        int depthCnt = 0;
        mRefFrame->obs().clear(); currFrame->obs().clear();
        for (int i = 0; i < best_inliers.size(); i++) {
            if (!best_inliers[i])
                continue;
            
            float depth = calcuDepth(Rcw, tcw, ref_obs[i], cur_obs[i]);
            // LOG(INFO) << ">>> [second frame] depth: " << depth;
            if (depth <= 0) {
                continue;
            }
            depthCnt ++;
            Vector3f xyz = ref_obs[i]*depth;
            LandMarkPtr ldmk = new LandMark(xyz);
            ldmk->type = LandMark::LANDMARK_TYPR::GOOD;

            Vector2f px1(mFirstCorners[i].x, mFirstCorners[i].y), px2(nextCorners[i].x, nextCorners[i].y);
            FeaturePtr ref_feature = new Feature(mRefFrame.get(), ldmk, px1, ref_obs[i], mFirstCornersLevel[i]);
            FeaturePtr cur_feature = new Feature(currFrame.get(), ldmk, px2, cur_obs[i], mFirstCornersLevel[i]);

            mRefFrame->addFeature(ref_feature);
            currFrame->addFeature(cur_feature);

            ldmk->addFeature(ref_feature);
            ldmk->addFeature(cur_feature);
        }

        #if SHOW_MATCH
        testDepth(currFrame);
        #endif
        LOG(INFO) << ">>> [addSecondFrame] good feature " << depthCnt << "/" << best_inlier_cnt;
        return SUCCESS;
    }

    float KltHomographyInit::findFundamental(vector<Point2f> &refPoints, vector<Point2f> &curPoints, vector<bool>& inliers, cv::Mat &F21) {
        // Number of putative matches
        const int N = refPoints.size();

        // Normalize coordinates
        Mat T1, T2;
        vector<Point2f> vPn1, vPn2;
        normalize(refPoints,vPn1, T1);
        normalize(curPoints,vPn2, T2);
        Mat T2t = T2.t();

        // Best Results variables
        float score = 0.0f, currentScore = 0.0f;
        inliers = vector<bool>(N,false);

        // Iteration variables
        Mat F21i;
        set<int> seens;
        vector<Point2f> vPn1i(8);
        vector<Point2f> vPn2i(8);
        vector<bool> vInliers(N, false);

        // Perform all RANSAC iterations and save the solution with highest score
        for(int it = 0; it<N/3; it++) {
            seens.clear();
            // Select a minimum set
            for(int j=0; j<8; j++) {
                int idx = 0;
                do {
                    idx = theRNG().uniform(0, N);
                } while (seens.count(idx));
                vPn1i[j] = vPn1[idx];
                vPn2i[j] = vPn2[idx];
                seens.insert(idx);
            }

            Mat Fn = computeF21(vPn1i,vPn2i);
            F21i = T2t*Fn*T1;
            currentScore = checkFundamental(F21i, refPoints, curPoints, vInliers, 1.0f);

            if(currentScore > score) {
                F21     = F21i.clone();
                inliers = move(vInliers);
                score   = currentScore;
            }
        }
        return score;
    }

    Mat KltHomographyInit::computeF21(const vector<Point2f> &vP1, const vector<Point2f> &vP2) {
        const int N = vP1.size();
        cv::Mat A(N,9,CV_32F);
        for(int i=0; i<N; i++) {
            const float u1 = vP1[i].x;
            const float v1 = vP1[i].y;
            const float u2 = vP2[i].x;
            const float v2 = vP2[i].y;

            A.at<float>(i,0) = u2*u1;
            A.at<float>(i,1) = u2*v1;
            A.at<float>(i,2) = u2;
            A.at<float>(i,3) = v2*u1;
            A.at<float>(i,4) = v2*v1;
            A.at<float>(i,5) = v2;
            A.at<float>(i,6) = u1;
            A.at<float>(i,7) = v1;
            A.at<float>(i,8) = 1;
        }

        Mat u,w,vt;
        SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        Mat Fpre = vt.row(8).reshape(0, 3);
        SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        w.at<float>(2)=0;
        return  u*cv::Mat::diag(w)*vt;
    }

    void KltHomographyInit::normalize(vector<Point2f> &vKeys, vector<Point2f> &vNormalizedPoints, Mat &T) {
        float meanX = 0;
        float meanY = 0;
        const int N = vKeys.size();

        vNormalizedPoints.resize(N);

        for(int i=0; i<N; i++) {
            meanX += vKeys[i].x;
            meanY += vKeys[i].y;
        }

        meanX = meanX/N;
        meanY = meanY/N;

        float meanDevX = 0;
        float meanDevY = 0;

        for(int i=0; i<N; i++) {
            vNormalizedPoints[i].x = vKeys[i].x - meanX;
            vNormalizedPoints[i].y = vKeys[i].y - meanY;

            meanDevX += fabs(vNormalizedPoints[i].x);
            meanDevY += fabs(vNormalizedPoints[i].y);
        }

        meanDevX = meanDevX/N;
        meanDevY = meanDevY/N;

        float sX = 1.0/meanDevX;
        float sY = 1.0/meanDevY;

        for(int i=0; i<N; i++)
        {
            vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
            vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
        }

        T = cv::Mat::eye(3,3,CV_32F);
        T.at<float>(0,0) = sX;
        T.at<float>(1,1) = sY;
        T.at<float>(0,2) = -meanX*sX;
        T.at<float>(1,2) = -meanY*sY;
    }

    void KltHomographyInit::decomposeE(cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t) {
        cv::Mat u,w,vt;
        cv::SVD::compute(E,w,u,vt);

        u.col(2).copyTo(t);
        t=t/cv::norm(t);

        cv::Mat W(3,3,CV_32F,cv::Scalar(0));
        W.at<float>(0,1)=-1;
        W.at<float>(1,0)=1;
        W.at<float>(2,2)=1;

        R1 = u*W*vt;
        if(cv::determinant(R1)<0)
            R1=-R1;

        R2 = u*W.t()*vt;
        if(cv::determinant(R2)<0)
            R2=-R2;
    }

    float KltHomographyInit::checkFundamental(const Mat &F21, vector<Point2f> &refPoints, vector<Point2f> &curPoints, vector<bool> &vbMatchesInliers, float sigma) {
        const int N = refPoints.size();

        const float f11 = F21.at<float>(0,0);
        const float f12 = F21.at<float>(0,1);
        const float f13 = F21.at<float>(0,2);
        const float f21 = F21.at<float>(1,0);
        const float f22 = F21.at<float>(1,1);
        const float f23 = F21.at<float>(1,2);
        const float f31 = F21.at<float>(2,0);
        const float f32 = F21.at<float>(2,1);
        const float f33 = F21.at<float>(2,2);

        vbMatchesInliers.resize(N);

        float score = 0;

        const float th = 3.841;
        const float thScore = 5.991;

        const float invSigmaSquare = 1.0/(sigma*sigma);

        for(int i=0; i<N; i++)
        {
            bool bIn = true;

            const Point2f &kp1 = refPoints[i];
            const Point2f &kp2 = curPoints[i];

            const float u1 = kp1.x;
            const float v1 = kp1.y;
            const float u2 = kp2.x;
            const float v2 = kp2.y;

            // Reprojection error in second image
            // l2=F21x1=(a2,b2,c2)

            const float a2 = f11*u1+f12*v1+f13;
            const float b2 = f21*u1+f22*v1+f23;
            const float c2 = f31*u1+f32*v1+f33;

            const float num2 = a2*u2+b2*v2+c2;

            const float squareDist1 = num2*num2/(a2*a2+b2*b2);

            const float chiSquare1 = squareDist1*invSigmaSquare;

            if(chiSquare1>th)
                bIn = false;
            else
                score += thScore - chiSquare1;

            // Reprojection error in second image
            // l1 =x2tF21=(a1,b1,c1)

            const float a1 = f11*u2+f21*v2+f31;
            const float b1 = f12*u2+f22*v2+f32;
            const float c1 = f13*u2+f23*v2+f33;

            const float num1 = a1*u1+b1*v1+c1;

            const float squareDist2 = num1*num1/(a1*a1+b1*b1);

            const float chiSquare2 = squareDist2*invSigmaSquare;

            if(chiSquare2>th)
                bIn = false;
            else
                score += thScore - chiSquare2;

            if(bIn)
                vbMatchesInliers[i]=true;
            else
                vbMatchesInliers[i]=false;
        }

        return score;
    }

    int KltHomographyInit::computeInliers(Mat& R21, Mat& t21, const Mat& K, vector<Point2f>& pts1, vector<Point2f>& pts2, \
                                        vector<uchar>& inliers, vector<float>& depthes, float th) {
        th *= th;
        int N = pts1.size();
        inliers.resize(N, 0);
        depthes.resize(N, 0);

        const float fx = Config::fx();
        const float fy = Config::fy();
        const float cx = Config::cx();
        const float cy = Config::cy();

        Mat O1 = Mat::zeros(3, 1, CV_32F);
        Mat O2 = -R21.t()*t21;

        Mat T1(3, 4, CV_32F, cv::Scalar::all(0));
        Mat T2(3, 4, CV_32F, cv::Scalar::all(0));

        K.copyTo(  T1.colRange(0, 3));
        R21.copyTo(T2.colRange(0, 3));
        t21.copyTo(T2.col(3));
        T2 = K*T2;

        int good = 0;
        int j = 0;
        for (int i = 0; i < N; i++) {
            Point2f& p1 = pts1[i];
            Point2f& p2 = pts2[i];

            Mat Pw;
            Mat pt1 = (Mat_<float>(3,1) << (p1.x-cx)/fx, (p1.y-cy)/fy, 1);
            Mat pt2 = (Mat_<float>(3,1) << (p2.x-cx)/fx, (p2.y-cy)/fy, 1);
            pt1 /= cv::norm(pt1); pt2 /= cv::norm(pt2); 
            calcu3DPoint(R21, t21, pt1, pt2, Pw);
            const float Pwx   = Pw.at<float>(0);
            const float Pwy   = Pw.at<float>(1);
            const float depth = Pw.at<float>(2);
            if (depth <= 0 or std::isnan(depth) or std::isinf(depth)) {
                continue;
            }

            Mat line1 = Pw - O1;
            Mat line2 = Pw - O2;
            float theta = line1.dot(line2)/(cv::norm(line1)*cv::norm(line2));
            if (theta > 0.99998) {
                continue;
            }

            float im1x, im1y;
            float invZ1 = 1.0/Pw.at<float>(2);
            im1x = fx*Pw.at<float>(0)*invZ1 + cx;
            im1y = fy*Pw.at<float>(1)*invZ1 + cy;

            float error1 = (im1x-p1.x)*(im1x-p1.x)+(im1y-p1.y)*(im1y-p1.y);
            if (error1 > th) {
                continue;
            }

            Mat Pc2 = R21*Pw + t21;
            float im2x, im2y;
            float invZ2 = 1.0/Pc2.at<float>(2);
            im2x = fx*Pc2.at<float>(0)*invZ2 + cx;
            im2y = fy*Pc2.at<float>(1)*invZ2 + cy;

            float error2 = (im2x-p2.x)*(im2x-p2.x)+(im2y-p2.y)*(im2y-p2.y);
            if (Pc2.at<float>(2) <= 0 || error2 > th) {
                continue;
            }
            inliers[i] = 1;
            depthes[j] = depth;
            good += 1;
            j ++;
        }
        depthes.resize(j);
        return good;
    }

    void KltHomographyInit::reset() { 
        mRefFrame.reset();
        mFirstCorners.clear();
        mFirstCornersLevel.clear();
    }

    int KltHomographyInit::testMatch(Mat& refImage, Mat& curImage, vector<cv::Point2f>& refPoints, vector<cv::Point2f>& curPoints) {
        Mat mergeImage;
        addWeighted(refImage, 0.5, curImage, 0.5, 0.0, mergeImage);
        cvtColor(mergeImage, mergeImage, COLOR_GRAY2BGR);
        for (int i = 0; i < refPoints.size(); i++) {
            Scalar color(theRNG().uniform(0, 255), theRNG().uniform(0, 255));
            line(mergeImage, refPoints[i], curPoints[i], color);
        }
        imshow("[init] match", mergeImage);
        waitKey();
        // destroyWindow("[init] match");
    }

    int KltHomographyInit::testDepth(FramePtr curFrame) {
        Quaternionf Rcw = curFrame->Rwc().inverse();
        Vector3f&   twr = curFrame->twc();

        Mat  mergeImage;
        Mat& curImage = curFrame->imagePyr()[0];
        cvtColor(curImage, mergeImage, COLOR_GRAY2BGR);

        int num = 0;
        auto& features = curFrame->obs();
        for (auto it = features.begin(); it != features.end(); it++) {
            FeaturePtr feature = *it;
            if (!feature->mLandmark) {
                continue;
            }

            Vector3f Pw = feature->mLandmark->xyz();
            Vector2f Pcxy = curFrame->world2uv(Pw);
            Vector2f Pcuv = feature->mPx;
            // Vector2f Pcuv = feature->mPx;

            if (!curFrame->isVisible(Pcxy, 0)) {
                continue;
            }

            Scalar color(theRNG().uniform(0, 255), theRNG().uniform(0, 255), theRNG().uniform(0, 255));
            line(mergeImage, Point(Pcuv(0), Pcuv(1)), Point(Pcxy(0), Pcxy(1)), color);
            num++;
        }

        imshow("[init] test depth", mergeImage);
        waitKey();
        // destroyWindow("[init] test depth");
        return true;
    }
}

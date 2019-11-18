#include "matcher.hpp"

namespace mSVO {
    float calcuDepth(Matrix3f& Rcw, Vector3f& tcw, Vector3f& f1, Vector3f& f2) {
        Matrix<float, 3, 2> A;
        Vector3f b;
        A.col(0) = Rcw*f1;
        A.col(1) = -f2;
        b = tcw;

        Matrix2f ATA = A.transpose()*A;
        Vector2f ATb = A.transpose()*b;
        if (ATA.determinant() == 0.0f) {
            return -1.0f;
        }
        Vector2f zs = -ATA.inverse() * ATb;
        return zs(0);
    }

    bool calcu3DPoint(Mat& Rcw, Mat& tcw, Mat& p1, Mat& p2, Mat& point) {
        // Mat A(4, 4, CV_32F);
        // A.row(0) = p1.x*T1.row(2) - T1.row(0);
        // A.row(1) = p1.y*T1.row(2) - T1.row(1);
        // A.row(2) = p2.x*T2.row(2) - T2.row(0);
        // A.row(3) = p2.y*T2.row(2) - T2.row(1);

        // Mat U, W, VT;
        // SVD svd;
        // svd.compute(A, W, U, VT, SVD::MODIFY_A|SVD::FULL_UV);
        // Mat pw = VT.row(3).t();
        // float pwx = pw.at<float>(0);
        // float pwy = pw.at<float>(1);
        // float pwz = pw.at<float>(2);
        // float pww = pw.at<float>(3);
        // point  = pw.rowRange(0, 3)/pw.at<float>(3);
        Mat A(3, 2, CV_32F);
        Mat b(3, 1, CV_32F);
        A.col(0) = Rcw*p1;
        A.col(1) = -p2;
        b = tcw;
        Mat ATA = A.t()*A;
        Mat ATb = A.t()*b;
        Mat zs = -ATA.inv() * ATb;
        if (isinf(zs.at<float>(0)) or isnan(zs.at<float>(0)) or 
            isinf(zs.at<float>(1)) or isnan(zs.at<float>(1))) {
            point = (Mat_<float>(3, 1) << 0,0,0);
            return false;
        }
        Mat pw1 = p1*zs.at<float>(0);
        Mat pc2 = p2*zs.at<float>(1);
        Mat pw2 = Rcw.t()*(pc2 - tcw);
        point = 0.5f*(pw1 + pw2);
        return true;
    }

    float calcuProjError(Mat& R, Mat& t, Mat& K, Point3f& point, Point2f uv) {
        Mat pw  = (Mat_<float>(3,1) << point.x, point.y, point.z);
        Mat xyz = (R*pw + t);
        Mat xy_ = (Mat_<float>(3, 1) << xyz.at<float>(0)/xyz.at<float>(2), xyz.at<float>(1)/xyz.at<float>(2), 1);
        Mat uv_ = K*xy_;
        cv::Point2f xy(uv_.at<float>(0), uv_.at<float>(1));
        return cv::norm(uv - xy);
    }

    int Matcher::patchSize      = 8;
    int Matcher::patchArea      = 64;
    int Matcher::halfPatchSize  = 4;

    bool Matcher::warpAffine(cv::Mat& image, Vector2f& px, Matrix2f& Acr, int level, int searchLevel, int halfPatchSize, uint8_t* patchPtr) {
        Matrix2f Arc = Acr.inverse();
        if (isnan(Arc(0, 0))) {
            // LOG(INFO) << ">>> Warper is bad." << endl;
            return false;
        }

        Vector2f pyrPx = px/(1<<level);
        for (int row = -halfPatchSize; row < halfPatchSize; ++row)
        for (int col = -halfPatchSize; col < halfPatchSize; ++col, patchPtr++) {
            Vector2f pxPatch = Vector2f(col, row)*(1<<searchLevel);
            Vector2f puv(Arc*pxPatch + pyrPx);
            if (puv[0]<0 || puv[1]<0 || puv[0]>=image.cols-1 || puv[1]>=image.rows-1) {
                *patchPtr = 0;
            } else {
                *patchPtr = (uint8_t)Alignment::interpolateU8(image, puv);
            }
        }
        return true;
    }

    int Matcher::getBestSearchLevel(const Matrix2f& Acr, const int max_level) {
        // Compute patch level in other image
        int   search_level = 0;
        float D = Acr.determinant();
        while(D > 3.0 && search_level < max_level) {
            search_level += 1;
            D *= 0.25;
        }
        return search_level;
    }

    bool Matcher::getWarpAffineMatrix(Frame* refFrame, Frame* curFrame, Quaternionf& Rcr, Vector3f& tcr, 
                                        Vector3f& refDirect, float depth, int level, Matrix2f& warpMatrix) {
        const int halfPatchSize = 4+1;
        Vector3f refXYZ = refDirect*depth;
        Vector2f refCuv = refFrame->camera()->world2cam(refXYZ);

        // calculate the ref px
        Vector2f refDuCuv = refCuv + Vector2f(halfPatchSize, 0)*(1<<level);
        Vector2f refDvCuv = refCuv + Vector2f(0, halfPatchSize)*(1<<level);
        Vector3f refDuXYZ = refFrame->camera()->cam2world(refDuCuv);
        Vector3f refDvXYZ = refFrame->camera()->cam2world(refDvCuv);
        refDuXYZ *= refXYZ(2)/refDuXYZ(2);
        refDvXYZ *= refXYZ(2)/refDvXYZ(2);

        // project to the current frame
        Vector3f curXYZ   = Rcr*refXYZ   + tcr;
        Vector3f curDuXYZ = Rcr*refDuXYZ + tcr;
        Vector3f curDvXYZ = Rcr*refDvXYZ + tcr;
        Vector2f curCuv   = curFrame->camera()->world2cam(curXYZ);
        Vector2f curDuCuv = curFrame->camera()->world2cam(curDuXYZ);
        Vector2f curDvCuv = curFrame->camera()->world2cam(curDvXYZ);

        warpMatrix.col(0) = (curDuCuv - curCuv)/halfPatchSize;
        warpMatrix.col(1) = (curDvCuv - curCuv)/halfPatchSize;
        return true;
    }

    bool Matcher::createPatchFromBorderPatch(uint8_t* patch, uint8_t* patchWithBorder, int patchSize) {
        for (int i = 1; i < patchSize+1; i++) {
            uint8_t* patchBorder = patchWithBorder + i*(patchSize+2) + 1;
            for (int j = 0; j < patchSize; j++, patch++, patchBorder++) {
                *patch = *patchBorder;
            }
        }
        return true;
    }

    bool Matcher::findDirectMatch(Frame* curFrame, LandMarkPtr landmark, Vector2f& px) {
        const int patchArea = patchSize * patchSize;
        
        // TODO: find the closest frame(feature)
        FeaturePtr refFeature   = NULL;
        Vector3f&  curFramePose = curFrame->twc();
        landmark->findClosestObs(curFramePose, refFeature);
        if (!refFeature) {
            return false; 
        }

        // TODO: calculate the warp matrix because the distance between them is very large
        Matrix2f Acr;
        uint8_t patchWithBorder[(patchSize+2)*(patchSize+2)] = {0};
        uint8_t patch[patchArea] = {0};
        const float refDepth = (landmark->xyz() - refFeature->mFrame->twc()).norm();
        Quaternionf Rcr = curFrame->Rwc().inverse()* refFeature->mFrame->Rwc();
        Vector3f    tcr = curFrame->Rwc().inverse()*(refFeature->mFrame->twc() - curFrame->twc());

        if (!getWarpAffineMatrix(refFeature->mFrame, curFrame, Rcr, tcr, refFeature->mDirect, refDepth, refFeature->mLevel, Acr)) {
            return false;
        }
        int bestSearchLevel = getBestSearchLevel(Acr, Config::pyramidNumber()-1);
        warpAffine(refFeature->mFrame->imagePyr()[refFeature->mLevel], refFeature->mPx, 
                    Acr, refFeature->mLevel, bestSearchLevel, 
                    halfPatchSize+1, patchWithBorder);
        createPatchFromBorderPatch(patch, patchWithBorder, patchSize);

        // TODO: align them
        Vector2f levelPx = px / (1<<bestSearchLevel);
        if (!Alignment::align2D(curFrame->imagePyr()[bestSearchLevel], levelPx, Config::alignIterCnt(), patchSize, patch, patchWithBorder)) {
            return false;
        }

        px = levelPx * (1<<bestSearchLevel);
        return true;
    }

    bool Matcher::findEpipolarMatch(Frame* curFrame, FeaturePtr feature, Quaternionf& Rcr, Vector3f& tcr, float minDepth, float depth, float maxDepth, float& z) {
        z = -1.0f;
        Frame* refFrame = feature->mFrame;
        // get the warp
        Matrix2f Acr;
        getWarpAffineMatrix(refFrame, curFrame, Rcr, tcr, feature->mDirect, depth, feature->mLevel, Acr);
        
        int bestSearchLevel = getBestSearchLevel(Acr, Config::pyramidNumber()-1);
        float searchscale   = 1.0f/(1<<bestSearchLevel);

        uint8_t patchWithBorder[(patchSize+2)*(patchSize+2)] = {0};
        uint8_t patch[patchArea] = {0};
        warpAffine(refFrame->imagePyr()[feature->mLevel], feature->mPx, Acr, feature->mLevel, 
                    bestSearchLevel, halfPatchSize+1, patchWithBorder);
        createPatchFromBorderPatch(patch, patchWithBorder, patchSize);

        // max xyz and min xyz
        Vector3f maxRXYZ = feature->mDirect*maxDepth;
        Vector3f minRXYZ = feature->mDirect*minDepth;

        // project them to plane.
        Vector3f maxCXYZ = Rcr*maxRXYZ + tcr;
        Vector3f minCXYZ = Rcr*minRXYZ + tcr;
        Vector2f maxCuv  = curFrame->camera()->world2cam(maxCXYZ)*searchscale;
        Vector2f minCuv  = curFrame->camera()->world2cam(minCXYZ)*searchscale;
        
        // epipolar
        cv::Mat& curImage = curFrame->imagePyr()[bestSearchLevel];
        Vector2f epipolar = maxCuv - minCuv;
        float epipolarLen = epipolar.norm();
        if (epipolarLen < 2.0) {
            // epipolar is too close
            Vector2f levelPx = (maxCuv+minCuv)/2.0f;
            if (curFrame->isVisible(levelPx, patchSize, bestSearchLevel)) {
                return false;
            }
            
            if (!Alignment::align2D(curImage, levelPx, Config::alignIterCnt(), patchSize, patch, patchWithBorder)) {
                return false;
            }
            
            Vector2f cuv  = levelPx/searchscale;
            Vector3f cxyz = curFrame->camera()->cam2world(cuv);
            Matrix3f MRcr = Rcr.toRotationMatrix();
            cxyz.normalize();
            z = calcuDepth(MRcr, tcr, feature->mDirect, cxyz);
            return true;
        }

        // TODO: align them
        const int nStep = int(epipolarLen/0.7f);
        Vector2f   step = epipolar / nStep;

        if (nStep > Config::depthFilterIterCnt()) {
            LOG(INFO) << ">>> depth filter too many step";
            return false;
        }

        float bestNCC = -1;
        Vector2f bestuv(0, 0);
        Vector2f uv = minCuv;
        Vector2i uvLast(0.0f, 0.0f);
        
        for (int i = 0; i < nStep; ++i, uv += step) {
            // check frame visible
            Vector2i iuv(int(uv(0)+0.5f), int(uv(1)+0.5f));
            if (uvLast == iuv) {
                continue;
            }
            uvLast = iuv;
            if (!curFrame->isVisible(uv, patchSize, bestSearchLevel)) {
                continue;
            }
            // NCC score
            uint8_t* curPatch = new uint8_t[patchArea];
            createPatch(curImage, iuv, curPatch, halfPatchSize);
            float score = NCCScore(patch, curPatch, patchArea);
            if (score > bestNCC) {
                bestNCC = score;
                bestuv  = uv;
            }
            delete(curPatch);
            curPatch = NULL;
        }

        /// align, the threshold is too high?
        /// TODO: add the scale?
        if (bestNCC < Config::depthFilterNCCScore()*patchArea) {
            return false;
        }
        
        if (!Alignment::align2D(curImage, bestuv, Config::alignIterCnt(), patchSize, patch, patchWithBorder)) {
            return false;
        }

        mCuv  = bestuv/searchscale;
        Vector3f cxyz = curFrame->camera()->cam2world(mCuv);
        Matrix3f MRcr = Rcr.toRotationMatrix();
        cxyz.normalize();
        z = calcuDepth(MRcr, tcr, feature->mDirect, cxyz);
        return true;
    }

    bool Matcher::createPatch(cv::Mat& image, Vector2i& uv, uint8_t* patch, int halfPatchSize) {
        const int stride = image.cols;
        const int x = uv(0), y = uv(1);
        for (int i = -halfPatchSize; i < halfPatchSize; i++) {
            uint8_t* ptr = (uint8_t*)image.data + (y + i)*stride + x - halfPatchSize;
            for (int j = 0; j < 2*halfPatchSize; j++, ptr++, patch++) {
                *patch = *ptr;
            }
        }
        return true;
    }

    float Matcher::NCCScore(uint8_t* refPatch,  uint8_t* curPatch, int patchArea) const {
        int sumA  = 0, sumB  = 0;
        int sumAA = 0, sumBB = 0;
        int sumAB = 0;
        for(int r = 0; r < patchArea; r++) {
            const uint8_t refPixel = refPatch[r];
            const uint8_t curPixel = curPatch[r];
            sumA  += refPixel;
            sumB  += curPixel;
            sumAA += refPixel*refPixel;
            sumBB += curPixel*curPixel;
            sumAB += curPixel*refPixel;
        }
        return sumAA - 2*sumAB + sumBB - (sumA*sumA - 2*sumA*sumB + sumB*sumB)/patchArea;
    }
}

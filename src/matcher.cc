#include "matcher.hpp"

namespace mSVO {
    float calcuDepth(Matrix3f& Rcw, Vector3f& tcw, Vector3f& f1, Vector3f& f2) {
        Matrix<float, 3, 2> A;
        Vector3f b;
        A.col(0) = Rcw*f1;
        A.col(1) = f2;
        b = -tcw;

        Matrix2f ATA = A.transpose()*A;
        Vector2f ATb = A.transpose()*b;
        if (ATA.determinant() == 0.0f) {
            return -1.0f;
        }
        Vector2f zs = ATA.inverse() * ATb;
        return zs(0);
    }

    float calcuProjError(Matrix3f& R, Vector3f& t, Vector3f& point, Vector3f& xyz) {
        Vector3f xyz_ = R*point + t;
        Vector2f xy  = xyz.head<2>()/xyz(2);
        Vector2f xy_ = xyz_.head<2>()/xyz_(2);
        Vector2f dis = xy - xy_;
        return dis.norm();
    }

    bool Matcher::warpAffine(cv::Mat& image, Vector2f& px, Matrix2f& Acr, int level, int searchLevel, int halfPatchSize, uint8_t* patchPtr) {
        Matrix2f Arc = Acr.inverse();
        if (isnan(Arc(0, 0)) {
            LOG(INFO) << ">>> Warper is bad." << endl;
            return false;
        }

        Vector2f pyrPx = px/(1<<level);
        for (int row = -halfPatchSize; row <= halfPatchSize; ++row)
        for (int col = -halfPatchSize; col <= halfPatchSize; ++col, patchPtr++) {
            Vector2f pxPatch = Vector2f(col, row)*(1<<searchLevel);
            Vector2f px(Arc*pxPatch + pyrPx);
            if (px[0]<0 || px[1]<0 || px[0]>=image.cols-1 || px[1]>=image.rows-1) {
                *patchPtr = 0;
            } else {
                *patchPtr = (uint8_t)Alignment::interpolateU8(image, px);
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

    bool Matcher::getWarpAffineMatrix(FramePtr refFrame, FramePtr curFrame, Vector3f& refDirect, float depth, int level, Matrix2f& warpMatrix) {
        const int halfPatchSize = 4+1;
        Vector3f refXYZ = refDirect*depth;
        Vector2f refCuv  = refFrame->camera()->world2cam(refXYZ);

        // calculate the ref px
        Vector2f refDuCuv = refCuv + Vector2f(halfPatchSize, 0)*(1<<level);
        Vector2f refDvCuv = refCuv + Vector2f(0, halfPatchSize)*(1<<level);
        Vector3f refDuXYZ = refFrame->camera()->cam2world(refDuCuv);
        Vector3f refDvXYZ = refFrame->camera()->cam2world(refDvCuv);
        refDuXYZ *= refXYZ(2)/refDuXYZ(2);
        refDvXYZ *= refXYZ(2)/refDvXYZ(2);

        // calculate relative transform
        Sophus::SE3 Tcr = curFrame->pose().inverse() * refFrame->pose();

        // project to the current frame
        Vector3f curXYZ = Tcr*refXYZ;
        Vector3f curDuXYZ = Tcr*refDuXYZ;
        Vector3f curDvXYZ = Tcr*refDvXYZ;
        Vector2f curCuv = curFrame->camera()->world2cam(curXYZ);
        Vector2f curDuCuv = curFrame->camera()->world2cam(curDuXYZ);
        Vector2f curDvCuv = curFrame->camera()->world2cam(curDvXYZ);

        warpMatrix.col(0) = (curDuCuv - curCuv)/halfPatchSize;
        warpMatrix.col(1) = (curDvCuv - curCuv)/halfPatchSize;
        return true;
    }

    bool Matcher::createPatchFromBorderPatch(uint8_t* patch, const uint8_t const* patchWithBorder, const int patchSize) {
        for (int i = 1; i < patchSize+1; i++) {
            uint8_t* patchBorder = patchWithBorder + i*(patchSize+2) + 1;
            for (int j = 0; j < patchSize; j++, patch++, patchBorder++) {
                *patch = *patchBorder;
            }
        }
        return true;
    }

    bool Matcher::findDirectMatch(FramePtr curFrame, LandMarkPtr landmark, Vector2f& px) {
        // TODO: find the closest frame(feature)
        FeaturePtr refFeature = NULL;
        Vector3f   curFramePose = curFrame->twc();
        landmark->findClosestObs(curFramePose, refFeature);
        if (!refFeature) {
            return false;
        }

        // TODO: calculate the warp matrix because the distance between them is very large
        Matrix2f Acr;
        uint8_t* patchWithBorder = new uint8_t[(patchSize+2)*(patchSize+2)];
        uint8_t* patch           = new uint8_t[patchArea];
        const float refDepth = (landmark->xyz() - refFeature->mFrame->twc()).norm();
        if (!getWarpAffineMatrix(refFeature->mFrame, curFrame, refFeature->mDirect, refDepth, Arc)) {
            return false;
        }
        const bestSearchLevel = getBestSearchLevel(Acr, Config::pyramidNumber()-1);
        warpAffine(refFeature->mFrame->imagePyr()[refFeature->mLevel], refFeature->mPx, Acr, refFeature->mLevel, searchLevel, halfPatchSize+1, patchWithBorder);
        createPatchFromBorderPatch(patch, patchWithBorder, patchSize);

        // TODO: align them
        Vector2f levelPx = px / (1<<bestSearchLevel);
        if (!Alignment::align2D(curFrame->imagePyr()[bestSearchLevel], levelPx, Config::alignIterCnt(), patch, patchWithBorder)) {
            return false;
        }

        px = levelPx * (1<<bestSearchLevel);
        return true;
    }
}

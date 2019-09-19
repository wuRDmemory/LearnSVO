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

    bool Matcher::findDirectMatch(FramePtr curFrame, LandMarkPtr landmark, Vector2f& px) {
        // TODO: find the closest frame(feature)
        FeaturePtr refFeature = NULL;
        Vector3f   curFramePose = curFrame->twc();
        landmark->findClosestObs(curFramePose, refFeature);

        if (!refFeature) {
            return false;
        }

        // TODO: calculate the warp matrix because the distance between them is very large
        

        // TODO: align them
        return true;
    }
}

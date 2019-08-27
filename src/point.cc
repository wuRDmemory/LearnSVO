#include "point.hpp"

namespace mSVO {
    float LandMark::calcuDepth(Matrix3f& Rcw, Vector3f& tcw, Vector3f& f1, Vector3f& f2) {
        Vector3d f2 = R * feature2;
        Vector2d b;
        b[0] = t.dot(feature1);
        b[1] = t.dot(f2);
        Matrix2d A;
        A(0,0) = feature1.dot(feature1);
        A(1,0) = feature1.dot(f2);
        A(0,1) = -A(1,0);
        A(1,1) = -f2.dot(f2);
        Vector2d lambda = A.inverse() * b;
        Vector3d xm = lambda[0] * feature1;
        Vector3d xn = t + lambda[1] * f2;
        return ( xm + xn )/2;
    }
}

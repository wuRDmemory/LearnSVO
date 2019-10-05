#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "math_utils.hpp"
#include <ceres/ceres.h>

namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    // class PoseLocalParameterization : public ceres::LocalParameterization {
    //     virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const { 
    //         Eigen::Map<const Eigen::Vector3d> tcw_old(x);
    //         Eigen::Map<const Eigen::Quaterniond> qcw_old(x + 3);

    //         Eigen::Map<const Eigen::Vector3d> deltat(delta[0], delta[1], delta[2]);
    //         Eigen::Quaterniond deltaq = deltaQuaternion(Map<Vector3d>(delta[3], delta[4], delta[5]));

    //         Eigen::Map<Eigen::Vector3d> tcw_new(x_plus_delta);
    //         Eigen::Map<Eigen::Quaterniond> qcw_new(x_plus_delta + 3);

    //         tcw_new = trw_old + deltat;
    //         qcw_new = (qcw_old * deltaq).normalized();
    //         return true; 
    //     }

    //     virtual bool ComputeJacobian(const double *x, double *jacobian) const { 
    //         Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    //         j.topRows<6>().setIdentity();
    //         j.bottomRows<1>().setZero();
    //         return true;
    //     }
    //     virtual int GlobalSize() const { return 7; };
    //     virtual int LocalSize() const { return 6; };
    // };

    // class PoseCostFunction : public ceres::SizedCostFunction<2, 7> {
    // private:
    //     Vector2d muv;
    //     Vector3d mxyz;
   
    // public:
    //     PoseCostFunction(Vector2f& uv, Vector3f& xyz) {
    //         muv  = uv.cast<double>();
    //         mxyz = xyz.cast<double>();
    //     };
    //     virtual ~PoseCostFunction() {}
    //     virtual bool Evaluate(double const* const* parameters,
    //                             double* residuals, double** jacobians) const {
    //             Map<Vector3d>    tcw(parameters[0][0], parameters[0][1], parameters[0][2]);
    //             Map<Quaterniond> Rcw(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    //             Vector3d Pc = Rcw * mxyz + tcw;
    //             Vector2d xy(Pc(0)/Pc(2), Pc(1)/Pc(2));

    //             Map<Vector2d> residual(residuals);
    //             residual = muv - xy;

    //             if (jacobians) {
    //                 Eigen::Matrix<double, 2, 3> reduce(2, 3);

    //                 reduce << 1.0/Pc(2),         0, -Pc(0)/(Pc(2) * Pc(2)),
    //                                   0, 1.0/Pc(2), -Pc(1)/(Pc(2) * Pc(2));
    //                 if (jacobians[0]) {
    //                     Map<Matrix<double, 2, 7> > jacobian_pose(jacobians[0]);

    //                     Matrix<double, 3, 6> jacobian;
    //                     jacobian.leftCols<3>()  = Eigen::Matrix3d::Identity();
    //                     jacobian.rightCols<3>() = -symmetricMatrix<double>(Rcw * mxyz);

    //                     jacobian_pose.leftCols<6>() = reduce*jacobians;
    //                     jacobian_pose.rightCols<1>().setZero();
    //                 }
    //             }
    //             return true;
    //         }
    //     };

    //     class StructCostFunction : public ceres::SizedCostFunction<2, 3> {
    //         private:
    //             Vector2d muv;
    //             Quaterniond mRcw;
    //             Vector3d mtcw;
        
    //         public:
    //             StructCostFunction(Vector2f& uv, Quaternionf& Rcw, Vector3f& tcw) {
    //                 muv  = uv.cast<double>();
    //                 mRcw = Rcw.cast<double>();
    //                 mtcw = tcw.cast<double>();
    //             };
    //             virtual ~StructCostFunction() {}
    //             virtual bool Evaluate(double const* const* parameters,
    //                             double* residuals, double** jacobians) const {
    //             Map<Vector3d> xyz(parameters[0][0], parameters[0][1], parameters[0][2]);

    //             Vector3d Pc = mRcw*xyz + mtcw;
    //             Vector2d xy(Pc(0)/Pc(2), Pc(1)/Pc(2));

    //             Map<Vector2d> residual(residuals);
    //             residual = muv - xy;

    //             if (jacobians) {
    //                 Eigen::Matrix<double, 2, 3> reduce(2, 3);

    //                 reduce << 1.0/Pc(2),         0, -Pc(0)/(Pc(2) * Pc(2)),
    //                                   0, 1.0/Pc(2), -Pc(1)/(Pc(2) * Pc(2));
    //                 if (jacobians[0]) {
    //                     Map<Matrix<double, 2, 3> > jacobian_pose(jacobians[0]);
    //                     jacobian_pose = reduce*mRcw.toRotationMatrix();
    //                 }
    //             }
    //             return true;
    //         }
    //     };
}

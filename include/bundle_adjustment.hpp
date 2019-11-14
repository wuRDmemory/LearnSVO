#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <ceres/ceres.h>

#include "map.hpp"
#include "math_utils.hpp"


namespace mSVO {
    using namespace std;
    using namespace cv;
    using namespace Eigen;

    class PoseLocalParameterization : public ceres::LocalParameterization {
        virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const {
            const Eigen::Vector3d    tcw_old(x);
            const Eigen::Quaterniond qcw_old(x+3);

            Eigen::Map<const Eigen::Vector3d> deltat(delta);
            Eigen::Map<const Eigen::Vector3d> deltatheta(delta+3);
            Eigen::Quaterniond deltaq = deltaQuaternion<double>(deltatheta);

            Eigen::Map<Eigen::Vector3d>    tcw_new(x_plus_delta);
            Eigen::Map<Eigen::Quaterniond> qcw_new(x_plus_delta+3);

            tcw_new = tcw_old + deltat;
            qcw_new = (deltaq * qcw_old).normalized();
            return true; 
        }

        virtual bool ComputeJacobian(const double *x, double *jacobian) const { 
            Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
            j.topRows<6>().setIdentity();
            j.bottomRows<1>().setZero();
            return true;
        }
        virtual int GlobalSize() const { return 7; };
        virtual int LocalSize() const { return 6; };
    };

    class BACostFunction : public ceres::SizedCostFunction<2, 7, 3> {
    private:
        Vector2d muv;
   
    public:
        BACostFunction(Vector2f& uv) {
            muv  = uv.cast<double>();
        };
        virtual ~BACostFunction() {}
        virtual bool Evaluate(double const* const* parameters,
                                double* residuals, double** jacobians) const {
            Vector3d    tcw(parameters[0][0], parameters[0][1], parameters[0][2]);
            Quaterniond Rcw(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
            Vector3d    Pw( parameters[1][0], parameters[1][1], parameters[1][2]);

            Vector3d Pc = Rcw * Pw + tcw;
            Vector2d xy(Pc(0)/Pc(2), Pc(1)/Pc(2));

            Eigen::Map<Vector2d> residual(residuals);
            residual = xy - muv;

            if (jacobians) {
                Eigen::Matrix<double, 2, 3, Eigen::RowMajor> reduce(2, 3);

                reduce << 1.0/Pc(2), 0, -Pc(0)/(Pc(2) * Pc(2)), 
                          0, 1.0/Pc(2), -Pc(1)/(Pc(2) * Pc(2));
                
                if (jacobians[0]) {
                    Eigen::Map<Matrix<double, 2, 7, Eigen::RowMajor> > jacobian_pose(jacobians[0]);

                    Matrix<double, 3, 6> jacobian;
                    jacobian.leftCols<3>()  = Eigen::Matrix3d::Identity();
                    jacobian.rightCols<3>() = symmetricMatrix<double>(Rcw*Pw)*-1;

                    jacobian_pose.leftCols<6>() = reduce*jacobian;
                    jacobian_pose.rightCols<1>().setZero();
                }

                if (jacobians[1]) {
                    Eigen::Map<Matrix<double, 2, 3, Eigen::RowMajor> > jacobian_point(jacobians[1]);

                    jacobian_point = reduce * Rcw.toRotationMatrix();
                }
            }
            return true;
        }
    };

    class BundleAdjustment {
    public:
        BundleAdjustment(int maxIter, MapPtr map);

        bool run();

    private:
        int mMaxIter;
        MapPtr mMap;
    };

    typedef BundleAdjustment* BundleAdjustmentPtr;
}

#include "image_align.hpp"
#include "utils.hpp"

namespace {
    // class ImageAlignCostFunction : public ceres::SizedCostFunction<1, 7> {
    // public:
    //     ImageAlignCostFunction(const double refInstense, const double curInstense, 
    //                             Eigen::Matrix<double, 6, 1>& J) {
    //         mRefBlock = refBlock;
    //         mCurBlock = curBlock;
    //         mXYZ = xyz;
    //         mRefFrame = refFrame;
    //     }
    //     virtual ~ImageAlignCostFunction() {}
    //     virtual bool Evaluate(double const* const* parameters,
    //                             double* residuals,
    //                             double** jacobians) const {
    //         residuals[0] =  mCurInstense - mRefInstense;

    //         // Compute the Jacobian if asked for.
    //         if (jacobians != NULL){
    //             if (jacobians[0] != NULL) {
    //                 jacobians[0][0] = -1;
    //             }
    //         }
    //         return true;
    //     }
    // private:
    //     double mRefInstense, mCurInstense;
    //     Eigen::MatrixXf mJ;
    // };
}

namespace mSVO {
    #define INSIDEIMAGE(x, dx, min, max) ((x-dx) >= (min) and (x+dx) < (max))

    ImageAlign::ImageAlign(int minlevel, int maxlevel, int iterCnt): mMinLevel(minlevel), mMaxLevel(maxlevel), mIterCnt(iterCnt){
        meps = 1e-7;
        mMaxFailThr = 5;
    }

    ImageAlign::~ImageAlign() {
        ;
    }

    void ImageAlign::run(FramePtr refFrame, FramePtr curFrame) {
        mTc_r_new = curFrame->pose().inverse()*refFrame->pose();
        mTc_r_old = mTc_r_new;
        for (int level = mMinLevel; level <mMaxLevel; level++) {
            optimize(refFrame, curFrame, level);
        }
        Sophus::SE3 Twc = refFrame->pose()*mTc_r_new.inverse();
        curFrame->pose() = Twc;
    }

    void ImageAlign::prepareData(FramePtr refFrame, int level) {
        const int border  = halfPatchSize+1;
        cv::Mat& ref_img  = refFrame->imagePyr()[level];
        const int stride  = ref_img.cols;
        const int width   = ref_img.cols, height = ref_img.rows;
        const float scale = 1.0f/(1<<level);
        const float fx    = refFrame->camera()->errorMultiplier2();
        Vector3f pos      = refFrame->pose().translation().cast<float>();

        auto& features = refFrame->obs();
        mVisables.resize(features.size(), true);
        mRefPatchCache.resize(features.size(),   patchArea);
        mJacobianCache.resize(features.size()*patchArea, 6);
        int feature_cnt = 0;
        for (auto begin = features.begin(); begin != features.end(); ++begin, ++feature_cnt) {
            Vector2f fPx = (*begin)->mPx*scale;
            const float u_fref = fPx(0),             v_fref = fPx(1);
            const int   u_iref = (int)floor(fPx(0)), v_iref = (int)floor(fPx(1));
            // if the feature has no landmark or this point can not calculate the gradient of image
            if (!(*begin) or 
                !INSIDEIMAGE(u_iref, border, 0, width) or 
                !INSIDEIMAGE(v_iref, border, 0, height)) {
                mVisables[feature_cnt] = false;
                continue;
            }
            
            // calculate the pw in ref frame
            Vector3f& wxyz = (*begin)->mLandmark->xyz();
            float depth = (wxyz - pos).norm();
            Vector3f cxyz = (*begin)->mDirect*depth;

            Matrix<float,2,6> frame_jac;
            Frame::jacobian_uv2se3(cxyz, frame_jac);

            const float subpix_u_ref = u_fref-u_iref;
            const float subpix_v_ref = v_fref-v_iref;
            const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
            const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
            const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;

            int pixel_count = 0;
            for (int i = 0; i < patchSize; i++) {
                uchar* img_ptr = (uchar*)ref_img.data + (v_iref+i-halfPatchSize)*stride + (u_iref-halfPatchSize);
                for (int j = 0; j < patchSize; j++, img_ptr++, pixel_count++) {
                    mRefPatchCache(feature_cnt, pixel_count) = \
                        w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[1] + w_ref_bl*img_ptr[stride] + w_ref_br*img_ptr[stride+1];

                    float dx = 0.5f*((w_ref_tl*img_ptr[1]  + w_ref_tr*img_ptr[2] + w_ref_bl*img_ptr[stride+1] + w_ref_br*img_ptr[stride+2])
                                    -(w_ref_tl*img_ptr[-1] + w_ref_tr*img_ptr[0] + w_ref_bl*img_ptr[stride-1] + w_ref_br*img_ptr[stride]));
                    float dy = 0.5f*((w_ref_tl*img_ptr[stride]  + w_ref_tr*img_ptr[1+stride] + w_ref_bl*img_ptr[stride*2] + w_ref_br*img_ptr[stride*2+1])
                                    -(w_ref_tl*img_ptr[-stride] + w_ref_tr*img_ptr[1-stride] + w_ref_bl*img_ptr[0]        + w_ref_br*img_ptr[1]));

                    mJacobianCache.row(feature_cnt*patchArea + pixel_count) = \
                        (dx*frame_jac.row(0)+dy*frame_jac.row(1))*fx;
                    
                    // if (Eigen::isnan(mJacobianCache.row(feature_cnt*patchArea + pixel_count))) 
                    //     cout << "intense nan" << endl;
                }
            }
        }
    }

    float ImageAlign::computeError(FramePtr refFrame, FramePtr curFrame, int level, bool linearSystem, bool useWeight) { 
        const int border  = halfPatchSize+1;
        cv::Mat& cur_img  = curFrame->imagePyr()[level];
        const int stride  = cur_img.cols;
        const int width   = cur_img.cols, height = cur_img.rows;
        const float scale = 1.0f/(1<<level);
        const float fx    = curFrame->camera()->errorMultiplier2();
        Vector3f pos      = refFrame->pose().translation().cast<float>();

        auto& features = refFrame->obs();
        float chi2 = 0;
        int feature_cnt = 0;
        for (auto begin = features.begin(); begin != features.end(); ++begin, ++feature_cnt) {
            if (!mVisables[feature_cnt])
                continue;
            
            Vector3f& wxyz = (*begin)->mLandmark->xyz();
            float depth = (wxyz - pos).norm();
            Vector3d rxyz = ((*begin)->mDirect*depth).cast<double>();

            Vector3d d_cxyz = mTc_r_new*rxyz;
            Vector3f f_cxyz = d_cxyz.cast<float>();
            Vector2f cuv    = curFrame->camera()->world2cam(f_cxyz);
            Vector2f fPx    = cuv*scale;
            const float u_fref = fPx(0),             v_fref = fPx(1);
            const int   u_iref = (int)floor(fPx(0)), v_iref = (int)floor(fPx(1));
            // if the feature has no landmark or this point can not calculate the gradient of image
            if (!(*begin) or 
                !INSIDEIMAGE(u_iref, border, 0, width) or 
                !INSIDEIMAGE(v_iref, border, 0, height)) {
                continue;
            }

            const float subpix_u_ref = u_fref-u_iref;
            const float subpix_v_ref = v_fref-v_iref;
            const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
            const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
            const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;

            int pixel_count = 0;
            for (int i = 0; i < patchSize; i++) {
                uchar* img_ptr = (uchar*)cur_img.data + (v_iref+i-halfPatchSize)*stride + (u_iref-halfPatchSize);
                for (int j = 0; j < patchSize; j++, img_ptr++, pixel_count++) {
                    float instensy = \
                        w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[1] + w_ref_bl*img_ptr[stride] + w_ref_br*img_ptr[stride+1];
                    float res = instensy - mRefPatchCache(feature_cnt, pixel_count);
                    float weight = 1.0f;
                    if (useWeight) 
                        weight = weightFunction(res/scale);

                    if (isnan(res))    cout << "res is nan" << endl;
                    if (isnan(weight)) cout << "weight is nan" << endl;

                    chi2 += res*res*weight;
                    mInliersCnt++;
                    if (linearSystem) {
                        // mJacobianCache.row()
                        Matrix<float, 1, 6> J(mJacobianCache.row(feature_cnt*patchArea + pixel_count));
                        mH.noalias() += J.transpose()*J*weight;
                        mb.noalias() -= J.transpose()*res*weight;
                    }
                }
            }
        }
        return chi2 / (mInliersCnt+0.0001f);
    }

    bool ImageAlign::solve() {
        mDeltaX = mH.ldlt().solve(mb);
        if (isnan(mDeltaX[0]) or isinf(mDeltaX[0])) {
            return false;
        }
        return true;
    }

    bool ImageAlign::update(Sophus::SE3& new_model, Sophus::SE3& old_model) {
        Matrix<double, 6, 1> ddeltaX = mDeltaX.cast<double>();
        new_model = old_model*Sophus::SE3::exp(-ddeltaX); 
    }

    float ImageAlign::maxLimit(Matrix<float, 6, 1>& x) {
        float max_ele = 0;
        for (int i = 0; i < 6; i++)
        if (fabs(x[i]) > max_ele)
            max_ele = fabs(x[i]);
        return max_ele;
    }

    bool ImageAlign::reset() { 
        mInliersCnt = 0; 
        mH.setZero();
        mb.setZero();
        return true;
    }

    float ImageAlign::weightFunction(float res) {
        const float k = 1.345f;
        const float res_abs = std::abs(res);
        if(res_abs < k)
            return 1.0f;
        else
            return k / res_abs;
    }

    void ImageAlign::optimize(FramePtr refFrame, FramePtr curFrame, int level) {
        bool verbose = Config::verbose();
        // prepare data
        prepareData(refFrame, level);
        // main loop
        reset();
        chi2 = computeError(refFrame, curFrame, level, false, true);
        float mu = -1, ro = 0, v = 2;
        if(mu < 0){
            double H_max_diag = 0;
            double tau = 1e-4;
            for(size_t j=0; j<mH.rows(); ++j)
                H_max_diag = std::max(H_max_diag, (double)fabs(mH(j,j)));
            mu = tau*H_max_diag;
        }

        bool stop = false;
        float rho = -1;
        for (int i=0; i<mIterCnt; i++) {
            int failCnt = 0;
            do {
                reset();
                // build linear function H b
                computeError(refFrame, curFrame, level, true, true);
                // LM
                mH += (mH.diagonal()*mu).asDiagonal();
                // solve 
                float chi2_new = 0;
                
                if (solve()) {
                    update(mTc_r_new, mTc_r_old);
                    // success
                    mInliersCnt = 0;
                    // recalculate the F(x)
                    chi2_new = computeError(refFrame, curFrame, level, false, true);
                    rho = chi2 - chi2_new;
                } else {
                    rho = -1;
                }
                // adjust the mu
                if (rho > 0) {
                    if (verbose) {
                        LOG(INFO) << ">>> " << i    << " update success: old chi2: " << chi2 \
                                  << "  new chi2: " << chi2_new \
                                  << "  rho: "      << rho \
                                  << "  inlier: "   << mInliersCnt; 
                    }
                    // update old model
                    mTc_r_old = mTc_r_new;
                    // update chi2, chi2 mean the value of F(x)
                    chi2 = chi2_new;
                    // stop condition
                    stop = maxLimit(mDeltaX)<=meps;
                    // update mu and v
                    mu  *= max(1./3., min(1.-pow(2*rho-1,3), 2./3.));
                    v   = 2.;
                } else {
                    if (verbose) {
                        LOG(INFO) << ">>> " << i << " update failed: old chi2: " << chi2 \
                                  << "  new chi2: " << chi2_new \
                                  << "  rho: " << rho;
                    }
                    // reset the Tcr
                    mTc_r_new = mTc_r_old;
                    // update mu
                    mu = mu*v;
                    v  = v*2;
                    if (failCnt++ > mMaxFailThr)
                        stop = true;
                }
            } while (!(rho>0 || stop));
            // if stop, break
            if (stop) break;
        }
    }
}

#include "image_align.hpp"
#include "config.hpp"

#define SHOW_MATCH 0

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
        mRc_r_new = curFrame->Rwc().inverse() * refFrame->Rwc();
        mtc_r_new = curFrame->Rwc().inverse() * (refFrame->twc()-curFrame->twc());
        mRc_r_old = mRc_r_new; mtc_r_old = mtc_r_new;

        LOG(INFO) << ">>> [ImageAlign] Begin Align image";
        for (int level = mMaxLevel-1; level >= mMinLevel; level--) {
            optimize(refFrame, curFrame, level);
        }
        curFrame->Rwc() = refFrame->Rwc()*mRc_r_new.inverse();
        curFrame->twc() = refFrame->twc() - curFrame->Rwc()*mtc_r_new;
        LOG(INFO) << ">>> [ImageAlign] Finish Align image";
        LOG(INFO) << ">>> [image align] " << curFrame->twc().transpose();

        testImageAlign(refFrame, curFrame);
    }

    void ImageAlign::prepareData(FramePtr refFrame, int level) {
        const int border  = halfPatchSize+1;
        cv::Mat& ref_img  = refFrame->imagePyr()[level];
        const int stride  = ref_img.cols;
        const int width   = ref_img.cols;
        const int height  = ref_img.rows;
        const float scale = 1.0f/(1<<level);
        const float fx    = refFrame->camera()->errorMultiplier2();
        Vector3f& pos     = refFrame->twc();

        auto& features = refFrame->obs();
        mVisables.resize(features.size(), true);
        mRefPatchCache.resize(features.size(),   patchArea);
        mJacobianCache.resize(features.size()*patchArea, 6);
        int feature_cnt = 0;
        for (auto begin  = features.begin(); begin != features.end(); ++begin, ++feature_cnt) {
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
            float depth    = (wxyz - pos).norm();
            Vector3f cxyz  = (*begin)->mDirect*depth;

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
                        (dx*frame_jac.row(0)+dy*frame_jac.row(1))*(fx*scale);
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
        Vector3f& pos     = refFrame->twc();

        auto& features = refFrame->obs();
        float chi2 = 0;
        int feature_cnt = 0;
        for (auto begin = features.begin(); begin != features.end(); ++begin, ++feature_cnt) {
            if (!mVisables[feature_cnt])
                continue;
            Vector2f  rPx     = (*begin)->mPx*scale;
            Vector3f& wxyz    = (*begin)->mLandmark->xyz();
            const float depth = (wxyz - pos).norm();
            Vector3f rxyz     = (*begin)->mDirect*depth;

            Vector3f f_cxyz = mRc_r_new*rxyz + mtc_r_new;
            Vector2f cuv    = curFrame->camera()->world2cam(f_cxyz);
            Vector2f fPx    = cuv*scale;
            const float u_fref = fPx(0),             v_fref = fPx(1);
            const int   u_iref = (int)floorf(fPx(0)), v_iref = (int)floorf(fPx(1));
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
                        weight = weightFunction(res);

                    chi2 += res*res*weight;
                    mInliersCnt++;
                    if (linearSystem) {
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

    bool ImageAlign::update(Quaternionf& new_Rcw, Vector3f& new_tcw, Quaternionf& old_Rcw, Vector3f& old_tcw) {
        Vector3f deltaTheta = -mDeltaX.tail<3>();
        Vector3f deltaTrans = -mDeltaX.head<3>();

        Quaternionf deltaQ(1, deltaTheta(0)/2, deltaTheta(1)/2, deltaTheta(2)/2);

        new_Rcw = old_Rcw * deltaQ;
        new_tcw = old_tcw + deltaTrans;
        new_Rcw.normalize();

        // Matrix<double, 6, 1> ddelta = mDeltaX.cast<double>();
        // Sophus::SE3 deltaT = Sophus::SE3::exp(-ddelta);
        // Quaternionf deltaQ = deltaT.unit_quaternion().cast<float>();
        // Vector3f    deltaTrans = deltaT.translation().cast<float>();

        // new_Rcw = old_Rcw * deltaQ;
        // new_tcw = old_tcw + old_Rcw * deltaTrans;

        // Sophus::SE3 old_model(old_Rcw.cast<double>(), old_tcw.cast<double>());
        // Sophus::SE3 new_model = old_model * Sophus::SE3::exp(-ddelta);
        // new_Rcw = new_model.unit_quaternion().cast<float>();
        // new_tcw = new_model.translation().cast<float>();
        return true; 
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
        mInliersCnt = 0;
        chi2 = computeError(refFrame, curFrame, level, false, false);
        float mu = 0.1f, v = 2, rho = -1;
        if(mu < 0){
            double H_max_diag = 0;
            double tau = 1e-4;
            for(size_t j=0; j<mH.rows(); ++j)
                H_max_diag = std::max(H_max_diag, (double)fabs(mH(j, j)));
            mu = tau*H_max_diag;
        }

        bool stop = false;
        for (int i=0; i<mIterCnt; i++) {
            int failCnt = 0;
            do {
                reset();
                // build linear function H b
                computeError(refFrame, curFrame, level, true, false);
                // LM
                mH += (mH.diagonal()*mu).asDiagonal();
                // solve 
                float chi2_new = 0;
                if (solve()) {
                    update(mRc_r_new, mtc_r_new, mRc_r_old, mtc_r_old);
                    mInliersCnt = 0;
                    chi2_new = computeError(refFrame, curFrame, level, false, false);
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
                    mRc_r_old = mRc_r_new;
                    mtc_r_old = mtc_r_new;
                    chi2 = chi2_new;
                    stop = (maxLimit(mDeltaX)<=meps);
                    mu  *= max(1./3., min(1.-pow(2*rho-1,3), 2./3.));
                    v   = 2.;
                } else {
                    if (verbose) {
                        LOG(INFO) << ">>> " << i << " update failed: old chi2: " << chi2 \
                                  << "  new chi2: " << chi2_new \
                                  << "  rho: " << rho;
                    }
                    // reset the Tcr
                    mRc_r_new = mRc_r_old;
                    mtc_r_new = mtc_r_old;
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

    bool ImageAlign::testImageAlign(FramePtr refFrame, FramePtr curFrame) {
        Quaternionf Rrw = refFrame->Rwc().inverse();
        Quaternionf Rcw = curFrame->Rwc().inverse();
        Vector3f&   twr = refFrame->twc();
        Vector3f&   twc = curFrame->twc();

        Mat& refImage = refFrame->imagePyr()[0];
        Mat& curImage = curFrame->imagePyr()[0];
        Mat  mergeImage;
        addWeighted(refImage, 0.4, curImage, 0.6, 0, mergeImage);
        cvtColor(mergeImage, mergeImage, COLOR_GRAY2BGR);

        int num = 0;
        auto& features = refFrame->obs();
        for (auto it = features.begin(); it != features.end(); it++) {
            FeaturePtr feature = *it;
            if (!feature->mLandmark) {
                continue;
            }

            Vector3f Pw = feature->mLandmark->xyz();
            Vector2f Pruv = refFrame->world2uv(Pw);
            Vector2f Pcuv = curFrame->world2uv(Pw);

            if (!(refFrame->isVisible(Pruv, 0) && curFrame->isVisible(Pcuv, 0))) {
                continue;
            }

            Scalar color(theRNG().uniform(0, 255), theRNG().uniform(0, 255), theRNG().uniform(0, 255));
            line(mergeImage, Point(Pruv(0), Pruv(1)), Point(Pcuv(0), Pcuv(1)), color);
            num++;
        }

        LOG(INFO) << "[testImageAlign] image align " << num << "/" << features.size() << endl;
        imshow("[Image Align] image alignment", mergeImage);
        waitKey();
        return true;
    }
}

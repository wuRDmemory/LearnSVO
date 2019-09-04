#include "image_align.hpp"

namespace mSVO {
    #define INSIDEIMAGE(x, dx, min, max) return (x-dx) >= (min) and (x+dx) < (max)

    ImageAlign::ImageAlign(int minlevel, int maxlevel, int iterCnt): mMinLevel(minlevel), mMaxLevel(maxlevel), mIterCnt(iterCnt){
        ;
    }

    ImageAlign::~ImageAlign() {
        ;
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
        mRefPatchCache = MatrixXf(features.size(),   patchArea);
        mJacobianCache = MatrixXf(features.size()*patchArea, 6);
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
                    mRefPatchCache[feature_cnt, pixel_count] = \
                        w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[1] + w_ref_bl*img_ptr[stride] + w_ref_br*img_ptr[stride+1];
                    
                    float dx = 0.5f*((w_ref_tl*img_ptr[1]  + w_ref_tr*img_ptr[2] + w_ref_bl*img_ptr[stride+1] + w_ref_br*img_ptr[stride+2])
                                    -(w_ref_tl*img_ptr[-1] + w_ref_tr*img_ptr[0] + w_ref_bl*img_ptr[stride-1] + w_ref_br*img_ptr[stride]));
                    float dy = 0.5f*((w_ref_tl*img_ptr[stride]  + w_ref_tr*img_ptr[1+stride] + w_ref_bl*img_ptr[stride*2] + w_ref_br*img_ptr[stride*2+1])
                                    -(w_ref_tl*img_ptr[-stride] + w_ref_tr*img_ptr[1-stride] + w_ref_bl*img_ptr[0]        + w_ref_br*img_ptr[1]));

                    mJacobianCache.row(feature_cnt*patchArea + pixel_count) = \
                        (dx*frame_jac.row(0)+dy*frame_jac.row(1))*fx;
                }
            }
        }
    }

    float ImageAlign::computeError(FramePtr refFrame, FramePtr curFrame, int level) { 
        const int border  = halfPatchSize+1;
        cv::Mat& cur_img  = curFrame->imagePyr()[level];
        const int stride  = cur_img.cols;
        const int width   = cur_img.cols, height = cur_img.rows;
        const float scale = 1.0f/(1<<level);
        const float fx    = curFrame->camera()->errorMultiplier2();
        Vector3f pos      = refFrame->pose().translation().cast<float>();

        auto& features = refFrame->obs();
        float chi2 = 0;
        int feature_cnt = 0, n_cnt = 0;
        for (auto begin = features.begin(); begin != features.end(); ++begin, ++feature_cnt) {
            if (!mVisables[feature_cnt])
                continue;
            
            Vector3f& wxyz = (*begin)->mLandmark->xyz();
            float depth = (wxyz - pos).norm();
            Vector3f rxyz = (*begin)->mDirect*depth;

            Vector3f cxyz = mTnewc_r*rxyz;
            Vector2f cuv  = cxyz.head<2>()/cxyz(2);
            Vector2f fPx  = cuv*scale;
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
                    float res = instensy - mRefPatchCache[feature_cnt, pixel_count];
                    float weight = weightFunction(res/scale);
                    chi2 += res*res*weight;
                    n_cnt ++;
                    Matrix<float, 1, 6>& J = mJacobianCache.row(feature_cnt*patchArea + pixel_count);
                    mH.noalias() += J.translation()*J*weight;
                    mb.noalias() += J.translation()*e*weight;
                }
            }
        }
        return chi2 / n_cnt;
    }

    void ImageAlign::optimize(FramePtr refFrame, FramePtr curFrame, int level) {
        // prepare data
        prepareData(refFrame, level);
        // main loop
        float old_chi2 = 0;
        for (int i=0; i<mIterCnt; i++) {
            mH.setZero();
            mb.setZero();

            computeError();
        }
    }
}

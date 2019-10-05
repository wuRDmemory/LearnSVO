#include "alignment.hpp"

namespace mSVO {
    float Alignment::interpolateU8(cv::Mat& image, Vector2f& px) {
        // compute interpolation weights
        const int stride = image.cols;
        const int u_r = floor(px[0]);
        const int v_r = floor(px[1]);
        const float subpix_u = px[0]-u_r;
        const float subpix_v = px[1]-v_r;
        const float wTL = (1.0-subpix_u)*(1.0-subpix_v);
        const float wTR = subpix_u * (1.0-subpix_v);
        const float wBL = (1.0-subpix_u)*subpix_v;
        const float wBR = subpix_u * subpix_v;

        uint8_t* img_ptr = (uint8_t*)image.data  + v_r*image.cols + u_r;
        return wTL*img_ptr[0] + wTR*img_ptr[1] + wBL*img_ptr[stride] + wBR*img_ptr[stride+1];
    }

    bool Alignment::align2D(cv::Mat& image, Vector2f& px, int iterCnt, int patchSize, 
                            uint8_t* patch, uint8_t* patchWithBorder) {
        
        // prepare the Jacobian
        const int halfPatchSize = patchSize/2;
        const int patchArea     = patchSize*patchSize;
        const int patchSizeBorder = patchSize+2;
        bool converged = false;

        float __attribute__((__aligned__(16))) refPatchDx[patchArea] = {0};
        float __attribute__((__aligned__(16))) refPatchDy[patchArea] = {0};

        Matrix3f H; H.setZero();
        // compute gradient and hessian

        float* it_dx = refPatchDx;
        float* it_dy = refPatchDy;
        for(int y = 0; y < patchSize; ++y) {
            uint8_t* it = patchWithBorder + (y+1)*patchSizeBorder + 1;
            for(int x = 0; x < patchSize; ++x, ++it, ++it_dx, ++it_dy) {
                Vector3f J;
                J[0] = 0.5 * (it[1] - it[-1]);
                J[1] = 0.5 * (it[patchSizeBorder] - it[-patchSizeBorder]);
                J[2] = 1;
                *it_dx = J[0];
                *it_dy = J[1];
                H += J*J.transpose();
            }
        }
        Matrix3f Hinv = H.inverse();
        float mean_diff = 0;

        // Compute pixel location in new image:
        float u = px.x();
        float v = px.y();

        // termination condition
        const float min_update_squared = 0.03f * 0.03f;
        const int cur_step = image.step.p[0];

        //  float chi2 = 0;
        Vector3f update; update.setZero();
        for(int iter = 0; iter < iterCnt; ++iter) {
            int u_r = floor(u);
            int v_r = floor(v);
            if(u_r < halfPatchSize || v_r < halfPatchSize || u_r >= image.cols - halfPatchSize || v_r >= image.rows - halfPatchSize)
                break;

            if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
                return false;

            // compute interpolation weights
            float subpix_x = u-u_r;
            float subpix_y = v-v_r;
            float wTL = (1.0-subpix_x)*(1.0-subpix_y);
            float wTR = subpix_x * (1.0-subpix_y);
            float wBL = (1.0-subpix_x)*subpix_y;
            float wBR = subpix_x * subpix_y;

            // loop through search_patch, interpolate
            uint8_t* it_ref = patch;
            float* it_ref_dx = refPatchDx;
            float* it_ref_dy = refPatchDy;
            //    float new_chi2 = 0.0;
            Vector3f Jres; Jres.setZero();
            for(int y = 0; y < patchSize; ++y) {
                uint8_t* it = (uint8_t*) image.data + (v_r + y - halfPatchSize) * cur_step + u_r - halfPatchSize;
                for(int x = 0; x < patchSize; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy) {
                    float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
                    float res = search_pixel - *it_ref + mean_diff;
                    Jres[0] -= res*(*it_ref_dx);
                    Jres[1] -= res*(*it_ref_dy);
                    Jres[2] -= res;
                    // new_chi2 += res*res;
                }
            }

            update = Hinv * Jres;
            u += update[0];
            v += update[1];
            mean_diff += update[2];

            if(update[0]*update[0]+update[1]*update[1] < min_update_squared) {
                converged=true;
                break;
            }
        }

        px << u, v;
        return converged;
    }
}



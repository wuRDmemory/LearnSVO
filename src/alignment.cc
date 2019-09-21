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
                            const uint8_t* patch, const uint8_t* patchWithBorder) {
        
        // prepare the Jacobian
        
        
    }
}



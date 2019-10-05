#include "depth_filter.hpp"
#include "config.hpp"

namespace mSVO {
    DepthFilter::DepthFilter(MapPtr map): mMap(map) {
        ;
    }

    DepthFilter::~DepthFilter() {

    }

    bool DepthFilter::addNewFrame(FramePtr frame) {

    }

    bool DepthFilter::addNewKeyFrame(FramePtr frame) {
        
    }

    bool DepthFilter::runFilter(FramePtr frame) {
        
        float pxNoise      = 1.0f;
        float focalLength  = frame->camera()->errorMultiplier2();
        float pxErrorAngle = atan(px_noise/(2.0*focal_length))*2.0; 
        
        for (auto begin = mSeedList.begin(); begin != mSeedList.end(); ++begin) {
            SeedPtr seed = *begin;
            // check this seed is or not too old
            if (frame->ID() - seed->seenFrameID > Config::depthFilterTrackMaxGap()) {
                mSeedList->erase(seed);
                continue;
            }

            // get releative pose
            FeaturePtr feature = seed->feature;
            Quaternionf Rcr = frame->Rwc().inverse()*feature->mFrame->Rwc();
            Vector3f    tcr = frame->Rwc().inverse()*(feature->mFrame->twc() - frame->twc());

            Vector3f   rxyz = feature->mDirect/seed->mu;
            Vector3f   cxyz = Rcr*rxyz + tcr;
            if (cxyz(2) < 0) {
                continue;
            }

            // project the cxyz to plane
            Vector2f cuv = frame->camera()->world2cam(cxyz);
            frame->isVisible(cuv, 0);

            // project the depth min and max to plane.
            float sigma = sqrt(seed->sigma2);
            float minDepth = seed->mu + sigma;
            float maxDepth = max(seed->mu - sigma, 0.0001f);

            
            // update seed's seen id
            seed->seenFrameID = frame->ID();
        }
    }
}

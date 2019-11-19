#include "frame.hpp"
#include "config.hpp"
#include "feature.hpp"
#include "landmark.hpp"

namespace mSVO {
    int Frame::id = 0;
    
    Frame::Frame(double timestamp, CameraModel* camera, const cv::Mat& img):
        mTimestamp(timestamp), mCamera(camera), mID(id++), mIsKeyFrame(false) {
        initFrame(img);
    }

    Frame::~Frame() {
        cout << "Destory Frame id " << mID << " FeatureCnt: " << mObs.size() << endl;
        // std::for_each(mObs.begin(), mObs.end(), [](Feature* ftr) {
        //     delete(ftr);
        //     ftr = NULL;
        // });
        
        auto it = mObs.begin();
        while (it != mObs.end()) {
            delete(*it);
            it = mObs.erase(it);
        }
    }

    void Frame::initFrame(const cv::Mat& img) {
        if (img.empty()) {
            throw std::runtime_error(">>> [ERROR] feed image error");
        }
        std::for_each(mObs.begin(), mObs.end(), [](Feature* ftr) {
            ftr = NULL;
        });

        float scale = 1.0f/Config::pyramidFactor();
        mImagePyr.resize(Config::pyramidNumber());
        mImagePyr[0] = img.clone();
        for (int i=1, N = Config::pyramidNumber(); i<N; i++) {
            cv::pyrDown(mImagePyr[i-1], mImagePyr[i]);
        }
    }

    void Frame::addFeature(Feature* feature) {
        mObs.push_back(feature);
    }

    bool Frame::isVisible(const Vector3f& xyz) {
        Vector2f cxy = world2uv(xyz);
        return cxy(0) >= 0 && cxy(0) < mCamera->width() && cxy(1) >= 0 && cxy(1) < mCamera->height();
    }

    bool Frame::isVisible(const Vector2f& uv, int border, int level) {
        const float scale = 1.0f/(1<<level);
        const int   width = mCamera->width() *scale;
        const int  height = mCamera->height()*scale;
        return uv(0) >= border && uv(0) <  width-border && uv(1) >= border && uv(1) < height-border;
    }

    bool Frame::getDepth(float& minDepth, float& meanDepth) {
        
        minDepth = FLT_MAX;
        meanDepth = 0;

        int depthCnt = 0;
        auto it = mObs.begin();
        while (it != mObs.end()) {
            FeaturePtr feature = *it;
            LandMark* ldmk = feature->mLandmark;
            if (!ldmk || ldmk->type == LandMark::LANDMARK_TYPR::DELETE) {
                continue;
            }
            Vector3f& xyz = ldmk->xyz();
            float z = (xyz - mtwc).norm();
            minDepth = std::min(z, minDepth);
            meanDepth += z;
            depthCnt++;
            it++;
        }

        meanDepth /= depthCnt;
        return true;
    }

    Vector2f Frame::world2uv(const Vector3f& XYZ) {
        Vector3f cxyz = mRwc.inverse()*(XYZ - mtwc);
        return mCamera->world2cam(cxyz);
    }

    Vector3f Frame::world2camera(const Vector3f& XYZ) {
        return mRwc.inverse()*(XYZ - mtwc);
    }
}

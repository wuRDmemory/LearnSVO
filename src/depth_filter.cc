#include "depth_filter.hpp"
#include "config.hpp"

namespace mSVO {
    int Seed::ID      = 0;
    int Seed::batchID = 0;

    Seed::Seed(FeaturePtr ftr, float depthMin, float depthMean) {
        id = ID++;
        feature = ftr;
        mu     = 1.0f/depthMean;
        zRange = 1.0f/depthMin;
        a = b  = 10;
        sigma2 = (zRange*zRange)/36;
        seenFrameID = batchID;
    }

    DepthFilter::DepthFilter(MapPtr map): mMap(map), mStop(false), mNewKeyFrameFlag(false) {
        mDetector = new Detector(Config::width(), Config::height(), Config::gridCellNumber(), Config::pyramidNumber(), 10.0f);
    }

    DepthFilter::~DepthFilter() {
        mStop = true;
        mThread->join();
        delete mThread;
        delete mDetector;
    }

    bool DepthFilter::startThread() { 
        mStop = false;
        mSeedUpdateHalt = false;
        mThread = new thread(&DepthFilter::mainloop, this);
        return true;
    }

    bool DepthFilter::addNewFrame(FramePtr frame) {
        {
            unique_lock<mutex> lock(mAddFrameLock);
            if (mAddFrames.size() > 2) {
                mAddFrames.pop();
            }
            mAddFrames.push(frame);
        }
        mSeedUpdateHalt = false;
        mConditionVariable.notify_one();
    }

    bool DepthFilter::addNewKeyFrame(FramePtr frame, float minDepth, float meanDepth) {
        assert(frame->isKeyFrame());
        mDepthMin  = minDepth;
        mDepthMean = meanDepth;
        mNewKeyFrameFlag = true;
        mSeedUpdateHalt  = true;
        mNewKeyFrame     = frame;
        mConditionVariable.notify_one();
    }

    bool DepthFilter::clearFrameList() { 
        while (!mAddFrames.empty()) {
            mAddFrames.pop();
        }
        return true;
    }

    bool DepthFilter::mainloop() {
        while(!mStop) {
            FramePtr frame;
            {
                unique_lock<mutex> lock(mAddFrameLock);
                while (mAddFrames.empty() && !mNewKeyFrameFlag) {
                    mConditionVariable.wait(lock);
                }
                // check new key frame first
                if(mNewKeyFrameFlag) {
                    mNewKeyFrameFlag = false;
                    mSeedUpdateHalt  = false;
                    clearFrameList();
                    frame = mNewKeyFrame;
                } else {
                    frame = mAddFrames.front();
                    mAddFrames.pop();
                }
            }
            runFilter(frame);
            if(frame->isKeyFrame()) {
                initialKeyFrame(frame);
            }
        }
    }

    bool DepthFilter::initialKeyFrame(FramePtr& keyframe) {
        mSeedUpdateHalt = true;
        Seed::batchID++;

        vector<Point2f> corners;
        vector<int>     cornersLevel;
        mDetector->setMask(keyframe->obs());
        mDetector->detect(keyframe.get(), corners, cornersLevel);
        unique_lock<mutex> lock(mAddSeedLock);
        for (int i = 0, N = corners.size(); i < N; i++) {
            Vector2f px(corners[i].x, corners[i].y);
            FeaturePtr f = new Feature(keyframe.get(), px, cornersLevel[i]);
            SeedPtr newSeed = new Seed(f, mDepthMin, mDepthMean);
            mSeedList.push_back(newSeed);
        }
        mSeedUpdateHalt = false;
        LOG(INFO) << ">>> [initialKeyFrame] init " << corners.size() << " new seeds";
        return true;
    }

    bool DepthFilter::runFilter(FramePtr frame) {
        mNFailNum = 0;
        mNMatched = 0;
        float pxNoise      = 1.0f;
        float focalLength  = frame->camera()->errorMultiplier2();
        float pxErrorAngle = atan(pxNoise/(2.0*focalLength))*2.0; 
        
        for (auto begin = mSeedList.begin(); begin != mSeedList.end();) {
            if (mSeedUpdateHalt) {
                break;
            }

            SeedPtr seed = NULL;
            {
                unique_lock<mutex> lock(mAddSeedLock);
                seed = *begin;
            }
            // check this seed is too old
            if (Seed::batchID - seed->seenFrameID > Config::depthFilterTrackMaxGap()) {
                begin = mSeedList.erase(begin);
                continue;
            }

            // get releative pose
            FeaturePtr feature = seed->feature;
            Frame*    refFrame = feature->mFrame;
            Quaternionf Rcr = frame->Rwc().inverse()*refFrame->Rwc();
            Vector3f    tcr = frame->Rwc().inverse()*(refFrame->twc() - frame->twc());

            Vector3f   rxyz = feature->mDirect/seed->mu;
            Vector3f   cxyz = Rcr*rxyz + tcr;
            if (cxyz(2) < 0) {
                begin++;
                continue;
            }

            // project the cxyz to plane
            Vector2f cuv = frame->camera()->world2cam(cxyz);
            if (!frame->isVisible(cuv, 0)) {
                begin++;
                continue;
            }

            // project the depth min and max to plane.
            float sigma = sqrt(seed->sigma2);
            float minDepth = 1.0f/(seed->mu + sigma);
            float maxDepth = 1.0f/max(seed->mu - sigma, 0.0001f);
            float z = -1.0f;
            if (!mMatcher.findEpipolarMatch(frame.get(), feature, Rcr, tcr, minDepth, 1.0f/seed->mu, maxDepth, z) || z == -1.0f) {
                seed->b++;
                mNFailNum++;
                begin++;
                continue;
            }

            // compute tau
            float tau = computeTau(Rcr, tcr, feature->mDirect, z, pxErrorAngle);
            float tauInverse = 0.5f * (1.0f/max(0.0000001f, z - tau) - 1.0f/(z + tau));

            // update the estimate
            updateSeed(1.0f/z, tauInverse*tauInverse, seed);
            mNMatched++;

            // update seed's seen id
            seed->seenFrameID = Seed::batchID;
            if(frame->isKeyFrame()) {
                // The feature detector should not initialize new seeds close to this location
                mDetector->setMask(mMatcher.getUV());
            }

            // TODO: check the coverage
            if (seed->sigma2 < Config::depthFilterSigmaThr()) {
                assert(seed->feature->mLandmark == NULL); // TODO this should not happen anymore
                Vector3f wxyz(feature->mFrame->Rwc()*(feature->mDirect*(1.0/seed->mu)) + feature->mFrame->twc());
                LandMarkPtr point = new LandMark(wxyz, feature);
                seed->feature->mLandmark = point;
                {
                    // add the landmark to the candidate list
                    mMap->candidatePointManager().addCandidateLandmark(point, frame);
                }
                begin = mSeedList.erase(begin);
            } else if (isnan(minDepth)) {
                begin = mSeedList.erase(begin);
            } else {
                begin++;
            }
        }
        return true;
    }

    float DepthFilter::computeTau(Quaternionf& Rcr, Vector3f& tcr, Vector3f& direct, float z, float noiseAngle) {
        Vector3f a = direct*z - tcr;
        float t_norm = tcr.norm();
        float a_norm = a.norm();
        float alpha  = acos(direct.dot(tcr)/t_norm); // dot product
        float beta   = acos(a.dot(-tcr)/(t_norm*a_norm)); // dot product
        float beta_plus  = beta + noiseAngle;
        float gamma_plus = M_PI - alpha-beta_plus; // triangle angles sum to PI
        float z_plus     = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
        return (z_plus - z); // tau
    }

    bool DepthFilter::updateSeed(const float x, const float tau2, Seed* seed) {
        float norm_scale = sqrt(seed->sigma2 + tau2);
        if(std::isnan(norm_scale))
            return false;
        
        std::normal_distribution<float> nd(seed->mu, norm_scale);
        float s2 = 1./(1./seed->sigma2 + 1./tau2);
        float m = s2*(seed->mu/seed->sigma2 + x/tau2);
        float C1 = seed->a/(seed->a+seed->b) * nd(generator);
        float C2 = seed->b/(seed->a+seed->b) * 1.0f/seed->zRange;
        float normalization_constant = C1 + C2;
        C1 /= normalization_constant;
        C2 /= normalization_constant;
        float f = C1*(seed->a+1.)/(seed->a+seed->b+1.) + C2*seed->a/(seed->a+seed->b+1.);
        float e = C1*(seed->a+1.)*(seed->a+2.)/((seed->a+seed->b+1.)*(seed->a+seed->b+2.))
                + C2*seed->a*(seed->a+1.0f)/((seed->a+seed->b+1.0f)*(seed->a+seed->b+2.0f));

        // update parameters
        float mu_new = C1*m+C2*seed->mu;
        seed->sigma2 = C1*(s2 + m*m) + C2*(seed->sigma2 + seed->mu*seed->mu) - mu_new*mu_new;
        seed->mu = mu_new;
        seed->a = (e-f)/(f-e/f);
        seed->b = seed->a*(1.0f-f)/f;

        return true;
    }
}

#include "initialization.hpp"
#include "feature.hpp"
#include "frame.hpp"
#include "utils.hpp"

namespace mSVO {
    KltHomographyInit::KltHomographyInit() { 
        mGridCell  = Config::getGridCellNumber();
        mImWidth   = Config::getImageWidth();
        mImHeight  = Config::getImageHeight();
        mFtrNumber = Config::getFeatureNumber();
        mCellFtrNumber = mFtrNumber/mGridCell;
        // initial grid cell
        int row_step = mImHeight / mGridCell;
        int col_step = mImWidth  / mGridCell;
        mGridCellRoi.resize(mGridCell*mGridCell);
        for (int i=0; i<mGridCell; i++) {
            for (int j=0; j<mGridCell; j++) {
                mGridCellRoi[i*mGridCell + j] = \
                    cv::Rect(cv::Point(i*col_step, j*row_step), 
                             cv::Point(i*col_step, (j+1)*row_step));
            }
        }
    }

    InitResult KltHomographyInit::addFirstFrame(FramePtr frameRef) {
        reset();
        mFirstCorners.clear();
        detectCorner(frameRef, mFirstCorners);
        
        if (mFirstCorners.size() < 200) {
            LOG(ERROR) << ">>> Too few corner detected " << mFirstCorners.size();
            return FAILURE;
        }
        mRefFrame = frameRef;
        return SUCCESS;
    }

    InitResult KltHomographyInit::addSecondFrame(FramePtr frameRef) {
        vector<cv::Point2f> nextPoints;
        vector<uchar> status;
        vector<float> error;
        cv::calcOpticalFlowPyrLK(mRefFrame->mImagePyr[0], frameRef->mImagePyr[0], mFirstCorners, nextPoints, status, error);

        int j = 0;
        for (int i = 0; i < status.size(); i++) if (status[i]) {
            Vector2f px(nextPoints[i].x(), nextPoints[i].y());
            mFirstCorners[j] = mFirstCorners[i];
            nextPoints[j++]  = nextPoints[i];
            mRefFrame->addFeature(new Feature(frameRef, px, 1));
        }

        // use fundamental matrix to shift some outlier
        cv::findFundamentalMat(mFirstCorners, nextPoints, cv::Mat(), cv::FM_RANSAC, 3.0, 0.99, status);
        

        return SUCCESS;
    }

    bool KltHomographyInit::detectCorner(FramePtr frame, vector<cv::Point2f>& points) {
        cv::Mat& image = frame->mImagePyr[0];
        for (int i=0; i<mGridCell; i++) {
            for (int j=0; j<mGridCell; j++) {
                cv::Rect rroi = mGridCellRoi[i*mGridCell + j];
                cv::Point tl = rroi.tl();
                cv::Point br = rroi.br();
                cv::Mat roi  = image.rowRange(tl.y, br.y).colRange(tl.x, br.x);
                // find fast corner
                vector<cv::Point2f> corner;
                cv::goodFeaturesToTrack(roi, corner, mCellFtrNumber, 0.1, 10);
                for (int i=0; i<corner.size(); i++) {
                    points.emplace_back(corner[i]);
                    const Vector2f px(corner[i].x, corner[i].y);
                    frame->mObs.push_back(new Feature(frame, px, 1));
                }
            }
        }
        return true;
    }

    void KltHomographyInit::reset() { 
        mRefFrame = NULL;
        mFirstCorners.clear();
    }
}

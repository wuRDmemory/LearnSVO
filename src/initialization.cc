#include "initialization.hpp"
#include "utils.hpp"

namespace mSVO {
    KltHomographyInit::KltHomographyInit() { 
        mDetector  = cv::FastFeatureDetector::create();
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
                mGridCell[i*mGridCell + j] = \
                    cv::Rect(cv::Point(i*col_step, j*row_step), 
                             cv::Point(i*col_step, (j+1)*row_step));
            }
        }
    }

    InitResult KltHomographyInit::addFirstFrame(FramePtr frameRef) {
        reset();

    }


    bool KltHomographyInit::detectCorner(FramePtr frame, 
                      vector<cv::Point>& points, 
                      vector<Eigen::Vector3f>& ftrs) {
        cv::Mat& image = frame->mImagePyr[0];

        vector<>
        for (int i=0; i<mGridCell; i++) {
            for (int j=0; j<mGridCell; j++) {
                cv::Rect rroi = mGridCell[i*mGridCell + j];
                cv::Point tl = rroi.tl();
                cv::Point br = rroi.br();
                cv::Mat& roi  = image.rowRange(tl.y, br.y).colRange(tl.x, br.x);
                // find fast corner
                vector<cv::Point2f> corner;
                cv::goodFeaturesToTrack(roi, corner, mCellFtrNumber, 0.1, 10);


            }
        }
        
    }
}

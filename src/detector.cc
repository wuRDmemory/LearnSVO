#include "detector.hpp"
#include <glog/logging.h>

namespace mSVO {
    Detector::Detector(int width, int height, int step, int levels, float threshold): 
        mWidth(width), mHeight(height), mStep(step), 
        mLevels(levels), mThreshold(threshold), mBorder(3) {
        
        /// some value
        mRows = mHeight / mStep;
        mCols = mWidth  / mStep;

        /// ROIs 
        mGridCellRoi.resize(mStep*mStep);
        for (int i=0; i<mStep; i++)
        for (int j=0; j<mStep; j++) {
            mGridCellRoi[i*mStep + j] = \
                cv::Rect(cv::Point((i+0)*mCols,   (j+0)*mRows  ), 
                         cv::Point((i+1)*mCols-1, (j+1)*mRows-1));
        }
        reset();
    }

    bool Detector::detect(Frame* frame, vector<Point2f>& keyPoints, vector<int>& pointsLevel) {
        vector<Mat>& pyrImages = frame->imagePyr();
        assert(pyrImages.size() == mLevels);
        
        for (int l = 0; l < mLevels; l++) {
            vector<KeyPoint> kps;
            FAST(pyrImages[l], kps, mThreshold);

            /*  find the most strong response key point  */
            for (int j = 0; j < kps.size(); j++) {
                KeyPoint& kp = kps[j];
                Point2f   pt = kp.pt*(1 << l);
                if (pt.x < mBorder          || pt.y < mBorder          || 
                    pt.x > mWidth - mBorder || pt.y > mHeight - mBorder) {
                    continue;
                }

                const int x  = pt.x / mStep;
                const int y  = pt.y / mStep;
                
                if (mOccupied[y*mCols + x]) {
                    continue;
                }

                if (mCells[y*mCols + x].score < kp.response) {
                    mCells[y*mCols + x].score = kp.response;
                    mCells[y*mCols + x].px    = pt;
                    mCells[y*mCols + x].level = l;
                }
            }
        }
        
        vector<int> summary(mLevels+1, 0);

        const int N = mRows * mCols;
        pointsLevel.reserve(N);
        keyPoints.reserve(N);
        for (int i = 0; i < N; i++) {
            CellElem& cell = mCells[i];
            if (cell.level < 0)
                continue;
            keyPoints.push_back(cell.px);
            pointsLevel.push_back(cell.level);

            summary[cell.level]++;
            summary.back()++;
        }

        LOG(INFO) << ">>> [Detect] Summary : Level: " << summary[0] << "  " << summary[1] << "  " << summary[2] << "  total: " << summary.back();

        // reset all occupied cell
        reset();
        return true;
    }

    bool Detector::setMask(const Vector2f& uv) {
        int x = int(uv(0)) / mStep;
        int y = int(uv(1)) / mStep;
        mOccupied[y*mCols + x] = true;
        // if (mMask.at<uchar>(int(uv.y()), int(uv.y()))) {
        //     cv::circle(mMask, Point(uv.x(), uv.y()), 15, cv::Scalar::all(0), -1);
        // }
        return true;
    }

    bool Detector::setMask(const Features& obs) {
        for (auto it = obs.begin(), end = obs.end(); it != end; it++) {
            Vector2f uv = (*it)->mPx;
            int x = int(uv(0)) / mStep;
            int y = int(uv(1)) / mStep;
            mOccupied[y*mCols + x] = true;
        }
        // cv::imshow("mask", mMask);
        // cv::waitKey();
        return true;
    }

    bool Detector::reset() {
        mCells.clear();
        mCells.resize(mCols*mRows, CellElem());
        mOccupied.clear();
        mOccupied.resize(mRows*mCols, false);
        // mMask = Mat(mHeight, mWidth, CV_8U, Scalar::all(255));
        return true;
    }
    
    float Detector::calculateScore(const Mat& img, int u, int v) {
        assert(img.type() == CV_8UC1);
        float dXX = 0.0;
        float dYY = 0.0;
        float dXY = 0.0;
        const int halfbox_size = 4;
        const int box_size = 2*halfbox_size;
        const int box_area = box_size*box_size;
        const int x_min = u-halfbox_size;
        const int x_max = u+halfbox_size;
        const int y_min = v-halfbox_size;
        const int y_max = v+halfbox_size;

        if(x_min < 1 || x_max >= img.cols-1 || y_min < 1 || y_max >= img.rows-1)
            return 0.0; // patch is too close to the boundary

        const int stride = img.step.p[0];
        for( int y=y_min; y<y_max; ++y ) {
            const uint8_t* ptr_left   = img.data + stride*y + x_min - 1;
            const uint8_t* ptr_right  = img.data + stride*y + x_min + 1;
            const uint8_t* ptr_top    = img.data + stride*(y-1) + x_min;
            const uint8_t* ptr_bottom = img.data + stride*(y+1) + x_min;
            for(int x = 0; x < box_size; ++x, ++ptr_left, ++ptr_right, ++ptr_top, ++ptr_bottom) {
                float dx = *ptr_right - *ptr_left;
                float dy = *ptr_bottom - *ptr_top;
                dXX += dx*dx;
                dYY += dy*dy;
                dXY += dx*dy;
            }
        }

        // Find and return smaller eigenvalue:
        dXX = dXX / (2.0 * box_area);
        dYY = dYY / (2.0 * box_area);
        dXY = dXY / (2.0 * box_area);
        return 0.5 * (dXX + dYY - sqrt( (dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY) ));
    }
}

/*
        vector<Point2f> corners;
        for (int lv = 0; lv < mLevels; lv++) {
            corners.clear();
            Mat& img = frame->imagePyr()[lv];
            goodFeaturesToTrack(img, corners, 500, 0.1, 5);

            const int scale = 1<<lv;
            for (int i = 0, N = corners.size(); i < N; i++) {
                int x = int(corners[i].x*scale) % mStep;
                int y = int(corners[i].y*scale) % mStep;

                if (mOccupied[y*mCols + x])
                    continue;
                
                float score = calculateScore(img, corners[i].x, corners[i].y);
                CellElem& elem = mGridCell[y*mCols + x];
                if ( score > elem.score ) {
                    elem.score = score;
                    elem.level = lv;
                    elem.uv    = Point2f(corners[i].x*scale, corners[i].y*scale);
                    size++;
                }
            }
        }

        keyPoints.clear(); pointsLevel.clear();
        keyPoints.reserve(size); pointsLevel.reserve(size);
        for (auto& elem : mGridCell) {
            if (elem.score > mThreshold) {
                pointsLevel.push_back(elem.level);
                keyPoints.push_back(elem.uv);
            }
        }
        keyPoints.resize(keyPoints.size());
        pointsLevel.resize(pointsLevel.size());
*/

#include "detector.hpp"

namespace mSVO {
    Detector::Detector(int width, int height, int step, int levels, float threshold): 
        mWidth(width), mHeight(height), mStep(step), mLevels(levels), mThreshold(threshold) {
        mDetector = cv::FastFeatureDetector::create(20);
        mRows = mHeight / mStep;
        mCols = mWidth  / mStep;

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
        cv::Mat& image = frame->imagePyr()[0];
        for (int i=0; i < mStep; i++)
        for (int j=0; j < mStep; j++) {
            cv::Rect& rroi = mGridCellRoi[i*mStep + j];
            cv::Point tl    = rroi.tl();
            cv::Point br    = rroi.br();
            cv::Mat roi     = image.rowRange(tl.y, br.y).colRange(tl.x, br.x);
            // find fast corner
            vector<cv::Point2f> corner;
            cv::goodFeaturesToTrack(roi, corner, 5, 0.1, 10);
            for (int i=0; i<corner.size(); i++) {
                Point px(corner[i].x + tl.x, corner[i].y+tl.y);
                if (0 == mMask.at<uchar>(px)) {
                    continue;
                }
                keyPoints.emplace_back(corner[i].x + tl.x, corner[i].y + tl.y);
                pointsLevel.emplace_back(0);
            }
        }
        reset();
        return true;
    }

    bool Detector::setMask(const Vector2f& uv) {
        // int x = int(uv(0)) % mStep;
        // int y = int(uv(1)) % mStep;
        // mOccupied[y*mCols + x] = true;
        if (mMask.at<uchar>(int(uv.y()), int(uv.y()))) {
            cv::circle(mMask, Point(uv.x(), uv.y()), 5, cv::Scalar::all(0), -1);
        }
        return true;
    }

    bool Detector::setMask(const Features& obs) {
        for (auto it = obs.begin(), end = obs.end(); it != end; it++) {
            Vector2f uv = (*it)->mPx;
            if (mMask.at<uchar>(int(uv.y()), int(uv.y()))) {
                cv::circle(mMask, Point(uv.x(), uv.y()), 5, cv::Scalar::all(0), -1);
            }
        }
        return true;
    }

    bool Detector::reset() {
        // mGridCell.resize(mCols*mRows, CellElem());
        // mOccupied.resize(mRows*mCols, false);
        mMask = Mat(mHeight, mWidth, CV_8U, Scalar::all(255));
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

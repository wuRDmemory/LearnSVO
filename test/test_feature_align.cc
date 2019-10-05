#include <ostream>
#include <opencv2/opencv.hpp>

#include "alignment.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

void createPatchWithBorder(cv::Mat& image, Vector2f& px, uint8_t* patchWithBorder, int halfPatchSize) {
    Vector2f pyrPx = px;
    for (int row = -halfPatchSize; row < halfPatchSize; ++row)
    for (int col = -halfPatchSize; col < halfPatchSize; ++col, patchWithBorder++) {
        Vector2f pxn = px + Vector2f(col, row);
        if (pxn[0]<0 || pxn[1]<0 || pxn[0]>=image.cols-1 || pxn[1]>=image.rows-1) {
            *patchWithBorder = 0;
        } else {
            *patchWithBorder = (uint8_t)mSVO::Alignment::interpolateU8(image, pxn);
        }
    }
}

void patchFromBorder(uint8_t* patchWithBorder, uint8_t* patch, int patchSize) {
    for (int i = 1; i < patchSize+1; i++) {
        uint8_t* patchBorder = patchWithBorder + i*(patchSize+2) + 1;
        for (int j = 0; j < patchSize; j++, patch++, patchBorder++) {
            *patch = *patchBorder;
        }
    }
}

void printPatch(uint8_t* patch, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            cout << (int)patch[i * size + j] << "\t";
        }
        cout << endl;
    }
}

int main(int argc, char* argv[]) {
    string imagePath = "/home/ubuntu/Downloads/Opencv/opencv-3.4.3/samples/data/graf1.png";

    Mat image = imread(imagePath, 0);
 #if 0
    vector<Point2f> corners;
    cv::goodFeaturesToTrack(image, corners, 10, 0.01, 10, Mat(), 3, 3, 0, 0.04);

    for (Point2f& p : corners) {
        cout << p.x << "  " << p.y << endl;
    }
#endif

    const int patchSize     = 8;
    const int halfPatchSize = 4;
    const int patchArea     = 64;
    Vector2f truthPx(492, 476);

    uint8_t patchWithBorder[(patchSize+2)*(patchSize+2)] = {0};
    uint8_t patch[patchSize*patchSize] = {0};

    createPatchWithBorder(image, truthPx, patchWithBorder, halfPatchSize+1);
    patchFromBorder(patchWithBorder, patch, patchSize);

#if 0
    printPatch(patchWithBorder, patchSize+2);
    cout << endl;
    printPatch(patch, patchSize);
#endif

    Vector2f pxError(2.5, 0.8);  
    Vector2f pxEst = truthPx - pxError;
    cout << "Truth     : " << truthPx.x() << " " << truthPx.y() << endl;
    cout << "before Est: " << pxEst.x() << " " << pxEst.y() << endl;
    bool converge = mSVO::Alignment::align2D(image, pxEst, 5, patchSize, patch, patchWithBorder);
    cout << "after  Est: " << pxEst.x() << " " << pxEst.y() << endl;
    cout << "Converge :" << converge << "  all done!!" << endl;
    return 0;
}

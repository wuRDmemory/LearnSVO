#include "viewer.hpp"
#include "config.hpp"
#include <time.h>

namespace mSVO {
    Viewer::Viewer(MapPtr map): mMap(map) {
        mCameraSize      = Config::viewCameraSize();
        mCameraLineWidth = Config::viewCameraLineWidth();
        mPointSize       = Config::viewPointSize();
        LOG(INFO) << ">>> [Viewer] start viewer";
    }

    Viewer::~Viewer() {
        LOG(INFO) << ">>> [Viewer] stop viewer";
    }

    bool Viewer::addCurrentFrame(FramePtr curFrame) {
        {
            unique_lock<mutex> lock(mAddFrameMutex);
            mCurFrame  = curFrame;
        }
        return true;
    }

    bool Viewer::setup() {
        mStop = false;
        mThread = new thread(&Viewer::run, this);
        return true;
    }

    bool Viewer::showFrame(Frame* frame) {
        cv::Mat image = frame->imagePyr()[0];
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

        auto& obs = frame->obs();
        auto  it  = obs.begin();
        while (it != obs.end()) {
            FeaturePtr  ftr  = (*it);
            LandMarkPtr ldmk = ftr->mLandmark;
            Vector2f    px   = ftr->mPx;
            if (!ldmk || ldmk->type == LandMark::LANDMARK_TYPR::DELETE)
                continue;
            
            Scalar color;
            if (ldmk->type == LandMark::LANDMARK_TYPR::GOOD) {
                color = Scalar(0, 255, 0);
            } else {
                color = Scalar(0, 0, 255);
            }
            cv::circle(image, Point(px(0), px(1)), 3, color, 1);
            it++;
        }
        cv::imshow("Current Image", image);
        cv::waitKey(30);
        return true;
    } 

    void Viewer::run() {
        pangolin::CreateWindowAndBind("SVO: Map Viewer",1024,768);

        // 3D Mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);

        // Issue specific OpenGl we might need
        glEnable (GL_BLEND);
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam(
                    pangolin::ProjectionMatrix(1024, 768, Config::viewViewpointF(),Config::viewViewpointF(), 512, 389, 0.1, 1000),
                    pangolin::ModelViewLookAt(Config::viewViewpointX(), Config::viewViewpointY(), Config::viewViewpointZ(), 0, 0, 0, 0.0, -1.0, 0.0)
                    );

        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
                .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        while (!mStop) {
            // 
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            FramePtr frame = NULL;
            {
                unique_lock<mutex> lock(mAddFrameMutex);
                frame = mCurFrame;
            }

            if (!frame.get()) {
                continue;
            }

            convertMatrix(frame->Rwc(), frame->twc(), Twc);
            s_cam.Follow(Twc);
            d_cam.Activate(s_cam);
            glClearColor(1.0f,1.0f,1.0f,1.0f);
            
            drawFrame(Twc);
            drawPoint(frame->obs(), true);

            auto& keyFrames = mMap->keyFrames();
            for (auto iter = keyFrames.begin(), end = keyFrames.end(); iter != end; ++iter) {
                FramePtr& frame_ = *iter;
                if (frame_.get() == frame.get()) {
                    continue;
                }
                convertMatrix(frame_->Rwc(), frame_->twc(), Twc);
                drawFrame(Twc);
                drawPoint(frame_->obs(), false);
            }

            pangolin::FinishFrame();

            showFrame(frame.get());
        }
    }

    bool Viewer::convertMatrix(Quaternionf& Qwc, Vector3f& twc, pangolin::OpenGlMatrix& M) {
        Matrix3f Rwc = Qwc.toRotationMatrix();
        M.m[0] = Rwc(0,0);
        M.m[1] = Rwc(1,0);
        M.m[2] = Rwc(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc(0,1);
        M.m[5] = Rwc(1,1);
        M.m[6] = Rwc(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc(0,2);
        M.m[9] = Rwc(1,2);
        M.m[10] = Rwc(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc(0);
        M.m[13] = twc(1);
        M.m[14] = twc(2);
        M.m[15]  = 1.0;
        return true;
    }

    bool Viewer::drawFrame(pangolin::OpenGlMatrix &Twc) {
        const float &w = mCameraSize;
        const float h = w*0.75;
        const float z = w*0.6;

        glPushMatrix();
        glMultMatrixd(Twc.m);

        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f,1.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }

    bool Viewer::drawPoint(Features& features, bool active) {
        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        if (active) {
            glColor3f(1.0,0.0,0.0);
        } else {
            glColor3f(0.0,0.0,0.0);
        }
        
        for (auto iter = features.begin(); iter != features.end(); iter++) {
            FeaturePtr feature = *iter;
            LandMarkPtr landmark = feature->mLandmark;
            if (!landmark || landmark->type == LandMark::DELETE) {
                continue;
            }
            Vector3f& xyz = landmark->xyz();
            glVertex3f(xyz(0), xyz(1), xyz(2));
        }
        glEnd();
        return true;
    }
}

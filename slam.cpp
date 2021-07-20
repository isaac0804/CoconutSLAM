#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/calib3d/calib3d.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
#include <mutex>

#include <pangolin/pangolin.h>

using namespace std;
using namespace cv;

mutex mtx;

class CocoPoint 
{
public: 
  int id;
  vector<Point2f> coords;
  Point3d loc = Point3d(0,0,0);
  vector<int> frames_id;

  CocoPoint(int point_id, int frame_id, Point2f coord) {
    id = point_id;
    frames_id.push_back(frame_id);
    coords.push_back(coord);
  }
};

class Frame 
{
public: 
  int id;
  Mat image;
  vector<KeyPoint> kps; 
  Mat dps; 
  vector<int> pts_id;
  Mat pose;
  
  Frame(int frame_id, 
      Mat input_image,
      vector<KeyPoint> frame_kps,
      Mat frame_dps)
  {
    id = frame_id;
    image = input_image;
    kps = frame_kps;
    dps = frame_dps;
  }
};


class Extractor 
{
public: 
  int MAX_FEATURES = 8000;
  double scale = 0.5; // must be integers..... also this affects the time needed for each frame drastically. how can i improve?
  double thresh = 0.75;
  double dist_thresh = 0.1;

  Extractor (int MAX_FEATURES = 8000, 
            double scale = 0.5,
            double thresh = 0.75,
          double dist_thresh = 0.1) 
  {
    MAX_FEATURES = MAX_FEATURES;
    scale = scale;
    thresh = thresh;
    dist_thresh = dist_thresh;
  }

  void getFeatures(Mat image, vector<KeyPoint> &frame_kps, Mat &frame_dps) 
  {
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints; 
    Mat descriptors; 
    vector<Point2f> corners; 

    int frame_width = image.cols * scale;
    int frame_height = image.rows * scale;

    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);

    for(size_t y = 0; y < 1/scale; y++)
    {
      for(size_t x = 0; x < 1/scale; x++) 
      {
        vector<Point2f> curr_corners;
        Rect roi = Rect(x*frame_width, y*frame_height, frame_width, frame_height);
        Mat mask = Mat::zeros(image.size(), CV_8UC1);
        mask(roi).setTo(Scalar::all(255));
        goodFeaturesToTrack(gray, curr_corners, MAX_FEATURES*scale*scale, 0.1, 8, mask, 3, false, 0.04);
        corners.insert(corners.end(), curr_corners.begin(), curr_corners.end());
      }
    }
    
    cout << "Detected corners : " << corners.size() << "\n";
    // Find keypoints and Compute descriptors
    for(size_t i = 0; i < corners.size(); i++) 
    {
      keypoints.push_back(KeyPoint(corners[i], 20));
    }

    orb->compute(image, keypoints, descriptors);
    frame_kps = keypoints;
    frame_dps = descriptors;
  }

  void matchFeatures(vector<Frame> &frames, vector<CocoPoint> &points) 
  {
    Frame &prev_frame = frames[frames.size()-2];
    Frame &curr_frame = frames[frames.size()-1];
    
    // Match keypoints based on descriptors
    BFMatcher matcher;
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(curr_frame.dps, prev_frame.dps, knn_matches, 2);

    // Filter invalid matches 
    vector<DMatch> good_matches;
    for(size_t i = 0; i < knn_matches.size(); i++) 
    {
      if(knn_matches[i][0].distance < thresh * knn_matches[i][1].distance) 
      {
        if(sqrt((curr_frame.kps[knn_matches[i][0].queryIdx].pt.x - prev_frame.kps[knn_matches[i][0].trainIdx].pt.x)*(curr_frame.kps[knn_matches[i][0].queryIdx].pt.x - prev_frame.kps[knn_matches[i][0].trainIdx].pt.x) + (curr_frame.kps[knn_matches[i][0].queryIdx].pt.y - prev_frame.kps[knn_matches[i][0].trainIdx].pt.y)*(curr_frame.kps[knn_matches[i][0].queryIdx].pt.y - prev_frame.kps[knn_matches[i][0].trainIdx].pt.y)) < dist_thresh*sqrt(curr_frame.image.cols*curr_frame.image.cols+curr_frame.image.rows*curr_frame.image.rows)) 
        {
          good_matches.push_back(knn_matches[i][0]);
        }
      }
    }
    
    cout << "Good Matches     : " << good_matches.size() << "\n";
    // Put good matches into each frame 
    int addNew = 0;
    int addOld = 0;
    for(size_t i = 0; i < good_matches.size(); i++) 
    {
      if(prev_frame.pts_id[good_matches[i].trainIdx] == -1) 
      {
        addNew++;
        CocoPoint point = CocoPoint(points.size(), prev_frame.id, prev_frame.kps[good_matches[i].trainIdx].pt);
        points.push_back(point);
        points[points.size()-1].frames_id.push_back(curr_frame.id);
        points[points.size()-1].coords.push_back(curr_frame.kps[good_matches[i].queryIdx].pt);
        //temp_pts_id.push_back(points.size()-1);
        curr_frame.pts_id[i] = (points.size()-1);
      } 
      else 
      {
        addOld++;
        points[prev_frame.pts_id[good_matches[i].trainIdx]].frames_id.push_back(curr_frame.id);
        points[prev_frame.pts_id[good_matches[i].trainIdx]].coords.push_back(curr_frame.kps[good_matches[i].queryIdx].pt);
        //temp_pts_id.push_back(prev_frame.pts_id[good_matches[i].trainIdx]);
        curr_frame.pts_id[i] = (prev_frame.pts_id[good_matches[i].trainIdx]);
      }
    }

    cout << "New Points       : " << addNew << "\n";
    cout << "Old Points       : " << addOld << "\n";
  }
};


class Map
{
public:
  vector<Frame> frames;
  vector<CocoPoint> points;  
  Extractor extractor = Extractor();
  Mat K = (Mat_<double>(3, 3) << 9.842439e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00, 9.808141e+02, 2.331966e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00);
  Mat dist = (Mat_<double>(1, 5) << -3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02);

  void initFirstFrame(Mat &first_frame) 
  {
    vector<KeyPoint> keypoints; 
    Mat descriptors;
    Mat pose = Mat::eye(4, 4, CV_64F);
    
    // Find and draw features
    extractor.getFeatures(first_frame, keypoints, descriptors);

    // Put first frame into frames
    Frame frame = Frame(frames.size(), first_frame, keypoints, descriptors);
    frame.pose = pose;
    frames.push_back(frame);

    for(size_t i = 0; i < frames[0].kps.size(); i++) 
    {
      CocoPoint point  = CocoPoint(points.size(), 0, frames[0].kps[i].pt);
      points.push_back(point);
      frames[0].pts_id.push_back(i);
    }
  }
  
  void extractAndMatch(Mat &curr_frame)
  {
    vector<KeyPoint> keypoints; 
    Mat descriptors; 

    // Find the features evenly by doing the grid
    extractor.getFeatures(curr_frame, keypoints, descriptors);

    // Put current frame into frames
    Frame frame = Frame(frames.size(), curr_frame, keypoints, descriptors);
    frames.push_back(frame);

    for(size_t i = 0; i < frames[frames.size()-1].kps.size(); i++) 
    {
      frames[frames.size()-1].pts_id.push_back(-1);
    }
    // Match features with previous frame and directly add into Map 
    extractor.matchFeatures(frames, points); 
  }

  void estimatePose() 
  {

    vector<Point2f> curr_points;
    vector<Point2f> prev_points;
    for(size_t i = 0; i < frames[frames.size()-1].pts_id.size(); i++)
    {
      if(frames[frames.size()-1].pts_id[i] != -1)
      {
        for(size_t j = 0; j < points[frames[frames.size()-1].pts_id[i]].frames_id.size(); j++)
        {
          if(points[frames[frames.size()-1].pts_id[i]].frames_id[j] == frames.size()-2) 
          {
            curr_points.push_back(points[frames[frames.size()-1].pts_id[i]].coords[points[frames[frames.size()-1].pts_id[i]].coords.size()-1]);
            prev_points.push_back(points[frames[frames.size()-1].pts_id[i]].coords[points[frames[frames.size()-1].pts_id[i]].coords.size()-2]);
          }
        }
      }
    }

    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(curr_points, prev_points, CV_FM_8POINT);

    Point2d principal_point(K.at<double>(0, 2), K.at<double>(1, 2));
    double focal_length = K.at<double>(1, 1);
    Mat essential_matrix;

    essential_matrix = findEssentialMat(curr_points, prev_points, focal_length, principal_point);

    Mat R;
    Mat t; 
    recoverPose(essential_matrix, curr_points, prev_points, R, t, focal_length, principal_point);
    
    Mat T;
    Mat btm = (Mat_<double>(1, 4) << 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00);
    hconcat(R, t, T);
    vconcat(T, btm, T);
    T.convertTo(T, CV_64F);

    frames[frames.size()-1].pose = T*frames[frames.size()-2].pose;
  }

  void displayVideo() 
  {

    Mat image; 
    image = frames[frames.size()-1].image.clone();
    for(auto keypoint : frames[frames.size()-1].kps) 
    {
      circle(image, keypoint.pt, 4, Scalar(0, 255, 0));  
    }

    imshow("Videos", image);  
    cout << "Current Pose     : \n" << frames[frames.size()-1].pose.size() << "\n";
    cout << "Total Frames     : " << frames.size() << "\n";
    cout << "Total Points     : " << points.size() << "\n";
  }

  void triangulate() 
  {
    Frame prev_frame = frames[frames.size()-2];
    Frame curr_frame = frames[frames.size()-1];
      
    Mat T1 = prev_frame.pose.rowRange(0, 3).clone();
    // Mat T1 = (Mat_<double>(3,4) <<
    //   1, 0, 0, 0,
    //   0, 1, 0, 0,
    //   0, 0, 1, 0);
    Mat T2 = curr_frame.pose.rowRange(0, 3).clone();

    vector<Point2f> pts_1, pts_2;
    for(size_t i = 0; i < curr_frame.kps.size(); i++) 
    {
      if(curr_frame.pts_id[i] != -1  && points[curr_frame.pts_id[i]].loc == Point3d(0,0,0))
      {
        pts_1.push_back(points[curr_frame.pts_id[i]].coords[points[curr_frame.pts_id[i]].coords.size()-2]);
        pts_2.push_back(points[curr_frame.pts_id[i]].coords[points[curr_frame.pts_id[i]].coords.size()-1]);
      }
    }
    undistortPoints(pts_1, pts_1, K, dist);
    undistortPoints(pts_2, pts_2, K, dist);

    Mat pts_4d;
    triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for(size_t i = 0; i < pts_4d.cols; i++)
    {
      Mat temp = pts_4d.col(i);
      temp = temp / temp.at<float>(3,0);
      Point3d p(
          temp.at<float>(0,0),
          temp.at<float>(1,0),
          temp.at<float>(2,0)
      );
      points[curr_frame.pts_id[i]].loc = p;
    }
  }
};

void display3D(Map &world) 
{
  pangolin::BindToContext("3d View");
  glEnable(GL_DEPTH_TEST);

  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024,768,420,420,512,389,0.1,1000),
    pangolin::ModelViewLookAt(0, 10, -10, 0,0,0, 0.0,1.0,0.0)
  );

  // Create Interactive View in window
  pangolin::Handler3D handler(s_cam);
  pangolin::View& d_cam = pangolin::CreateDisplay()
          .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f/768.0f)
          .SetHandler(&handler);
  
  //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  //d_cam.Activate(s_cam);
  while( !pangolin::ShouldQuit() )
  {
    // Lock the data while in use 
    mtx.lock();

    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    //s_cam.Follow(Twc);
   
    double w=1.0;
    double h_ratio=0.75;
    double z_ratio=0.6;
    double h = w * h_ratio;
    double z = w * z_ratio;

    // Draw camera movements  
    for (size_t i = 0; i < world.frames.size(); i++)
    {
      Mat pose; 
      if(!world.frames[i].pose.empty())
      {
        pose = world.frames[i].pose.clone();
      }
      else
      {
        break;
      }
      pangolin::OpenGlMatrix Twc;
      Twc.SetIdentity();
      
      Mat Rwc(3, 3, CV_64F);
      Mat twc(3, 1, CV_64F);
      
      Rwc = pose.rowRange(0, 3).colRange(0, 3).t();
      //twc = -Rwc*pose.rowRange(0, 3).col(3);
      twc = Rwc*pose.rowRange(0, 3).col(3);

      Twc.m[0] = Rwc.at<double>(0,0);
      Twc.m[1] = Rwc.at<double>(1,0);
      Twc.m[2] = Rwc.at<double>(2,0);
      Twc.m[3]  = 0.0;

      Twc.m[4] = Rwc.at<double>(0,1);
      Twc.m[5] = Rwc.at<double>(1,1);
      Twc.m[6] = Rwc.at<double>(2,1);
      Twc.m[7]  = 0.0;

      Twc.m[8] = Rwc.at<double>(0,2);
      Twc.m[9] = Rwc.at<double>(1,2);
      Twc.m[10] = Rwc.at<double>(2,2);
      Twc.m[11]  = 0.0;

      Twc.m[12] = twc.at<double>(0);
      Twc.m[13] = twc.at<double>(1);
      Twc.m[14] = twc.at<double>(2);
      Twc.m[15]  = 1.0;

      glPushMatrix();
      glMultMatrixd(Twc.m);

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
    for (size_t i = 0; i < world.points.size(); i++)
    {
      if(world.points[i].loc == Point3d(0,0,0))
      {
        continue;
      }
      pangolin::OpenGlMatrix Twc;
      Twc.SetIdentity();
    
      glPushMatrix();
      glMultMatrixd(Twc.m);

      glPointSize(10);
      glBegin(GL_POINTS);
      glColor3f(1.0,0.0,0.0);
      glVertex3d(world.points[i].loc.x, world.points[i].loc.y, world.points[i].loc.z);
      glEnd();

      glPopMatrix();
      cout << "Drawing point " << i << endl;
      cout << world.points[i].loc.x << "\n" << world.points[i].loc.y << "\n" << world.points[i].loc.z << endl;
    }

  // Render OpenGL Cube
  pangolin::glDrawColouredCube();
  pangolin::glDrawAxis(3);

  // Swap frames and Process Events
  pangolin::FinishFrame();

  // Unlock the data after use
  mtx.unlock();
  }

  pangolin::GetBoundWindow()->RemoveCurrent();
}


int main(int argc, char **argv) 
{
  // Check input
  if(argv[1] == nullptr) 
  {
    cerr << "./slam <Video.mp4>\n";
    return -1;
  }

  // Check video type
  VideoCapture cap(argv[1]);
  if(!cap.isOpened())
  {
    cerr << "Error opening video stream or file.\n";
    return -1;
  }

  Mat K = (Mat_<double>(3, 3) << 9.842439e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00, 9.808141e+02, 2.331966e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00);
  Mat dist = (Mat_<double>(1, 5) << -3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02);
  Map World = Map(); 
  
  pangolin::CreateWindowAndBind("3d View", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  pangolin::GetBoundWindow()->RemoveCurrent();

  thread render(display3D, ref(World));
    
  while(1) 
  {
    // Load frame from video
    Mat frame;
    cap >> frame;
    if(frame.empty()) break;
    cout << "******************Frame " << cap.get(CAP_PROP_POS_FRAMES) << "*****************\n";

    // Process image 
    chrono::steady_clock::time_point start = chrono::steady_clock::now();


    Mat image; 
    undistort(frame, image, K, dist);
    if(cap.get(CAP_PROP_POS_FRAMES) > 1) 
    {
      World.extractAndMatch(image);
      World.estimatePose();
      World.triangulate();
    }
    else World.initFirstFrame(image);
      
    // Display
    World.displayVideo();

    char c = (char)waitKey(1000/cap.get(CAP_PROP_FPS)); // display according to original fps
    if(c==27) break; // Esc 

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> interval = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "Time used        : " << interval.count() << "\n";
  }
  render.join();

  cap.release();
  destroyAllWindows();
  return 0; 
}

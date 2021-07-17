#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;


class CocoPoint 
{
public: 
  int id;
  vector<double> coord;
  vector<int> frames_id;
  vector<int> idxs; // indexes of Point in frame.pts

  CocoPoint(int point_id, int frame_id, int index) {
    id = point_id;
    frames_id.push_back(frame_id);
    idxs.push_back(index);
  }
};

class Frame 
{
public: 
  int id;
  Mat image;
  vector<KeyPoint> kps; 
  Mat dps; 
  vector<CocoPoint> pts;
  vector<int> kps_idxs;
  
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
  int MAX_FEATURES = 4000;
  double scale = 1.0; // must be integers..... also this affects the time needed for each frame drastically. how can i improve?
  double thresh = 0.75;
  double dist_thresh = 0.1;

  Extractor (int MAX_FEATURES = 4000, 
            double scale = 1.0,
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

        cout << "Detected corners : " << curr_corners.size() << endl;
      }
    }
    
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
    Frame prev_frame = frames[frames.size()-2];
    Frame curr_frame = frames[frames.size()-1];
    
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
    
    cout << "Good Matches     : " << good_matches.size() << endl;
    // Put good matches into each frame 
    int addNew = 0;
    int addOld = 0;
    for(size_t i = 0; i < good_matches.size(); i++) 
    {
      if(prev_frame.kps_idxs[good_matches[i].trainIdx] == -1) 
      {
        addNew++;
        CocoPoint point = CocoPoint(points.size(), prev_frame.id, good_matches[i].trainIdx);
        points.push_back(point);

        // put the index of the point in frame.pts into frame.kps_idxs and point.idxs
        frames[frames.size()-2].kps_idxs[good_matches[i].trainIdx] = frames[frames.size()-2].pts.size();
        frames[frames.size()-1].kps_idxs[good_matches[i].queryIdx] = frames[frames.size()-1].pts.size();
        points[points.size()-1].idxs.push_back(frames[frames.size()-2].pts.size()-1);
        points[points.size()-1].idxs.push_back(frames[frames.size()-1].pts.size()-1);

        // append the point into the frame.pts
        frames[frames.size()-2].pts.push_back(point);
        frames[frames.size()-1].pts.push_back(point);

        // put the frame id into the point.frames_id
        points[points.size()-1].frames_id.push_back(frames[frames.size()-2].id);
        points[points.size()-1].frames_id.push_back(frames[frames.size()-1].id);
      } 
      else 
      {
        addOld++;
        curr_frame.kps_idxs[good_matches[i].queryIdx] = curr_frame.pts.size();
        curr_frame.pts.push_back(points[prev_frame.pts[prev_frame.kps_idxs[good_matches[i].trainIdx]].id]);
      }
    }
    cout << "New Points       : " << addNew << endl;
    cout << "Old Points       : " << addOld << endl;
  }
};


class Map
{
public:
  vector<Frame> frames;
  vector<CocoPoint> points;  
  Extractor extractor = Extractor();
  Mat K = (Mat_<double>(3, 3) << 9.842439e+02, 0.000000e+00, 6.900000e+02, 0.000000e+00, 9.808141e+02, 2.331966e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00);

  void initFirstFrame(Mat &first_frame) 
  {
    vector<KeyPoint> keypoints; 
    Mat descriptors;
    
    // Find and draw features
    extractor.getFeatures(first_frame, keypoints, descriptors);

    // Put first frame into frames
    Frame frame = Frame(frames.size(), first_frame, keypoints, descriptors);
    frames.push_back(frame);

    for(size_t i = 0; i < frames[frames.size()-1].kps.size(); i++) 
    {
      CocoPoint point  = CocoPoint(points.size(), frames.size(), i);
      frames[frames.size()-1].kps_idxs.push_back(i);
      frames[frames.size()-1].pts.push_back(point);
      points.push_back(point);
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
      frames[frames.size()-1].kps_idxs.push_back(-1);
    }
    // Match features with previous frame and directly add into Map 
    extractor.matchFeatures(frames, points); 
  }

  void display() 
  {
    Mat image; 
    image = frames[frames.size()-1].image.clone();
    for(auto keypoint : frames[frames.size()-1].kps) 
    {
      circle(image, keypoint.pt, 4, Scalar(0, 255, 0));  
    }

    imshow("Videos", image);  
    cout << "Total Frames     : " << frames.size() << endl;
    cout << "Total Points     : " << points.size() << endl;
  }
};




int main(int argc, char **argv) 
{

  // Check input
  if(argv[1] == nullptr) 
  {
    cerr << "./slam <Video.mp4>" << endl;
    return -1;
  }

  // Check video type
  VideoCapture cap(argv[1]);
  if(!cap.isOpened())
  {
    cerr << "Error opening video stream or file." << endl;
    return -1;
  }

  Map World = Map(); 

  while(1) 
  {
    // Load frame from video
    Mat frame;
    cap >> frame;
    if(frame.empty()) break;
    cout << "******************Frame " << cap.get(CAP_PROP_POS_FRAMES) << "*****************" << endl;

    // Feature Extraction using ORB
    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    if(cap.get(CAP_PROP_POS_FRAMES) > 1) World.extractAndMatch(frame);
    else World.initFirstFrame(frame);
      
    // Display
    World.display();
    char c = (char)waitKey(1000/cap.get(CAP_PROP_FPS)); // display according to original fps
    if(c==27) break; // Esc 

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> interval = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "Time used        : " << interval.count() << endl;
  }

  cap.release();
  destroyAllWindows();
  return 0; 
}


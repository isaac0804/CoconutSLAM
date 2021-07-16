#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;


class Frame 
{
  public: 
    int id;
    Mat image;
    vector<KeyPoint> kps; 
    Mat dps; 
    vector<int> kps_idxs;
    //vector<CocoPoint> pts;
    
  Frame(int frame_id, 
      Mat input_image,
      vector<KeyPoint> frame_kps,
      Mat frame_dps)
  {
    id = frame_id;
    image = input_image;
    kps = frame_kps;
    dps = frame_dps;
    vector<int> kps_idxs(kps.size());
    fill(kps_idxs.begin(), kps_idxs.end(), -1);
  }

  //void addPoints(CocoPoint &point, int kps_index, int pts_index) {
  //  pts.push_back(point); 
  //  kps_idxs[kps_index] = pts_index; // pts_index is index of pts or pts id
  void addPoints(int point_id, int kps_index) {
    kps_idxs[kps_index] = point_id; // pts_index is index of pts or pts id
  }
};

class CocoPoint 
{
  public: 
    int id;
    vector<double> coord;
    vector<Frame> frames;
    vector<int> idxs; // indexes of Point in frame.pts
  
  CocoPoint(int point_id, Frame point_in_frames, int index) {
    id = point_id;
    frames.push_back(point_in_frames);
    idxs.push_back(index); 
  }
  
  void addObservation(Frame frame, int index) {
    frames.push_back(frame);
    idxs.push_back(index);
  }
};

class Extractor 
{
  public: 
    int MAX_FEATURES;
    double scale;
    double thresh;
    double dist_thresh;

  Extractor (int MAX_FEATURES = 8000, 
            double scale = 0.25, // must be integers..... also this affects the time needed for each frame drastically. how can i improve?
            double thresh = 0.75,
            double dist_thresh = 0.1) {
    MAX_FEATURES = MAX_FEATURES;
    scale = scale;
    thresh = thresh;
    dist_thresh = dist_thresh;
  }

  void getFeatures(Mat image, vector<KeyPoint> &frame_kps, Mat &frame_dps) {
    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints; 
    Mat descriptors; 
    vector<Point2f> corners; 

    int frame_width = image.cols * scale;
    int frame_height = image.rows * scale;

    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);

    for(size_t y = 0; y < 1/scale; y++)
    for(size_t x = 0; x < 1/scale; x++) {
      vector<Point2f> curr_corners;
      Rect roi = Rect(x*frame_width, y*frame_height, frame_width, frame_height);
      Mat mask = Mat::zeros(image.size(), CV_8UC1);
      mask(roi).setTo(Scalar::all(255));
      goodFeaturesToTrack(gray, curr_corners, MAX_FEATURES*scale*scale, 0.1, 8, mask, 3, false, 0.04);
      corners.insert(corners.end(), curr_corners.begin(), curr_corners.end());
    }
    
    // Find keypoints and Compute descriptors
    for(size_t i = 0; i < corners.size(); i++) {
      keypoints.push_back(KeyPoint(corners[i], 20));
    }

    orb->compute(image, keypoints, descriptors);
    frame_kps = keypoints;
    frame_dps = descriptors;
  }

  void matchFeatures(vector<Frame> &frames, vector<CocoPoint> &points) {
    Frame prev_frame = frames[frames.size()-2];
    Frame curr_frame = frames[frames.size()-1];
    
    // Match keypoints based on descriptors
    BFMatcher matcher;
    vector<vector<DMatch>> knn_matches;
    matcher.knnMatch(curr_frame.dps, prev_frame.dps, knn_matches, 2);

    // Filter invalid matches 
    vector<DMatch> good_matches;
    for(size_t i = 0; i < knn_matches.size(); i++) {
      if(knn_matches[i][0].distance < thresh * knn_matches[i][1].distance) {
        if(sqrt((curr_frame.kps[knn_matches[i][0].queryIdx].pt.x - prev_frame.kps[knn_matches[i][0].trainIdx].pt.x)*(curr_frame.kps[knn_matches[i][0].queryIdx].pt.x - prev_frame.kps[knn_matches[i][0].trainIdx].pt.x) + (curr_frame.kps[knn_matches[i][0].queryIdx].pt.y - prev_frame.kps[knn_matches[i][0].trainIdx].pt.y)*(curr_frame.kps[knn_matches[i][0].queryIdx].pt.y - prev_frame.kps[knn_matches[i][0].trainIdx].pt.y)) < dist_thresh*sqrt(curr_frame.image.cols*curr_frame.image.cols+curr_frame.image.rows*curr_frame.image.rows)) {
          good_matches.push_back(knn_matches[i][0]);
        }
      }
    }
     
    // Put good matches into each frame 
    for(size_t i = 0; i < good_matches.size(); i++) {
      if(prev_frame.kps_idxs[good_matches[i].trainIdx] == -1) {
        points.push_back(CocoPoint(points.size(), prev_frame, good_matches[i].trainIdx));
        points[points.size()-1].addObservation(curr_frame, good_matches[i].queryIdx);
        prev_frame.addPoints(points.size()-1, good_matches[i].trainIdx);
        curr_frame.addPoints(points.size()-1, good_matches[i].queryIdx);
      } else {
        points[prev_frame.kps_idxs[good_matches[i].trainIdx]].addObservation(curr_frame, good_matches[i].queryIdx);
      }
    }
  }
};


class Map {
  public:
    vector<Frame> frames;
    vector<CocoPoint> points;  
    Extractor extractor = Extractor();

  Map() {
    Extractor extractor = Extractor();
    vector<Frame> frames;
    vector<Point> points;  
  }

  void initFirstFrame(Mat first_frame) {
    vector<KeyPoint> keypoints; 
    Mat descriptors; 
    
    // Find and draw features
    extractor.getFeatures(first_frame, keypoints, descriptors);

    // Put first frame into frames
    frames.push_back(Frame(frames.size(), first_frame, keypoints, descriptors));
  }
  
  void extractAndMatch(Mat curr_frame) {
    vector<KeyPoint> keypoints; 
    Mat descriptors; 

    // Find the features evenly by doing the grid
    extractor.getFeatures(curr_frame, keypoints, descriptors);

    // Put current frame into frames
    frames.push_back(Frame(frames.size(), curr_frame, keypoints, descriptors));

    // Match features with previous frame and directly add into Map 
    extractor.matchFeatures(frames, points); 
  }

  void display() {
    Mat image; 
    image = frames[frames.size()-1].image.clone();
    for(auto keypoint : frames[frames.size()-1].kps) {
      circle(image, keypoint.pt, 4, Scalar(0, 255, 0));  
    }

    imshow("Videos", image); 
    cout << "Total Frames: " << frames.size();
    cout << "Total Points: " << points.size();
  }
};




int main(int argc, char **argv) {

  // Check input
  if(argv[1] == nullptr) {
    cerr << "./slam <Video.mp4>" << endl;
    return -1;
  }

  // Check video type
  VideoCapture cap(argv[1]);
  if(!cap.isOpened()){
    cerr << "Error opening video stream or file." << endl;
    return -1;
  }

  Map World = Map(); 

  while(1) {
    // Load frame from video
    Mat frame;
    cap >> frame;
    if(frame.empty()) break;

    // Feature Extraction using ORB
    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    if(cap.get(CAP_PROP_POS_FRAMES) > 1) {
      World.extractAndMatch(frame);  
    } else World.initFirstFrame(frame);

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> interval = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "Time used         : " << interval.count() << endl;
      
    // Display
    World.display();
    char c = (char)waitKey(1000/cap.get(CAP_PROP_FPS)); // display according to original fps
    if(c==27) break; // Esc 
  }

  cap.release();
  destroyAllWindows();
  return 0; 
}


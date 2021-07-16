#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;


class Frame {
  public: 
    int index;
    Mat image;
    vector<KeyPoint> kps; 
    Mat dps; 
    
    Frame(int frame_index, Mat input_image) {
      index = frame_index;
      image = input_image;
    }
};

void extractAndMatch(Frame &curr_frame, Frame &prev_frame) {
  int MAX_FEATURES = 3000;
  float scale = 0.25; // must be integers..... also this affects the time needed for each frame drastically. how can i improve?
  int frame_width = curr_frame.image.cols * scale;
  int frame_height = curr_frame.image.rows * scale;

  Mat gray;
  cvtColor(curr_frame.image, gray, CV_BGR2GRAY);

  vector<KeyPoint> keypoints; 
  Mat descriptors; 
  vector<Point2f> corners; 
  Ptr<ORB> orb = ORB::create();

  // Find the features evenly by doing the grid
  for(size_t y = 0; y < 1/scale; y++)
  for(size_t x = 0; x < 1/scale; x++) {
    vector<Point2f> curr_corners; 
    Rect roi = Rect(x*frame_width, y*frame_height, frame_width, frame_height);
    Mat mask = Mat::zeros(curr_frame.image.size(), CV_8UC1);
    mask(roi).setTo(Scalar::all(255));
    goodFeaturesToTrack(gray, curr_corners, MAX_FEATURES*scale*scale, 0.1, 8, mask, 3, false, 0.04);
    corners.insert(corners.end(), curr_corners.begin(), curr_corners.end());
  }

  // Find keypoints and Compute descriptors
  for(size_t i = 0; i < corners.size(); i++) {
    keypoints.push_back(KeyPoint(corners[i], 20));
  }
  orb->compute(curr_frame.image, keypoints, descriptors);
  curr_frame.kps = keypoints;
  curr_frame.dps = descriptors;

  // Match keypoints based on descriptors
  BFMatcher matcher;
  vector<vector<DMatch>> knn_matches;
  matcher.knnMatch(descriptors, prev_frame.dps, knn_matches, 2);

  // Filter invalid matches 
  const double thresh = 0.7;
  vector<DMatch> good_matches;
  for(size_t i = 0; i < knn_matches.size(); i++) {
    if(knn_matches[i][0].distance < thresh * knn_matches[i][1].distance) {
        good_matches.push_back(knn_matches[i][0]);
    }
  }

  // Draw matches 
  for(size_t i = 0; i < good_matches.size(); i++) {
    circle(curr_frame.image, prev_frame.kps[good_matches[i].trainIdx].pt, 4, Scalar(255, 0, 0));
    line(curr_frame.image, prev_frame.kps[good_matches[i].trainIdx].pt, keypoints[good_matches[i].queryIdx].pt, Scalar(255, 0, 0));
    circle(curr_frame.image, keypoints[good_matches[i].queryIdx].pt, 4, Scalar(0, 255, 0));
  }
 
  cout << "Number of features: " << corners.size() << " -> " << good_matches.size() << endl;
}

void initFirstFrame(Frame &first_frame) {
  int MAX_FEATURES = 3000;
  float scale = 0.25; // must be integers..... 
  int frame_width = first_frame.image.cols * scale;
  int frame_height = first_frame.image.rows * scale;

  Mat gray;
  cvtColor(first_frame.image, gray, CV_BGR2GRAY);
  vector<KeyPoint> keypoints; 
  Mat descriptors; 
  vector<Point2f> corners; 
  Ptr<ORB> orb = ORB::create();

  // Find and draw features
  for(size_t y = 0; y < 1/scale; y++)
  for(size_t x = 0; x < 1/scale; x++) {
    vector<Point2f> curr_corners; 
    Rect roi = Rect(x*frame_width, y*frame_height, frame_width, frame_height);
    Mat mask = Mat::zeros(first_frame.image.size(), CV_8UC1);
    mask(roi).setTo(Scalar::all(255));
    goodFeaturesToTrack(gray, curr_corners, MAX_FEATURES*scale*scale, 0.1, 8, mask, 3, false, 0.04);
    corners.insert(corners.end(), curr_corners.begin(), curr_corners.end());
  }

  // Find keypoints and Compute descriptors
  for(size_t i = 0; i < corners.size(); i++) {
    keypoints.push_back(KeyPoint(corners[i], 20));
  }
  orb->compute(first_frame.image, keypoints, descriptors);
  first_frame.kps = keypoints;
  first_frame.dps = descriptors;

  // Draw points
  for(size_t i = 0; i < corners.size(); i++) {
    circle(first_frame.image, corners[i], 4, Scalar(0, 255, 0));
  }
}

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

  vector<Frame> frames;
  while(1) {
    // Load frame from video
    Mat frame;
    cap >> frame;
    if(frame.empty()) break;
    frames.push_back(Frame(cap.get(CAP_PROP_POS_FRAMES)-1, frame));

    // Feature Extraction using ORB
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    if(cap.get(CAP_PROP_POS_FRAMES) > 1) {
      extractAndMatch(frames[cap.get(CAP_PROP_POS_FRAMES)-1], frames[cap.get(CAP_PROP_POS_FRAMES)-2]);  
    } else initFirstFrame(frames[cap.get(CAP_PROP_POS_FRAMES)-1]);
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> interval = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "Time used         : " << interval.count() << endl;
      
    // Display
    resize(frames[cap.get(CAP_PROP_POS_FRAMES)-1].image, frame, Size(frame.cols/2, frame.rows/2));
    imshow("Video", frame);
    char c = (char)waitKey(1000/cap.get(CAP_PROP_FPS)); // display according to original fps
    if(c==27) break; // Esc 
  }

  cap.release();
  destroyAllWindows();
  return 0; 
}


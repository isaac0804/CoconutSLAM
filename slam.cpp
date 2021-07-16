#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/features2d/features2d.hpp>

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
  int MAX_FEATURES = 4000;
  float scale = 0.2; // must be integers..... 
  int frame_width = curr_frame.image.cols * scale;
  int frame_height = curr_frame.image.rows * scale;

  vector<KeyPoint> keypoints; 
  Mat descriptors; 
  Ptr<ORB> orb = ORB::create(MAX_FEATURES);

  // Find the features evenly by doing the grid
  //for(size_t y = 0; y < 1/scale; y++)
  //for(size_t x = 0; x < 1/scale; x++) {
  //  vector<KeyPoint> curr_keyPoints; 
  //  Rect roi = Rect(x*frame_width, y*frame_height, frame_width, frame_height);
  //  Mat mask = Mat::zeros(curr_frame.image.size(), CV_8UC1);
  //  mask(roi).setTo(Scalar::all(255));
  //  orb->detect(curr_frame.image, curr_keyPoints, mask);
  //  for(auto curr_keypoint : curr_keyPoints) {
  //    curr_keypoint.pt = curr_keypoint.pt + Point2f(x*frame_width, y*frame_height);
  //    keyPoints.push_back(curr_keypoint);
  //  }
  //}

  // Find keypoints and Compute descriptors
  orb->detectAndCompute(curr_frame.image, noArray(), keypoints, descriptors, false);
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
  
  cout << "Number of features: " << keypoints.size() << " -> " << good_matches.size() << endl;
}

void initFirstFrame(Frame &first_frame) {
  int MAX_FEATURES = 4000;
  float scale = 0.1; // must be integers..... 
  int frame_width = first_frame.image.cols * scale;
  int frame_height = first_frame.image.rows * scale;

  vector<KeyPoint> keyPoints; 
  Mat descriptors; 
  Ptr<ORB> orb = ORB::create(MAX_FEATURES*scale*scale);

  // Find and draw features
  for(size_t y = 0; y < 1/scale; y++)
  for(size_t x = 0; x < 1/scale; x++) {
    vector<KeyPoint> curr_keyPoints; 
    Rect roi = Rect(x*frame_width, y*frame_height, frame_width, frame_height);
    orb->detect(first_frame.image(roi), curr_keyPoints);

    for(auto curr_keypoint : curr_keyPoints) {
      curr_keypoint.pt = curr_keypoint.pt + Point2f(x*frame_width, y*frame_height);
      circle(first_frame.image, curr_keypoint.pt, 4, Scalar(0, 255, 0));
    }
    keyPoints.insert(keyPoints.end(), curr_keyPoints.begin(), curr_keyPoints.end()); 
  }
  first_frame.kps = keyPoints;
  orb->compute(first_frame.image, keyPoints, descriptors);
  first_frame.dps = descriptors;
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


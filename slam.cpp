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
    vector<Point2f> uv; 
    
    Frame(int frame_index, Mat input_image) {
      index = frame_index;
      image = input_image;
    }
};

void extractAndMatch(Frame &curr_frame, Frame &prev_frame) {
  int MAX_FEATURES = 4000;
  float scale = 0.1; // must be integers..... 
  int frame_width = curr_frame.image.cols * scale;
  int frame_height = curr_frame.image.rows * scale;

  vector<Point2f> coords; 
  vector<KeyPoint> keyPoints; 
  Mat descriptors; 
  Ptr<ORB> orb = ORB::create(MAX_FEATURES*scale*scale);

  // Find and draw features
  for(size_t y = 0; y < 1/scale; y++)
  for(size_t x = 0; x < 1/scale; x++) {
    vector<KeyPoint> curr_keyPoints; 
    Mat curr_descriptors; 
    Rect roi = Rect(x*frame_width, y*frame_height, frame_width, frame_height);
    orb->detect(curr_frame.image(roi), curr_keyPoints);

    for(auto keypoint : curr_keyPoints) {
      keypoint.pt = keypoint.pt + Point2f(x*frame_width, y*frame_height);
      coords.push_back(keypoint.pt); 
      circle(curr_frame.image, keypoint.pt, 4, Scalar(0, 255, 0));
    }

    for(auto uv : prev_frame.uv) {
      circle(curr_frame.image, uv, 4, Scalar(255, 0, 0));
    }
    keyPoints.insert(keyPoints.end(), curr_keyPoints.begin(), curr_keyPoints.end()); 
  }
  curr_frame.uv = coords; 

  // Compute descriptors
  // orb->compute(curr_frame.image, coords, 
  
  cout << "Number of features: " << coords.size() << endl;
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
    extractAndMatch(frames[cap.get(CAP_PROP_POS_FRAMES)-1], frames[cap.get(CAP_PROP_POS_FRAMES)-2]);  
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


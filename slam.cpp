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
    
    Frame(int frame_index, Mat input_image) {
      index = frame_index;
      image = input_image;
      cout << "Register Frame " << index << endl;
    }
};

void extractAndMatch(Mat &image) {
  int MAX_FEATURES = 1000;
  vector<KeyPoint> keyPoints; 
  Mat descriptors; 
  Ptr<ORB> orb = ORB::create(MAX_FEATURES);
  orb->detect(image, keyPoints);
  for(auto keypoint : keyPoints) {
    circle(image, keypoint.pt, 4, Scalar(0, 255, 0));
  }
  cout << "Number of features: " << keyPoints.size() << endl;
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
    frames.push_back(Frame(cap.get(1)-1, frame));

    // Feature Extraction using ORB
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    extractAndMatch(frame);  
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    chrono::duration<double> interval = chrono::duration_cast<chrono::duration<double>>(end - start);
    cout << "Time used         : " << interval.count() << endl;
      
    // Display
    resize(frames[cap.get(1)-1].image, frame, Size(frame.cols/2, frame.rows/2));
    imshow("Video", frame);
    char c = (char)waitKey(1000/cap.get(5)); // display according to original fps
    if(c==27) break; // Esc 
  }

  cap.release();
  destroyAllWindows();
  return 0; 
}


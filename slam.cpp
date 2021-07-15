#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp> 
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;


void extract(Mat &image) {
  int MAX_FEATURES = 50000;
  vector<KeyPoint> keyPoints; 
  Mat descriptors; 
  Ptr<ORB> orb = ORB::create(MAX_FEATURES);
  orb->detect(image, keyPoints);
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

  while(1) {
    // Load frame from video
    Mat frame;
    cap >> frame;
    if(frame.empty()) break;

    // Feature Extraction using ORB
    extract(frame);  
      
    // Display
    imshow("Video", frame);
    char c = (char)waitKey(1000/cap.get(5)); // display according to original fps
    if(c==27) break; // Esc 
  }

  cap.release();
  destroyAllWindows();
  return 0; 
}


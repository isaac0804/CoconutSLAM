#include <iostream>
#include <chrono> 

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
  // Read image in argv[1]
  cv::Mat image;
  image = cv::imread(argv[1]);

  // Check if the file exists
  if (image.data == nullptr) {
    cerr << "file" << argv[1] << " not exist." << endl;
    return 0;
  }

  // Print some basic information
  cout << "Image cols: " << image.cols << endl;
  cout << "Image rows: " << image.rows << endl;
  cout << "Image channels: " << image.channels() << endl;
  
  // Show image
  cv::imshow("Image", image);
  cv::waitKey(0);

  // Check image type (grayscale and RGB only)
  if(image.type() != CV_8UC1 && image.type() != CV_8UC3) {
    cout << "Image type incorrect." << endl;
    return 0;
  }

  // Calculate time
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for(size_t y = 0; y < image.rows; y++) {
    // row_ptr is the pointer of y-th row
    unsigned char *row_ptr = image.ptr<unsigned char>(y);
    for(size_t x = 0; x < image.cols; x++) {
      // data_ptr is the pointer to the data at image[x][y]
      unsigned char *data_ptr = &row_ptr[x * image.channels()];
      for(int c = 0; c != image.channels(); c++) {
        // data is the pixel in each channel
        unsigned char data = data_ptr[c];
      }
    }
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1);
  cout << "Time used: " << time_used.count() << " seconds." << endl;

  // copying cv::Mat 
  // operator = will not copy the image data, but only the reference 
  cv::Mat image_copy = image;
  // set top-left 100*100 block to zero 
  image_copy(cv::Rect(0, 0, 100, 100)).setTo(0);
  cv::imshow("Image", image);
  cv::waitKey(0);

  cv::Mat image_clone = image.clone();
  image_clone(cv::Rect(10, 10, 100, 100)).setTo(255);
  cv::imshow("Image", image);
  cv::imshow("Cloned Image", image_clone);
  cv::waitKey(0);

  cv::destroyAllWindows();
  return 0;
}

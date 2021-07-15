#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;

int main(int argc, char **argv){
  // intrinsics
  double fx = 718.856, fy = 718.856, cx = 412.5, cy = 140;
  // baseline
  double b = 0.573;

  cv::Mat left = cv::imread(argv[1]);
  cv::Mat right = cv::imread(argv[2]);

  // compute point cloud 
  vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud; 
  cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 120, 9, 16 * 9 * 9, 64 * 9 * 9, 1, 80, 5, 100, 16);
  cv::Mat disparity_sgbm, disparity;
  sgbm->compute(left, right, disparity_sgbm);
  disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0F );

  for(int v = 0; v < left.rows; v++)
  for(int u = 0; u < right.rows; u++){
    if(disparity.at<float>(v, u) <= 5.0 || disparity.at<float>(v, u) >= 80.0) continue;

    Eigen::Vector4d point(0, 0, 0, left.at<uchar>(v, u) / 255.0);

    double x = (u - cx) / fx;
    double y = (v - cy) / fy;
    double depth = fx * b / (disparity.at<float>(v, u));
    point[0] = x * depth;
    point[1] = y * depth;
    point[2] = depth;

    pointcloud.push_back(point);
  }

  cv::imshow("Disparity", disparity / 80.0);
  cv::waitKey(0);
  // showPointCloud(pointcloud);
  return 0;
}

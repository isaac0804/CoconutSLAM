# Introduction 

CoconutSLAM is a SLAM written in bad C++ code during a holiday. It is bad and the code is terrible, but it is a SLAM. 


# General 
- Shi-Tomasi corner detection
- Orb compute descriptors
- BFMatcher match points 
- extract R, t from Essential matrix
- triangulate based on R, t
- a thread for pangolin

# Results
![Kitti Dataset](videos/CoconutSLAM-2021-08-17_20.41.25.gif)

# Library
- opencv 
  - image processing, feature extraction, matching features
- pangolin 
  - 3d visualisation
- g2o coming soon
  - graph optimisation, bundle adjustment


# Todo 
- refactor code in standard directories (include, src, bin...)
- use other data structure to store points (dictionary or map?) to avoid vector + complex indexing
- g2o optimization
- save and load point maps
- filter points behind camera
- use better matching algorithms (done, HAMMING 2)
- if it doesn't work, we use ORB and implement modifications in ORB SLAM 2
- essential matrix is weird, why
- triangulate isn't great
- is there a way to estimate intrinsic matrix without additional information
- more test set (drone, driving, indoor) 


# Todo (Far future)
- generate mesh using points
- deep learning models to regenerate 3d structure based on points
- deep learning models to filter?

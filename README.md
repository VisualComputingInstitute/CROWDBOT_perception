# CROWDBOT_perception
This is the perception pipeline for the [CROWDBOT project](http://crowdbot.eu/), featuring person detection and tracking from multi-sensor modalities.
The project is based on earlier works from the [SPENCER project](https://github.com/spencer-project/spencer_people_tracking).
The new features include:
- Adaption for operating on Jetson AGX 
- YOLOv3 with TensorRT acceleration for fast RGB-based person detection
- DROW3/DR-SPAAM for person detection with 2D LiDAR sensors
- [OPTIONAL] Optical flow aided tracking
- [OPTIONAL] 3D person pose analysis
- [OPTIONAL] Person ReID

For setting up, refer to `rwth_crowdbot_launch/README.md`.

# Acknowledgement
- SPENCER tracking pipeline [`[repo]`](https://github.com/spencer-project/spencer_people_tracking), [`[repo]`](https://github.com/sbreuers/detta)
- [YOLOv3 with TensorRT](https://github.com/lewes6369/TensorRT-Yolov3)
- [darknet_ros](https://github.com/leggedrobotics/darknet_ros) for person detection
- [DROW3 and DR-SPAAM](https://github.com/VisualComputingInstitute/2D_lidar_person_detection) for 2D-LiDAR-based person detection 
- [MeTRAbs](https://www.vision.rwth-aachen.de/publication/00203/) for monocular 3D pose estimation
- [pwc_net](https://github.com/NVlabs/PWC-Net) for optical flow estimation

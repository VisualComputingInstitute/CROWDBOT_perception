## Jetson AGX Xavier - Prerequisites Installation Steps

The following instructions and information might be incomplete, so please insert any steps and details you find missing.

The setup works with CUDA 10.

1. For installing python modules, create a virtual environment (using virtualenv) allowing site packages, and activate it.
```
pip install virtualenv  # if not installed already (may require sudo)
virtualenv -p /usr/bin/python2 --system-site-packages ros_venv
```

2. Install the python modules listed in the file ``jetson_agx_xavier_requirements_local.txt`` (which is in the same directory as this README). 
As this list might be too big, one might prefer to skip this step and install only the packages that turn out to be missing later on (causing runtime errors).

3. Install PyTorch 1.3 via the pre-built wheel from [here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/).

4. Install ROS Melodic following the instructions on ros.org (package: ros-melodic-ros-base).

5. Install additional ros-packages: 
```
sudo apt install ros-melodic-geometry2 ros-melodic-cv-bridge ros-melodic-image-geometry libimage-transport-dev libcv-bridge-dev ros-melodic-rviz
```

6. Install OpenCV via the install script from [here](https://github.com/mdegans/nano_build_opencv).

7. Create a catkin workspace if there is not one already
```
mkdir -p tracker_ws/src
```

8. Clone darknet_ros and keep only the message package
```
cd tracker_ws/src
git clone --recursive https://github.com/leggedrobotics/darknet_ros.git
rm -rf darknet_ros/darknet_ros darknet_ros/darknet
```

9. Clone and build the tracker (`darknet_ros` package will fail and that's ok)
```
git clone --branch crowdbot_master --recursive https://git.rwth-aachen.de/sabarinath.mahadevan/frame_soft.git
catkin build -cs rwth_crowdbot_launch
```

10. Set up the network weights for DROW (the tracker's laser scan based person detector)
	1. From [google drive](https://drive.google.com/open?id=1_GQHN45QCj5pat44qbtfX-4lCOVyY97d), download the weigths file `dr_spaam_e40.pth` or `drow_e40.pth`.
	1. Set the path to this file in the config file `rwth_crowdbot_launch/config/PLATFORM/drow_ros/drow_ros.yaml` (`PLATFORM` is the robotic platform, `qolo` for example).

11. Download and set the path to the TensorRT engine file in the launch file `rwth_crowdbot_launch/launch/PLATFORM_onboard.launch` (under `<!-- YOLO (with TensorRT) -->`).

12. Launch the tracker
```
roslaunch rwth_crowdbot_launch PLATFORM_onboard.launch
```

13. Optional: The TensorRT engine depends on platform, cuda version, and JetPack version. 
If the downloaded engine file is incompatible, use following steps to generate new engine file for your platform:
	1. Clone [this repo](https://github.com/lewes6369/TensorRT-Yolov3)
	1. In its sub-folder tensorRTWrapper, apply the modifications which are documented in ``jetson_agx_xavier_tensorRTWrapper_modifications.txt``.
	1. In the repo's root folder, apply the modifications which are documented in ``jetson_agx_xavier_TensorRT-Yolov3_modifications.txt``.
	1. Build and install the repo's code according to the instructions in its README.
	1. Download ``yolov3_416.caffemodel`` and ``yolov3_416_trt.prototxt`` from the Google drive linked by the repo's README.
	1. Download any picture (e.g. showing cars and people), place it in the repo and name it ``pic_test.jpeg``.
	1. Create the engine file by executing ``./install/runYolov3 --caffemodel=./yolov3_416.caffemodel --prototxt=./yolov3_416_trt.prototxt --input=./pic_test.jpeg --W=416 --H=416 --class=80 --mode=fp16``
	1. The engine file is created as ``yolov3_fp16.engine`` (it will be used in the last step).


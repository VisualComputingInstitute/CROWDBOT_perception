## Jetson AGX Xavier - Prerequisites Installation Steps

The following instructions and information might be incomplete, so please insert any steps and details you find missing.

The setup works with CUDA 10.

1. For installing python modules, create a virtual environment (using virtualenv) allowing site packages, and activate it.
1. Install the python modules listed in the file ``jetson_agx_xavier_requirements_local.txt`` (which is in the same directory as this README). As this list might be too big, one might prefer to skip this step and install only the packages that turn out to be missing later on (causing runtime errors).
1. Install PyTorch 1.3 via the pre-built wheel from [here](https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/).
1. Install ROS Melodic following the instructions on ros.org (package: ros-melodic-ros-base).
1. Install additional ros-packages: ``sudo apt install ros-melodic-geometry2 ros-melodic-cv-bridge ros-melodic-image-geometry libimage-transport-dev libcv-bridge-dev ros-melodic-rviz``
1. Install OpenCV via the install script from [here](https://github.com/mdegans/nano_build_opencv).
1. Create from the trained YOLO network the TensorRT engine file for your platform as follows:
	1. Clone [this repo](https://github.com/lewes6369/TensorRT-Yolov3)
	1. In its sub-folder tensorRTWrapper, apply the modifications which are documented in ``jetson_agx_xavier_tensorRTWrapper_modifications.txt``.
	1. In the repo's root folder, apply the modifications which are documented in ``jetson_agx_xavier_TensorRT-Yolov3_modifications.txt``.
	1. Build and install the repo's code according to the instructions in its README.
	1. Download ``yolov3_416.caffemodel`` and ``yolov3_416_trt.prototxt`` from the Google drive linked by the repo's README.
	1. Download any picture (e.g. showing cars and people), place it in the repo and name it ``pic_test.jpeg``.
	1. Create the engine file by executing ``./install/runYolov3 --caffemodel=./yolov3_416.caffemodel --prototxt=./yolov3_416_trt.prototxt --input=./pic_test.jpeg --W=416 --H=416 --class=80 --mode=fp16``
	1. The engine file is created as ``yolov3_fp16.engine`` (it will be used in the last step).
1. Create somewhere the catkin workspace for building the tracker using ``mkdir -p tracker_ws/src``
1. Clone darknet_ros and keep only the message package
	1. ``cd tracker_ws/src``
	1. ``git clone --recursive https://github.com/leggedrobotics/darknet_ros.git``
	1. ``rm -rf darknet_ros/darknet_ros darknet_ros/darknet``
1. Build the tracker
	1. ``git clone --branch qolo https://git.rwth-aachen.de/sabarinath.mahadevan/frame_soft.git``
	1. ``cd ..``
	1. ``catkin build -c rwth_crowdbot_launch``
1. Set up the network weights for DROW (the tracker's laserscan-based person detector)
	1. From [here](https://drive.google.com/drive/u/0/folders/1jHisLdOQ8bMnYWE-Dwji60ngIpK_seX8), download the weigths file ``single_ckpt_e40.pth``.
	1. Set the path to this file in the config file ``config/qolo/drow_ros/drow_ros.yaml``
1. Set the path to the TensorRT engine file (created previously) in the launch file ``qolo_onboard.launch`` under ``<!-- YOLO (with TensorRT) -->``.

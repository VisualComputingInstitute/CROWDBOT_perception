cmake_minimum_required(VERSION 2.8.3)
project(detection_fusion)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
  COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  rwth_perception_people_msgs
  message_filters
  frame_msgs
  darknet_ros_msgs
  image_transport
  cv_bridge
  tf)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


set(CMAKE_BUILD_TYPE Release)
add_compile_options(-std=c++11)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-O3)        ## Optimize
endif()

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES convert_yolo
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(detection_fusion src/detection_fusion.cpp)

add_dependencies(detection_fusion ${catkin_EXPORTED_TARGETS})

target_link_libraries(detection_fusion
 ${catkin_LIBRARIES}
 X11
)
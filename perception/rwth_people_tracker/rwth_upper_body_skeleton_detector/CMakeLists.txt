cmake_minimum_required(VERSION 2.8.3)
project(rwth_upper_body_skeleton_detector)
set(ROS_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  rwth_perception_people_msgs
  rwth_upper_body_detector
  sensor_msgs
  std_msgs
  message_generation 
)

find_package(OpenCV)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
add_service_files(
  FILES
  GetUpperBodySkeleton.srv
)

generate_messages(DEPENDENCIES std_msgs)


#catkin_package(CATKIN_DEPENDS)

include_directories(include ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${OpenMP_INCLUDE_DIRS})

add_executable(rwth_upper_body_skeleton_detector src/main.cpp src/body_pose_regressor_utilities.cpp src/forest_utilities.cpp src/mean_shift_utilities.cpp src/tree_utilities.cpp)
target_link_libraries(rwth_upper_body_skeleton_detector ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
add_dependencies(rwth_upper_body_skeleton_detector rwth_upper_body_skeleton_detector_gencpp)

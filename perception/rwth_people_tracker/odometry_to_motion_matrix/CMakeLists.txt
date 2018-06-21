cmake_minimum_required(VERSION 2.8.3)
project(odometry_to_motion_matrix)

find_package(PkgConfig REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  eigen_conversions
  nav_msgs
  roscpp
  tf
  tf_conversions
  message_generation sensor_msgs std_msgs message_filters
)

# Include Eigen
find_package(Eigen REQUIRED) 
include_directories(${EIGEN_INCLUDE_DIRS}) 
add_definitions(${EIGEN_DEFINITIONS})

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  VisualOdometry.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
 )


###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS
    eigen_conversions
    nav_msgs
    roscpp
    tf
    tf_conversions
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)


add_executable(tf2visual src/odom2visual/tf2visual.cpp)
add_dependencies(tf2visual ${catkin_EXPORTED_TARGETS})
target_link_libraries(tf2visual
  ${catkin_LIBRARIES}
)



#############
## Install ##
#############

install(TARGETS tf2visual
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

cmake_minimum_required(VERSION 2.8.3)
project(rwth_crowdbot_launch)

find_package(catkin REQUIRED)

catkin_package()

install(PROGRAMS scripts/run_pwc_net_ros.sh
  DESTINATION
  ${CATKIN_PACKAGE_BIN_DESTINATION})


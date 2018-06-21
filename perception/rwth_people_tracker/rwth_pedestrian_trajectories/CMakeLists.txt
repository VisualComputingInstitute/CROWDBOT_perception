cmake_minimum_required(VERSION 2.8.3)
project(rwth_pedestrian_trajectories)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS cmake_modules cv_bridge message_filters roscpp sensor_msgs std_msgs frame_msgs tf)

include_directories(${catkin_INCLUDE_DIRS})

# If we can use C++11, we do want to make use of it.
# Otherwise, currently, we implicitly fallback on boost.
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    add_definitions(-std=c++11)
elseif(COMPILER_SUPPORTS_CXX0X)
    add_definitions(-std=c++0x)
endif()


find_package(Qt4 REQUIRED COMPONENTS QtCore QtGui)
include(${QT_USE_FILE})

set(CMAKE_BUILD_TYPE Debug)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
endif()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES tracking
#  CATKIN_DEPENDS cv_bridge message_filters roscpp sensor_msgs std_msgs rwth_perception_people_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)


## Declare a cpp executable
add_executable(pedestrian_trajectories src/main.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
#add_dependencies(pedestrian_trajectories rwth_perception_people_msgs_generate_messages_cpp rwth_perception_people_msgs_gencpp)

## Specify libraries to link a library or executable target against
 target_link_libraries(pedestrian_trajectories
   ${catkin_LIBRARIES}
   ${QT_LIBRARIES}
   ${QT_QTCORE_LIBRARY}
   ${QT_QTGUI_LIBRARY}
   X11
 )

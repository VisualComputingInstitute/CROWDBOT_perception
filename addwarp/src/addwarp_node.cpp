#include "addwarp/addwarp.h"
#include <ros/ros.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "addwarp");
  ros::NodeHandle nh, nh_private("~");

  addwarp::AddWarp awp(nh, nh_private);

  ros::spin();
  return 0;
}

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "robot_tf_publisher");
  ros::NodeHandle n;

  ros::Rate r(100);

  tf::TransformBroadcaster broadcaster;

  while(n.ok()){
    broadcaster.sendTransform(
      tf::StampedTransform(
        tf::Transform(tf::Quaternion(0.0, 0.069, 0.0, 0.997), tf::Vector3(0.0, 0.0, 1.46)),
        ros::Time::now(),"base_link", "camera_link"));
    r.sleep();
  }
}

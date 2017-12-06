#include "ros/ros.h"
#include "rwth_perception_people_msgs/AnnotatedFrame.h"
#include "rwth_perception_people_msgs/Annotation.h"

#include <iostream>

void annotationCallback(const rwth_perception_people_msgs::Annotation& anno)
{
  // Just print the annotation to the standard output
  std::cout << anno.frame << " " << anno.id << " " << anno.tlx << " " << anno.tly << " " << anno.brx << " " << anno.bry << std::endl;
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "annotationreceiver");

  ros::NodeHandle n;

  n.subscribe("annotation_boxes", 1000, annotationCallback);

  ros::spin();

  return 0;
}

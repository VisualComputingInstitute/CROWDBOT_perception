#pragma once

#include "frame_msgs/DetectedPerson.h"
#include "frame_msgs/DetectedPersons.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/String.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

namespace addwarp {

class AddWarp {
public:
  /**
   * @brief      Constructor.
   */
  AddWarp(ros::NodeHandle &nh, ros::NodeHandle &nh_private);

  /**
   * @brief      Destructor.
   */
  ~AddWarp();

private:
  /**
   * @brief      Read parameter from ROS parameter server.
   *
   * @return     Success.
   */
  bool readParameters();

  /**
   * @brief      Initialize ROS connection.
   *
   * @return     Success.
   */
  bool init();

  /**
   * @brief      Convenience function to read parameters for a subscriber.
   *
   * @return     Success.
   */
  bool readSubscriberParam(const std::string &name, std::string &topic,
                           int &queue_size);

  /**
   * @brief      Convenience function to read parameters for a publisher.
   *
   * @return     Success.
   */
  bool readPublisherParam(const std::string &name, std::string &topic,
                          int &queue_size, bool &latch);

  /**/

  float imageDifference(const cv_bridge::CvImagePtr image_ptr,
                        const frame_msgs::DetectedPerson &detection);

  void
  detectionsCallback(const frame_msgs::DetectedPersonsConstPtr &detections_msg);

  void imageCallback(const sensor_msgs::ImageConstPtr &msg);

  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  image_transport::ImageTransport it_;

  ros::Publisher detected_persons_with_warp_pub_;

  ros::Subscriber detected_persons_sub_;

  image_transport::Subscriber warp_image_sub_;

  int min_frame_rate_;
  ros::Duration max_time_difference_;

  cv_bridge::CvImagePtr crop_image_ptr_;

  std::vector<sensor_msgs::Image> img_msg_array_;

  int i;
  int j;
};

} // namespace addwarp

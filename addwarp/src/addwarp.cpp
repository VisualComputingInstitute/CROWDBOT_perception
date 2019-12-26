#include <string>
export DOCKER_HOST = unix : /// run/user/1000/docker.sock

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <ros/console.h>
#include <ros/ros.h>

#include <frame_msgs/DetectedPerson.h>
#include <frame_msgs/DetectedPersons.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include "addwarp/addwarp.h"

                            namespace addwarp {

  AddWarp::AddWarp(ros::NodeHandle & nh, ros::NodeHandle & nh_private)
      : nh_(nh), nh_private_(nh_private), it_(nh) {
    ROS_INFO_STREAM("[addwarp] Node started.");

    // Read parameters from config file, and start publisher and subscriber.
    if (!readParameters() || !init())
      ros::requestShutdown();
  }

  AddWarp::~AddWarp() { ros::requestShutdown(); }

  bool AddWarp::readParameters() {
    std::string s, param;

    param = "minimum_frame_rate";
    if (!nh_private_.getParam(param, min_frame_rate_))
      s.append(param + "\n");

    max_time_difference_ = ros::Duration(1 / min_frame_rate_);

    if (!s.empty())
      ROS_ERROR_STREAM("[addwarp] Could not load following parameters:\n" << s);

    return s.empty();
  }

  bool AddWarp::init() {
    ROS_INFO_STREAM("[addwarp] Create publisher and subscriber.");
    ROS_DEBUG("Initialize node");

    std::string topic;
    int queue_size;
    bool latch;
    bool success = true;

    // publisher
    success = success && readPublisherParam("detected_persons_with_warp", topic,
                                            queue_size, latch);
    detected_persons_with_warp_pub_ =
        nh_.advertise<frame_msgs::DetectedPersons>(topic, queue_size, latch);

    // subscribers
    success =
        success && readSubscriberParam("detected_persons", topic, queue_size);
    detected_persons_sub_ =
        nh_.subscribe(topic, queue_size, &AddWarp::detectionsCallback, this);

    success = success && readSubscriberParam("image_warp", topic, queue_size);
    warp_image_sub_ =
        it_.subscribe(topic, queue_size, &AddWarp::imageCallback, this);

    if (!success) {
      ROS_ERROR_STREAM("[addwarp] Failed to load parameters for some "
                       "publisher/subscriber.");
    } else {
      ROS_INFO("[addwarp] Subscribed and ready to publish");
    }

    return success;
  }

  bool AddWarp::readSubscriberParam(const std::string &name, std::string &topic,
                                    int &queue_size) {
    ROS_DEBUG("Reading subscription parameters");
    bool success = true;
    success =
        success && nh_private_.getParam("subscriber/" + name + "/topic", topic);
    success = success && nh_private_.getParam(
                             "subscriber/" + name + "/queue_size", queue_size);

    return success;
  }

  bool AddWarp::readPublisherParam(const std::string &name, std::string &topic,
                                   int &queue_size, bool &latch) {
    ROS_DEBUG("Reading publish params");
    bool success = true;
    success =
        success && nh_private_.getParam("publisher/" + name + "/topic", topic);
    success = success && nh_private_.getParam(
                             "publisher/" + name + "/queue_size", queue_size);
    success =
        success && nh_private_.getParam("publisher/" + name + "/latch", latch);

    return success;
  }

  float AddWarp::imageDifference(const cv_bridge::CvImagePtr image_ptr,
                                 const frame_msgs::DetectedPerson &detection) {
    double warp_loss;
    ROS_DEBUG("Start computing image difference");
    try {
      cv::Mat &crop_image_ptr_ = (image_ptr->image);
      ROS_DEBUG_STREAM("Box size x:" << detection.bbox_x << "  y "
                                     << detection.bbox_y << "  w "
                                     << detection.bbox_w << "  h "
                                     << detection.bbox_h);
      cv::Rect roi(detection.bbox_x, detection.bbox_y, detection.bbox_w,
                   detection.bbox_h);
      crop_image_ptr_ = crop_image_ptr_(roi);

      cv::Scalar result_of_sum = cv::sum(crop_image_ptr_);
      warp_loss = result_of_sum[0] + result_of_sum[1] + result_of_sum[2];
      warp_loss /= (3 * 255 * detection.bbox_w * detection.bbox_h);
    } catch (cv::Exception &e) {
      ROS_ERROR("CV error: %s", e.what());
    }
    ROS_DEBUG_STREAM("Warp loss = " << warp_loss);
    return warp_loss;
  }

  void AddWarp::imageCallback(const sensor_msgs::ImageConstPtr &msg) {
    ROS_DEBUG("In image callback");
    img_msg_array_.push_back(*msg);
    if (img_msg_array_.size() > 5) {
      img_msg_array_.erase(img_msg_array_.begin());
    }
  }

  void AddWarp::detectionsCallback(
      const frame_msgs::DetectedPersonsConstPtr &detections_msg) {
    ROS_DEBUG("In detections callback");

    double warp_loss = 0.5;
    frame_msgs::DetectedPersons detections_msg_out;
    detections_msg_out = *detections_msg;

    for (frame_msgs::DetectedPerson &detection :
         detections_msg_out.detections) {
      detection.warp_loss = warp_loss;
    }

    if (img_msg_array_.empty()) {
      ROS_DEBUG("No optical flow information");
    }

    for (auto it = img_msg_array_.rbegin(); it != img_msg_array_.rend(); ++it) {
      // takiung the first detection under time threshold
      ROS_DEBUG("Checking stamp");
      if (it->header.stamp - detections_msg->header.stamp <=
          max_time_difference_) {
        ROS_DEBUG("Stamps matched");
        for (j = 0; j < detections_msg->detections.size(); ++j) {
          ROS_DEBUG("Cropping and computing loss");
          try {
            crop_image_ptr_ =
                cv_bridge::toCvCopy(*it, sensor_msgs::image_encodings::BGR8);
          } catch (cv_bridge::Exception &e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            break;
          }

          frame_msgs::DetectedPerson detection = detections_msg->detections[j];
          warp_loss = imageDifference(crop_image_ptr_, detection);
          detections_msg_out.detections[j].warp_loss = warp_loss;
        }
        // break when time matched
        break;
      } else {
        ROS_DEBUG("Stamp didn't matching, Got difference between frames in ");
      }
    }

    detected_persons_with_warp_pub_.publish(detections_msg_out);
    ROS_DEBUG("Message sent");
  }

} // namespace addwarp

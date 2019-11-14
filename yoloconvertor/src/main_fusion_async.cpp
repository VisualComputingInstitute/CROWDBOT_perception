#include <string.h>
#include <cmath>

#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <frame_msgs/DetectedPersons.h>

#include "Matrix.h"
#include "Vector.h"


// For multi-sensor use in SPENCER
int detection_id_increment_;
int detection_id_offset_;
int current_detection_id_;

double pose_variance_; // used in output frame_msgs::DetectedPerson.pose.covariance
double overlap_thresh_; // used in detecting overlapping, default value is 0.5
int fusion_rate_;

string world_frame_;
double world_scale_;  // for computing 3D positions from BBoxes

std::shared_ptr<tf::TransformListener> tf_;
ros::Publisher detected_persons_pub_;

typedef frame_msgs::DetectedPersons DP;
typedef message_filters::Subscriber<DP> DPSub;

std::vector<DP::ConstPtr> dp_queue_;  // Keep only the latest detection msg for each sensor
std::vector<std::shared_ptr<DPSub> > dp_sub_queue_;

// Keep track if there has been new detection since last fusion. If no, don't publish.
// Should be set to true after receiving new detection (in detection callback).
// Should be set to false after publising fused detections (in main loop).
ros::Time latest_detection_stamp_, prev_fused_detection_stamp_;

/*
subscrible:
    tf:
    3 detected person:
publish:
    1 detection person
  */

int findDuplicate(const frame_msgs::DetectedPerson& det_query,
                  const DP& dets,
                  const int latest)

{
    for (int i = 0; i < latest; ++i)
    {
        const frame_msgs::DetectedPerson& det = dets.detections.at(i);

        // Check position
        const Vector<double> p1(det_query.pose.pose.position.x,
                                det_query.pose.pose.position.y,
                                det_query.pose.pose.position.z);
        const Vector<double> p2(det.pose.pose.position.x,
                                det.pose.pose.position.y,
                                det.pose.pose.position.z);
        // Vector<double> p1, p2;
        // p1.setSize(3);
        // p2.setSize(3);
        // p1[0] = det_query.pose.pose.position.x;
        // p1[1] = det_query.pose.pose.position.y;
        // p1[2] = det_query.pose.pose.position.z;
        // p2[0] = det.pose.pose.position.x;
        // p2[1] = det.pose.pose.position.y;
        // p2[2] = det.pose.pose.position.z;
        if ((p1 - p2).norm() < overlap_thresh_)
        {
            return i;
        }

        // // Check appearance
        // const Vector<float> a1(det_query.embed_vector);
        // const Vector<float> a2(det.embed_vector);
        // if ((a1 - a2).norm() < 40)
        // {
        //     return i;
        // }
    }

    return -1;
}


void fuseDetections(DP& dp_fused)
{
    int detection_count = 0;
    for (const DP::ConstPtr& dp : dp_queue_)
    {
        if (dp)
        {
            detection_count += dp->detections.size();
        }
    }

    dp_fused.detections.reserve(detection_count);
    dp_fused.header.stamp = latest_detection_stamp_;  // use stamp from latest detection
    dp_fused.header.frame_id = world_frame_;

    // Keep track of number of fused detections after detections from one sensor
    // has been processed. Since we only need to check duplication across sensors,
    // this number can be used to speed-up fusion.
    int number_fused_detections = 0;

    for (auto it = dp_queue_.rbegin(); it != dp_queue_.rend(); it++)
    {
        const DP::ConstPtr& dp = *it;

        if (!dp)
        {
            continue;
        }

        const ros::Time dp_time = dp->header.stamp;
        const string dp_frame = dp->header.frame_id;

        tf::StampedTransform transform;
        try
        {
            // it may cannot find the tf in exact the same time as the input image.. so, maybe someway, lets see
            tf_->waitForTransform(world_frame_, dp_frame, dp_time, ros::Duration(1.0));
            tf_->lookupTransform(world_frame_, dp_frame, dp_time, transform);  //from camera_frame to world_frame_
        }
        catch (tf::TransformException ex)
        {
            ROS_WARN_THROTTLE(20.0, "Failed transform lookup in camera frame to world frame, ignore detections.", ex.what());
            continue;
        }

        for (const frame_msgs::DetectedPerson& d : dp->detections)
        {
            frame_msgs::DetectedPerson d_w = d;

            const tf::Vector3 pos_in_cam(d.pose.pose.position.x,
                                         d.pose.pose.position.y,
                                         d.pose.pose.position.z);
            const tf::Vector3 pos_in_world = transform * pos_in_cam;
            d_w.pose.pose.position.x = pos_in_world.getX();
            d_w.pose.pose.position.y = pos_in_world.getY();
            d_w.pose.pose.position.z = pos_in_world.getZ();

            if (findDuplicate(d_w, dp_fused, number_fused_detections) >= 0)
            {
                // check if replace duplicate
                continue;
            }

            Matrix<double> rotation;
            Matrix<double> covariance;
            covariance.set_size(3, 3, 0.0);
            rotation.set_size(3, 3, 0.0);
            covariance(0,0) = d.pose.covariance[0]; //0.05;
            covariance(1,1) = d.pose.covariance[7];//0.05;
            covariance(2,2) = d.pose.covariance[14];//0.05;
            covariance(0,1) = d.pose.covariance[6];//0;
            covariance(0,2) = d.pose.covariance[12];//0;
            covariance(1,0) = d.pose.covariance[1];//0;
            covariance(1,2) = d.pose.covariance[13];//0;
            covariance(2,0) = d.pose.covariance[2];//0;
            covariance(2,1) = d.pose.covariance[8];//0;
            tf::Quaternion rot_q = transform.getRotation();
            tf::Matrix3x3 rot(rot_q);
            rotation(0,0) = rot.getColumn(0).getX();
            rotation(0,1) = rot.getColumn(0).getY();
            rotation(0,2) = rot.getColumn(0).getZ();
            rotation(1,0) = rot.getColumn(1).getX();
            rotation(1,1) = rot.getColumn(1).getY();
            rotation(1,2) = rot.getColumn(1).getZ();
            rotation(2,0) = rot.getColumn(2).getX();
            rotation(2,1) = rot.getColumn(2).getY();
            rotation(2,2) = rot.getColumn(2).getZ();
            //std::cout << "cov before:" << std::endl;
            //covariance.Show();
            covariance = rotation * covariance;
            rotation.Transpose();
            covariance = covariance * rotation;
            //std::cout << "cov after:" << std::endl;
            //covariance.Show();

            d_w.pose.covariance[0] = covariance(0,0); //0.05;
            d_w.pose.covariance[7] = covariance(1,1);//0.05;
            d_w.pose.covariance[14] = covariance(2,2);//0.05;
            d_w.pose.covariance[6] = covariance(0,1);//0;
            d_w.pose.covariance[12] = covariance(0,2);//0;
            d_w.pose.covariance[1] = covariance(1,0);//0;
            d_w.pose.covariance[13] = covariance(1,2);//0;
            d_w.pose.covariance[2] = covariance(2,0);//0;
            d_w.pose.covariance[8] = covariance(2,1);//0;

            // additional nan check
            if (!std::isnan(d_w.pose.pose.position.x) &&
                !std::isnan(d_w.pose.pose.position.y) &&
                !std::isnan(d_w.pose.pose.position.z))
            {
                d_w.detection_id = current_detection_id_;  // reasign detection id
                dp_fused.detections.push_back(d_w);
                current_detection_id_ += detection_id_increment_;
            }
            else
            {
                ROS_DEBUG("A detection has been discarded because of nan values during fusion!");
            }
        }  // for (const frame_msgs::DetectedPerson& d : dp->detections)

        number_fused_detections = dp_fused.detections.size();
        // *it = nullptr;
    }  // for (auto it = dp_queue_.rbegin(); it != dp_queue_.rend(); it++)
}


void detectedPersonsCallback(const DP::ConstPtr& msg, const int subscriber_id)
{
    dp_queue_.at(subscriber_id) = msg;
    if (msg->header.stamp.toSec() > latest_detection_stamp_.toSec())
        latest_detection_stamp_ = msg->header.stamp;
}


// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(ros::Subscriber& connect_sub,
                     std::vector<std::shared_ptr<DPSub> >& sub_queue)
{
    if (!detected_persons_pub_.getNumSubscribers())
    {
        ROS_DEBUG("yoloconvertor: No subscribers. Unsubscribing.");
        connect_sub.shutdown();
        for (auto& sub : dp_sub_queue_)
        {
            sub->unsubscribe();
        }
    }
    else
    {
        ROS_DEBUG("yoloconvertor: New subscribers. Subscribing.");
        for (auto& sub : dp_sub_queue_)
        {
            sub->subscribe();
        }
    }
}

int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "fuse_yolo_async");
    ros::NodeHandle n, private_node_handle_("~");

    latest_detection_stamp_ = ros::Time::now();
    prev_fused_detection_stamp_ = latest_detection_stamp_;

    // create a tf listener
    tf_ = std::make_shared<tf::TransformListener>();

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string detected_persons_left;
    string detected_persons_right;
    string detected_persons_rear;
    string pub_topic_detected_persons;

    // Initialize node parameters from launch file or command line.
    private_node_handle_.param("queue_size", queue_size, int(1));
    private_node_handle_.param("detected_persons_left", detected_persons_left, string("oops!need param for left"));
    private_node_handle_.param("detected_persons_right", detected_persons_right, string("oops!need param for right"));
    private_node_handle_.param("detected_persons_rear", detected_persons_rear, string("oops!need param for rear"));

    // For SPENCER DetectedPersons message
    private_node_handle_.param("world_scale", world_scale_, 1.0); // default for ASUS sensors
    private_node_handle_.param("detection_id_increment", detection_id_increment_, 1);
    private_node_handle_.param("detection_id_offset", detection_id_offset_, 0);
    private_node_handle_.param("pose_variance", pose_variance_, 0.05);
    private_node_handle_.param("overlap_thresh", overlap_thresh_, 0.50);  //this overlap_thresh is for overlapping detection
    private_node_handle_.param("fusion_rate", fusion_rate_, 7);  //the publish rate for the fused detections

    private_node_handle_.param("world_frame", world_frame_, string("/robot/OdometryFrame"));


    current_detection_id_ = detection_id_offset_;


    // Create a subscriber.
    std::vector<string> sub_topics;
    sub_topics.push_back(detected_persons_left);
    if (detected_persons_right != detected_persons_left)
    {
        sub_topics.push_back(detected_persons_right);
    }
    if (detected_persons_rear != detected_persons_left)
    {
        sub_topics.push_back(detected_persons_rear);
    }

    dp_sub_queue_.reserve(sub_topics.size());
    for (const string& t : sub_topics)
    {
        dp_sub_queue_.push_back(std::make_shared<DPSub>(n, t.c_str(), 1));
        // std::cout << dp_sub_queue_.size() << " " << t.c_str() << std::endl;
        dp_sub_queue_.back()->unsubscribe();
        dp_queue_.push_back(nullptr);
    }

    ros::Subscriber connect_sub;
    ros::SubscriberStatusCallback connect_cb = boost::bind(&connectCallback,
                                                           boost::ref(connect_sub),
                                                           boost::ref(dp_sub_queue_));

    for (int i = 0; i < dp_sub_queue_.size(); ++i)
    {
        dp_sub_queue_.at(i)->registerCallback(
                boost::bind(&detectedPersonsCallback, _1, i));
    }

    // Create publishers
    private_node_handle_.param("total_detected_persons", pub_topic_detected_persons, string("/total_detected_persons"));
    detected_persons_pub_ = n.advertise<frame_msgs::DetectedPersons>(pub_topic_detected_persons, 1, connect_cb, connect_cb);

    ros::Rate r(fusion_rate_); // 10 hz
    while (ros::ok())
    {
        const bool new_detection_received =
                latest_detection_stamp_.toSec() > prev_fused_detection_stamp_.toSec();
        if (new_detection_received && detected_persons_pub_.getNumSubscribers())
        {
            prev_fused_detection_stamp_ = latest_detection_stamp_;
            DP fused_detections;
            fuseDetections(fused_detections);
            detected_persons_pub_.publish(fused_detections);
        }

        ros::spinOnce();
        r.sleep();
    }

    return 0;
}




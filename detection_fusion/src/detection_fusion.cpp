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
bool enforce_all_;

string world_frame_;
double world_scale_;  // for computing 3D positions from BBoxes

std::shared_ptr<tf::TransformListener> tf_;
ros::Publisher detected_persons_pub_;

typedef frame_msgs::DetectedPersons DP;
typedef message_filters::Subscriber<DP> DPSub;

/* ------------------

Helper func

------------------ */

float getDist(const float x0, const float y0, const float z0,
              const float x1, const float y1, const float z1)
{
    const float x = x1 - x0;
    const float y = y1 - y0;
    const float z = z1 - z0;

    return sqrt(x * x + y * y + z * z);
}

float getDist(const frame_msgs::DetectedPerson& d0, 
                 const frame_msgs::DetectedPerson& d1)
{
    return getDist(d0.pose.pose.position.x,
                   d0.pose.pose.position.y,
                   d0.pose.pose.position.z,
                   d1.pose.pose.position.x,
                   d1.pose.pose.position.y,
                   d1.pose.pose.position.z);
}

float getDist(const frame_msgs::DetectedPerson& d)
{
    return getDist(d.pose.pose.position.x,
                   d.pose.pose.position.y,
                   d.pose.pose.position.z,
                   0.0,
                   0.0,
                   0.0);
}

void detectedPersonsCallback(const DP::ConstPtr& msg, const int subscriber_id);

class DetectionSource {
  public:
    DetectionSource(ros::NodeHandle& nh,
                    const std::string topic, 
                    const int id, 
                    const float range_min, 
                    const float range_max) 
    {
        id_ = id;
        topic_ = topic;

        ros_sub_ = std::make_shared<DPSub>(nh, topic.c_str(), 1);
        range_min_ = range_min;
        range_max_ = range_max;

        ros_sub_->unsubscribe();
        updated_ = false;
        dets_ptr_ = nullptr;

        std::cout << "[DetectionFusion] Add sensor source"
                  << " id: " << id_
                  << " topic: " << topic_ 
                  << " range_min: " << range_min_
                  << " range_max: " << range_max_
                  << std::endl;
    };

    void subscribe() {
        ros_sub_->subscribe();
    };

    void unsubscribe() {
        ros_sub_->unsubscribe();
    };

    void registerCallback() {
        ros_sub_->registerCallback(boost::bind(&detectedPersonsCallback, _1, id_));
    };


    void update(const DP& msg)
    {
        // std::cout << "[DetectionFusion] DetectionSource " << id_ 
        //           << " update triggered" << std::endl;

        DP msg_new;

        // std::cout << "[DetectionFusion] access message header " << std::endl;

        msg_new.header = msg.header;

        // std::cout << "[DetectionFusion] ready to start for loop " << std::endl;

        for (const frame_msgs::DetectedPerson& d : msg.detections)
        {
            // std::cout << "[DetectionFusion] compute " << std::endl;
            const float bev_dist = getDist(d);
            // std::cout << bev_dist << std::endl;
            if (bev_dist < range_max_ and bev_dist >= range_min_)
            {
                msg_new.detections.push_back(d);
                // std::cout << "added" << std::endl;
            }
        }

        dets_ptr_ = std::make_shared<DP>(msg_new);
        updated_ = true;

        // std::cout << "[DetectionFusion] DetectionSource (id: " << id_ 
        //           << ") update finished" << std::endl;
    }

    void outdate() { updated_ = false; };

    bool isUpdated() const { return updated_; };

    int getSize() const
    {
        if (dets_ptr_) {
            return dets_ptr_->detections.size();
        } else {
            return 0;
        }
    };

    std::shared_ptr<DP> getDetections() const { return dets_ptr_; };
  
  private:
    int id_;
    std::string topic_;
    std::shared_ptr<DP> dets_ptr_;
    std::shared_ptr<DPSub> ros_sub_;
    float range_min_;
    float range_max_;
    bool updated_;
} ;

// // TODO these should be organized into a struct
// std::vector<DP::ConstPtr> dp_queue_;  // Keep only the latest detection msg for each sensor
// std::vector<std::shared_ptr<DPSub> > dp_sub_queue_;  // Holder for all subscriber
// std::vector<bool> dp_new_queue_;  // Keep track of new detections from each sensor
// std::vector<float> range_min_queue_, range_max_queue_;

std::vector<DetectionSource> ds_queue_;

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
        const float dist = getDist(det_query, det);
        if (dist < overlap_thresh_)
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


// int findDuplicate(const frame_msgs::DetectedPerson& det_query,
//                   const DP& dets,
//                   const int latest)

// {
//     for (int i = 0; i < latest; ++i)
//     {
//         const frame_msgs::DetectedPerson& det = dets.detections.at(i);

//         // Check position
//         const Vector<double> p1(det_query.pose.pose.position.x,
//                                 det_query.pose.pose.position.y,
//                                 det_query.pose.pose.position.z);
//         const Vector<double> p2(det.pose.pose.position.x,
//                                 det.pose.pose.position.y,
//                                 det.pose.pose.position.z);
//         // Vector<double> p1, p2;
//         // p1.setSize(3);
//         // p2.setSize(3);
//         // p1[0] = det_query.pose.pose.position.x;
//         // p1[1] = det_query.pose.pose.position.y;
//         // p1[2] = det_query.pose.pose.position.z;
//         // p2[0] = det.pose.pose.position.x;
//         // p2[1] = det.pose.pose.position.y;
//         // p2[2] = det.pose.pose.position.z;
//         if ((p1 - p2).norm() < overlap_thresh_)
//         {
//             return i;
//         }

//         // // Check appearance
//         // const Vector<float> a1(det_query.embed_vector);
//         // const Vector<float> a2(det.embed_vector);
//         // if ((a1 - a2).norm() < 40)
//         // {
//         //     return i;
//         // }
//     }

//     return -1;
// }


void fuseDetections(DP& dp_fused)
{
    // std::cout << "[DetectionFusion] fuseDetections triggered" << std::endl;

    int detection_count = 0;
    for (const auto& ds : ds_queue_)
    {
        detection_count += ds.getSize();
    }
    // std::cout << "detection_count: " << detection_count << std::endl;

    dp_fused.detections.reserve(detection_count);
    dp_fused.header.stamp = latest_detection_stamp_;  // use stamp from latest detection
    dp_fused.header.frame_id = world_frame_;

    // Keep track of number of fused detections after detections from one sensor
    // has been processed. Since we only need to check duplication across sensors,
    // this number can be used to speed-up fusion.
    int number_fused_detections = 0;

    // Iterate over detections from each sensor
    for (auto it = ds_queue_.begin(); it != ds_queue_.end(); it++)
    {
        const std::shared_ptr<DP>& dp = it->getDetections();

        if (it->getSize() <= 0)
        {
            // std::cout << "No detections, continue" << std::endl;
            continue;
        }

        const ros::Time dp_time = dp->header.stamp;
        const string dp_frame = dp->header.frame_id;

        // std::cout << dp_time << std::endl;
        // std::cout << dp_frame << std::endl;

        tf::StampedTransform transform;
        try
        {
            tf_->waitForTransform(world_frame_, dp_frame, dp_time, ros::Duration(1.0));
            tf_->lookupTransform(world_frame_, dp_frame, dp_time, transform);  //from camera_frame to world_frame_
        }
        catch (tf::TransformException ex)
        {
            ROS_WARN("[DetectionFusion] Failed transform lookup from %s to %s, ignore detections. %s", 
                     dp_frame.c_str(), world_frame_.c_str(), ex.what());
            continue;
        }

        // Iterate over each detection from this sensor
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
                // std::cout << "Duplicate" << std::endl;
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
        it->outdate();
    }  // for (auto it = ds_queue_.rbegin(); it != ds_queue_.rend(); it++)
}


void detectedPersonsCallback(const DP::ConstPtr& msg, const int subscriber_id)
{
    // std::cout << "[DetectionFusion] detectedPersonsCallback triggered, id:" 
    //           << subscriber_id << std::endl;

    // std::shared_ptr<DP> msg_ptr = std::make_shared<DP>(*msg);

    ds_queue_.at(subscriber_id).update(*msg);
    // dp_queue_.at(subscriber_id) = msg;
    // dp_new_queue_.at(subscriber_id) = true;
    // Detection may not arrive sequentially
    if (msg->header.stamp.toSec() > latest_detection_stamp_.toSec())
        latest_detection_stamp_ = msg->header.stamp;
}


// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(ros::Subscriber& connect_sub)
{
    // std::cout << "[DetectionFusion] connectCallback triggered" << std::endl;

    if (!detected_persons_pub_.getNumSubscribers())
    {
        ROS_INFO("[DetectionFusion] No subscribers. Unsubscribing.");
        connect_sub.shutdown();
        for (auto& sub : ds_queue_)
        {
            sub.unsubscribe();
        }
    }
    else
    {
        ROS_INFO("[DetectionFusion] New subscribers. Subscribing.");
        for (auto& sub : ds_queue_)
        {
            sub.subscribe();
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

    // These params matter and should be set up in ros config
    private_node_handle_.param("fusion_rate", fusion_rate_, 7);  //the publish rate for the fused detections
    private_node_handle_.param("enforce_all", enforce_all_, false);  //the publish rate for the fused detections
    private_node_handle_.param("world_frame", world_frame_, string("/robot/OdometryFrame"));
    private_node_handle_.param("overlap_thresh", overlap_thresh_, 0.50);  //this overlap_thresh is for overlapping detection

    // These params do not matter much and default value would be ok.
    private_node_handle_.param("world_scale", world_scale_, 1.0); // default for ASUS sensors
    private_node_handle_.param("detection_id_increment", detection_id_increment_, 1);
    private_node_handle_.param("detection_id_offset", detection_id_offset_, 0);
    private_node_handle_.param("pose_variance", pose_variance_, 0.05);

    current_detection_id_ = detection_id_offset_;

    // Subscribers
    std::vector<string> sub_topics;
    int num_detection_source;
    private_node_handle_.getParam("number_of_detection_source", num_detection_source);

    ds_queue_.reserve(num_detection_source);
    for (int idx = 0; idx < num_detection_source; ++idx)
    {
        const std::string sub_name = "subscriber/detections" + std::to_string(idx);
        std::string sub_topic;
        private_node_handle_.getParam(sub_name + "/topic", sub_topic);

        float range_min, range_max;
        private_node_handle_.param<float>(sub_name + "/range_min", range_min, 0.0);
        private_node_handle_.param<float>(sub_name + "/range_max", range_max, 1000.0);

        ds_queue_.push_back(DetectionSource(n, sub_topic, idx, range_min, range_max));

        // dp_sub_queue_.push_back(std::make_shared<DPSub>(n, sub_topic.c_str(), 1));
        // range_min_queue_.push_back(range_min);
        // range_max_queue_.push_back(range_max);
        // std::cout << "[main_fusion_async] Add sensor source"
        //           << " id: " << idx
        //           << " topic: " << sub_topic 
        //           << " range_min: " << range_min
        //           << " range_max: " << range_max
        //           << std::endl;
        // dp_sub_queue_.back()->unsubscribe();
        // dp_queue_.push_back(nullptr);
        // dp_new_queue_.push_back(false);
    }

    ros::Subscriber connect_sub;
    ros::SubscriberStatusCallback connect_cb = boost::bind(&connectCallback,
                                                           boost::ref(connect_sub));

    for (int idx = 0; idx < ds_queue_.size(); ++idx)
    {
        ds_queue_.at(idx).registerCallback();
    }

    // Publisher
    std::string pub_topic;
    private_node_handle_.param("publisher/detections/topic", pub_topic, std::string("/detected_persons_synchronized"));
    detected_persons_pub_ = n.advertise<frame_msgs::DetectedPersons>(pub_topic, 1, connect_cb, connect_cb);

    // Main loop
    ros::Rate r(fusion_rate_); // 10 hz
    while (ros::ok())
    {
        ros::spinOnce();

        bool do_fusion = true;
        if (!detected_persons_pub_.getNumSubscribers())
        {
            // ROS_INFO("No subscriber.");
            r.sleep();
            continue;
        }

        // No new detection
        if (latest_detection_stamp_.toSec() <= prev_fused_detection_stamp_.toSec())
        {
            // ROS_INFO("No new detection.");
            r.sleep();
            continue;
        }

        // Not all sensors are updated
        if (enforce_all_)
        {
            bool all_detections_are_new = true;
            for (const auto& ds : ds_queue_)
            {
                all_detections_are_new = all_detections_are_new && ds.isUpdated();
            }

            if (!all_detections_are_new)
            {
                // ROS_INFO("Not all detections are new.");
                r.sleep();
                continue;
            }
        }

        // Perform fusion
        prev_fused_detection_stamp_ = latest_detection_stamp_;
        DP fused_detections;
        fuseDetections(fused_detections);
        detected_persons_pub_.publish(fused_detections);

        r.sleep();
    }

    return 0;
}




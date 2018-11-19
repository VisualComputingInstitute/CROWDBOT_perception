// ROS includes.
#include <ros/ros.h>
#include <ros/time.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <string.h>
#include <cmath>
#include <algorithm>    // std::min_element, std::max_element

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <rwth_perception_people_msgs/GroundPlane.h>
#include <frame_msgs/DetectedPersons.h>

#include "Matrix.h"
#include "Vector.h"

using namespace std;
using namespace message_filters;
using namespace rwth_perception_people_msgs;
using namespace frame_msgs;

tf::TransformListener* listener;
ros::Publisher pub_detected_persons;
double worldScale; // for computing 3D positions from BBoxes

int detection_id_increment, detection_id_offset, current_detection_id; // added for multi-sensor use in SPENCER
double pose_variance; // used in output frame_msgs::DetectedPerson.pose.covariance
double overlap_thresh; // used in detecting overlapping, default value is 0.5

string world_frame;

/*
subscrible:
    tf:
    3 detected person:
publish:
    1 detection person
  */


void transfer_detected_persons_to_world_cord(const DetectedPersonsConstPtr &sub_dp, DetectedPersons &pub_dp, string camera_frame)
{
    ros::Time detection_time(sub_dp->header.stamp);
    tf::StampedTransform transform;
    try {
        listener->waitForTransform(world_frame, camera_frame, detection_time, ros::Duration(1.0));
        listener->lookupTransform(world_frame,camera_frame,detection_time, transform);  //from camera_frame to world_frame
        // it may cannot find the tf in exact the same time as the input image.. so, maybe someway, lets see
    }
    catch (tf::TransformException ex){
       ROS_WARN_THROTTLE(20.0, "Failed transform lookup in camera frame to world frame", ex.what());
    }

    for(unsigned int i=0;i<(sub_dp->detections.size());i++)
    {
        // only process the 3d pose.
        // get its tf, and do transformation
        // other things are the same with the dp_left
        // questions. about the detection id...
        DetectedPerson detected_person(sub_dp->detections[i]);

        tf::Vector3 pos_vector_in_camera(detected_person.pose.pose.position.x, detected_person.pose.pose.position.y, detected_person.pose.pose.position.z);
        tf::Vector3 pos_vector_in_world = transform*pos_vector_in_camera;  // may need from world to camera, lets see.

        Vector<double> pos3D;
        pos3D.setSize(3);
        pos3D[0] = pos_vector_in_world.getX();
        pos3D[1] = pos_vector_in_world.getY();
        pos3D[2] = pos_vector_in_world.getZ();

        detected_person.pose.pose.position.x = pos3D(0);
        detected_person.pose.pose.position.y = pos3D(1);
        detected_person.pose.pose.position.z = pos3D(2);

        pub_dp.detections.push_back(detected_person);
    }
}

bool is_overlaping(const DetectedPerson& dp1,const DetectedPerson& dp2)
{
    Vector<double> pos1;
    Vector<double> pos2;
    pos1.setSize(3);
    pos2.setSize(3);

    pos1[0] = dp1.pose.pose.position.x;
    pos1[1] = dp1.pose.pose.position.y;
    pos1[2] = dp1.pose.pose.position.z;
    pos2[0] = dp2.pose.pose.position.x;
    pos2[1] = dp2.pose.pose.position.y;
    pos2[2] = dp2.pose.pose.position.z;

    bool flag = false;

    if((pos1 - pos2).norm() < overlap_thresh )   // set 0.05, if two people is closer than 5 cm, we say they should be exactly the same person
        flag = true;
    else
        flag = false;

    return flag;
}




bool compare_stamp(ros::Time & i, ros::Time& j) { return i.toSec()<j.toSec(); }



void add_new_camera_detection(DetectedPersons& dps_new, DetectedPersons& dps_dst)
{
    // it should only compare with the first k dps_dst's elements, which means all detection from the original camera
    size_t k = dps_dst.detections.size();

    // a k size index vector, to indicate if this index's detection in dps_dst has been replaced,
    // if it has been raplaced, which means now the detection in dps_dst is from the new camera.
    // since we don't want to replace detetions from the same camera.
    // so In this case we will simply push_back the new detection into dps_dst, but not replace the one in dps_dst.
    // true: can replace
    // false: don't replace
    std::vector<bool> index(k,true);

    for(int i=0;i<dps_new.detections.size();++i)
    {
        DetectedPerson new_det(dps_new.detections[i]);
        // we have actually 3 cases for each detection from dps_new:
        // 1. the new det overlapping and better and can replace the det in dps_dst -> replace
        // 2. the new det overlapping and better but cannot replace the det in dps_dst-> push_back
        // 3. the new det not overlapping -> push_back
        // 4. throw away this detetction.
        bool overlaping_flag(false);
        bool better_det_flag(false);
        bool replace_flag(false);
        // it should only compare with the first k dps_dst's elements, which are come from the first camera
        for(int j=0;j<k;++j)
        {
                if(is_overlaping(new_det,dps_dst.detections[j]))
                {
                    overlaping_flag = true;
                    // if this new detection from this camera is better, replace the original one by this new_det
                    if(new_det.confidence > dps_dst.detections[j].confidence)
                    {
                        better_det_flag = true;
                        // see if we can replace it
                        if(index[j])
                        {
                            dps_dst.detections[j] = new_det;
                            // if we have replaced, we should simply go to the next detection in new camera( go to the next iteration in out loop)
                            replace_flag = true;
                            break;
                        }
                    }
                }
        }
        if(replace_flag) // case 1
        {
           continue;
        }
        else if(overlaping_flag && better_det_flag) //case 2
        {
            dps_dst.detections.push_back(new_det);
        }
        else if(!overlaping_flag) //case 3
        {
            dps_dst.detections.push_back(new_det);
        }
    }

}


void combine_three_camera_detection(DetectedPersons& dps0, DetectedPersons& dps1, DetectedPersons& dps2, DetectedPersons& dps_dst)
{
    // again use greedy method
    // we first deal with the camera 0
    for(int i = 0;i<dps0.detections.size();++i)
    {
        // since there is no detection from other camera, we see every detection here as best detection and push_back.
        dps_dst.detections.push_back(dps0.detections[i]);
    }

    //camera 1
    add_new_camera_detection(dps1,dps_dst);
    //camera 2
    add_new_camera_detection(dps2,dps_dst);

    // now re assign detection ID
    for(int i=0;i<dps_dst.detections.size();++i)
    {
        dps_dst.detections[i].detection_id = current_detection_id;
        current_detection_id += detection_id_increment;
    }
}

void yolofusioncallback(const DetectedPersonsConstPtr &dp_left, const DetectedPersonsConstPtr &dp_right, const DetectedPersonsConstPtr &dp_rear )
{
    // debug output, to show latency from yolo_v3
    //ROS_DEBUG_STREAM("current time:" << ros::Time::now());

    if(pub_detected_persons.getNumSubscribers()) {

        // these three DetectedPersons are in world frame
        frame_msgs::DetectedPersons left_detected_persons;
        frame_msgs::DetectedPersons right_detected_persons;
        frame_msgs::DetectedPersons rear_detected_persons;
        transfer_detected_persons_to_world_cord(dp_left, left_detected_persons, dp_left->header.frame_id);
        transfer_detected_persons_to_world_cord(dp_right, right_detected_persons, dp_right->header.frame_id);
        transfer_detected_persons_to_world_cord(dp_rear, rear_detected_persons, dp_rear->header.frame_id);


        // three modification
        // 1. only do this remove overlapping between camera and camera
        // 2. when overlapping happen, take the highest confidence detection
        // 3. use an arg, not a constant value as overlap_thresh.
        // remove the overlaping detection
        frame_msgs::DetectedPersons detected_persons;
        // reserve enough memory for this final detected_persons msg.
        detected_persons.detections.reserve(left_detected_persons.detections.size()+right_detected_persons.detections.size()+rear_detected_persons.detections.size());
        combine_three_camera_detection(left_detected_persons,right_detected_persons,rear_detected_persons, detected_persons);

        // this is for using std::max_element to get the latest time stamp
        std::vector<ros::Time> stamp_vec;
        stamp_vec.push_back(dp_left->header.stamp);
        stamp_vec.push_back(dp_right->header.stamp);
        stamp_vec.push_back(dp_rear->header.stamp);
        detected_persons.header.stamp = *std::max_element(stamp_vec.begin(),stamp_vec.end(),compare_stamp);


        detected_persons.header.frame_id = world_frame;
        // Publish
        pub_detected_persons.publish(detected_persons);
    }
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(ros::Subscriber &sub_msg,
                     ros::NodeHandle &n,
                     Subscriber<DetectedPersons> &sub_dp_left,
                     Subscriber<DetectedPersons> &sub_dp_right,
                     Subscriber<DetectedPersons> &sub_dp_rear){
    if(!pub_detected_persons.getNumSubscribers()) {
        ROS_DEBUG("yoloconvertor: No subscribers. Unsubscribing.");
        sub_msg.shutdown();
        sub_dp_left.unsubscribe();
        sub_dp_rear.unsubscribe();
        sub_dp_right.unsubscribe();
    } else {
        ROS_DEBUG("yoloconvertor: New subscribers. Subscribing.");
        sub_dp_left.subscribe();
        sub_dp_right.subscribe();
        sub_dp_rear.subscribe();
    }

}

int main(int argc, char **argv)
{


    // Set up ROS.
    ros::init(argc, argv, "fuse_yolo");
    ros::NodeHandle n;

    // create a tf listener
    listener = new tf::TransformListener();

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string detected_persons_left;
    string detected_persons_right;
    string detected_persons_rear;
    string pub_topic_detected_persons;




    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("detected_persons_left", detected_persons_left, string("oops!need param for left"));
    private_node_handle_.param("detected_persons_right", detected_persons_right, string("oops!need param for right"));
    private_node_handle_.param("detected_persons_rear", detected_persons_rear, string("oops!need param for rear"));

    // For SPENCER DetectedPersons message
    private_node_handle_.param("world_scale", worldScale, 1.0); // default for ASUS sensors
    private_node_handle_.param("detection_id_increment", detection_id_increment, 1);
    private_node_handle_.param("detection_id_offset",    detection_id_offset, 0);
    private_node_handle_.param("pose_variance",    pose_variance, 0.05);
    private_node_handle_.param("overlap_thresh",    overlap_thresh, 0.50);  //this overlap_thresh is for overlapping detection

    private_node_handle_.param("world_frame", world_frame, string("oops!need param for world frame"));


    current_detection_id = detection_id_offset;


    // Create a subscriber.
    // Name the topic, message queue, callback function with class name, and object containing callback function.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    ros::Subscriber sub_message; //Subscribers have to be defined out of the if scope to have affect.
    Subscriber<DetectedPersons> subscriber_detected_persons_left(n, detected_persons_left.c_str(), 1); subscriber_detected_persons_left.unsubscribe();
    Subscriber<DetectedPersons> subscriber_detected_persons_right(n, detected_persons_right.c_str(),1); subscriber_detected_persons_right.unsubscribe();
    Subscriber<DetectedPersons> subscriber_detected_persons_rear(n, detected_persons_rear.c_str(),1); subscriber_detected_persons_rear.unsubscribe();



    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(sub_message),
                                                       boost::ref(n),
                                                       boost::ref(subscriber_detected_persons_left),
                                                       boost::ref(subscriber_detected_persons_right),
                                                       boost::ref(subscriber_detected_persons_rear));




    //The real queue size for synchronisation is set here.
    sync_policies::ApproximateTime<DetectedPersons,DetectedPersons, DetectedPersons> MySyncPolicy(queue_size);
    MySyncPolicy.setAgePenalty(1000); //set high age penalty to publish older data faster even if it might not be correctly synchronized.


    const sync_policies::ApproximateTime<DetectedPersons, DetectedPersons, DetectedPersons> MyConstSyncPolicy = MySyncPolicy;
    Synchronizer< sync_policies::ApproximateTime<DetectedPersons, DetectedPersons, DetectedPersons> > sync(MyConstSyncPolicy,
                                                                                        subscriber_detected_persons_left,
                                                                                        subscriber_detected_persons_right,
                                                                                        subscriber_detected_persons_rear);
    sync.registerCallback(boost::bind(&yolofusioncallback, _1, _2, _3));


    // Create publishers
    private_node_handle_.param("total_detected_persons", pub_topic_detected_persons, string("/total_detected_persons"));
    pub_detected_persons = n.advertise<frame_msgs::DetectedPersons>(pub_topic_detected_persons, 10, con_cb, con_cb);



    ros::spin();

    return 0;
}




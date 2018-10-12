// ROS includes.
#include <ros/ros.h>
#include <ros/time.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <string.h>
#include <cmath>


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
double overlap_thresh;
string left_camera_frame;
string right_camera_frame;
string rear_camera_frame;
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

bool is_overlapping(const DetectedPerson& dp1,const DetectedPerson& dp2, double thresh)
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

    double dist = (pos1 - pos2).norm();
    //std::cout << "dets: " << dp1.detection_id << ", " << dp2.detection_id <<" dist: " << dist << std::endl;

    bool flag = false;
    if(dist < thresh ){   // set 0.05, if two people is closer than 5 cm, we say they should be exactly the same person
        flag = true;
        //std::cout << "overlapping!" << std::endl;
    }
    else{
        flag = false;
    }

    return flag;
}


void yolofusioncallback(const DetectedPersonsConstPtr &dp_left, const DetectedPersonsConstPtr &dp_right, const DetectedPersonsConstPtr &dp_rear )
{

    // debug output, to show latency from yolo_v3
    ROS_DEBUG_STREAM("current time:" << ros::Time::now());

    if(pub_detected_persons.getNumSubscribers()) {
        frame_msgs::DetectedPersons total_detected_persons;
        //total_detected_persons.header = dp_left->header; // which actually header should we get? We may do a comparsion, to get the latest time stamp, and take its header.
        total_detected_persons.header.stamp = ros::Time::now();
        total_detected_persons.header.frame_id = world_frame;

        transfer_detected_persons_to_world_cord(dp_left,total_detected_persons,left_camera_frame);  // this function will write the detected_persons into total_detected_persons
        transfer_detected_persons_to_world_cord(dp_right,total_detected_persons, right_camera_frame);
        transfer_detected_persons_to_world_cord(dp_rear,total_detected_persons, rear_camera_frame);

        // remove the overlapping detection
        frame_msgs::DetectedPersons fused_detected_persons;
        fused_detected_persons.header = total_detected_persons.header;
        for(int i = 0; i < total_detected_persons.detections.size();i++)
        {
            DetectedPerson detected_person(total_detected_persons.detections[i]);
            bool overlapping_flag = false;
            for(int j=0; j < fused_detected_persons.detections.size(); j++)
            {
                if(is_overlapping(detected_person, fused_detected_persons.detections[j], overlap_thresh))
                {
                    overlapping_flag = true;
                    break;
                }
            }
            if(overlapping_flag == false)
            {
                detected_person.detection_id = current_detection_id;    //set the detection id
                current_detection_id += detection_id_increment;
                fused_detected_persons.detections.push_back(detected_person);
	
            }
        }

        // take the latest time stamp
        /*if(dp_left->header.stamp.toSec() > dp_rear->header.stamp.toSec() )
        {
            if(dp_left->header.stamp.toSec() > dp_right->header.stamp.toSec())
                fused_detected_persons.header.stamp = dp_left->header.stamp;
            else
                fused_detected_persons.header.stamp = dp_right->header.stamp;
        }
        else
        {
            if(dp_rear->header.stamp.toSec() > dp_right->header.stamp.toSec())
                fused_detected_persons.header.stamp = dp_rear->header.stamp;
            else
                fused_detected_persons.header.stamp = dp_right->header.stamp;
        }
        fused_detected_persons.header.stamp = ros::Time::now();
        fused_detected_persons.header.frame_id = world_frame;*/ 
        // Publish
        pub_detected_persons.publish(fused_detected_persons);
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
    private_node_handle_.param("pose_variance", pose_variance, 0.05);

    private_node_handle_.param("overlap_thresh", overlap_thresh, 0.05);

    private_node_handle_.param("left_camera_frame", left_camera_frame, string("oops!need param for left camera frame"));
    private_node_handle_.param("right_camera_frame", right_camera_frame, string("oops!need param for right camera frame"));
    private_node_handle_.param("rear_camera_frame", rear_camera_frame, string("oops!need param for rear camera frame"));
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



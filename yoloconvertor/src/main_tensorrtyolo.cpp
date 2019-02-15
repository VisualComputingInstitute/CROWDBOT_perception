// ROS includes.
#include <ros/ros.h>
#include <iostream>
#include <cmath>
#include <ros/time.h>
#include <image_transport/subscriber_filter.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <string.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>

#include "Vector.h"
#include "Visual.h"


using namespace darknet_ros_msgs;
using namespace message_filters;

ros::Publisher pub_boundingboxes;


//void connectCallback(ros::Subscriber &sub_msg,
//                     image_transport::Subscriber &sub_img,
//                     image_transport::ImageTransport &it){
//    if(!pub_boundingboxes.getNumSubscribers()) {
//        ROS_DEBUG("yoloconvertor: No subscribers. Unsubscribing.");
//        sub_msg.shutdown();
//        sub_img.unsubscribe();

//    } else {
//        ROS_DEBUG("yoloconvertor: New subscribers. Subscribing.");
//        sub_img.subscribe(it,sub_img.getTopic().c_str(),1);
//        it.subscribe();
//            }
//}

void Callback(const sensor_msgs::ImageConstPtr& img)
{
    ROS_INFO("get image");
}


int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "tensorrt_yolo");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string image_topic;
    string boundingboxes;

    //debug


    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("image", image_topic, string("/hardware/video/valeo/rectificationNIKRLeft/PanoramaImage"));
    private_node_handle_.param("bounding_boxes", boundingboxes, string("darknet_ros/bounding_boxes"));



    ROS_DEBUG("yoloconvertor: Queue size for synchronisation is set to: %i", queue_size);

    // Image transport handle
    image_transport::ImageTransport it(private_node_handle_);
    ros::Subscriber sub_message; //Subscribers have to be defined out of the if scope to have affect.
    image_transport::Subscriber subscriber_img = it.subscribe(image_topic.c_str(),queue_size,Callback);

//    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
//                                                       boost::ref(sub_message),
//                                                       boost::ref(subscriber_img),
//                                                       boost::ref(it));



//    //The real queue size for synchronisation is set here.
//    sync_policies::ApproximateTime<Image> MySyncPolicy(queue_size);

//    const sync_policies::ApproximateTime< Image> MyConstSyncPolicy = MySyncPolicy;
//    Synchronizer< sync_policies::ApproximateTime<Image> > sync(MyConstSyncPolicy,
//                                                               subscriber_img);

//    // Decide which call back should be used.
//    if(strcmp(ground_plane.c_str(), "") == 0) {
//        ROS_FATAL("ground_plane: need ground_plane to produce 3d information");
//    } else {
//        sync.registerCallback(boost::bind(&Callback, _1));
//    }

    // Create publishers
    private_node_handle_.param("bounding_boxs", boundingboxes, string("/bounding_boxs"));
    pub_boundingboxes = n.advertise<darknet_ros_msgs::BoundingBoxes>(boundingboxes, 10);/* con_cb, con_cb)*/;



    ros::spin();

    return 0;
}

// ROS includes.
#include <ros/ros.h>
#include <ros/time.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>

#include <Eigen/Geometry>

#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include "rwth_visual_odometry/VisualOdometry.h"

ros::Publisher pub_message;

void callback(const nav_msgs::OdometryConstPtr& odom) {
    Eigen::Affine3d eigenTr;
    tf::poseMsgToEigen(odom->pose.pose, eigenTr);

    // transpose, just to have P=[R|t] format
    Eigen::Matrix4d m = eigenTr.matrix();
    ROS_WARN_STREAM("m: \n" << m ); 

    // for ros coordinate system to tracker coordinate system, rotate 270° around y and 90° around (then) x counter-clock-wise
    Eigen::Affine3d coordinateTransform = Eigen::Affine3d::Identity();
    coordinateTransform.rotate( Eigen::AngleAxisd(270.0 / 180.0 * M_PI, Eigen::Vector3d::UnitY() ) );
    coordinateTransform.rotate( Eigen::AngleAxisd(90.0  / 180.0 * M_PI, Eigen::Vector3d::UnitX() ) );

    Eigen::Matrix4d ros2tracker_R = coordinateTransform.matrix();
    //ros2tracker_R << 0,-1,0,0, 0,0,-1,0, 1,0,0,0, 0,0,0,1;

    ROS_WARN_STREAM("R: \n" << ros2tracker_R ); 

    Eigen::Affine3d rotatedOdometry(ros2tracker_R * m);

    ROS_WARN_STREAM("rotatedOdometry: \n" << rotatedOdometry.matrix() ); 

    // and mirror at x-z-plane (turns a +0 in a -0 in t and does sth. to R, but still...)
    Eigen::Matrix3d A_rh = rotatedOdometry.linear();
    Eigen::Vector3d b_rh = rotatedOdometry.translation();

    Eigen::Matrix3d S_y; S_y << 1, 0, 0,   0, -1, 0,  0, 0, 1;

    Eigen::Matrix3d A_lh = S_y * A_rh * S_y;
    Eigen::Vector3d b_lh = S_y * b_rh;

    Eigen::Affine3d reflectedOdometry;
    reflectedOdometry.linear() = A_lh;
    reflectedOdometry.translation() = b_lh;

    ROS_WARN_STREAM("reflectedOdometry: \n" << reflectedOdometry.matrix() ); 


    //DEBUG
    /*for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(i==j){
                m(i,j) = 1;
            }
            else{
                m(i,j) = 0;
            }
        }
    }*/
    // switch y,z in t
    /*float temp_val = m(3,1);
    m(3,1) = m(3,2);
    m(3,2) = temp_val;

    // switch y,z in R
    temp_val = m(2,2);
    m(2,2) = m(1,1);
    m(1,1) = temp_val;

    temp_val = m(2,0);
    m(2,0) = m(1,0);
    m(1,0) = temp_val;

    temp_val = m(0,2);
    m(0,2) = m(0,1);
    m(0,1) = temp_val;

    temp_val = m(1,2);
    m(1,2) = m(2,1);
    m(2,1) = temp_val;

    // switch x,z in t
    temp_val = m(3,0);
    m(3,0) = m(3,2);
    m(3,2) = temp_val;

    // switch x,z in R
    temp_val = m(2,2);
    m(2,2) = m(0,0);
    m(0,0) = temp_val;

    temp_val = m(2,0);
    m(2,0) = m(0,2);
    m(0,2) = temp_val;

    temp_val = m(1,0);
    m(1,0) = m(1,2);
    m(1,2) = temp_val;

    temp_val = m(0,1);
    m(0,1) = m(2,1);
    m(2,1) = temp_val;*/

    // negate x in t
    //m(3,0) = -1.0*m(3,0);

   // "negate" x in R (roatate around ? 180°)
   //m(2,0) = -1.0*m(2,0);
   //m(0,0) = -1.0*m(0,0);
   //m(1,1) = -1.0*m(1,1);
   //m(0,2) = -1.0*m(0,2);
    
    /*temp_val = m(3,0);
    m(3,0) = m(3,2);
    m(3,2) = temp_val;
    //DEBUG-END*/


    // transpose for correct array-filling below 
    Eigen::Matrix4d finalOdometry = reflectedOdometry.matrix().transpose();

    //ROS_WARN_STREAM("R*m^-1: \n" << mi );

    /*for(int i = 0; i < 4*4; i++){
        printf("%.2f ",mi.data()[i]);
        if((i+1)%4 == 0)
        printf("\n");
    }*/

    rwth_visual_odometry::VisualOdometry fovis_info_msg;
    fovis_info_msg.header = odom->header;
    fovis_info_msg.motion_estimate_valid = true;
    fovis_info_msg.transformation_matrix.resize(4*4);
    for(int i = 0; i < 4*4; i++)
        fovis_info_msg.transformation_matrix[i] = finalOdometry.data()[i];
     pub_message.publish(fovis_info_msg);
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(ros::Subscriber &sub_odom,
                     ros::NodeHandle &n,
                     std::string odom_topic){
    if(!pub_message.getNumSubscribers()) {
        ROS_DEBUG("Odom: No subscribers. Unsubscribing.");
        sub_odom.shutdown();
    } else {
        ROS_DEBUG("Odom: New subscribers. Subscribing.");
        sub_odom = n.subscribe(odom_topic.c_str(), 1, &callback);
    }
}

int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "odom2visual");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    std::string odom_topic;
    std::string pub_topic;

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("odom", odom_topic, std::string("/odom"));

    // Create a subscriber.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    ros::Subscriber sub_odom; //Subscribers have to be defined out of the if scope to have effect.

    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(sub_odom),
                                                       boost::ref(n),
                                                       odom_topic);

    private_node_handle_.param("motion_parameters", pub_topic, std::string("/spencer/sensors/visual_odometry/motion_matrix"));
    pub_message = n.advertise<rwth_visual_odometry::VisualOdometry>(pub_topic.c_str(), 10, con_cb, con_cb);

    ros::spin();

    return 0;
}

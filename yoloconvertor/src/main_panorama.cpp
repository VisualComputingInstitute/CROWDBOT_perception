// ROS includes.
#include <ros/ros.h>


#include <cmath>
#include <ros/time.h>
#include <sensor_msgs/CameraInfo.h>
#include <nav_msgs/OccupancyGrid.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <string.h>
#include <stdio.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <rwth_perception_people_msgs/GroundPlane.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <frame_msgs/DetectedPersons.h>

#include "Matrix.h"
#include "Vector.h"
#include "PanoramaCameraModel.h"

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace rwth_perception_people_msgs;
using namespace darknet_ros_msgs;
using namespace cv_bridge;

tf::TransformListener* listener;


const double eps(1e-5);

ros::Publisher pub_detected_persons;
double worldScale; // for computing 3D positions from BBoxes
int detection_id_increment, detection_id_offset, current_detection_id; // added for multi-sensor use in SPENCER
double pose_variance; // used in output frame_msgs::DetectedPerson.pose.covariance

nav_msgs::OccupancyGrid oc_map;


extern mira::camera::PanoramaCameraIntrinsic panorama_intrinsic;// the defination and all hardcode parameter is in PanoramaCamearIntrinsic.h




// this call back only get called once, and get the map.
void map_callback(const nav_msgs::OccupancyGridConstPtr& ogptr)
{
    oc_map = *ogptr;
}




void intersectPlane(const Vector<double>& gp, double gpd, const Vector<double>& ray1, const Vector<double>& ray2, Vector<double>& point)
{
    Vector<double> diffRay;
    diffRay = ray1;
    diffRay -= ray2;

    double den = DotProduct(gp, diffRay);
    double t = (DotProduct(gp, ray1) + gpd) / den;

    point = ray1;
    diffRay = (ray2);
    diffRay -= (ray1);
    diffRay *= t;
    point += diffRay;
}

void calc3DPosFromBBox( const Vector<double>& GPN_, double GPD_, double x, double y, double w, double h, double ConvertScale, Vector<double>& pos3D)
{
    // bottom_center is point of the BBOX
    Vector<double> bottom_center(3, 1.0);
    bottom_center(0) = x + w/2.0;
    bottom_center(1) = y + h;
    // Backproject through base point
    Vector<double> ray_bot_center_1(3,0.0);
    Vector<double> ray_bot_center_2;

    mira::camera::PanoramaCameraModel::projectPixelTo3dRay(bottom_center, ray_bot_center_2, panorama_intrinsic);

    // Intersect with ground plane
    Vector<double> gpPointCenter;
    intersectPlane(GPN_, GPD_, ray_bot_center_1, ray_bot_center_2, gpPointCenter);

    // Compute 3D Position of BBOx
    double posX = gpPointCenter(0) * ConvertScale;
    double posY = gpPointCenter(1) * ConvertScale;
    double posZ = gpPointCenter(2) * ConvertScale;

    pos3D.setSize(3);
    pos3D(0) = posX;
    pos3D(1) = posY;
    pos3D(2) = posZ;
}

// here all ray_1 somehow are totally useless. Originally they are used to transfer the detection's frame from
// camera to world. But now we do this in another node.
// so here we simply set all ray_1 as 0. means we computer the result in camera frame:
double calcHeightfromRay( const Vector<double>& GPN_, double GPD_, const BoundingBox& curBox )
{

        double x = curBox.xmin;
        double y = curBox.ymin;
        double w = curBox.xmax - x;
        double h = curBox.ymax - y;

        // bottom_left and bottom_right are the point of the BBOX
        Vector<double> bottom_left(3, 1.0);
        bottom_left(0) = x + w/2.0;
        bottom_left(1) = y + h;

        Vector<double> bottom_right(3, 1.0);
        bottom_right(0) = x + w;
        bottom_right(1) = y + h;

        Vector<double> ray_bot_left_1(3,0.0);
        Vector<double> ray_bot_left_2;

        Vector<double> ray_bot_right_1(3,0.0);
        Vector<double> ray_bot_right_2;

        // Backproject through base point
        mira::camera::PanoramaCameraModel::projectPixelTo3dRay(bottom_left, ray_bot_left_2, panorama_intrinsic); //here ray_upper_center is a unit vector
        mira::camera::PanoramaCameraModel::projectPixelTo3dRay(bottom_right, ray_bot_right_2, panorama_intrinsic); //here ray_upper_center is a unit vector
//        getRay(bottom_left, ray_bot_left_1, ray_bot_left_2);
//        getRay(bottom_right, ray_bot_right_1, ray_bot_right_2);

        Vector<double> gpPointLeft;
        Vector<double> gpPointRight;

        intersectPlane(GPN_, GPD_, ray_bot_left_1, ray_bot_left_2, gpPointLeft);
        intersectPlane(GPN_, GPD_, ray_bot_right_1, ray_bot_right_2, gpPointRight);

        // Find top point
        Vector<double> ray_top_1(3,0.0);
        Vector<double> ray_top_2;

        Vector<double> aux(3, 1.0);
        aux(0) = x + w/2.0; //FIXED: move top point in middle, s.t. height is computed correctly lateron
        aux(1) = y;

        mira::camera::PanoramaCameraModel::projectPixelTo3dRay(aux, ray_top_2, panorama_intrinsic); //here ray_upper_center is a unit vector

        //getRay(aux, ray_top_1, ray_top_2);

        // Vertical plane through base points + normal
        Vector<double> point3;
        point3 = gpPointLeft;
        point3 -= (GPN_);
        Vector<double> vpn(3,0.0);
        Vector<double> diffGpo1Point3;
        Vector<double> diffGpo2Point3;

        diffGpo1Point3 = gpPointLeft;
        diffGpo1Point3 -=(point3);

        diffGpo2Point3 = gpPointRight;
        diffGpo2Point3 -= point3;

        vpn = cross(diffGpo1Point3,diffGpo2Point3);
        double vpd = (-1.0)*DotProduct(vpn, point3);  // here may 1000*!

        Vector<double> gpPointTop;
        intersectPlane(vpn, vpd, ray_top_1, ray_top_2, gpPointTop);

        // Results
        gpPointTop -= gpPointLeft;

        // Compute Size
        double dSize = gpPointTop.norm();

        return dSize;
}



void yoloConvertorCallback(const BoundingBoxesConstPtr &boxes,const GroundPlaneConstPtr &gp/*,const ImageConstPtr &color*/)
{
    ROS_DEBUG("entered yoloconvertor cb");
    // debug output, to show latency from yolo_v3
    //ROS_DEBUG_STREAM("time stamep of input image:" << boxes->header);
    //ROS_DEBUG_STREAM("current time:" << ros::Time::now());
    //ROS_DEBUG_STREAM("-----------------------------------------");


    // Get GP
    Vector<double> GPN(3, (double*) &gp->n[0]);
    double GPd = ((double) gp->d);
    //std::string camera_frame_id = color->header.frame_id;
    std::string camera_frame_id = boxes->image_header.frame_id;
    std::string gp_frame_id = gp->header.frame_id;
    if(camera_frame_id != gp_frame_id){

        geometry_msgs::Vector3Stamped normalVectorStamped;
        geometry_msgs::Vector3Stamped normalVectorStampedCamera;

        geometry_msgs::PointStamped distancePointStamped;
        geometry_msgs::PointStamped distancePointStampedCamera;

        normalVectorStamped.header.frame_id = gp_frame_id;
        normalVectorStamped.header.stamp = ros::Time();
        normalVectorStamped.vector.x = GPN[0];
        normalVectorStamped.vector.y = GPN[1];
        normalVectorStamped.vector.z = GPN[2];

        distancePointStamped.header.frame_id = camera_frame_id;
        distancePointStamped.header.stamp = ros::Time();
        distancePointStamped.point.x = 0.0;
        distancePointStamped.point.y = 0.0;
        distancePointStamped.point.z = 0.0;

        try {
            listener->waitForTransform(camera_frame_id, gp_frame_id, ros::Time(), ros::Duration(1.0));
            listener->transformVector(camera_frame_id, normalVectorStamped, normalVectorStampedCamera);
            listener->waitForTransform(gp_frame_id, camera_frame_id, ros::Time(), ros::Duration(1.0));
            listener->transformPoint(gp_frame_id, distancePointStamped, distancePointStampedCamera);

            GPN[0] = normalVectorStampedCamera.vector.x;
            GPN[1] = normalVectorStampedCamera.vector.y;
            GPN[2] = normalVectorStampedCamera.vector.z;
            GPd = distancePointStampedCamera.point.z;
        }
        catch(tf::TransformException ex) {
            ROS_WARN_THROTTLE(20.0, "Failed transform lookup in yoloconvertor_pinhole -- maybe the RGB-D drivers are not yet running!? Reason: %s. Message will re-appear within 20 seconds.", ex.what());
            return;
        }
    }

    //
    // get the transformation from camera frame to map frame
    //
    tf::StampedTransform camera2map; //this transform do transformation from camera frame to map frame.
    string map_frame_id = oc_map.header.frame_id;
    ros::Time tracked_time(boxes->image_header.stamp);
    try {
        listener->waitForTransform(map_frame_id, camera_frame_id, tracked_time, ros::Duration(1.0));
        listener->lookupTransform(map_frame_id, camera_frame_id, tracked_time, camera2map);  //from camera to map
    }
    catch (tf::TransformException ex){
       ROS_WARN_THROTTLE(20.0, "Failed transform lookup from camera frame to map frame. The map data is empty:%s", oc_map.data.empty() ? "true" : "false", ex.what());
    }


    //
    // Now create 3D coordinates for SPENCER DetectedPersons msg
    //
    if(pub_detected_persons.getNumSubscribers()) {
        frame_msgs::DetectedPersons detected_persons;
        detected_persons.header = boxes->image_header;


        for(unsigned int i=0;i<(boxes->bounding_boxes.size());i++)
        {
            BoundingBox curBox(boxes->bounding_boxes[i]);
            float x = curBox.xmin;
            float y = curBox.ymin;
            float width = curBox.xmax - x;
            float height = curBox.ymax - y;

            Vector<double> pos3D;
            calc3DPosFromBBox( GPN, GPd, x, y, width, height, worldScale, pos3D);

            // DetectedPerson for SPENCER
            frame_msgs::DetectedPerson detected_person;
            detected_person.modality = frame_msgs::DetectedPerson::MODALITY_GENERIC_MONOCULAR_VISION;
            // use the probability from yolo detector
            detected_person.confidence = curBox.probability;
            detected_person.pose.pose.position.x = pos3D(0);
            detected_person.pose.pose.position.y = pos3D(1);
            detected_person.pose.pose.position.z = pos3D(2);
            detected_person.pose.pose.orientation.w = 1.0;


            // compute this bounding box's height
            double detection_height = calcHeightfromRay(GPN, GPd, curBox);
            detected_person.height = detection_height;

            const double LARGE_VARIANCE = 999999999;
            detected_person.pose.covariance[0*6 + 0] = pose_variance;
            detected_person.pose.covariance[1*6 + 1] = pose_variance; // up axis (since this is in sensor frame!)
            detected_person.pose.covariance[2*6 + 2] = pose_variance;
            detected_person.pose.covariance[3*6 + 3] = LARGE_VARIANCE;
            detected_person.pose.covariance[4*6 + 4] = LARGE_VARIANCE;
            detected_person.pose.covariance[5*6 + 5] = LARGE_VARIANCE;

            detected_person.detection_id = current_detection_id;
            current_detection_id += detection_id_increment;


            detected_person.bbox_x = x;
            detected_person.bbox_y = y;
            detected_person.bbox_w = width;
            detected_person.bbox_h = height;

            // now check if this bounding box can exsit in the map
            // 1. transform this pos3D to map frame;
            tf::Vector3 pos3D_incam(detected_person.pose.pose.position.x,detected_person.pose.pose.position.y, detected_person.pose.pose.position.z);
            tf::Vector3 pos3D_inmap = camera2map*pos3D_incam;

            // 2. get the index for the map. Since the map's left corner (0,0 cordinate in map image) is not exactly the origin in map frame, it is biasd.
            double map_resolution = oc_map.info.resolution;
            tf::Vector3 map_origin(oc_map.info.origin.position.x, oc_map.info.origin.position.y, 0.0);  // this is in meter unit
            //oc_map.info.origin.orientation;  // we assume no rotation bias, since most systm ignore this map rotation
            pos3D_inmap -= map_origin;
            unsigned int map_ind_x = static_cast<unsigned int>(pos3D_inmap.x()*1.0/map_resolution);
            unsigned int map_ind_y = static_cast<unsigned int>(pos3D_inmap.y()*1.0/map_resolution);

            //3. see if it is occupanied in map
            auto grid_value = oc_map.data[map_ind_x+map_ind_y*oc_map.info.width]; //row-major
            if(grid_value == 100) // now with the map from map_serve, 100 is occupied
            {
//                cout<<"hey grid_value is 100, id is "<<current_detection_id<<endl;
//                cout<<"x= "<<map_ind_x/10<<" y= "<<map_ind_y/10<<endl;
                continue;   // do not pass the check, directly go to the next bounding box.
            }

            // additional nan check
            if(!isnan(detected_person.pose.pose.position.x) && !isnan(detected_person.pose.pose.position.y) && !isnan(detected_person.pose.pose.position.z)){
                detected_persons.detections.push_back(detected_person);
            }else{
                std::cout << "A detection has been discarded because of nan values!" << std::endl;
                ROS_DEBUG("A detection has been discarded because of nan values in panorama conversion!");
            }


        }

        // Publish
        pub_detected_persons.publish(detected_persons);

    }
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(ros::Subscriber &sub_msg,
                     ros::NodeHandle &n,
                     string gp_topic,
                     Subscriber<GroundPlane> &sub_gp,
                     Subscriber<BoundingBoxes> &sub_boxes){
    if(!pub_detected_persons.getNumSubscribers()) {
        ROS_DEBUG("yoloconvertor: No subscribers. Unsubscribing.");
        sub_msg.shutdown();
        sub_gp.unsubscribe();
        sub_boxes.unsubscribe();
    } else {
        ROS_DEBUG("yoloconvertor: New subscribers. Subscribing.");
        if(strcmp(gp_topic.c_str(), "") == 0) {
             ROS_FATAL("ground_plane: need ground_plane to produce 3d information");
        }
        sub_gp.subscribe();
        sub_boxes.subscribe();
    }

}

int main(int argc, char **argv)
{

    // Set up ROS.
    ros::init(argc, argv, "convert_yolo_pano");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string ground_plane;
    //string pano_image;
    string pub_topic_detected_persons;
    string boundingboxes;
    string map_topic;

    //debug
    string pub_topic_result_image;


    listener = new tf::TransformListener();

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("ground_plane", ground_plane, string(""));
    private_node_handle_.param("bounding_boxes", boundingboxes, string("darknet_ros/bounding_boxes"));

    // For SPENCER DetectedPersons message
    private_node_handle_.param("world_scale", worldScale, 1.0); // default for ASUS sensors
    private_node_handle_.param("detection_id_increment", detection_id_increment, 1);
    private_node_handle_.param("detection_id_offset",    detection_id_offset, 0);
    private_node_handle_.param("pose_variance",    pose_variance, 0.05);
    current_detection_id = detection_id_offset;

    //map
    private_node_handle_.param("map", map_topic, string("/map"));
    ros::Subscriber sub_map = n.subscribe(map_topic, 1, map_callback);



    // Create a subscriber.
    // Name the topic, message queue, callback function with class name, and object containing callback function.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    ros::Subscriber sub_message; //Subscribers have to be defined out of the if scope to have affect.
    Subscriber<GroundPlane> subscriber_ground_plane(n, ground_plane.c_str(), 1); subscriber_ground_plane.unsubscribe();
    Subscriber<BoundingBoxes> subscriber_bounding_boxes(n,boundingboxes.c_str(),1); subscriber_bounding_boxes.unsubscribe();



    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(sub_message),
                                                       boost::ref(n),
                                                       ground_plane,
                                                       boost::ref(subscriber_ground_plane),
                                                       boost::ref(subscriber_bounding_boxes));






    //The real queue size for synchronisation is set here.
    //sync_policies::ApproximateTime<BoundingBoxes, GroundPlane, Image> MySyncPolicy(queue_size);
    sync_policies::ApproximateTime<BoundingBoxes, GroundPlane> MySyncPolicy(queue_size);
    MySyncPolicy.setAgePenalty(1000); //set high age penalty to publish older data faster even if it might not be correctly synchronized.


    const sync_policies::ApproximateTime<BoundingBoxes, GroundPlane> MyConstSyncPolicy = MySyncPolicy;
    Synchronizer< sync_policies::ApproximateTime<BoundingBoxes, GroundPlane> > sync(MyConstSyncPolicy, subscriber_bounding_boxes, subscriber_ground_plane);

    // Decide which call back should be used.
    if(strcmp(ground_plane.c_str(), "") == 0) {
        ROS_FATAL("ground_plane: need ground_plane to produce 3d information");
    } else {
        //sync.registerCallback(boost::bind(&yoloConvertorCallback, _1, _2, _3));
        sync.registerCallback(boost::bind(&yoloConvertorCallback, _1, _2));
    }

    // Create publishers
    private_node_handle_.param("detected_persons", pub_topic_detected_persons, string("/detected_persons"));
    pub_detected_persons = n.advertise<frame_msgs::DetectedPersons>(pub_topic_detected_persons, 10, con_cb, con_cb);


    //double min_expected_frequency, max_expected_frequency;
    //private_node_handle_.param("min_expected_frequency", min_expected_frequency, 8.0);
    //private_node_handle_.param("max_expected_frequency", max_expected_frequency, 100.0);

    /*pub_detected_persons.setExpectedFrequency(min_expected_frequency, max_expected_frequency);
    pub_detected_persons.setMaximumTimestampOffset(0.3, 0.1);
    pub_detected_persons.finalizeSetup();*/

    ros::spin();

    return 0;
}



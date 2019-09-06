// ROS includes.
#include <ros/ros.h>

#include <ros/time.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/OccupancyGrid.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <string.h>

#include <cv_bridge/cv_bridge.h>

#include <rwth_perception_people_msgs/GroundPlane.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <frame_msgs/DetectedPersons.h>

#include "Matrix.h"
#include "Vector.h"
//#include "MapFunctions.hpp"

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace rwth_perception_people_msgs;
using namespace darknet_ros_msgs;

tf::TransformListener* listener;
//MapFunctions* g_map_func;

//for debug image
//image_transport::Publisher pub_result_image;
//#include <QImage>
//#include <QPainter>
//cv_bridge::CvImagePtr cv_depth_ptr;	// cv_bridge for depth image
cv::Mat img_depth_;


ros::Publisher pub_detected_persons;
double worldScale; // for computing 3D positions from BBoxes

int detection_id_increment, detection_id_offset, current_detection_id; // added for multi-sensor use in SPENCER
double pose_variance; // used in output frame_msgs::DetectedPerson.pose.covariance
double depth_scale; // used in output frame_msgs::DetectedPerson.pose.covariance

const double eps(1e-5);


void getRay(const Matrix<double>& K, const Vector<double>& x, Vector<double>& ray1, Vector<double>& ray2)
{
    Matrix<double> Kinv = K;
    Kinv.inv();

    ray1 = Vector<double>(3, 0.0);

    Matrix<double> rot = Eye<double>(3);
    rot *= Kinv;
    ray2 = rot * x;
    ray2 += ray1;
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

void calc3DPosFromBBox(const Matrix<double>& K, const Vector<double>& GPN_, double GPD_, double x, double y, double w, double h, double ConvertScale, Vector<double>& pos3D)
{
    // bottom_center is point of the BBOX
    Vector<double> bottom_center(3, 1.0);
    bottom_center(0) = x + w/2.0;
    bottom_center(1) = y + h;

    // Backproject through base point
    Vector<double> ray_bot_center_1;
    Vector<double> ray_bot_center_2;
    getRay(K, bottom_center, ray_bot_center_1, ray_bot_center_2);
    
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

// we already has depth from Kinect, can we directly use this depth and the ground plane
// directly get the new person's plane?
double calcHeightfromRay(const Matrix<double>& K, const Vector<double>& GPN_, double GPD_, const BoundingBox& curBox )
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
        getRay(K, bottom_left, ray_bot_left_1, ray_bot_left_2);
        getRay(K, bottom_right, ray_bot_right_1, ray_bot_right_2);

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

        getRay(K, aux, ray_top_1, ray_top_2);

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




void yoloConvertorCallback(const BoundingBoxesConstPtr &boxes, const CameraInfoConstPtr &camera_info,
                              const GroundPlaneConstPtr &gp, const ImageConstPtr &depth, const CameraInfoConstPtr &dep_info)
{
    //if (filter_frame_id!="" && filter_frame_id!=boxes->header.frame_id){
    //    ROS_DEBUG("YoloBoxes have been filtered in YoloConvertor!")
    //    return;
    //}
    // Get GP
    Vector<double> GPN(3, (double*) &gp->n[0]);
    double GPd = ((double) gp->d);

    Matrix<double> K(3,3, (double*)&camera_info->K[0]);
    Matrix<double> K_d(3,3, (double*)&dep_info->K[0]);
    std::string camera_frame_id = camera_info->header.frame_id;
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

    //g_map_func->updateCamera2frameTransform(camera_frame_id,boxes->image_header.stamp,listener);


    //
    // Now create 3D coordinates for SPENCER DetectedPersons msg
    //
    if(pub_detected_persons.getNumSubscribers()) {
        frame_msgs::DetectedPersons detected_persons;
        detected_persons.header = boxes->image_header;
        detected_persons.header.stamp = ros::Time::now();

        //debug image
        //convert depth to rgb image for display
        cv_bridge::CvImagePtr cv_depth_ptr(cv_bridge::toCvCopy(depth,"32FC1"));
        /*if (depth->encoding == "16UC1" || depth->encoding == "32FC1") {
            cv_depth_ptr->image *= 0.001;
        }*/
        //cv_depth_ptr->image *= depth_scale;
        img_depth_ = cv_depth_ptr->image;
        //cv::Mat tmp_depth_mat;
        //img_depth_.cv::Mat::convertTo(tmp_depth_mat,CV_8U);
        //cv::Mat depth_mat_rgb;
        //cv::cvtColor(tmp_depth_mat,depth_mat_rgb,CV_GRAY2RGB);
        //cout<< "depth now image coding "<< depth_mat_rgb.type()<<endl;  //cv8uc3


        for(unsigned int i=0;i<(boxes->bounding_boxes.size());i++)
        {
            BoundingBox curBox(boxes->bounding_boxes[i]);
            float x = curBox.xmin;
            float y = curBox.ymin;
            float width = curBox.xmax - x;
            float height = curBox.ymax - y;

            Vector<double> pos3D;
            calc3DPosFromBBox(K, GPN, GPd, x, y, width, height, worldScale, pos3D);


            // get readings in rectanular region from (registered!) depth image

            // only take 50% box.
            vector<double> vector_depth;
            vector_depth.clear();
            vector_depth.reserve(width*height/2);
            int x_50 = (int)(x+0.25*width);
            int y_50 = (int)(y+0.25*height);
            int height_50 = (int)(0.50*height);
            int width_50 = (int)(0.50*width);
            for (int r = y_50 ; r < y_50 + height_50 ; r++){
                for (int c = x_50 ; c < x_50 + width_50 ; c++) {
                    double depth_value = img_depth_.at<float>(r,c);
                    if((!isfinite(depth_value))||(abs(depth_value)<=eps*abs(depth_value) )) // check if 0 or inf
                    {
                        continue;
                    }
                    else
                    {
                        vector_depth.push_back(depth_value);
                    }
               }
            }
            int len = vector_depth.size();
            float good_pixel_ratio = (float)len / (height_50*width_50);
            bool use_depth_image = true;
            if(good_pixel_ratio <= 0.25 )  // if only less than 25% pixel is good( not 0 or NaN), use ray casting
            {
                use_depth_image = false;
            }
            else{
                use_depth_image = true;
            }
            nth_element( vector_depth.begin(), vector_depth.begin()+len/2,vector_depth.end() );
            double med_depth_50 = vector_depth[len/2];


            // DetectedPerson for SPENCER
            frame_msgs::DetectedPerson detected_person;
            detected_person.modality = frame_msgs::DetectedPerson::MODALITY_GENERIC_MONOCULAR_VISION;
            // use the probability from yolo detector
            detected_person.confidence = curBox.probability;
            detected_person.pose.pose.position.x = pos3D(0);
            detected_person.pose.pose.position.y = pos3D(1);
            if(med_depth_50 < 30 && med_depth_50 > eps && use_depth_image){
                detected_person.pose.pose.position.z = med_depth_50;//min(med_depth_50, pos3D(2));
                detected_person.pose.pose.position.x = (pos3D(0)*med_depth_50)/pos3D(2);
            }else{
                detected_person.pose.pose.position.z = pos3D(2);
            }
            detected_person.pose.pose.orientation.w = 1.0;

            // check map occupancy
            //tf::Vector3 pos3D_incam(detected_person.pose.pose.position.x,detected_person.pose.pose.position.y, detected_person.pose.pose.position.z);
            //if(g_map_func->isPosOccupied(pos3D_incam))
            //    continue;


            // additional nan check
            if(std::isnan(detected_person.pose.pose.position.x) || std::isnan(detected_person.pose.pose.position.y) || std::isnan(detected_person.pose.pose.position.z)){
                ROS_DEBUG("A detection has been discarded because of nan values in standard conversion!");
                continue;
            }
            // additional in front of cam check
            if(detected_person.pose.pose.position.z<0){
                ROS_DEBUG("A detection has been discarded because it was not in front of the camera (z<0)!");
                continue;
             }

            // scale uncertainty (covariance) of detection with increasing distance to camera, also "rotate" accordingly (off-diagonal)
            const double LARGE_VARIANCE = 999999999;
            detected_person.pose.covariance[0*6 + 0] = pose_variance*(1+std::log(1+detected_person.pose.pose.position.z));// /8; // x ("l/r") in sensor frame
            detected_person.pose.covariance[1*6 + 1] = pose_variance*(1+std::log(1+detected_person.pose.pose.position.z)); // y (up axis), in sensor frame!)
            detected_person.pose.covariance[2*6 + 2] = pose_variance*(1+std::log(1+detected_person.pose.pose.position.z));// /2; // z ("depth") in sensor frame
            /*detected_person.pose.covariance[0*6 + 2] = ((detected_person.pose.covariance[0*6 + 0]+detected_person.pose.covariance[2*6 + 2])/2)
                                                        * ((detected_person.pose.pose.position.x)
                                                        / (sqrt(detected_person.pose.pose.position.x*detected_person.pose.pose.position.x
                                                        + detected_person.pose.pose.position.z*detected_person.pose.pose.position.z)));
            detected_person.pose.covariance[2*6 + 0] = detected_person.pose.covariance[0*6 + 2];
            /*detected_person.pose.covariance[0*6 + 0] = pose_variance; // x ("l/r") in sensor frame
            detected_person.pose.covariance[1*6 + 1] = pose_variance; // y (up axis), in sensor frame!)
            detected_person.pose.covariance[2*6 + 2] = pose_variance; // z ("depth") in sensor frame
            detected_person.pose.covariance[0*6 + 2] = 0.0;
            detected_person.pose.covariance[2*6 + 0] = detected_person.pose.covariance[0*6 + 2];*/
            detected_person.pose.covariance[3*6 + 3] = LARGE_VARIANCE;
            detected_person.pose.covariance[4*6 + 4] = LARGE_VARIANCE;
            detected_person.pose.covariance[5*6 + 5] = LARGE_VARIANCE;

            detected_person.detection_id = current_detection_id;
            current_detection_id += detection_id_increment;

            //std::cout << "--detID--" << detected_person.detection_id << std::endl;
            //std::cout << "alpha" << acos(detected_person.pose.covariance[0*6 + 2])/(pose_variance) << std::endl;

            detected_person.bbox_x = x;
            detected_person.bbox_y = y;
            detected_person.bbox_w = width; 
            detected_person.bbox_h = height;


            // compute this bounding box's height
            double detection_height = calcHeightfromRay(K ,GPN, GPd, curBox);
            detected_person.height = detection_height;

            detected_persons.detections.push_back(detected_person);

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
                     Subscriber<CameraInfo> &sub_cam,
		             Subscriber<BoundingBoxes> &sub_boxes,
                     image_transport::SubscriberFilter &sub_dep,
                     message_filters::Subscriber<CameraInfo> &sub_d_cam,
                     image_transport::ImageTransport &it){
    if(!pub_detected_persons.getNumSubscribers()) {
        ROS_DEBUG("yoloconvertor: No subscribers. Unsubscribing.");
        sub_msg.shutdown();
        sub_gp.unsubscribe();
        sub_cam.unsubscribe();
        sub_boxes.unsubscribe();
        sub_dep.unsubscribe();
        sub_d_cam.unsubscribe();
    } else {
        ROS_DEBUG("yoloconvertor: New subscribers. Subscribing.");
        if(strcmp(gp_topic.c_str(), "") == 0) {
             ROS_FATAL("ground_plane: need ground_plane to produce 3d information");
        }
        sub_cam.subscribe();
        sub_gp.subscribe();
    	sub_boxes.subscribe();
        sub_dep.subscribe(it,sub_dep.getTopic().c_str(),1);
        sub_d_cam.subscribe();
    }
}

int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "convert_yolo");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string ground_plane;
    string camera_ns;
    string pub_topic_detected_persons;
    string boundingboxes;

    //debug
    string pub_topic_result_image;

    listener = new tf::TransformListener();

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("camera_namespace", camera_ns, string("/head_xtion"));
    private_node_handle_.param("ground_plane", ground_plane, string(""));
    private_node_handle_.param("bounding_boxes", boundingboxes, string("darknet_ros/bounding_boxes"));

    // For SPENCER DetectedPersons message
    private_node_handle_.param("world_scale", worldScale, 1.0); // default for ASUS sensors
    private_node_handle_.param("detection_id_increment", detection_id_increment, 1);
    private_node_handle_.param("detection_id_offset",    detection_id_offset, 0);
    private_node_handle_.param("pose_variance",    pose_variance, 0.05);
    private_node_handle_.param("depth_scale",    depth_scale, 1.0);
    current_detection_id = detection_id_offset;

    //string image_color = camera_ns + "/hd/image_color_rect";
    string camera_info = camera_ns + "/hd/camera_info";
    string topic_depth_info = camera_ns + "/sd/camera_info";
    string topic_depth_image = camera_ns + "/hd/image_depth_rect";

    //map
    //string map_topic;
    double opt;
    int half_length;
    //private_node_handle_.param("map", map_topic, string("/map"));
    private_node_handle_.param("occupancy_threshold",opt, 75.0);  // from 0 to 100
    private_node_handle_.param("occupancy_check_box_half_length",half_length, int(1));
    //ros::Subscriber sub_map = n.subscribe(map_topic, 1, map_callback);
    //g_map_func = new MapFunctions(n,map_topic,opt,half_length);

    ROS_DEBUG("yoloconvertor: Queue size for synchronisation is set to: %i", queue_size);

    // Image transport handle
    image_transport::ImageTransport it(private_node_handle_);

    // Create a subscriber.
    // Name the topic, message queue, callback function with class name, and object containing callback function.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    ros::Subscriber sub_message; //Subscribers have to be defined out of the if scope to have affect.
    Subscriber<GroundPlane> subscriber_ground_plane(n, ground_plane.c_str(), 10); subscriber_ground_plane.unsubscribe();
    
//    image_transport::SubscriberFilter subscriber_color;
//    subscriber_color.subscribe(it, image_color.c_str(), 1); subscriber_color.unsubscribe();
    Subscriber<CameraInfo> subscriber_camera_info(n, camera_info.c_str(), 10); subscriber_camera_info.unsubscribe();
    Subscriber<BoundingBoxes> subscriber_bounding_boxes(n,boundingboxes.c_str(),5); subscriber_bounding_boxes.unsubscribe();
    image_transport::SubscriberFilter subscriber_depth;
    subscriber_depth.subscribe(it, topic_depth_image.c_str(),1); subscriber_depth.unsubscribe();
    message_filters::Subscriber<CameraInfo> subscriber_depth_info(n, topic_depth_info.c_str(), 10); subscriber_depth_info.unsubscribe();

    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(sub_message),
                                                       boost::ref(n),
                                                       ground_plane,
                                                       boost::ref(subscriber_ground_plane),
                                                       boost::ref(subscriber_camera_info),
                                                       boost::ref(subscriber_bounding_boxes),
                                                       boost::ref(subscriber_depth),
                                                       boost::ref(subscriber_depth_info),
                                                       boost::ref(it));



    //The real queue size for synchronisation is set here.
    sync_policies::ApproximateTime<BoundingBoxes, CameraInfo, GroundPlane, Image, CameraInfo> MySyncPolicy(queue_size);

    const sync_policies::ApproximateTime<BoundingBoxes, CameraInfo, GroundPlane, Image, CameraInfo> MyConstSyncPolicy = MySyncPolicy;
    Synchronizer< sync_policies::ApproximateTime<BoundingBoxes, CameraInfo, GroundPlane, Image, CameraInfo> > sync(MyConstSyncPolicy,
                                                                                        subscriber_bounding_boxes,
                                                                                        subscriber_camera_info,
                                                                                        subscriber_ground_plane,
                                                                                        subscriber_depth,
                                                                                        subscriber_depth_info);

    // Decide which call back should be used.
    if(strcmp(ground_plane.c_str(), "") == 0) {
        ROS_FATAL("ground_plane: need ground_plane to produce 3d information");
    } else {
        sync.registerCallback(boost::bind(&yoloConvertorCallback, _1, _2, _3, _4, _5));
    }

    // Create publishers
    private_node_handle_.param("detected_persons", pub_topic_detected_persons, string("/detected_persons"));
    pub_detected_persons = n.advertise<frame_msgs::DetectedPersons>(pub_topic_detected_persons, 1, con_cb, con_cb);



    ros::spin();

    return 0;
}



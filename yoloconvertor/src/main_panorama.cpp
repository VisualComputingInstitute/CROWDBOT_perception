#include <cmath>
#include <string.h>
#include <stdio.h>

#include <ros/ros.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <frame_msgs/DetectedPersons.h>
#include <rwth_perception_people_msgs/GroundPlane.h>

#include "Matrix.h"
#include "Vector.h"
#include "PanoramaCameraModel.h"

// Tf
std::shared_ptr<tf::TransformListener> tf_;

// Store topic
rwth_perception_people_msgs::GroundPlane::ConstPtr ground_plane_;

// Pub
ros::Publisher detected_persons_pub_;

// Param
double world_scale_; // for computing 3D positions from BBoxes
double pose_variance_; // used in output frame_msgs::DetectedPerson.pose.covariance
double depth_scale_; // used in output frame_msgs::DetectedPerson.pose.covariance

// Detecion ID
int detection_id_increment_;
int detection_id_offset_;
int current_detection_id_;

int prev_darknet_image_seq_ = -1;

extern mira::camera::PanoramaCameraIntrinsic panorama_intrinsic;// the defination and all hardcode parameter is in PanoramaCamearIntrinsic.h


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

void calc3DPosFromBBox(const Vector<double>& GPN_,
                       double GPD_,
                       const darknet_ros_msgs::BoundingBox& box,
                       double ConvertScale,
                       Vector<double>& pos3D)
{
    const double x = box.xmin;
    const double y = box.ymin;
    const double w = box.xmax - x;
    const double h = box.ymax - y;

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
double calcHeightfromRay( const Vector<double>& GPN_, double GPD_, const darknet_ros_msgs::BoundingBox& curBox )
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

void groundPlaneCallback(const rwth_perception_people_msgs::GroundPlane::ConstPtr& msg)
{
    ground_plane_ = msg;
}

bool checkReady()
{
    if (!ground_plane_)
    {
        ROS_WARN("Missing ground plane.");
        return false;
    }

    return true;
}

bool getGroundPlane(const ros::Time& stamp,
                    const std::string& camera_frame,
                    Vector<double>& normal,
                    double& d)
{
    normal = Vector<double>(3, (double*) &ground_plane_->n[0]);
    d = (double) ground_plane_->d;

    const std::string gp_frame_id = ground_plane_->header.frame_id;

    if (camera_frame == gp_frame_id)
        return true;

    geometry_msgs::Vector3Stamped normalVectorStamped;
    geometry_msgs::Vector3Stamped normalVectorStampedCamera;

    geometry_msgs::PointStamped distancePointStamped;
    geometry_msgs::PointStamped distancePointStampedCamera;

    normalVectorStamped.header.frame_id = gp_frame_id;
    normalVectorStamped.header.stamp = stamp;
    normalVectorStamped.vector.x = normal[0];
    normalVectorStamped.vector.y = normal[1];
    normalVectorStamped.vector.z = normal[2];

    distancePointStamped.header.frame_id = camera_frame;
    distancePointStamped.header.stamp = stamp;
    distancePointStamped.point.x = 0.0;
    distancePointStamped.point.y = 0.0;
    distancePointStamped.point.z = 0.0;

    try
    {
        tf_->waitForTransform(camera_frame, gp_frame_id, stamp, ros::Duration(1.0));
        tf_->transformVector(camera_frame, normalVectorStamped, normalVectorStampedCamera);
        // tf_->waitForTransform(gp_frame_id, camera_frame, stamp, ros::Duration(1.0));
        tf_->transformPoint(gp_frame_id, distancePointStamped, distancePointStampedCamera);

        normal[0] = normalVectorStampedCamera.vector.x;
        normal[1] = normalVectorStampedCamera.vector.y;
        normal[2] = normalVectorStampedCamera.vector.z;
        d = distancePointStampedCamera.point.z;
    }
    catch(tf::TransformException ex)
    {
        ROS_WARN_THROTTLE(20.0, "Failed transform lookup in yoloconvertor_standard. Reason: %s. Message will re-appear within 20 seconds.", ex.what());
        return false;
    }

    return true;
}

void bboxCallback(const darknet_ros_msgs::BoundingBoxes::ConstPtr& msg)
{
    if (detected_persons_pub_.getNumSubscribers() == 0)
        return;

    if (msg->image_header.seq == prev_darknet_image_seq_)
        return;
    else
        prev_darknet_image_seq_ = msg->image_header.seq;

    if (!checkReady())
        return;

    // Ground plane
    Vector<double> GPN;
    double GPd;
    if (!getGroundPlane(msg->image_header.stamp, msg->image_header.frame_id, GPN, GPd))
    {
        ROS_WARN("Could not get groundplane. Discard all detections.");
        return;
    }

    // Published msg
    frame_msgs::DetectedPersons detected_persons;
    detected_persons.header = msg->image_header;

    for (const darknet_ros_msgs::BoundingBox& box : msg->bounding_boxes)
    {
        // Skip bounding box that is close to boarder, since the distortation is too large.
        const int boarder_margin = 100;
        const int mid_point_x = box.xmin + (box.xmax - box.xmin);
        if (mid_point_x < boarder_margin ||
                mid_point_x > panorama_intrinsic.width - boarder_margin)
        {
            // ROS_WARN("Discard detection close to image boarder of panoramaic cameras.");
            continue;
        }

        Vector<double> pos3D;
        calc3DPosFromBBox(GPN, GPd, box, world_scale_, pos3D);

        // DetectedPerson for SPENCER
        frame_msgs::DetectedPerson detected_person;
        detected_person.modality = frame_msgs::DetectedPerson::MODALITY_GENERIC_MONOCULAR_VISION;
        detected_person.confidence = box.probability;
        detected_person.pose.pose.position.x = pos3D(0);
        detected_person.pose.pose.position.y = pos3D(1);
        detected_person.pose.pose.position.z = pos3D(2);
        detected_person.pose.pose.orientation.w = 1.0;

        // additional nan check
        if (std::isnan(detected_person.pose.pose.position.x)
               || std::isnan(detected_person.pose.pose.position.y)
               || std::isnan(detected_person.pose.pose.position.z))
        {
            ROS_WARN("A detection has been discarded because of nan values in standard conversion!");
            continue;
        }

        // additional in front of cam check
        if (detected_person.pose.pose.position.z < 0)
        {
            ROS_WARN("A detection has been discarded because it was not in front of the camera (z<0)!");
            continue;
        }

        // scale uncertainty (covariance) of detection with increasing distance to camera, also "rotate" accordingly (off-diagonal)
        const double LARGE_VARIANCE = 999999999;
        const double var_tmp = pose_variance_ * min(detected_person.pose.pose.position.z, 15.0) / 2;

        detected_person.pose.covariance[0*6 + 0] = var_tmp;
        detected_person.pose.covariance[1*6 + 1] = var_tmp;
        detected_person.pose.covariance[2*6 + 2] = var_tmp;
        detected_person.pose.covariance[3*6 + 3] = LARGE_VARIANCE;
        detected_person.pose.covariance[4*6 + 4] = LARGE_VARIANCE;
        detected_person.pose.covariance[5*6 + 5] = LARGE_VARIANCE;

        detected_person.detection_id = current_detection_id_;
        current_detection_id_ += detection_id_increment_;

        detected_person.bbox_x = box.xmin;
        detected_person.bbox_y = box.ymin;
        detected_person.bbox_w = box.xmax - box.xmin;
        detected_person.bbox_h = box.ymax - box.ymin;

        // compute this bounding box's height
        detected_person.height = calcHeightfromRay(GPN, GPd, box);

        detected_persons.detections.push_back(detected_person);
    }

    // Publish
    detected_persons_pub_.publish(detected_persons);
}

int main(int argc, char **argv)
{
    // ROS
    ros::init(argc, argv, "convert_yolo_pano");
    ros::NodeHandle nh, nh_private("~");

    // Params (for SPENCER DetectedPersons message)
    nh_private.param("world_scale", world_scale_, 1.0); // default for ASUS sensors
    nh_private.param("detection_id_increment", detection_id_increment_, 1);
    nh_private.param("detection_id_offset",    detection_id_offset_, 0);
    nh_private.param("pose_variance",    pose_variance_, 0.05);
    current_detection_id_ = detection_id_offset_;

    // Publishers
    std::string pub_topic_detected_persons;
    nh_private.param("detected_persons", pub_topic_detected_persons, string("/detected_persons"));
    detected_persons_pub_ = nh.advertise<frame_msgs::DetectedPersons>(pub_topic_detected_persons, 1);

    // Tf
    tf_ = std::make_shared<tf::TransformListener>();

    // Subscribers
    std::string ground_plane;
    nh_private.param("ground_plane", ground_plane, std::string(""));
    ros::Subscriber ground_plane_sub = nh.subscribe(ground_plane.c_str(), 1, groundPlaneCallback);

    std::string boundingboxes;
    nh_private.param("bounding_boxes", boundingboxes, string("darknet_ros/bounding_boxes"));
    ros::Subscriber bbox_sub = nh.subscribe(boundingboxes.c_str(), 1, bboxCallback);

    ros::spin();

    return 0;
}



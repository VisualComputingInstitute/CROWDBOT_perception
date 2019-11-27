#include "Matrix.h"
#include "Vector.h"

#include <string.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/CameraInfo.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <rwth_perception_people_msgs/GroundPlane.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <frame_msgs/DetectedPersons.h>

// Tf
std::shared_ptr<tf::TransformListener> tf_;

// Store topic
sensor_msgs::CameraInfo::ConstPtr camera_info_, depth_camera_info_;
rwth_perception_people_msgs::GroundPlane::ConstPtr ground_plane_;
sensor_msgs::Image::ConstPtr depth_;

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

void calc3DPosFromBBox(const Matrix<double>& K,
                       const Vector<double>& GPN_,
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
double calcHeightfromRay(const Matrix<double>& K,
                         const Vector<double>& GPN_,
                         const double GPD_,
                         const darknet_ros_msgs::BoundingBox& curBox,
                         const float corr_depth)
{

        const float x = curBox.xmin;
        const float y = curBox.ymin;
        const float w = curBox.xmax - x;
        const float h = curBox.ymax - y;

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
        gpPointLeft(0) = (gpPointLeft(0)*corr_depth)/gpPointLeft(2);
        gpPointLeft(2) = corr_depth;
        gpPointRight(0) = (gpPointRight(0)*corr_depth)/gpPointRight(2);
        gpPointRight(2) = corr_depth;

        // Find top point
        Vector<double> ray_top_1(3, 0.0);
        Vector<double> ray_top_2;

        Vector<double> aux(3, 1.0);
        aux(0) = x + w/2.0; //FIXED: move top point in middle, s.t. height is computed correctly lateron
        aux(1) = y;

        getRay(K, aux, ray_top_1, ray_top_2);

        // Vertical plane through base points + normal
        Vector<double> point3;
        point3 = gpPointLeft;
        point3 -= (GPN_);
        Vector<double> vpn(3, 0.0);
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
        gpPointTop(0) = (gpPointTop(0)*corr_depth)/gpPointTop(2);
        gpPointTop(2) = corr_depth;

        // Results
        gpPointTop -= gpPointLeft;

        // Compute Size
        double dSize = gpPointTop.norm();

        return dSize;
}

void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
    camera_info_ = msg;
}

void groundPlaneCallback(const rwth_perception_people_msgs::GroundPlane::ConstPtr& msg)
{
    ground_plane_ = msg;
}

void depthCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    depth_ = msg;
}

void depthCameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg)
{
    depth_camera_info_ = msg;
}

bool checkReady()
{
    if (!camera_info_)
    {
        ROS_WARN("Missing camera info.");
        return false;
    }

    if (!ground_plane_)
    {
        ROS_WARN("Missing ground plane.");
        return false;
    }

	if (!depth_)
	{
		ROS_WARN("Missing depth.");
		return false;
	}

	if (!depth_camera_info_)
	{
		ROS_WARN("Missing depth camera info.");
		return false;
	}

    return true;
}

float distanceFromDepth(const darknet_ros_msgs::BoundingBox& box, const cv::Mat& depth)
{
    const int x = box.xmin;
    const int y = box.ymin;
    const int width = box.xmax - x;
    const int height = box.ymax - y;

    // Check bounding box size.
    if (false)
    {
        return -1;
    }

    const cv::Rect roi = cv::Rect((int) (x + 0.25 * width),
                                  (int) (y + 0.25 * height),
                                  (int) (0.5 * width),
                                  (int) (0.5 * height));
    const cv::Mat depth_roi = depth(roi);
    const int roi_size = roi.width * roi.height;

    // Take the value of middle roi.
    std::vector<float> vector_depth;
    vector_depth.clear();
    vector_depth.reserve(roi_size);

    for (cv::MatConstIterator_<float> it = depth_roi.begin<float>();
         it != depth_roi.end<float>();
         ++it)
    {
        if (isfinite(*it))
            vector_depth.push_back(*it);
    }

    const int num_good_pixel = vector_depth.size();
    const float good_pixel_ratio = (float) num_good_pixel / (float) roi_size;

    if (good_pixel_ratio < 0.25)
    {
        return -1;
    }

    const int median_idx = num_good_pixel / 2;
    std::nth_element(vector_depth.begin(), vector_depth.begin() + median_idx, vector_depth.end());
    float median_depth = vector_depth.at(median_idx);

    if (median_depth > 30.0 || median_depth < 1e-5)
    {
        return -1;
    }

    return median_depth;
}

bool getGroundPlane(const ros::Time& stamp, Vector<double>& normal, double& d)
{
    normal = Vector<double>(3, (double*) &ground_plane_->n[0]);
    d = (double) ground_plane_->d;

    const std::string camera_frame_id = camera_info_->header.frame_id;
    const std::string gp_frame_id = ground_plane_->header.frame_id;

    if (camera_frame_id == gp_frame_id)
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

    distancePointStamped.header.frame_id = camera_frame_id;
    distancePointStamped.header.stamp = stamp;
    distancePointStamped.point.x = 0.0;
    distancePointStamped.point.y = 0.0;
    distancePointStamped.point.z = 0.0;

    try
    {
        tf_->waitForTransform(camera_frame_id, gp_frame_id, stamp, ros::Duration(1.0));
        tf_->transformVector(camera_frame_id, normalVectorStamped, normalVectorStampedCamera);
        // tf_->waitForTransform(gp_frame_id, camera_frame_id, stamp, ros::Duration(1.0));
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
    if (!getGroundPlane(msg->image_header.stamp, GPN, GPd))
    {
        ROS_WARN("Could not get groundplane. Discard all detections.");
        return;
    }

    // Depth
    cv_bridge::CvImagePtr depth_ptr(cv_bridge::toCvCopy(depth_, "32FC1"));
    const cv::Mat cv_depth = depth_ptr->image;

    // Camera matrix
    Matrix<double> K(3,3, (double*) &camera_info_->K[0]);
    Matrix<double> K_d(3,3, (double*) &depth_camera_info_->K[0]);

    // Published msg
    frame_msgs::DetectedPersons detected_persons;
    detected_persons.header = msg->image_header;

    for (const darknet_ros_msgs::BoundingBox& box : msg->bounding_boxes)
    {
        Vector<double> pos3D;
        calc3DPosFromBBox(K, GPN, GPd, box, world_scale_, pos3D);

        const float depth_reading = distanceFromDepth(box, cv_depth);

        // DetectedPerson for SPENCER
        frame_msgs::DetectedPerson detected_person;
        detected_person.modality = frame_msgs::DetectedPerson::MODALITY_GENERIC_MONOCULAR_VISION;
        detected_person.confidence = box.probability;
        detected_person.pose.pose.position.x = pos3D(0);
        detected_person.pose.pose.position.y = pos3D(1);
        if (depth_reading > 0)
        {
            detected_person.pose.pose.position.z = depth_reading;//min(med_depth_50, pos3D(2));
            detected_person.pose.pose.position.x = (pos3D(0)*depth_reading)/pos3D(2);
        }
        else
        {
            detected_person.pose.pose.position.z = pos3D(2);
        }
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
        detected_person.height = calcHeightfromRay(K ,GPN, GPd, box, detected_person.pose.pose.position.z);

        detected_persons.detections.push_back(detected_person);
    }

    // Publish
    detected_persons_pub_.publish(detected_persons);
}

int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "convert_yolo");
    ros::NodeHandle nh, nh_private("~");

    // Params (for SPENCER DetectedPersons message)
    nh_private.param("world_scale", world_scale_, 1.0); // default for ASUS sensors
    nh_private.param("detection_id_increment", detection_id_increment_, 1);
    nh_private.param("detection_id_offset", detection_id_offset_, 0);
    nh_private.param("pose_variance", pose_variance_, 0.05);
    nh_private.param("depth_scale", depth_scale_, 1.0);
    current_detection_id_ = detection_id_offset_;

    // Publishers
    std::string pub_topic_detected_persons;
    nh_private.param("detected_persons", pub_topic_detected_persons, string("/detected_persons"));
    detected_persons_pub_ = nh.advertise<frame_msgs::DetectedPersons>(pub_topic_detected_persons, 1);

    // Tf
    tf_ = std::make_shared<tf::TransformListener>();

    // Subscribers
    std::string camera_ns;
    nh_private.param("camera_namespace", camera_ns, std::string("/head_xtion"));

    std::string camera_info;
    nh_private.param("camera_info", camera_info, std::string("/hd/camera_info"));
    camera_info = camera_ns + camera_info;
    ros::Subscriber camera_info_sub = nh.subscribe(camera_info.c_str(), 1, cameraInfoCallback);

    std::string boundingboxes;
    nh_private.param("bounding_boxes", boundingboxes, std::string("darknet_ros/bounding_boxes"));
    ros::Subscriber bbox_sub = nh.subscribe(boundingboxes.c_str(), 1, bboxCallback);

    std::string ground_plane;
    nh_private.param("ground_plane", ground_plane, std::string(""));
    ros::Subscriber ground_plane_sub = nh.subscribe(ground_plane.c_str(), 1, groundPlaneCallback);

    // Use depth
    std::string depth_camera_info;
    nh_private.param("depth_camera_info", depth_camera_info, std::string("/sd/camera_info"));
    depth_camera_info = camera_ns + depth_camera_info;
	ros::Subscriber depth_camera_info_sub = nh.subscribe(depth_camera_info.c_str(), 1, depthCameraInfoCallback);

    std::string depth_topic;
    nh_private.param("depth_topic", depth_topic, std::string("/hd/image_depth_rect"));
	depth_topic = camera_ns + depth_topic;
    image_transport::ImageTransport it(nh_private);
	image_transport::Subscriber depth_sub = it.subscribe(depth_topic.c_str(), 1, depthCallback);

    ros::spin();

    return 0;
}

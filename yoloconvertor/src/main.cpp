// ROS includes.
#include <ros/ros.h>


#include <ros/time.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>


#include <string.h>

#include <cv_bridge/cv_bridge.h>

#include <rwth_perception_people_msgs/GroundHOGDetections.h>
#include <rwth_perception_people_msgs/GroundPlane.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <frame_msgs/DetectedPersons.h>

#include "Matrix.h"
#include "Vector.h"

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace rwth_perception_people_msgs;
using namespace darknet_ros_msgs;

//for debug image
//image_transport::Publisher pub_result_image;
//#include <QImage>
//#include <QPainter>
cv_bridge::CvImagePtr cv_depth_ptr;	// cv_bridge for depth image
cv::Mat img_depth_;


ros::Publisher pub_detected_persons;
double worldScale; // for computing 3D positions from BBoxes

int detection_id_increment, detection_id_offset, current_detection_id; // added for multi-sensor use in SPENCER
double pose_variance; // used in output frame_msgs::DetectedPerson.pose.covariance


void getRay(const Matrix<double>& K, const Vector<double>& x, Vector<double>& ray1, Vector<double>& ray2)
{
    Matrix<double> Kinv = K;
    Kinv.inv();

    ray1 = Vector<double>(3, 0.0);

    Matrix<double> rot = Eye<double>(3);
    rot *= Kinv;
    ray2 = rot * x;
    //ray2 += ray1;
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


void yoloConvertorCallback(const BoundingBoxesConstPtr &boxes, const CameraInfoConstPtr &camera_info,
                              const GroundPlaneConstPtr &gp, const ImageConstPtr &depth, const CameraInfoConstPtr &dep_info)
{
    //    ROS_INFO("Entered yolo-convert callback");


    // Generate base camera
    //Matrix<float> R = Eye<float>(3);
    //Vector<float> t(3, 0.0);

    // Get GP
    Vector<double> GPN(3, (double*) &gp->n[0]);
    double GPd = ((double) gp->d)*(-1000.0); // GPd = -958.475;
    Matrix<double> K(3,3, (double*)&camera_info->K[0]);
    Matrix<double> K_d(3,3, (double*)&dep_info->K[0]);

    // NOTE: Using 0 1 0 does not work, apparently due to numerical problems in libCudaHOG (E(1,1) gets zero when solving quadratic form)
    //Vector<float> float_GPN(3);
    //float_GPN(0) = -0.0123896; //-float(GPN(0));
    //float_GPN(1) = 0.999417; //-float(GPN(1)); // swapped with z by Timm
    //float_GPN(2) = 0.0317988; //-float(GPN(2));

    //float float_GPd = (float) GPd;
    //Matrix<float> float_K(3,3);
    //float_K(0,0) = K(0,0); float_K(1,0) = K(1,0); float_K(2,0) = K(2,0);
    //float_K(1,1) = K(1,1); float_K(0,1) = K(0,1); float_K(2,1) = K(2,1);
    //float_K(2,2) = K(2,2); float_K(0,2) = K(0,2); float_K(1,2) = K(1,2);

    //ROS_WARN("Ground plane: %.2f %.2f %.2f d=%.3f", float_GPN(0), float_GPN(1), float_GPN(2), float_GPd);




    //
    // Now create 3D coordinates for SPENCER DetectedPersons msg
    //
    if(pub_detected_persons.getNumSubscribers()) {
        frame_msgs::DetectedPersons detected_persons;
        detected_persons.header = boxes->image_header;

        //debug image
        /*QImage image_rgb = QImage(&depth->data[0], depth->width, depth->height, depth->step, QImage::Format_RGB888).copy();
        QPainter painter(&image_rgb);
        QColor qColor;
        qColor.setRgb(255, 255, 255);
        QPen pen;
        pen.setColor(qColor);
        pen.setWidth(120.0);*/
        //debugend

        for(unsigned int i=0;i<(boxes->bounding_boxes.size());i++)
        {
            BoundingBox curBox(boxes->bounding_boxes[i]);
            float x = curBox.xmin;
            float y = curBox.ymin;
            float width = curBox.xmax - x;
            float height = curBox.ymax - y;
           
            //Vector<double> normal(3, 0.0);
            //normal(0) = GPN(0);
            //normal(1) = GPN(1);
            //normal(2) = GPN(2);

            Vector<double> pos3D;
            calc3DPosFromBBox(K, GPN, GPd, x, y, width, height, worldScale, pos3D);

            // get readings in rectanular region from (registered!) depth image

            // Get depth image as matrix
            cv_depth_ptr = cv_bridge::toCvCopy(depth, "32FC1");
            //if (depth->encoding == "16UC1" || depth->encoding == "32FC1") {
                //cv_depth_ptr->image *= 0.001;
            //}
            img_depth_ = cv_depth_ptr->image;

            const int len = (int)width*(int)height;
            vector<double> vector_depth(len);
            for (int r = y ; r < y+(int)height ; r++){
                for (int c = x ; c < x+(int)width ; c++) {
                    //std::cout << " " << (c-x)*(int)width+(r-y) << " " << std::endl;
                    //std::cout << " " << img_depth_.at<float>(r,c) << std::endl;
                    vector_depth[(c-x)*(int)height+(r-y)] = img_depth_.at<float>(r,c);
                }
            }
            nth_element( vector_depth.begin(), vector_depth.begin()+len/2,vector_depth.end() );
            double med_depth = vector_depth[len/2];
            //std::cout << "Median: " << med_depth << std::endl;

            // DetectedPerson for SPENCER
            frame_msgs::DetectedPerson detected_person;
            detected_person.modality = frame_msgs::DetectedPerson::MODALITY_GENERIC_MONOCULAR_VISION;
            // use the probability from yolo detector
            detected_person.confidence = curBox.probability;
            detected_person.pose.pose.position.x = -pos3D(0);
            detected_person.pose.pose.position.y = -pos3D(1);
            if(med_depth<99999 && med_depth > 0){
                detected_person.pose.pose.position.z = min(med_depth,-pos3D(2));
            }else{
                detected_person.pose.pose.position.z = -pos3D(2);
            }
            detected_person.pose.pose.orientation.w = 1.0;

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

            detected_persons.detections.push_back(detected_person);

            //debug
            // Calculate centres and corner points of bounding boxes in IR, world and RGB
            /*double depth_value = detected_person.pose.pose.position.z;
            double w_rgb = width;
            double h_rgb = height;
            double x_left_rgb = x;
            double y_top_rgb = y;
            double x_right_rgb = x_left_rgb + w_rgb;
            double y_down_rgb = y_top_rgb + h_rgb;

            double x_left_world = depth_value*((x_left_rgb-K(2,0))/K(0,0));
            double y_top_world = depth_value*((y_top_rgb-K(2,1))/K(1,1));
            double x_right_world = depth_value*((x_right_rgb-K(2,0))/K(0,0));
            double y_down_world = depth_value*((y_down_rgb-K(2,1))/K(1,1));

            //TODO: R/t?
            double x_left_d = (x_left_world * K_d(0,0) / depth_value) + K_d(2,0);
            double y_top_d = (y_top_world * K_d(1,1) / depth_value) + K_d(2,1);
            double x_right_d = (x_right_world * K_d(0,0) / depth_value) + K_d(2,0);
            double y_down_d = (y_down_world * K_d(1,1) / depth_value) + K_d(2,1);
            double w_d = x_right_d - x_left_d;
            double h_d = y_down_d - y_top_d;

            float x_d = x_left_d;
            float y_d = y_top_d;
            painter.drawLine(x_d,y_d, x_d+w_d,y_d);
            painter.drawLine(x_d,y_d, x_d,y_d+h_d);
            painter.drawLine(x_d+w_d,y_d, x_d+w_d,y_d+h_d);
            painter.drawLine(x_d,y_d+h_d, x_d+w_d,y_d+h_d);*/
            //debugend
        }

        // Publish
        pub_detected_persons.publish(detected_persons);

        //debug
        /*const uchar *bits = image_rgb.constBits();
        sensor_msgs::Image sensor_image;
        sensor_image.header = depth->header;
        sensor_image.height = image_rgb.height();
        sensor_image.width  = image_rgb.width();
        sensor_image.step   = image_rgb.bytesPerLine();
        sensor_image.data   = vector<uchar>(bits, bits + image_rgb.byteCount());
        sensor_image.encoding = "rgb8";//depth->encoding;
        pub_result_image.publish(sensor_image);*/
        //debugend
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
    current_detection_id = detection_id_offset;

    //string image_color = camera_ns + "/hd/image_color_rect";
    string camera_info = camera_ns + "/hd/camera_info";
    string topic_depth_info = camera_ns + "/sd/camera_info";
    string topic_depth_image = camera_ns + "/hd/image_depth_rect";



    ROS_DEBUG("yoloconvertor: Queue size for synchronisation is set to: %i", queue_size);

    // Image transport handle
    image_transport::ImageTransport it(private_node_handle_);

    // Create a subscriber.
    // Name the topic, message queue, callback function with class name, and object containing callback function.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    ros::Subscriber sub_message; //Subscribers have to be defined out of the if scope to have affect.
    Subscriber<GroundPlane> subscriber_ground_plane(n, ground_plane.c_str(), 1); subscriber_ground_plane.unsubscribe();
    
//    image_transport::SubscriberFilter subscriber_color;
//    subscriber_color.subscribe(it, image_color.c_str(), 1); subscriber_color.unsubscribe();
    Subscriber<CameraInfo> subscriber_camera_info(n, camera_info.c_str(), 10); subscriber_camera_info.unsubscribe();
    Subscriber<BoundingBoxes> subscriber_bounding_boxes(n,boundingboxes.c_str(),10); subscriber_bounding_boxes.unsubscribe();
    image_transport::SubscriberFilter subscriber_depth;
    subscriber_depth.subscribe(it, topic_depth_image.c_str(),10); subscriber_depth.unsubscribe();
    message_filters::Subscriber<CameraInfo> subscriber_depth_info(n, topic_depth_info.c_str(), 10); subscriber_depth_info.unsubscribe();

    // Neng, why we need this line? what is this for? also connectCallback
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

    /*image_transport::SubscriberStatusCallback image_cb = boost::bind(&connectCallback,
                                                                     boost::ref(sub_message),
                                                                     boost::ref(n),
                                                                     ground_plane,
                                                                     boost::ref(subscriber_ground_plane),
                                                                     boost::ref(subscriber_camera_info),
                                                                     boost::ref(subscriber_bounding_boxes),
                                                                     boost::ref(subscriber_depth),
                                                                     boost::ref(subscriber_depth_info),
                                                                     boost::ref(it));*/



    //The real queue size for synchronisation is set here.
    sync_policies::ApproximateTime<BoundingBoxes, CameraInfo, GroundPlane, Image, CameraInfo> MySyncPolicy(queue_size);
    MySyncPolicy.setAgePenalty(1000); //set high age penalty to publish older data faster even if it might not be correctly synchronized.

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
    pub_detected_persons = n.advertise<frame_msgs::DetectedPersons>(pub_topic_detected_persons, 10, con_cb, con_cb);

    //debug image publisher
    //private_node_handle_.param("yolo_depth_image", pub_topic_result_image, string("/yolo_depth_image"));
    //pub_result_image = it.advertise(pub_topic_result_image.c_str(), 1, image_cb, image_cb);

    //double min_expected_frequency, max_expected_frequency;
    //private_node_handle_.param("min_expected_frequency", min_expected_frequency, 8.0);
    //private_node_handle_.param("max_expected_frequency", max_expected_frequency, 100.0);
    
    /*pub_detected_persons.setExpectedFrequency(min_expected_frequency, max_expected_frequency);
    pub_detected_persons.setMaximumTimestampOffset(0.3, 0.1);
    pub_detected_persons.finalizeSetup();*/

    ros::spin();

    return 0;
}



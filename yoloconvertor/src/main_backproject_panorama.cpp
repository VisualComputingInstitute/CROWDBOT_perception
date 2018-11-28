// ROS includes.
#include <ros/ros.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>


#include <cmath>
#include <ros/time.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <string.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <frame_msgs/TrackedPersons.h>
#include <frame_msgs/TrackedPersons2d.h>

#include "Matrix.h"
#include "Vector.h"
#include "PanoramaCameraModel.h"
#include "Visual.h"
using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace cv_bridge;



tf::TransformListener* listener;
cv::Mat image_rgb;
image_transport::Publisher pub_result_image;

frame_msgs::TrackedPersons2dPtr tps2d_ptr; // this pointer point to the TrackedPersons2d msg, which we will publish in callback funciton.
// this pointer is actually for share this TrackedPersons2d msg between two callback function.

ros::Publisher pub_tracked_persons2d;

string camera_frame;
string world_frame;


float minPhi =  -1.775;
float maxPhi =  1.775;
float maxHeight = 1.400;
float minHeight = -1.400;
float iwidth = 1280;
float iheight = 800;

float hard_code_person_width = 0.7;

mira::camera::PanoramaCameraIntrinsic panorama_intrinsic(minPhi, maxPhi, minHeight, maxHeight, iwidth, iheight);

void transfer_tracked_persons_to_camera_cord(const frame_msgs::TrackedPersonsConstPtr &sub_tps, frame_msgs::TrackedPersons &dst_tps, string camera_frame)
{
    ros::Time tracked_time(sub_tps->header.stamp);
    tf::StampedTransform transform;
    try {
        listener->waitForTransform(camera_frame, world_frame, tracked_time, ros::Duration(1.0));
        listener->lookupTransform(camera_frame, world_frame, tracked_time, transform);  //from world to camera 
    }
    catch (tf::TransformException ex){
       ROS_WARN_THROTTLE(20.0, "Failed transform lookup from world frame to camera frame", ex.what());
    }

    for(unsigned int i=0;i<(sub_tps->tracks.size());i++)
    {
        frame_msgs::TrackedPerson tracked_person(sub_tps->tracks[i]);

        tf::Vector3 pos_vector_in_world(tracked_person.pose.pose.position.x, tracked_person.pose.pose.position.y, tracked_person.pose.pose.position.z);
        tf::Vector3 pos_vector_in_camera = transform*pos_vector_in_world; 

        Vector<double> pos3D;
        pos3D.setSize(3);
        pos3D[0] = pos_vector_in_camera.getX();
        pos3D[1] = pos_vector_in_camera.getY();
        pos3D[2] = pos_vector_in_camera.getZ();

        tracked_person.pose.pose.position.x = pos3D(0);
        tracked_person.pose.pose.position.y = pos3D(1);
        tracked_person.pose.pose.position.z = pos3D(2);

        dst_tps.tracks.push_back(tracked_person);
    }
}



void backProjectCallback(const frame_msgs::TrackedPersonsConstPtr &tpks_ptr)
{
    // debug output, to show latency
    ROS_DEBUG_STREAM("time stamep of input image:" << tpks_ptr);
    ROS_DEBUG_STREAM("current time:" << ros::Time::now());
    ROS_DEBUG_STREAM("-----------------------------------------");

    // intersection box
    int ibox_top_x;
    int ibox_top_y;
    int ibox_bot_x;
    int ibox_bot_y;
    int ibox_w;
    int ibox_h;

    if(pub_tracked_persons2d.getNumSubscribers()||pub_result_image.getNumSubscribers()) {
        frame_msgs::TrackedPersons tkps_in_camera_frame;
        // convert tracked person's position into camera frame
        transfer_tracked_persons_to_camera_cord(tpks_ptr, tkps_in_camera_frame, camera_frame);


        tps2d_ptr.reset(new frame_msgs::TrackedPersons2d); // make sure this dps_ptr will point to a new DetectedPersons object
        if(tps2d_ptr == 0)
        {
            std::cout<<"oops! no enough memory for this trackedpersons2d msg"<<std::endl;
            return;
        }
        frame_msgs::TrackedPersons2d& tracked_persons_2d(*tps2d_ptr);
        tracked_persons_2d.header = tpks_ptr->header;
        tracked_persons_2d.header.frame_id = camera_frame; 

        for(unsigned int i=0;i<(tkps_in_camera_frame.tracks.size());i++)
        {
            frame_msgs::TrackedPerson tracked_person(tkps_in_camera_frame.tracks[i]);
            Vector<double> bottom_center_3d(tracked_person.pose.pose.position.x,tracked_person.pose.pose.position.y,tracked_person.pose.pose.position.z);
            Vector<double> bottom_center_2d(2);
            mira::camera::PanoramaCameraModel::project3dRayToPixel(bottom_center_3d,bottom_center_2d,panorama_intrinsic); // input these ray3D to get the corresponding pixel.

            // we only consider bottom_center_2d which located within the image as this tracked person belong to this camera
            //if(bottom_center_2d[0]>0 && bottom_center_2d[0] < panorama_intrinsic.width && bottom_center_2d[1]>0 && bottom_center_2d[1]<panorama_intrinsic.height)
            //{
            frame_msgs::TrackedPerson2d tracked_person_2d;
            tracked_person_2d.person_height = tracked_person.height;

            // Note: now we are in the camera frame.
            Vector<double> top_center_3d(bottom_center_3d[0],bottom_center_3d[1]-tracked_person_2d.person_height,bottom_center_3d[2]);
            Vector<double> top_left_3d(top_center_3d);
            // very stupid here is, the image we use is fliped, which means the left in real world frame is the right in the image frame,,,
            top_left_3d[0] += hard_code_person_width/2.0;
            Vector<double> bottom_right_3d(bottom_center_3d);
            bottom_right_3d[0] -= hard_code_person_width/2.0;
       
            Vector<double> top_left_2d(2);
            mira::camera::PanoramaCameraModel::project3dRayToPixel(top_left_3d,top_left_2d,panorama_intrinsic);
            Vector<double> bottom_right_2d(2);
            mira::camera::PanoramaCameraModel::project3dRayToPixel(bottom_right_3d,bottom_right_2d,panorama_intrinsic);

            tracked_person_2d.depth = tracked_person.pose.pose.position.z;
            tracked_person_2d.x =(int)top_left_2d[0];
            tracked_person_2d.y =(int)top_left_2d[1];
            tracked_person_2d.h =(unsigned int)(bottom_right_2d[1]-top_left_2d[1]);
            tracked_person_2d.w = (unsigned int)(bottom_right_2d[0]-top_left_2d[0]);	
            
	        tracked_person_2d.track_id = tracked_person.track_id;

            ibox_top_x = min(max(tracked_person_2d.x, 0), (int)panorama_intrinsic.width);
            ibox_top_y = min(max(tracked_person_2d.y, 0), (int)panorama_intrinsic.height);
            ibox_bot_x = min(max((int)tracked_person_2d.x+(int)tracked_person_2d.w, 0), (int)panorama_intrinsic.width);
            ibox_bot_y = min(max((int)tracked_person_2d.y+(int)tracked_person_2d.h, 0), (int)panorama_intrinsic.height);
            ibox_w = ibox_bot_x - ibox_top_x;
            ibox_h = ibox_bot_y - ibox_top_y;

            // only add, if x% of box is visible and in front of camera!
            if (((float)ibox_w*ibox_h)/(tracked_person_2d.w*tracked_person_2d.h) >= 0.3 && tracked_person_2d.depth >= 0){
                tracked_persons_2d.boxes.push_back(tracked_person_2d);
            }
            //}


        }

        // Publish
        pub_tracked_persons2d.publish(tracked_persons_2d);
    }

}

void render_visualization(const frame_msgs::TrackedPersons2d& tracked_persons2d,const ImageConstPtr& color )
{
    CvImagePtr cv_color_ptr(toCvCopy(color));
    image_rgb = cv_color_ptr->image;

    for(unsigned int i=0;i<(tracked_persons2d.boxes.size());i++)
    {
        frame_msgs::TrackedPerson2d tracked_person2d(tracked_persons2d.boxes[i]);

        // since current we don't have the box's width and height from the tracked_persons2d.
        // So we hard code the height and width.
        float height = tracked_person2d.h;
        float width = tracked_person2d.w;
        float x =(float)std::max(tracked_person2d.x, 0);  // make sure x and y are in the image.
        float y = (float)std::max(tracked_person2d.y, 0);

        render_bbox_2D(x, y, width, height, image_rgb, 255, 0, 0);

        // show the track id
        string trackid_str = int_to_string((int)tracked_person2d.track_id);
        render_text(trackid_str, image_rgb, x, y, 0,255,0);

    }


    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(color->header, "bgr8", image_rgb).toImageMsg();
    //sensor_image.encoding = "rgb8";//depth->encoding;
    pub_result_image.publish(msg);
}



void backProjectwithImageCallback(const frame_msgs::TrackedPersonsConstPtr &tracked_persons, const ImageConstPtr &color)
{
    if(pub_tracked_persons2d.getNumSubscribers() || pub_result_image.getNumSubscribers()){
        camera_frame = color->header.frame_id;
        backProjectCallback(tracked_persons);
        render_visualization(*tps2d_ptr, color);
    }
}


// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(Subscriber<frame_msgs::TrackedPersons> &sub_tps,
                     image_transport::SubscriberFilter &sub_color,
                     image_transport::ImageTransport &it){
    // when both two topic no user subscribe.
    if((!pub_tracked_persons2d.getNumSubscribers())&&(!pub_result_image.getNumSubscribers())) {
        ROS_DEBUG("back projection: No subscribers. Unsubscribing.");
        sub_tps.unsubscribe();
        sub_color.unsubscribe();
    } else {
        ROS_DEBUG("back projection: New subscribers. Subscribing.");
        sub_tps.subscribe();
        sub_color.subscribe(it,sub_color.getTopic().c_str(),1);
    }

}



/* back project tracked persons to camera image.
 * For each camera, we run one this node to do back projection.
 * parameter:   camera frame
 * subscribe:   1.tracked persons
 *              2.camera image
 * publish:     tracked persons 2d. camera image with tracked person in image
 *
 *
***************************************************/
int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "back_project_panorama");
    ros::NodeHandle n;

    // create a tf listener
    listener = new tf::TransformListener();


    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string image_topic;

    //visualization
    string pub_topic_result_image;
    string pub_tracked_persons_2d;
    string tracked_persons;
    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("image_topic", image_topic, string("/hardware/video/valeo/rectificationNIKRLeft/PanoramaImage"));
    private_node_handle_.param("world_frame", world_frame, string("/robot/OdometryFrame"));
    private_node_handle_.param("tracked_persons", tracked_persons, string("/rwth_tracker/tracked_persons"));
    // For visualization output parameter

    string image_color = image_topic;

    // Image transport handle
    image_transport::ImageTransport it(private_node_handle_);

    // Create a subscriber.
    // Name the topic, message queue, callback function with class name, and object containing callback function.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    // Here these unsubscribes make at beginning this node doesn't subscribe any topic, until any other node subscribe this node, and want this node work.
    Subscriber<frame_msgs::TrackedPersons> subscriber_tracked_persons(n, tracked_persons.c_str(), 1); subscriber_tracked_persons.unsubscribe();
    image_transport::SubscriberFilter subscriber_color;
    subscriber_color.subscribe(it, image_color.c_str(), 1); subscriber_color.unsubscribe();

    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(subscriber_tracked_persons),
                                                       boost::ref(subscriber_color),
                                                       boost::ref(it));

    image_transport::SubscriberStatusCallback image_cb = boost::bind(&connectCallback,
                                                                     boost::ref(subscriber_tracked_persons),
                                                                     boost::ref(subscriber_color),
                                                                     boost::ref(it));

    //The real queue size for synchronisation is set here.
    // if we don't need visualization, we don't need to sync this image topic
    // I have to put ros::spin() in the same scope as sync, then it can guaratee that we can use sync to manager the message queue and get callback.
    // if I put ros::spin() out of this scope, then when we run into ros::spin(). The sync is already destructed, and cannot work.
    sync_policies::ApproximateTime<frame_msgs::TrackedPersons, Image> MySyncPolicy(queue_size);
    MySyncPolicy.setAgePenalty(1000); //set high age penalty to publish older data faster even if it might not be correctly synchronized.

    const sync_policies::ApproximateTime<frame_msgs::TrackedPersons, Image> MyConstSyncPolicy = MySyncPolicy;

    Synchronizer< sync_policies::ApproximateTime<frame_msgs::TrackedPersons, Image> > sync(MyConstSyncPolicy,
                                                                                           subscriber_tracked_persons,
                                                                                           subscriber_color);

    sync.registerCallback(boost::bind(&backProjectwithImageCallback, _1, _2));  //use the visualization callback

    // Create publishers
    private_node_handle_.param("tracked_person_2d", pub_tracked_persons_2d, string("/tracked_person_2d_pano"));
    pub_tracked_persons2d = n.advertise<frame_msgs::TrackedPersons2d>(pub_tracked_persons_2d, 10, con_cb, con_cb);

    //debug image publisher
    private_node_handle_.param("backproject_visual_image", pub_topic_result_image, string("/yoloconvertor_visual_image"));
    pub_result_image = it.advertise(pub_topic_result_image.c_str(), 1, image_cb, image_cb);

    ros::spin();



    return 0;
}




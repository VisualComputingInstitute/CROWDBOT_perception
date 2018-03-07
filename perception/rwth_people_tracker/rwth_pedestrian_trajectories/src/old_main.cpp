// ROS includes.
#include "ros/ros.h"
#include "ros/time.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <string.h>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/gregorian/gregorian_types.hpp>
#include <boost/circular_buffer.hpp>

#include <std_msgs/Float32.h>
#include <std_msgs/UInt16.h>

#include <iostream>
#include <fstream>

#include <cv_bridge/cv_bridge.h>

#include "frame_msgs/TrackedPersons.h"
#include "frame_msgs/PersonTrajectories.h"
#include "frame_msgs/PersonTrajectory.h"
#include "frame_msgs/PersonTrajectoryEntry.h"


using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace rwth_perception_people_msgs;

ros::Publisher pub_message;
image_transport::Publisher pub_image;
ros::Publisher pub_tracked_persons;
ros::Publisher pub_tracked_persons_2d;

cv::Mat img_depth_;
cv_bridge::CvImagePtr cv_depth_ptr;	// cv_bridge for depth image

void callbackWithoutHOG(const ImageConstPtr &color,
              const CameraInfoConstPtr &info,
              const GroundPlane::ConstPtr &gp,
              const UpperBodyDetector::ConstPtr &upper,
              const VisualOdometry::ConstPtr &vo)
{

    // ---update framerate + framerateVector---
    //printf("set new framerate to: %d / (%f-%f) \n", 1,color->header.stamp.toSec(),Globals::oldTimeForFPSUpdate);
    //printf("result: %d\n", (int) (1 / (color->header.stamp.toSec()-Globals::oldTimeForFPSUpdate)));
    // update framerate first after some tracking cycles, before use framerate from config file
    if (cnt>0) {
        double dt = color->header.stamp.toSec() - Globals::oldTimeForFPSUpdate;
        double fps = 1.0 / dt;

        if(!std::isfinite(fps) || fps < 1) {
            ROS_WARN("Abnormal frame rate detected: %f, dt: %f. Set to 1", fps, dt);
            Globals::frameRate = 1;
        }
        else {
            Globals::frameRate = (int) fps;
        }
        Globals::frameRateVector.swap();
        Globals::frameRateVector.pushBack(Globals::frameRate);
        Globals::frameRateVector.swap();
        Globals::frameRateVector.resize(min(cnt+1,100));
    }
    else{
        //first cycle, setup frameRateVector
        Globals::frameRateVector.setSize(1,0.0);
        Globals::frameRateVector(0) = Globals::frameRate;
    }
    //printf("---\n");
    //Globals::frameRateVector.show();
    //printf("---\n");
    // ---end update framerate+frameRatevector---

    //printf("set!\n");
    ROS_DEBUG("Entered callback without groundHOG data");
    Globals::render_bbox3D = (pub_image.getNumSubscribers() > 0) || (Globals::save_for_eval);

    // safe strings for eval
    std::string safe_string;
    char safe_string_char[128];

    // Get camera from VO and GP
    Vector<double> GP(3, (double*) &gp->n[0]);
    GP.pushBack((double) gp->d);
    Camera camera = createCamera(GP, vo, info);

    // Get detections from upper body
    Vector<double> single_detection(9);
    Vector<Vector< double > > detected_bounding_boxes;

    for(int i = 0; i < upper->pos_x.size(); i++)
    {
        single_detection(0) = cnt;
        single_detection(1) = i;
        single_detection(2) = 1;
        single_detection(3) = 1 - upper->dist[i]; // make sure that the score is always positive
        single_detection(4) = upper->pos_x[i];
        single_detection(5) = upper->pos_y[i];
        single_detection(6) = upper->width[i];
        single_detection(7) = upper->height[i] * 3;
        single_detection(8) = upper->median_depth[i];
        detected_bounding_boxes.pushBack(single_detection);
        //ROS_INFO("upper det %i bbox: (%f, %f ,%f ,%f): %f", i, single_detection(4), single_detection(5), single_detection(6), single_detection(7), single_detection(3) );
        //ROS_INFO("Depth for upper det: %f", single_detection(8));
    }

    get_image((unsigned char*)(&color->data[0]),info->width,info->height,cim);

    writeImageAndCamInfoToFile(color,info,camera);

    if (Globals::save_for_eval){
        // save color image, going to be processed
        safe_string = Globals::save_path_img;
        safe_string.append("img_%08d.jpg");
        sprintf(safe_string_char, safe_string.c_str(), cnt);
        cim.save(safe_string_char);
    }

    ///////////////////////////////////////////TRACKING///////////////////////////
    tracker.process_tracking_oneFrame(HyposAll, *det_comb, cnt, detected_bounding_boxes, cim, camera);
    ///////////////////////////////////////////TRACKING-END///////////////////////////

    if (Globals::save_for_eval){
        // save processed image
        safe_string = Globals::save_path_img;
        safe_string.append("track_%08d.jpg");
        sprintf(safe_string_char, safe_string.c_str(), cnt);
        cim.save(safe_string_char);
    }

    // publish tracks
    Vector<Hypo> hyposMDL = tracker.getHyposMDL();
    double currentTime = color->header.stamp.toSec();
    PedestrianTrackingArray allHypoMsg;
    allHypoMsg.header = color->header;

    // also prepare tracks
    frame_msgs::TrackedPersons trackedPersons;
    frame_msgs::TrackedPersons2d trackedPersons2d;
    trackedPersons.header.stamp = upper->header.stamp;
    trackedPersons.header.seq = ++track_seq;
    trackedPersons.header.frame_id = "/robot/OdometryFrame"; //FIXME: world frame, maybe should not be hardcoded

    trackedPersons2d.header.stamp = upper->header.stamp;
    trackedPersons2d.header.seq = track_seq;
    trackedPersons2d.header.frame_id = color->header.frame_id;

    Vector<Vector<double> > trajPts;
    Vector<double> dir;
    Vector<double> robot_frame_dir(3);
    for(int i = 0; i < hyposMDL.getSize(); i++)
    {
        PedestrianTracking oneHypoMsg;
        oneHypoMsg.header = color->header;
        hyposMDL(i).getTrajPts(trajPts);
        for(int j = 0; j < trajPts.getSize(); j++)
        {
            oneHypoMsg.traj_x.push_back(trajPts(j)(0));
            oneHypoMsg.traj_y.push_back(trajPts(j)(1));
            oneHypoMsg.traj_z.push_back(trajPts(j)(2));
            Vector<double> posInCamera = AncillaryMethods::fromWorldToCamera(trajPts(j), camera);

            oneHypoMsg.traj_x_camera.push_back(posInCamera(0));
            oneHypoMsg.traj_y_camera.push_back(posInCamera(1));
            oneHypoMsg.traj_z_camera.push_back(posInCamera(2));
        }

        oneHypoMsg.id = hyposMDL(i).getHypoID();
        oneHypoMsg.score = hyposMDL(i).getScoreMDL();
        oneHypoMsg.speed = hyposMDL(i).getSpeed();
        hyposMDL(i).getDir(dir);

        oneHypoMsg.dir.push_back(dir(0));
        oneHypoMsg.dir.push_back(dir(1));
        oneHypoMsg.dir.push_back(dir(2));
        allHypoMsg.pedestrians.push_back(oneHypoMsg);

        // Also publish tracks
        geometry_msgs::PoseWithCovariance pose;
        geometry_msgs::TwistWithCovariance twist;
        int curr_idx = trajPts.getSize()-1;
        Vector<Matrix<double> > C;
        hyposMDL(i).getStateCovMats(C);
        //printf("C:\n");
        //C(curr_idx).Show();

        // init one tracked person
        frame_msgs::TrackedPerson trackedPerson;
        frame_msgs::TrackedPerson2d trackedPerson2d;
        trackedPerson.track_id = hyposMDL(i).getHypoID();
        ros::Time currentCreationTime;
        hyposMDL(i).getCreationTime(currentCreationTime);
        trackedPerson.age = ros::Duration(ros::Time::now()-currentCreationTime);
        trackedPerson.is_occluded = false; // FIXME: available for mht tracker, yet?
        Vector<FrameInlier> frameInlier;
        hyposMDL(i).getIdx(frameInlier);
        //printf("hypo %i, number of inliers: %i\n", trackedPerson.track_id, frameInlier(frameInlier.getSize()-1).getNumberInlier());
        //printf("first:\n");
        //frameInlier(0).showFrameInlier();
        //printf("last:\n");
        //frameInlier(frameInlier.getSize()-1).showFrameInlier();
        Vector<int> currentInlier = frameInlier(frameInlier.getSize()-1).getInlier();
        if (currentInlier(0) > 0){
            trackedPerson.is_matched = true;
        }
        else{
            trackedPerson.is_matched = false;
        }
        trackedPerson.is_matched = true; // FIXME: available for mht tracker, yet? probably with getIdx() = Inlier detections
        // from kalman: !!det!!.getColorHist(frame, inl(0), newColHist); same with ID instead of colorhist? Look later...
        trackedPerson.detection_id = 0; // FIXME: available for mht tracker, yet? can we get it from Idx??

        trackedPerson2d.track_id = hyposMDL(i).getHypoID();
        trackedPerson2d.person_height = hyposMDL(i).getHeight();

        // prepare position and velocity of tracked person
        Vector<double> posInCamera = AncillaryMethods::fromWorldToCamera(trajPts(curr_idx), camera);

        Vector<double> bbox_topleftCornerInCam = posInCamera;
        Vector<double> bbox_bottomrightCornerInCam = posInCamera;
        bbox_topleftCornerInCam.pushBack(1);
        bbox_bottomrightCornerInCam.pushBack(1);
        bbox_topleftCornerInCam(0) -= Globals::pedSizeWVis/2.0;
        bbox_bottomrightCornerInCam(0) += Globals::pedSizeWVis/2.0;
        Vector<double> bbox_topleftCornerInWorld = AncillaryMethods::fromCameraToWorld(bbox_topleftCornerInCam, camera);
        Vector<double> bbox_bottomrightCornerInWorld = AncillaryMethods::fromCameraToWorld(bbox_bottomrightCornerInCam, camera);
        bbox_topleftCornerInWorld *= 1/bbox_topleftCornerInWorld(3);
        bbox_topleftCornerInWorld.resize(3);
        bbox_bottomrightCornerInWorld *= 1/bbox_bottomrightCornerInWorld(3);
        bbox_bottomrightCornerInWorld.resize(3);
        bbox_topleftCornerInWorld(1) -= hyposMDL(i).getHeight();
        Vector<double> bbox_topleftCornerInImage;
        Vector<double> bbox_bottomrightCornerInImage;
        camera.WorldToImage(bbox_topleftCornerInWorld, Globals::WORLD_SCALE, bbox_topleftCornerInImage);
        camera.WorldToImage(bbox_bottomrightCornerInWorld, Globals::WORLD_SCALE, bbox_bottomrightCornerInImage);
        trackedPerson2d.x = bbox_topleftCornerInImage(0);
        trackedPerson2d.y = bbox_topleftCornerInImage(1);
        trackedPerson2d.w = bbox_bottomrightCornerInImage(0) - bbox_topleftCornerInImage(0);
        trackedPerson2d.h = bbox_bottomrightCornerInImage(1) - bbox_topleftCornerInImage(1);
        trackedPerson2d.depth = posInCamera(2);

        robot_frame_dir(0) = dir(2);
        robot_frame_dir(1) = -dir(0);
        robot_frame_dir(2) = -dir(1);

        // Some constants for determining the pose
        //const double AVERAGE_ROTATION_VARIANCE = pow(10.0 / 180 * M_PI, 2); // FIXME: determine from vx, vy?
        const double INFINITE_VARIANCE = 9999999; // should not really use infinity here because then the covariance matrix cannot be rotated (singularities!)

        // Set pose (=position + orientation)
        pose.pose.position.x = trajPts(curr_idx)(2);
        pose.pose.position.y = -trajPts(curr_idx)(0);
        pose.pose.position.z = -trajPts(curr_idx)(1);

        // Set orientation
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(atan2(robot_frame_dir(1), robot_frame_dir(0))); // determine orientation from current velocity estimate
        pose.covariance.fill(0.0);
        pose.covariance[0 * 6 + 0] = C(curr_idx)(0,0); // variance of x position
        pose.covariance[1 * 6 + 1] = C(curr_idx)(1,1); // variance of y position
        pose.covariance[2 * 6 + 2] = INFINITE_VARIANCE; // variance of z position
        pose.covariance[3 * 6 + 3] = INFINITE_VARIANCE; // variance of x rotation
        pose.covariance[4 * 6 + 4] = INFINITE_VARIANCE; // variance of y rotation
        pose.covariance[5 * 6 + 5] = C(curr_idx)(2,2); // variance of z rotation

        // Set twist (=velocities)
        twist.twist.linear.x = robot_frame_dir(0);
        twist.twist.linear.y = robot_frame_dir(1);
        twist.twist.linear.z = robot_frame_dir(2);

        twist.covariance.fill(0.0);
        twist.covariance[0 * 6 + 0] = C(curr_idx)(3,3); // variance of x linear velocity
        twist.covariance[1 * 6 + 1] = C(curr_idx)(3,3); // variance of y linear velocity
        twist.covariance[2 * 6 + 2] = INFINITE_VARIANCE; // variance of z linear velocity
        twist.covariance[3 * 6 + 3] = C(curr_idx)(2,2); // variance of x angular velocity
        twist.covariance[4 * 6 + 4] = C(curr_idx)(2,2); // variance of y angular velocity
        twist.covariance[5 * 6 + 5] = INFINITE_VARIANCE; // variance of z angular velocity


        // set pose and twist and add to tracked persons
        trackedPerson.pose = pose;
        trackedPerson.twist = twist;
        trackedPersons.tracks.push_back(trackedPerson);
        trackedPersons2d.boxes.push_back(trackedPerson2d);


    }

    if(pub_image.getNumSubscribers()) {
        ROS_DEBUG("Publishing image");
        Image res_img;
        res_img.header = color->header;
        res_img.height = cim._height;
        res_img.width = cim._width;
        res_img.step   = color->step;
        for (std::size_t i = 0; i != cim._height*cim._width; ++i) {
            res_img.data.push_back(cim.data()[i+0*cim._height*cim._width]);
            res_img.data.push_back(cim.data()[i+1*cim._height*cim._width]);
            res_img.data.push_back(cim.data()[i+2*cim._height*cim._width]);
        }
        res_img.encoding = color->encoding;

        pub_image.publish(res_img);
    }

    pub_message.publish(allHypoMsg);
    pub_tracked_persons.publish(trackedPersons);
    pub_tracked_persons_2d.publish(trackedPersons2d);
    Globals::oldTimeForFPSUpdate = color->header.stamp.toSec(); //ros::Time::now().toSec();
    cnt++;
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(message_filters::Subscriber<CameraInfo> &sub_cam,
                     message_filters::Subscriber<GroundPlane> &sub_gp,
                     message_filters::Subscriber<GroundHOGDetections> &sub_hog,
                     message_filters::Subscriber<UpperBodyDetector> &sub_ubd,
                     message_filters::Subscriber<VisualOdometry> &sub_vo,
                     image_transport::SubscriberFilter &sub_col,
                     image_transport::ImageTransport &it){
    if(!pub_message.getNumSubscribers()
    && !pub_image.getNumSubscribers()
    && !pub_tracked_persons.getNumSubscribers()
    && !pub_tracked_persons_2d.getNumSubscribers()
    ) {
        ROS_DEBUG("Tracker: No subscribers. Unsubscribing.");
        sub_cam.unsubscribe();
        sub_gp.unsubscribe();
        sub_hog.unsubscribe();
        sub_ubd.unsubscribe();
        sub_vo.unsubscribe();
        sub_col.unsubscribe();
    } else {
        ROS_DEBUG("Tracker: New subscribers. Subscribing.");
        sub_cam.subscribe();
        sub_gp.subscribe();
        sub_hog.subscribe();
        sub_ubd.subscribe();
        sub_vo.subscribe();
        sub_col.subscribe(it,sub_col.getTopic().c_str(),1);
    }
}

int main(int argc, char **argv)
{
    //Globals::render_bbox2D = true;
    //Globals::render_tracking_numbers = true;

    // Set up ROS.
    ros::init(argc, argv, "pedestrian_trajectories");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string sub_topic_tracked_persons;

    string pub_topic_trajectories;

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));

    ROS_DEBUG("pedestrian_trajectories: Queue size for synchronisation is set to: %i", queue_size);

    // Create a subscriber.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    // The immediate unsubscribe is necessary to start without subscribing to any topic because message_filters does nor allow to do it another way.
    message_filters::Subscriber<TrackedPersons> subscriber_tracks(n, sub_topic_tracked_persons.c_str(), 1); subscriber_tracks.unsubscribe();

    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(subscriber_tracks));

    ///////////////////////////////////////////////////////////////////////////////////
    //Registering callback
    ///////////////////////////////////////////////////////////////////////////////////
    // With groundHOG
    sync_policies::ApproximateTime<Image, CameraInfo, GroundPlane,
            GroundHOGDetections, UpperBodyDetector, VisualOdometry> MySyncPolicyHOG(queue_size); //The real queue size for synchronisation is set here.
    MySyncPolicyHOG.setAgePenalty(1000); //set high age penalty to publish older data faster even if it might not be correctly synchronized.

    const sync_policies::ApproximateTime<Image, CameraInfo, GroundPlane,
            GroundHOGDetections, UpperBodyDetector, VisualOdometry> MyConstSyncPolicyHOG = MySyncPolicyHOG;

    Synchronizer< sync_policies::ApproximateTime<Image, CameraInfo, GroundPlane,
            GroundHOGDetections, UpperBodyDetector, VisualOdometry> >
            syncHOG(MyConstSyncPolicyHOG, subscriber_color, subscriber_camera_info, subscriber_gp,
                 subscriber_groundHOG, subscriber_upperbody, subscriber_vo);
    if(strcmp(topic_groundHOG.c_str(),"") != 0)
        syncHOG.registerCallback(boost::bind(&callbackWithHOG, _1, _2, _3, _4, _5, _6));

    // Create a topic publisher
    private_node_handle_.param("person_trajectories", pub_topic_tracked_persons, string("/frame/perception/tracked_persons"));
    pub_tracked_persons = n.advertise<frame_msgs::TrackedPersons>(pub_topic_tracked_persons, 10, con_cb, con_cb);


    ros::spin();
    return 0;
}


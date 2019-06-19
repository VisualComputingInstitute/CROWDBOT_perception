// ROS includes.
#include "ros/ros.h"
#include "ros/time.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <sensor_msgs/CameraInfo.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>

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

#include <QImage>
#include <QPainter>

#include "Matrix.h"
#include "Vector.h"
#include "Camera.h"
#include "Globals.h"
#include "ConfigFile.h"
#include "Hypo.h"
#include "Detections.h"
#include "AncillaryMethods.h"
#include "Tracker.h"

#include "rwth_perception_people_msgs/GroundPlane.h"
#include "rwth_perception_people_msgs/VisualOdometry.h"
#include "rwth_perception_people_msgs/PedestrianTracking.h"
#include "rwth_perception_people_msgs/PedestrianTrackingArray.h"
#include "frame_msgs/DetectedPersons.h"
#include "frame_msgs/TrackedPersons.h"
#include "frame_msgs/TrackedPersons2d.h"

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace rwth_perception_people_msgs;
using namespace frame_msgs;

ros::Publisher pub_message;
ros::Publisher pub_tracked_persons;

//cv::Mat img_depth_;
//cv_bridge::CvImagePtr cv_depth_ptr;	// cv_bridge for depth image

Vector< Hypo > HyposAll;
Detections *det_comb;
Tracker tracker;
tf::TransformListener* listener;
int cnt = 0;
unsigned long track_seq = 0;
int numAllDets = 0;


void ReadConfigFile(string path_config_file)
{

    ConfigFile config(path_config_file);

    //======================================
    // World scale
    //======================================
    config.readInto(Globals::WORLD_SCALE, "WORLD_SCALE");

    //======================================
    // height and width of images
    //======================================
    // SET VIA CAM_INFO, NOT CONFIG FILE!
    //Globals::dImHeight = config.read<int>("dImHeight");
    //Globals::dImWidth = config.read<int>("dImWidth");

    //====================================
    // Number of Frames / offset
    //====================================
    //Globals::numberFrames = config.read<int>("numberFrames");
    //Globals::nOffset = config.read<int>("nOffset");

    /////////////////////////////////TRACKING PART/////////////////////////

    // SET VIA ROS::TIME
    //Globals::frameRate = config.read<int>("frameRate");

    //======================================
    // Pedestrians width and height
    //======================================
    Globals::pedSizeWVis = config.read<double>("pedSizeWVis");
    Globals::pedSizeWCom = config.read<double>("pedSizeWCom");
    Globals::pedSizeHCom = config.read<double>("pedSizeHCom");

    //======================================
    // History
    //======================================
    Globals::history = config.read<int>("history");

    //======================================
    // Pedestrians parameter
    //======================================
    Globals::dObjHeight = config.read<double>("dObjHeight");
    Globals::dObjHVar = config.read<double>("dObjHVar");
    Globals::probHeight = config.read<double>("probHeight");

    //======================================
    // Adjustment for color histograms detections
    //======================================
    Globals::cutHeightBBOXforColor = config.read<double>("cutHeightBBOXforColor");
    Globals::cutWidthBBOXColor = config.read<double>("cutWidthBBOXColor");
    Globals::posponeCenterBBOXColor = config.read<double>("posponeCenterBBOXColor");
    config.readInto(Globals::binSize, "binSize");

    //======================================
    // Visualisation
    // Now handled by visualise parameter.
    //======================================
    //Globals::render_bbox3D = config.read("render_bbox3D", true);
    Globals::render_bbox2D = config.read("render_bbox2D", false);
    Globals::render_tracking_numbers = config.read("render_tracking_numbers", false);

    //======================================
    // MDL parameters for trajectories
    //======================================
    Globals::k1 = config.read<double>("k1");
    Globals::k2 = config.read<double>("k2");
    Globals::k3 = config.read<double>("k3");
    Globals::k4 = config.read<double>("k4");

    //======================================
    // Threshold for distinction between static/moving object
    //======================================
    Globals::minvel = config.read<double>("minvel");
    Globals::dMaxPedVel = config.read<double>("dMaxPedVel");

    //======================================
    // Threshold for identity management
    //======================================
    Globals::dSameIdThresh = config.read<double>("dSameIdThresh");
    Globals::reIdThresh_HypoLevel = config.read<double>("reIdThresh_HypoLevel");
    Globals::reIdThresh_DALevel = config.read<double>("reIdThresh_DALevel");

    //======================================
    // Trajectory
    //======================================
    Globals::threshLengthTraj = config.read<int>("threshLengthTraj");

    //======================================
    // Thresholds for accepted and displayed hypotheses
    //======================================
    Globals::dTheta2 = config.read<double>("dTheta2");

    //======================================
    // Time ant for temporal decay
    //======================================
    Globals::dTau = config.read<double>("dTau");

    //======================================
    // Time horizon for event cone search
    //======================================
    Globals::coneTimeHorizon = config.read<int>("coneTimeHorizon");
    Globals::maxHoleLen = config.read<int>("maxHoleLen");
    Globals::dHolePenalty = config.read<double>("dHolePenalty");

    // Q - the system covariance
    Globals::sysUncX = config.read<double>("sysUncX");
    Globals::sysUncY = config.read<double>("sysUncY");
    Globals::sysUncRot = config.read<double>("sysUncRot");
    Globals::sysUncVel = config.read<double>("sysUncVel");
    Globals::sysUncAcc = config.read<double>("sysUncAcc");

    /* P_init - the initial state covariance */
    Globals::initPX = config.read<double>("initPX");
    Globals::initPY = config.read<double>("initPY");
    Globals::initPVX = config.read<double>("initPVX");
    Globals::initPVY = config.read<double>("initPVY");

    Globals::kalmanObsMotionModelthresh = config.read<double>("kalmanObsMotionModelthresh");
    Globals::kalmanObsColorModelthresh = config.read<double>("kalmanObsColorModelthresh");

    Globals::accepted_frames_without_det = config.read<int>("accepted_frames_without_det");

    Globals::save_for_eval = config.read("save_for_eval", false);
    config.readInto(Globals::save_path_tracking, "save_path_tracking");
    config.readInto(Globals::save_path_img_info, "save_path_img_info");
    config.readInto(Globals::save_path_cam_info, "save_path_cam_info");
    config.readInto(Globals::save_path_img, "save_path_img");
    config.readInto(Globals::save_path_cam, "save_path_cam");
}

Camera createCamera(Vector<double>& GP,
                    const VisualOdometry::ConstPtr &vo,
                    const CameraInfoConstPtr &info) {
    // create camera from motion_matrix-, camera_info- and GP-topic
    //  * motion_matrix-topic need to have format [R|t] (+ 0 0 0 1 in last row)
    Matrix<double> motion_matrix(4,4, (double*) (&vo->transformation_matrix[0]));
    Matrix<double> R(motion_matrix, 0,2,0,2);
    Vector<double> t(motion_matrix(3,0), motion_matrix(3,1), motion_matrix(3,2));

    //  * K is read from camera_info-topic
    Matrix<double> K(3,3, (double*)&info->K[0]);

    //  * GP is read from GP-topic [n1 n2 n3 d] and transfered to World coordinates
    Camera camera(K, R, t, GP);
    Vector<double> GP_world = AncillaryMethods::PlaneToWorld(camera, GP);
    return Camera(K, R, t, GP_world);
}

void writeImageAndCamInfoToFile(const ImageConstPtr &color, const CameraInfoConstPtr &info, Camera camera)
{
        char safe_string_char[128];
        char nsec_int_char[128];
        ofstream aStream;
        sprintf(safe_string_char, Globals::save_path_img_info.c_str());
        aStream.open(safe_string_char, std::ios_base::app);
        if (cnt == 0) aStream << "image_nr frame_id seq stamp_sec.stamp_nsec" << endl;
        sprintf(nsec_int_char,"%09d", color->header.stamp.nsec);
        aStream << cnt << " ";
        aStream << color->header.frame_id << " ";
        aStream << color->header.seq << " ";
        aStream << color->header.stamp.sec << ".";
        aStream << nsec_int_char << endl;
        aStream.close();

        sprintf(safe_string_char, Globals::save_path_cam_info.c_str());
        aStream.open(safe_string_char, std::ios_base::app);
        if (cnt == 0) aStream << "image_nr frame_id seq stamp_sec.stamp_nsec" << endl;
        sprintf(nsec_int_char,"%09d", info->header.stamp.nsec);
        aStream << cnt << " ";
        aStream << info->header.frame_id<< " ";
        aStream << info->header.seq << " ";
        aStream << info->header.stamp.sec << ".";
        aStream << nsec_int_char << endl;
        aStream.close();

        sprintf(safe_string_char, Globals::save_path_cam.c_str(), cnt);

        camera.get_K().appendToTXT(safe_string_char);

        //distortion in info->D, but never used, so just add 0-row
        aStream.open(safe_string_char, std::ios_base::app);
        aStream << endl;
        aStream << "0.000000 0.000000 0.000000" << endl;
        aStream << endl;
        aStream.close();

        camera.get_R().appendToTXT(safe_string_char);
        aStream.open(safe_string_char, std::ios_base::app);
        aStream << endl;
        aStream.close();

        camera.get_t().appendToTXT(safe_string_char);
        aStream.open(safe_string_char, std::ios_base::app);
        aStream << endl;
        aStream.close();

        camera.get_GP().appendToTXT(safe_string_char);
}

Vector<double> projectPlaneToCam(Vector<double> p, Camera cam)
{
    Vector<double> gpInCam(4, 0.0);

    Vector<double> pv;
    pv.pushBack(p(0));
    pv.pushBack(p(1));
    pv.pushBack(p(2));

    Vector<double> camPos = cam.get_t();

    Matrix<double> camRot = cam.get_R();

    pv = Transpose(camRot)*pv;
    camRot *= -1.0;
    Vector<double> t = Transpose(camRot)*camPos;

    double d = p(3) - (pv(0)*t(0) + pv(1)*t(1) + pv(2)*t(2));

    gpInCam(0) = pv(0);
    gpInCam(1) = pv(1);
    gpInCam(2) = pv(2);
    gpInCam(3) = d;

    return gpInCam;
}

void callback(const DetectedPersons::ConstPtr &detections)
{
    ROS_DEBUG("Entered tracking callback");

    //std::cout << "===> cnt: " << cnt << std::endl;
    Globals::currentFrame = cnt;

    // ---update framerate + framerateVector OR dt + dtVector (descending order! dtVector(0) is the latest!)---
    //printf("set new framerate to: %d / (%f-%f) \n", 1,color->header.stamp.toSec(),Globals::oldTimeForFPSUpdate);
    //printf("result: %d\n", (int) (1 / (color->header.stamp.toSec()-Globals::oldTimeForFPSUpdate)));
    // update framerate first after some tracking cycles, before use framerate from config file
    if (cnt>0) {
        double dt = detections->header.stamp.toSec() - Globals::oldTimeForFPSUpdate;
        double fps = 1.0 / dt; //replaced with direct dt as discretization of dt for lower framerates is too coarse

        //framerate-based
        /*if(!std::isfinite(fps) || fps < 1) {
            ROS_WARN("Abnormal frame rate detected: %f, dt: %f. Set to 1", fps, dt);
            Globals::frameRate = 1;
        }
        else {
            Globals::frameRate = (int) fps;
        }*/

        //dt-based
        if(!std::isfinite(dt) || dt <= 0) {
            ROS_WARN("Abnormal dt detected: %f. Set to 1.0", dt);
            Globals::dt = 1.0;
        }
        else {
            Globals::dt = dt;
        }
        // fill dtVector
        if (Globals::dtVector.getSize() == 0 || Globals::history<2){

            Globals::dtVector.resize(1);
            Globals::dtVector(0) = Globals::dt;
        }
        else{
            Globals::dtVector.swap();
            Globals::dtVector.pushBack(Globals::dt);
            Globals::dtVector.swap();
            Globals::dtVector.resize(min(cnt,Globals::history));
        }
    }
    else{
        //first cycle, setup dtVector
        Globals::dtVector.clearContent();
    }
    //printf("---\n");
    //Globals::dtVector.show();
    //printf("---\n");
    // ---end update framerate+frameRatevector OR dt + dtVector---

    // OLD IMAGE STUFF (IMAGE NOT USED ANYMORE)
    //cim.resize(info->width,info->height,1,3);
    //Globals::dImHeight = info->height;
    //Globals::dImWidth = info->width;

    // safe strings for eval
    std::string safe_string;
    char safe_string_char[128];

    // Get camera from VO and GP (CAMERA+GP NOT USED ANYMORE)
    //Vector<double> GP(3, (double*) &gp->n[0]);
    //GP.pushBack((double) gp->d);
    //Camera camera = createCamera(GP, vo, info);

    // Get detections from upper body
    //Vector<double> single_detection(9);
    //Vector<Vector< double > > detected_bounding_boxes;

    /*
    for(int i = 0; i < upper->detections.size(); i++)
    {
        single_detection(0) = cnt;
        single_detection(1) = i;
        single_detection(2) = 1;
        //single_detection(3) = 1 - upper->dist[i]; // make sure that the score is always positive
        single_detection(3) = upper->detections[i].confidence;
        single_detection(4) = upper->detections[i].bbox_x;
        single_detection(5) = upper->detections[i].bbox_y;
        single_detection(6) = upper->detections[i].bbox_w;
        single_detection(7) = upper->detections[i].bbox_h;
        single_detection(8) = upper->detections[i].pose.pose.position.z;
        // ROS_INFO("upper det %i bbox: (%f, %f ,%f ,%f): %f", i, single_detection(4), single_detection(5), single_detection(6), single_detection(7), single_detection(3) );
        //ROS_INFO("Depth for upper det: %f", single_detection(8));
        //single_detection(9) = upper->detections[i].pose.pose.position.x;
        //single_detection(10) = upper->detections[i].pose.pose.position.y;
        //single_detection(11) = upper->detections[i].pose.pose.position.z;
        detected_bounding_boxes.pushBack(single_detection);
        //ROS_INFO("upper det %i bbox: (%f, %f ,%f ,%f): %f", i, single_detection(4), single_detection(5), single_detection(6), single_detection(7), single_detection(3) );
        //ROS_INFO("Depth for upper det: %f", single_detection(8));
    }
    */

    //get_image((unsigned char*)(&color->data[0]),info->width,info->height,cim);

    //writeImageAndCamInfoToFile(color,info,camera);

    /*if (Globals::save_for_eval){
        // save color image, going to be processed
        safe_string = Globals::save_path_img;
        safe_string.append("img_%08d.jpg");
        sprintf(safe_string_char, safe_string.c_str(), cnt);
        cim.save(safe_string_char);
    }*/

    ///////////////////////////////////////////TRACKING///////////////////////////
    tracker.process_tracking_oneFrame(HyposAll, *det_comb, cnt, detections/*, cim, camera*/);
    ///////////////////////////////////////////TRACKING-END///////////////////////////

    /*if (Globals::save_for_eval){
        // save processed image
        safe_string = Globals::save_path_img;
        safe_string.append("track_%08d.jpg");
        sprintf(safe_string_char, safe_string.c_str(), cnt);
        cim.save(safe_string_char);
    }*/

    // publish tracks
    Vector<Hypo> hyposMDL = tracker.getHyposMDL();
    //double currentTime = color->header.stamp.toSec();
    PedestrianTrackingArray allHypoMsg;
    allHypoMsg.header = detections->header;

    // also prepare tracks
    frame_msgs::TrackedPersons trackedPersons;
    trackedPersons.header.stamp = detections->header.stamp;
    trackedPersons.header.seq = ++track_seq;
    trackedPersons.header.frame_id = "/robot/OdometryFrame"; //FIXME: world frame, maybe should not be hardcoded

    Vector<Vector<double> > trajPts;
    Vector<double> dir;
    Vector<double> robot_frame_dir(3);
    for(int i = 0; i < hyposMDL.getSize(); i++)
    {
        PedestrianTracking oneHypoMsg;
        oneHypoMsg.header = detections->header;
        hyposMDL(i).getTrajPts(trajPts);
        for(int j = 0; j < trajPts.getSize(); j++)
        {
            oneHypoMsg.traj_x.push_back(trajPts(j)(0));
            oneHypoMsg.traj_y.push_back(trajPts(j)(1));
            oneHypoMsg.traj_z.push_back(trajPts(j)(2));
            //Vector<double> posInCamera = AncillaryMethods::fromWorldToCamera(trajPts(j), camera);

            /*oneHypoMsg.traj_x_camera.push_back(posInCamera(0));
            oneHypoMsg.traj_y_camera.push_back(posInCamera(1));
            oneHypoMsg.traj_z_camera.push_back(posInCamera(2));*/
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
        //printf("C(curr_idx):\n");
        //C(curr_idx).Show();
        //std::cout << "hypo " << hyposMDL(i).getHypoID() << " C size: " << C.getSize() << std::endl;

        // init one tracked person
        frame_msgs::TrackedPerson trackedPerson;
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
        trackedPerson.is_matched = false;
        trackedPerson.detection_id = 0; //actually 0 is wrong, as detection 0 exist, better use -1, but detection_id is unsigned
        if(frameInlier.getSize() > 0){
            Vector<int> currentInlier = frameInlier(frameInlier.getSize()-1).getInlier();
            if (frameInlier(frameInlier.getSize()-1).getFrame() == cnt){
                trackedPerson.is_matched = true;
                trackedPerson.detection_id = currentInlier(0) + numAllDets; //current position in detVector + num of all (previous) dets = current DetID
            }
        }
        trackedPerson.height = hyposMDL(i).getHeight();

        // update ReID embedding vector of person, only if it is currently matched (no averaging, just take latest detection embedding)
        // also update if not currently matched (last known detection embedding is used)
        //if(trackedPerson.is_matched){
            Vector<double> curr_emb_vec = hyposMDL(i).getEmb_vec();
            trackedPerson.embed_vector.clear();
            for(int evi=0; evi<curr_emb_vec.getSize(); evi++){
                trackedPerson.embed_vector.push_back(curr_emb_vec(evi));
            }
        //}

        // prepare position and velocity of tracked person
        Vector<double> vCurrVX;
        Vector<double> vCurrVY;
        double currVX;
        double currVY;
        hyposMDL(i).getVX(vCurrVX);
        hyposMDL(i).getVY(vCurrVY);
        currVX = vCurrVX(vCurrVX.getSize()-1);
        currVY = vCurrVY(vCurrVY.getSize()-1);

        robot_frame_dir(0) = currVX; //dir(0);//dir(2);
        robot_frame_dir(1) = currVY; //dir(1);//-dir(0);
        robot_frame_dir(2) = 0.0; //dir(2);//-dir(1);

        // Some constants for determining the pose
        const double AVERAGE_ROTATION_VARIANCE = pow(10.0 / 180 * M_PI, 2); // FIXME: determine from vx, vy?
        const double INFINITE_VARIANCE = 9999999; // should not really use infinity here because then the covariance matrix cannot be rotated (singularities!)

        // Set pose (=position + orientation)
        pose.pose.position.x = trajPts(curr_idx)(0);//trajPts(curr_idx)(2);
        pose.pose.position.y = trajPts(curr_idx)(1);//-trajPts(curr_idx)(0);
        pose.pose.position.z = 0;//trajPts(curr_idx)(2);//-trajPts(curr_idx)(1);

        // Set orientation
        //pose.pose.orientation = tf::createQuaternionMsgFromYaw(atan2(robot_frame_dir(1), robot_frame_dir(0))); // determine orientation from current velocity estimate
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(atan2(currVY, currVX)); // determine orientation from current velocity estimate
        pose.covariance.fill(0.0);
        pose.covariance[0 * 6 + 0] = C(curr_idx)(0,0); // variance of x position
        pose.covariance[1 * 6 + 1] = C(curr_idx)(1,1); // variance of y position
        pose.covariance[2 * 6 + 2] = INFINITE_VARIANCE; // variance of z position
        pose.covariance[3 * 6 + 3] = INFINITE_VARIANCE; // variance of x rotation
        pose.covariance[4 * 6 + 4] = INFINITE_VARIANCE; // variance of y rotation
        pose.covariance[5 * 6 + 5] = AVERAGE_ROTATION_VARIANCE; // variance of z rotation

        // Set twist (=velocities)
        twist.twist.linear.x = robot_frame_dir(0);
        twist.twist.linear.y = robot_frame_dir(1);
        twist.twist.linear.z = robot_frame_dir(2);

        twist.covariance.fill(0.0);
        twist.covariance[0 * 6 + 0] = C(curr_idx)(2,2); // variance of x linear velocity
        twist.covariance[1 * 6 + 1] = C(curr_idx)(3,3); // variance of y linear velocity
        twist.covariance[2 * 6 + 2] = INFINITE_VARIANCE; // variance of z linear velocity
        twist.covariance[3 * 6 + 3] = INFINITE_VARIANCE; // variance of x angular velocity
        twist.covariance[4 * 6 + 4] = INFINITE_VARIANCE; // variance of y angular velocity
        twist.covariance[5 * 6 + 5] = INFINITE_VARIANCE; // variance of z angular velocity


        // set pose and twist and add to tracked persons
        trackedPerson.pose = pose;
        trackedPerson.twist = twist;
        trackedPersons.tracks.push_back(trackedPerson);


    }

    pub_message.publish(allHypoMsg);
    pub_tracked_persons.publish(trackedPersons);
    Globals::oldTimeForFPSUpdate = detections->header.stamp.toSec(); //ros::Time::now().toSec();
    // number of detections over all Frames (for detID)
    numAllDets += detections->detections.size();
    cnt++;
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(message_filters::Subscriber<DetectedPersons> &sub_det){
    if(!pub_message.getNumSubscribers()
    && !pub_tracked_persons.getNumSubscribers()
    ) {
        ROS_DEBUG("Tracker: No subscribers. Unsubscribing.");
        sub_det.unsubscribe();
    } else {
        ROS_DEBUG("Tracker: New subscribers. Subscribing.");
        sub_det.subscribe();
    }
}

int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "pedestrian_tracking");
    ros::NodeHandle n;

    listener = new tf::TransformListener();

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string config_file;
    string topic_detections;

    string pub_topic;
    string pub_topic_tracked_persons;

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("config_file", config_file, string(""));

    private_node_handle_.param("detections", topic_detections, string("/detected_persons"));

    if(strcmp(config_file.c_str(),"") == 0) {
        ROS_ERROR("No config file specified.");
        ROS_ERROR("Run with: rosrun rwth_pedestrian_tracking pedestrian_tracking _config_file:=/path/to/config");
        exit(0);
    }

    ReadConfigFile(config_file);
    det_comb = new Detections(23, 0);

    ROS_DEBUG("pedestrian_tracker: Queue size for synchronisation is set to: %i", queue_size);

    // Image transport handle
    //image_transport::ImageTransport it(private_node_handle_);

    // Create a subscriber.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    // The immediate unsubscribe is necessary to start without subscribing to any topic because message_filters does nor allow to do it another way.
    message_filters::Subscriber<DetectedPersons> subscriber_detections(n, topic_detections.c_str(), 1); subscriber_detections.unsubscribe();

    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(subscriber_detections));

    ///////////////////////////////////////////////////////////////////////////////////
    //Registering callback
    ///////////////////////////////////////////////////////////////////////////////////

    // TRACKER NOW ONLY DEPENDS ON DETECTIONS: NO SYNC NEEDED
    //sync_policies::ApproximateTime<DetectedPersons> MySyncPolicy(queue_size); //The real queue size for synchronisation is set here.
    //MySyncPolicy.setAgePenalty(1000); //set high age penalty to publish older data faster even if it might not be correctly synchronized.

    //const sync_policies::ApproximateTime<DetectedPersons> MyConstSyncPolicy = MySyncPolicy;

    //Synchronizer< sync_policies::ApproximateTime<DetectedPersons> > sync(MyConstSyncPolicy, subscriber_detections);
    //sync.registerCallback(boost::bind(&callback, _1));

    subscriber_detections.registerCallback(boost::bind(&callback, _1));

    ///////////////////////////////////////////////////////////////////////////////////

    // Create a topic publisher
    private_node_handle_.param("pedestrian_array", pub_topic, string("/pedestrian_tracking/pedestrian_array"));
    pub_message = n.advertise<PedestrianTrackingArray>(pub_topic.c_str(), 10, con_cb, con_cb);

    private_node_handle_.param("tracked_persons", pub_topic_tracked_persons, string("/frame/perception/tracked_persons"));
    pub_tracked_persons = n.advertise<frame_msgs::TrackedPersons>(pub_topic_tracked_persons, 10, con_cb, con_cb);

    ros::spin();
    return 0;
}


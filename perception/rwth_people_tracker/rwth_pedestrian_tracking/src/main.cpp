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

#include "rwth_perception_people_msgs/UpperBodyDetector.h"
#include "rwth_perception_people_msgs/GroundPlane.h"
#include "rwth_perception_people_msgs/GroundHOGDetections.h"
#include "rwth_perception_people_msgs/VisualOdometry.h"
#include "rwth_perception_people_msgs/PedestrianTracking.h"
#include "rwth_perception_people_msgs/PedestrianTrackingArray.h"
#include "spencer_tracking_msgs/TrackedPersons.h"
#include "spencer_tracking_msgs/TrackedPersons2d.h"
#include <spencer_tracking_msgs/TrackingTimingMetrics.h>



using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace rwth_perception_people_msgs;

ros::Publisher pub_message;
image_transport::Publisher pub_image;
ros::Publisher pub_tracked_persons;
ros::Publisher pub_tracked_persons_2d;
ros::Publisher m_averageProcessingRatePublisher, m_averageCycleTimePublisher, m_trackCountPublisher, m_averageLoadPublisher, m_timingMetricsPublisher;

boost::circular_buffer<double> m_lastCycleTimes;
clock_t m_startClock;
clock_t m_clockBefore;
ros::WallTime m_startWallTime;
ros::WallTime m_wallTimeBefore;
bool m_timingInitialized;

cv::Mat img_depth_;
cv_bridge::CvImagePtr cv_depth_ptr;	// cv_bridge for depth image

//string path_config_file = "/home/mitzel/Desktop/sandbox/Demo/upper_and_cuda/bin/config_Asus.inp";

Vector< Hypo > HyposAll;
Detections *det_comb;
Tracker tracker;
tf::TransformListener* listener;
int cnt = 0;
unsigned long track_seq = 0;

//CImgDisplay* main_disp;
CImg<unsigned char> cim(1920,1080,1,3); //TODO: no hardcoded image dimensions

Vector<double> fromCam2World(Vector<double> posInCamera, Camera cam)
{
    Matrix<double> rotMat = cam.get_R();

    Vector<double> posCam = cam.get_t();

    Matrix<double> trMat(4,4,0.0);
    trMat(3,3) = 1;
    trMat(0,0) = rotMat(0,0);
    trMat(0,1) = rotMat(0,1);
    trMat(0,2) = rotMat(0,2);
    trMat(1,0) = rotMat(1,0);
    trMat(1,1) = rotMat(1,1);
    trMat(1,2) = rotMat(1,2);
    trMat(2,0) = rotMat(2,0);
    trMat(2,1) = rotMat(2,1);
    trMat(2,2) = rotMat(2,2);

    posCam *= Globals::WORLD_SCALE;

    trMat(3,0) = posCam(0);
    trMat(3,1) = posCam(1);
    trMat(3,2) = posCam(2);

    Vector<double> transpoint = trMat*posInCamera;
    return transpoint;

}

double computeDepthInCam(Vector<double> vbbox, Camera cam)
{
    Vector<double> pos3D;
    double distance;

    cam.bbToDetection(vbbox, pos3D, Globals::WORLD_SCALE, distance);
    return pos3D(2);
}

void get_image(unsigned char* b_image, uint w, uint h, CImg<unsigned char>& cim)
{
    unsigned char* ptr = b_image;
    for (unsigned int row = 0; row < h; ++row)
    {
        for (unsigned int col = 0; col < w; ++col)
        {
            // access the viewerImage as column, row
            cim(col,row,0,0) = *(ptr++); // red component
            cim(col,row,0,1) = *(ptr++); // green
            cim(col,row,0,2) = *(ptr++); // blue
        }
    }
}

void ReadConfigFile(string path_config_file)
{

    ConfigFile config(path_config_file);

    //=====================================
    // Input paths
    //=====================================
    config.readInto(Globals::camPath_left, "camPath_left");
    config.readInto(Globals::sImagePath_left, "sImagePath_left");
    config.readInto(Globals::tempDepthL, "tempDepthL");
    config.readInto(Globals::path_to_planes, "path_to_planes");

    //=====================================
    // Distance Range Accepted Detections
    //=====================================
    Globals::distance_range_accepted_detections = config.read<double>("distance_range_accepted_detections", 7);

    //======================================
    // ROI
    //======================================
    Globals::inc_width_ratio = config.read<double>("inc_width_ratio");
    Globals::inc_height_ratio = config.read<double>("inc_height_ratio");
    Globals::region_size_threshold = config.read<double>("region_size_threshold", 10);

    //======================================
    // Freespace Parameters
    //======================================
    Globals::freespace_scaleZ = config.read<double>("freespace_scaleZ", 20);
    Globals::freespace_scaleX = config.read<double>("freespace_scaleX", 20);
    Globals::freespace_minX = config.read<double>("freespace_minX", -20);
    Globals::freespace_minZ = config.read<double>("freespace_minZ", 0);
    Globals::freespace_maxX = config.read<double>("freespace_maxX", 20);
    Globals::freespace_maxZ = config.read<double>("freespace_maxZ", 30);
    Globals::freespace_threshold = config.read<double>("freespace_threshold", 120);
    Globals::freespace_max_depth_to_cons = config.read<int>("freespace_max_depth_to_cons", 20);

    //======================================
    // Evaluation Parameters
    //======================================
    Globals::evaluation_NMS_threshold = config.read<double>("evaluation_NMS_threshold",0.4);
    Globals::evaluation_NMS_threshold_LM = config.read<double>("evaluation_NMS_threshold_LM",0.4);
    Globals::evaluation_NMS_threshold_Border = config.read<double>("evaluation_NMS_threshold_Border",0.4);
    Globals::evaluation_inc_height_ratio = config.read<double>("evaluation_inc_height_ratio",0.2);
    Globals::evaluation_stride = config.read<int>("evaluation_stride",3);
    Globals::evaluation_scale_stride = config.read<double>("evaluation_scale_stride",1.03);
    Globals::evaluation_nr_scales = config.read<int>("evaluation_nr_scales",1);
    Globals::evaluation_inc_cropped_height = config.read<int>("evaluation_inc_cropped_height",20);
    Globals::evaluation_greedy_NMS_overlap_threshold = config.read<double>("evaluation_greedy_NMS_overlap_threshold", 0.1);
    Globals::evaluation_greedy_NMS_threshold = config.read<double>("evaluation_greedy_NMS_threshold", 0.25);
    //======================================
    // World scale
    //======================================
    config.readInto(Globals::WORLD_SCALE, "WORLD_SCALE");

    //======================================
    // height and width of images
    //======================================
    Globals::dImHeight = config.read<int>("dImHeight");
    Globals::dImWidth = config.read<int>("dImWidth");

    //======================================
    // Camera
    //======================================
    Globals::baseline = config.read<double>("baseline");

    //====================================
    // Number of Frames / offset
    //====================================
    Globals::numberFrames = config.read<int>("numberFrames");
    Globals::nOffset = config.read<int>("nOffset");

    //======================================
    // Console output
    //======================================
    //Globals::verbose = config.read("verbose", false);

    //=====================================
    // Determines if save bounding boxes or not
    //=====================================
    Globals::export_bounding_box = config.read("export_bounding_box", false);
    // Path of exported bounding boxes
    config.readInto(Globals::bounding_box_path, "bounding_box_path");

    //=====================================
    // Determines if save result images or not
    //=====================================
    Globals::export_result_images = config.read("export_result_images", false);
    config.readInto(Globals::result_images_path, "result_images_path");

    //====================================
    // Size of Template
    //====================================
    Globals::template_size = config.read<int>("template_size");


    /////////////////////////////////TRACKING PART/////////////////////////
    //======================================
    // Detections
    //======================================
    Globals::cutDetectionsUsingDepth = config.read("cutDetectionsUsingDepth", false);

    Globals::frameRate = config.read<int>("frameRate");

    //======================================
    // Camera
    //======================================
    Globals::farPlane = config.read<double>("farPlane");

    //======================================
    // World scale
    //======================================
    config.readInto(Globals::binSize, "binSize");

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

    //======================================
    // Adjustment for HOG detections
    //======================================
    Globals::cutHeightBBOXforColor = config.read<double>("cutHeightBBOXforColor");
    Globals::cutWidthBBOXColor = config.read<double>("cutWidthBBOXColor");
    Globals::posponeCenterBBOXColor = config.read<double>("posponeCenterBBOXColor");

    //======================================
    // Thresholds for combining the detection from left and right camera
    //======================================
    Globals::probHeight = config.read<double>("probHeight");

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

    Globals::kalmanObsMotionModelthresh = config.read<double>("kalmanObsMotionModelthresh");
    Globals::kalmanObsColorModelthresh = config.read<double>("kalmanObsColorModelthresh");

    /////////////////////////////////GP Estimator/////////////////////////
    Globals::nrInter_ransac = config.read<int>("nrInter_ransac");
    Globals::numberOfPoints_reconAsObstacle = config.read<int>("numberOfPoints_reconAsObstacle");

    //======================================
    // ROI Segmentation
    //======================================
    // Blurring parameters
    Globals::sigmaX = config.read<double>("sigmaX", 2.0);
    Globals::precisionX = config.read<double>("precisionX", 2.0);
    Globals::sigmaZ = config.read<double>("sigmaZ", 3.0);
    Globals::precisionZ = config.read<double>("precisionZ", 2.0);

    Globals::max_height = config.read<double>("max_height", 2.0);
    Globals::min_height = config.read<double>("min_height", 1.4);

    ///////////////////////////Recording /////////////////////
    Globals::from_camera = config.read("from_camera", true);
    config.readInto(Globals::from_file_path, "from_file_path");

    //////////////////////////Streaming///////////////////////
    config.readInto(Globals::stream_dest_IP, "stream_dest_IP");

    ////////////////////////HOG Detector////////////////////////
    Globals::hog_max_scale = config.read<float>("hog_max_scale",1.9);
    Globals::hog_score_thresh = config.read<float>("hog_score_thresh",0.4);

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
        //static string path_to_results_image_info = "/home/stefan/results/spencer_tracker/image_info.txt";
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

        //static string path_to_results_camera_info = "/home/stefan/results/spencer_tracker/camera_info.txt";
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

        //static string path_to_results_camera = "/home/stefan/results/spencer_tracker/camera/camera_%08d.txt";
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

void publishStatistics(ros::Time currentRosTime, const unsigned int numberTracks)
{
    // Get walltime since start
    ros::WallTime endTime = ros::WallTime::now();

    // Get cpu time since start
    clock_t endClock = clock();
    double wallTimeSinceStart = (endTime - m_startWallTime).toSec();
    double clockTimeSinceStart = ((double) (endClock - m_startClock)) / CLOCKS_PER_SEC;
    double avgLoad = (clockTimeSinceStart/wallTimeSinceStart)*100.0;

    double wallTimeCycle = (endTime - m_wallTimeBefore).toSec();
    double clockTimeCycle = ((double) (endClock - m_clockBefore)) / CLOCKS_PER_SEC;
    double currentLoad = (clockTimeCycle/wallTimeCycle)*100.0;

    // Calculate average
    double averageCycleTime = 0.0;
    foreach(float cycleTime, m_lastCycleTimes) {
        averageCycleTime += cycleTime;
    }
    averageCycleTime /= m_lastCycleTimes.size();

    // Publish average cycle time
    std_msgs::Float32 averageCycleTimeMsg;
    averageCycleTimeMsg.data = averageCycleTime;
    std_msgs::Float32 averageProcessingRateMsg;
    averageProcessingRateMsg.data = 1.0 / averageCycleTime;

    // Wait until buffer is full
    if(m_lastCycleTimes.size() == m_lastCycleTimes.capacity())
    {
        m_averageCycleTimePublisher.publish(averageCycleTimeMsg);

        // Publish average processing time
        m_averageProcessingRatePublisher.publish(averageProcessingRateMsg);
    }

    // Publish average cpu load
    std_msgs::Float32 averageLoadMsg;
    averageLoadMsg.data = avgLoad;
    m_averageLoadPublisher.publish(averageLoadMsg);

    // Publish timing metrics
    spencer_tracking_msgs::TrackingTimingMetrics timingMetrics;
    timingMetrics.header.seq = cnt;//m_tracker->getCurrentCycleNo();
    timingMetrics.header.frame_id = "odom";
    timingMetrics.header.stamp = currentRosTime;

    timingMetrics.cycle_no = cnt;//m_tracker->getCurrentCycleNo();
    timingMetrics.track_count = numberTracks;
    timingMetrics.average_cycle_time = averageCycleTime;
    timingMetrics.average_processing_rate = 1.0 / averageCycleTime;
    timingMetrics.cycle_time = m_lastCycleTimes.back();
    timingMetrics.elapsed_cpu_time = clockTimeSinceStart;
    timingMetrics.elapsed_time = wallTimeSinceStart;
    timingMetrics.cpu_load = currentLoad;
    timingMetrics.average_cpu_load = avgLoad;
    m_timingMetricsPublisher.publish(timingMetrics);

    m_wallTimeBefore = endTime;
    m_clockBefore = endClock;
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

void callbackWithoutHOG(const ImageConstPtr &color,
              const CameraInfoConstPtr &info,
              const GroundPlane::ConstPtr &gp,
              const UpperBodyDetector::ConstPtr &upper,
              const VisualOdometry::ConstPtr &vo)
{
    // for time benchmarking
    if (!m_timingInitialized)
    {
        m_startWallTime = m_wallTimeBefore = ros::WallTime::now();
        m_startClock = m_clockBefore =  clock();
        m_timingInitialized = true;
    }
    ros::Time currentRosTime = upper->header.stamp; // to make sure that timestamps remain exactly the same up to nanosecond precision (for ExactTime sync policy)

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

    // Save start time
    ros::WallTime startTime = ros::WallTime::now();
    clock_t startClock = clock();
    ///////////////////////////////////////////TRACKING///////////////////////////
    tracker.process_tracking_oneFrame(HyposAll, *det_comb, cnt, detected_bounding_boxes, cim, camera);
    ///////////////////////////////////////////TRACKING-END///////////////////////////
    // Save end time
    ros::WallTime endTime = ros::WallTime::now();
    double cycleTime = (endTime - startTime).toSec();
    m_lastCycleTimes.push_back( cycleTime );

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

    // also prepare tracks in Spencer format (+2d)
    spencer_tracking_msgs::TrackedPersons trackedPersons;
    spencer_tracking_msgs::TrackedPersons2d trackedPersons2d;
    trackedPersons.header.stamp = upper->header.stamp;
    trackedPersons.header.seq = ++track_seq;
    trackedPersons.header.frame_id = "/robot/OdometryFrame"; //FIXME: world frame, maybe should not be hardcoded

    trackedPersons2d.header.stamp = upper->header.stamp;
    trackedPersons2d.header.seq = track_seq;
    trackedPersons2d.header.frame_id = color->header.frame_id;

    Vector<Vector<double> > trajPts;
    Vector<double> dir;
    Vector<double> spencer_dir(3);
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

        // Also publish tracks in Spencer format
        geometry_msgs::PoseWithCovariance pose;
        geometry_msgs::TwistWithCovariance twist;
        int curr_idx = trajPts.getSize()-1;
        Vector<Matrix<double> > C;
        hyposMDL(i).getStateCovMats(C);
        //printf("C:\n");
        //C(curr_idx).Show();

        // init one tracked person
        spencer_tracking_msgs::TrackedPerson trackedPerson;
        spencer_tracking_msgs::TrackedPerson2d trackedPerson2d;
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

        spencer_dir(0) = dir(2);
        spencer_dir(1) = -dir(0);
        spencer_dir(2) = -dir(1);

        // Some constants for determining the pose
        //const double AVERAGE_ROTATION_VARIANCE = pow(10.0 / 180 * M_PI, 2); // FIXME: determine from vx, vy?
        const double INFINITE_VARIANCE = 9999999; // should not really use infinity here because then the covariance matrix cannot be rotated (singularities!)

        // Set pose (=position + orientation)
        pose.pose.position.x = trajPts(curr_idx)(2);
        pose.pose.position.y = -trajPts(curr_idx)(0);
        pose.pose.position.z = -trajPts(curr_idx)(1);

        // Set orientation
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(atan2(spencer_dir(1), spencer_dir(0))); // determine orientation from current velocity estimate
        pose.covariance.fill(0.0);
        pose.covariance[0 * 6 + 0] = C(curr_idx)(0,0); // variance of x position
        pose.covariance[1 * 6 + 1] = C(curr_idx)(1,1); // variance of y position
        pose.covariance[2 * 6 + 2] = INFINITE_VARIANCE; // variance of z position
        pose.covariance[3 * 6 + 3] = INFINITE_VARIANCE; // variance of x rotation
        pose.covariance[4 * 6 + 4] = INFINITE_VARIANCE; // variance of y rotation
        pose.covariance[5 * 6 + 5] = C(curr_idx)(2,2); // variance of z rotation

        // Set twist (=velocities)
        twist.twist.linear.x = spencer_dir(0);
        twist.twist.linear.y = spencer_dir(1);
        twist.twist.linear.z = spencer_dir(2);

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
    publishStatistics(currentRosTime, trackedPersons.tracks.size());
    Globals::oldTimeForFPSUpdate = color->header.stamp.toSec(); //ros::Time::now().toSec();
    cnt++;
}

void callbackWithHOG(const ImageConstPtr &color,
              const CameraInfoConstPtr &info,
              const GroundPlane::ConstPtr &gp,
              const GroundHOGDetections::ConstPtr& groundHOGDet,
              const UpperBodyDetector::ConstPtr &upper,
              const VisualOdometry::ConstPtr &vo)
{
    // for time benchmarking
    if (!m_timingInitialized)
    {
        m_startWallTime = m_wallTimeBefore = ros::WallTime::now();
        m_startClock = m_clockBefore =  clock();
        m_timingInitialized = true;
    }
    ros::Time currentRosTime = upper->header.stamp; // to make sure that timestamps remain exactly the same up to nanosecond precision (for ExactTime sync policy)

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

    ROS_DEBUG("Entered callback with groundHOG data");
    Globals::render_bbox3D = (pub_image.getNumSubscribers() > 0) || (Globals::save_for_eval);

    // safe strings for eval
    std::string safe_string;
    char safe_string_char[128];

    // Get camera from VO and GP
    Vector<double> GP(3, (double*) &gp->n[0]);
    GP.pushBack((double) gp->d);

    Camera camera = createCamera(GP, vo, info);

    //initial cam to compute HOG-depth
    Matrix<double> camRot_i = Eye<double>(3);
    Vector<double> camPos_i(3, 0.0);
    Vector<double> gp_i = camera.get_GP();
    Matrix<double> camInt_i = camera.get_K();
    Vector<double> planeInCam_i = projectPlaneToCam(gp_i, camera);
    Camera camI(camInt_i, camRot_i, camPos_i, planeInCam_i);

    // Get detections from HOG and upper body
    Vector<double> single_detection(9);
    Vector<Vector< double > > detected_bounding_boxes;

    for(int i = 0; i < groundHOGDet->pos_x.size(); i++)
    {
        single_detection(0) = cnt;
        single_detection(1) = i;
        single_detection(2) = 1;
        single_detection(3) = groundHOGDet->score[i];
        single_detection(4) = groundHOGDet->pos_x[i];
        single_detection(5) = groundHOGDet->pos_y[i];
        single_detection(6) = groundHOGDet->width[i];
        single_detection(7) = groundHOGDet->height[i];
        Vector<double> bbox(single_detection(4), single_detection(5), single_detection(6), single_detection(7));
        //ROS_INFO("HOG det %i bbox: (%f, %f ,%f ,%f): %f", i, single_detection(4), single_detection(5), single_detection(6), single_detection(7), single_detection(3) );
        single_detection(8) = computeDepthInCam(bbox, camI);
        //ROS_INFO("Depth for HOG det: %f", single_detection(8));
        detected_bounding_boxes.pushBack(single_detection);
    }

    for(int i = 0; i < upper->pos_x.size(); i++)
    {
        single_detection(0) = cnt;
        single_detection(1) = groundHOGDet->pos_x.size()+i;
        single_detection(2) = 1;
        single_detection(3) = 1 - upper->dist[i]; // make sure that the score is always positive
        single_detection(4) = upper->pos_x[i];
        single_detection(5) = upper->pos_y[i];
        single_detection(6) = upper->width[i];
        single_detection(7) = upper->height[i] * 3;
        single_detection(8) = upper->median_depth[i];
        // ROS_INFO("upper det %i bbox: (%f, %f ,%f ,%f): %f", i, single_detection(4), single_detection(5), single_detection(6), single_detection(7), single_detection(3) );
        //ROS_INFO("Depth for upper det: %f", single_detection(8));

        detected_bounding_boxes.pushBack(single_detection);
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

    // Save start time
    ros::WallTime startTime = ros::WallTime::now();
    clock_t startClock = clock();
    ///////////////////////////////////////////TRACKING///////////////////////////
    tracker.process_tracking_oneFrame(HyposAll, *det_comb, cnt, detected_bounding_boxes, cim, camera);
    ///////////////////////////////////////////TRACKING-END///////////////////////////
    // Save end time
    ros::WallTime endTime = ros::WallTime::now();
    double cycleTime = (endTime - startTime).toSec();
    m_lastCycleTimes.push_back( cycleTime );

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

    // also prepare tracks in Spencer format
    spencer_tracking_msgs::TrackedPersons trackedPersons;
    spencer_tracking_msgs::TrackedPersons2d trackedPersons2d;
    trackedPersons.header.stamp = upper->header.stamp;
    trackedPersons.header.seq = ++track_seq;
    trackedPersons.header.frame_id = "/robot/OdometryFrame"; //FIXME: world frame, maybe should not be hardcoded

    trackedPersons2d.header.stamp = upper->header.stamp;
    trackedPersons2d.header.seq = track_seq;
    trackedPersons2d.header.frame_id = color->header.frame_id;
    
    Vector<Vector<double> > trajPts;
    Vector<double> dir;
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

        // Also publish tracks in Spencer format
        geometry_msgs::PoseWithCovariance pose;
        geometry_msgs::TwistWithCovariance twist;
        int curr_idx = trajPts.getSize()-1;
        Vector<Matrix<double> > C;
        hyposMDL(i).getStateCovMats(C);

        // init one tracked person
        spencer_tracking_msgs::TrackedPerson trackedPerson;
        spencer_tracking_msgs::TrackedPerson2d trackedPerson2d;
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
        trackedPerson.detection_id = 1; // FIXME: available for mht tracker, yet?

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

        geometry_msgs::PoseStamped tempPoseStamped;
        geometry_msgs::PoseStamped tempPoseStampedWorld;
        geometry_msgs::Vector3Stamped tempVVectorStamped;
        geometry_msgs::Vector3Stamped tempVVectorStampedWorld;
        tempPoseStamped.header.stamp = ros::Time();
        tempPoseStamped.header.frame_id = upper->header.frame_id; //camera frame from upperbody
        tempPoseStamped.pose.position.x = posInCamera(0);
        tempPoseStamped.pose.position.y = posInCamera(1);
        tempPoseStamped.pose.position.z = posInCamera(2);
        tempVVectorStamped.header = tempPoseStamped.header;
        tempVVectorStamped.vector.x = dir(0);
        tempVVectorStamped.vector.y = dir(1);
        tempVVectorStamped.vector.z = dir(2);
        tempPoseStamped.pose.orientation = tf::createQuaternionMsgFromYaw(atan2(dir(2), dir(0))); // only dummy orientation, correct one is set below

        //Transform pose and velocity vector
        string target_frame = trackedPersons.header.frame_id;
        try {
            ROS_DEBUG("Transforming received position into %s coordinate system.", target_frame.c_str());
            listener->waitForTransform(tempPoseStamped.header.frame_id, target_frame, ros::Time(), ros::Duration(0.05));
            listener->transformPose(target_frame, tempPoseStamped, tempPoseStampedWorld);
            //listener->transformVector(target_frame, tempVVectorStamped, tempVVectorStampedWorld);
        }
        catch(tf::TransformException ex) {
            ROS_WARN("Failed transform: %s", ex.what());
            return;
        }
        //dir(0) = tempVVectorStampedWorld.vector.x;
        //dir(1) = tempVVectorStampedWorld.vector.y;
        //dir(2) = tempVVectorStampedWorld.vector.z;

        // Some constants for determining the pose
        const double AVERAGE_ROTATION_VARIANCE = pow(10.0 / 180 * M_PI, 2); // FIXME: determine from vx, vy?
        const double INFINITE_VARIANCE = 9999999; // should not really use infinity here because then the covariance matrix cannot be rotated (singularities!)

        // Set pose (=position + orientation)
        //pose.pose.position.x = tempPoseStampedWorld.pose.position.x;
        //pose.pose.position.y = tempPoseStampedWorld.pose.position.y;
        //pose.pose.position.z = 0.67;//tempPoseStampedWorld.pose.position.z; //FIXME: hard-coded groundplane height (also a bit off)
        pose.pose.position.x = trajPts(curr_idx)(2);
        pose.pose.position.y = -trajPts(curr_idx)(0);
        pose.pose.position.z = -trajPts(curr_idx)(1);

        // Set orientation
        Vector<double> spencer_dir(3, 0.0);
        spencer_dir(0) = dir(2);
        spencer_dir(1) = -dir(0);
        spencer_dir(2) = -dir(1);
        //unnormalize
        spencer_dir = spencer_dir*hyposMDL(i).getSpeed();

        pose.pose.orientation = tf::createQuaternionMsgFromYaw(atan2(spencer_dir(1), spencer_dir(0))); // determine orientation from current velocity estimate
        pose.covariance.fill(0.0);
        //printf("curr_idx: %i, size of C stateCov vector: %i, C:\n", curr_idx, C.getSize());
        //C(curr_idx).Show();
        pose.covariance[0 * 6 + 0] = C(curr_idx)(0,0); // variance of x position
        pose.covariance[1 * 6 + 1] = C(curr_idx)(1,1); // variance of y position
        pose.covariance[2 * 6 + 2] = INFINITE_VARIANCE; //C(curr_idx)(1,1); // variance of z position
        pose.covariance[3 * 6 + 3] = INFINITE_VARIANCE; // variance of x rotation
        pose.covariance[4 * 6 + 4] = INFINITE_VARIANCE; // variance of y rotation
        pose.covariance[5 * 6 + 5] = AVERAGE_ROTATION_VARIANCE; // variance of z rotation

        // Set twist (=velocities)
        twist.twist.linear.x = spencer_dir(0);
        twist.twist.linear.y = spencer_dir(1);
        twist.twist.linear.z = spencer_dir(2);

        twist.covariance.fill(0.0);
        twist.covariance[0 * 6 + 0] = C(curr_idx)(3,3); // variance of x linear velocity
        twist.covariance[1 * 6 + 1] = C(curr_idx)(3,3); // variance of y linear velocity
        twist.covariance[2 * 6 + 2] = INFINITE_VARIANCE; // variance of z linear velocity
        twist.covariance[3 * 6 + 3] = C(curr_idx)(2,2);; // variance of x angular velocity
        twist.covariance[4 * 6 + 4] = C(curr_idx)(2,2);; // variance of y angular velocity
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
        res_img.step   = color->step;
        res_img.width = cim._width;
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
    publishStatistics(currentRosTime, trackedPersons.tracks.size());
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
    ros::init(argc, argv, "pedestrian_tracking");
    ros::NodeHandle n;

    listener = new tf::TransformListener();

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string config_file;
    string cam_ns;
    string topic_gp;
    string topic_groundHOG;
    string topic_upperbody;
    string topic_vo;

    string pub_topic;
    string pub_image_topic;
    string pub_topic_tracked_persons;
    string pub_topic_tracked_persons_2d;

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("config_file", config_file, string(""));

    private_node_handle_.param("camera_namespace", cam_ns, string("/head_xtion"));
    private_node_handle_.param("ground_plane", topic_gp, string("/ground_plane"));
    private_node_handle_.param("ground_hog", topic_groundHOG, string("/groundHOG/detections"));
    private_node_handle_.param("upper_body_detections", topic_upperbody, string("/upper_body_detector/detections"));
    private_node_handle_.param("visual_odometry", topic_vo, string("/visual_odometry/motion_matrix"));

    string topic_color_image = cam_ns + "/hd/image_color_rect";
    string topic_camera_info = cam_ns + "/hd/camera_info";

    if(strcmp(config_file.c_str(),"") == 0) {
        ROS_ERROR("No config file specified.");
        ROS_ERROR("Run with: rosrun rwth_pedestrian_tracking pedestrian_tracking _config_file:=/path/to/config");
        exit(0);
    }

    ReadConfigFile(config_file);
    det_comb = new Detections(23, 0);

    ROS_DEBUG("pedestrian_tracker: Queue size for synchronisation is set to: %i", queue_size);

    // Image transport handle
    image_transport::ImageTransport it(private_node_handle_);

    // Create a subscriber.
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    // The immediate unsubscribe is necessary to start without subscribing to any topic because message_filters does nor allow to do it another way.
    image_transport::SubscriberFilter subscriber_color;
    subscriber_color.subscribe(it, topic_color_image.c_str(), 1); subscriber_color.unsubscribe(); //This subscribe and unsubscribe is just to set the topic name.
    message_filters::Subscriber<CameraInfo> subscriber_camera_info(n, topic_camera_info.c_str(), 1); subscriber_camera_info.unsubscribe();
    message_filters::Subscriber<GroundPlane> subscriber_gp(n, topic_gp.c_str(), 1); subscriber_gp.unsubscribe();
    message_filters::Subscriber<GroundHOGDetections> subscriber_groundHOG(n, topic_groundHOG.c_str(), 1); subscriber_groundHOG.unsubscribe();
    message_filters::Subscriber<UpperBodyDetector> subscriber_upperbody(n, topic_upperbody.c_str(), 1); subscriber_upperbody.unsubscribe();
    message_filters::Subscriber<VisualOdometry> subscriber_vo(n, topic_vo.c_str(), 1); subscriber_vo.unsubscribe();

    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(subscriber_camera_info),
                                                       boost::ref(subscriber_gp),
                                                       boost::ref(subscriber_groundHOG),
                                                       boost::ref(subscriber_upperbody),
                                                       boost::ref(subscriber_vo),
                                                       boost::ref(subscriber_color),
                                                       boost::ref(it));
    image_transport::SubscriberStatusCallback image_cb = boost::bind(&connectCallback,
                                                                     boost::ref(subscriber_camera_info),
                                                                     boost::ref(subscriber_gp),
                                                                     boost::ref(subscriber_groundHOG),
                                                                     boost::ref(subscriber_upperbody),
                                                                     boost::ref(subscriber_vo),
                                                                     boost::ref(subscriber_color),
                                                                     boost::ref(it));

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
    ///////////////////////////////////////////////////////////////////////////////////
    // Without groundHOG
    sync_policies::ApproximateTime<Image, CameraInfo, GroundPlane,
            UpperBodyDetector, VisualOdometry> MySyncPolicy(queue_size); //The real queue size for synchronisation is set here.
    MySyncPolicy.setAgePenalty(1000); //set high age penalty to publish older data faster even if it might not be correctly synchronized.

    const sync_policies::ApproximateTime<Image, CameraInfo, GroundPlane,
            UpperBodyDetector, VisualOdometry> MyConstSyncPolicy = MySyncPolicy;

    Synchronizer< sync_policies::ApproximateTime<Image, CameraInfo, GroundPlane,
            UpperBodyDetector, VisualOdometry> >
            sync(MyConstSyncPolicy, subscriber_color, subscriber_camera_info, subscriber_gp,
                 subscriber_upperbody, subscriber_vo);
    if(strcmp(topic_groundHOG.c_str(),"") == 0)
        sync.registerCallback(boost::bind(&callbackWithoutHOG, _1, _2, _3, _4, _5));
    ///////////////////////////////////////////////////////////////////////////////////

    // Create a topic publisher
    private_node_handle_.param("pedestrian_array", pub_topic, string("/pedestrian_tracking/pedestrian_array"));
    pub_message = n.advertise<PedestrianTrackingArray>(pub_topic.c_str(), 10, con_cb, con_cb);

    private_node_handle_.param("pedestrian_image", pub_image_topic, string("/pedestrian_tracking/image"));
    pub_image = it.advertise(pub_image_topic.c_str(), 1, image_cb, image_cb);

    private_node_handle_.param("tracked_persons", pub_topic_tracked_persons, string("/spencer/perception/tracked_persons"));
    pub_tracked_persons = n.advertise<spencer_tracking_msgs::TrackedPersons>(pub_topic_tracked_persons, 10, con_cb, con_cb);

    private_node_handle_.param("tracked_persons_2d", pub_topic_tracked_persons_2d, (pub_topic_tracked_persons + "_2d"));
    pub_tracked_persons_2d = n.advertise<spencer_tracking_msgs::TrackedPersons2d>(pub_topic_tracked_persons_2d, 10, con_cb, con_cb);

    // For benchmarking
    m_averageProcessingRatePublisher = private_node_handle_.advertise<std_msgs::Float32>("average_processing_rate", 1);
    m_averageCycleTimePublisher = private_node_handle_.advertise<std_msgs::Float32>("average_cycle_time", 1);
    m_trackCountPublisher = private_node_handle_.advertise<std_msgs::UInt16>("track_count", 1);
    m_averageLoadPublisher = private_node_handle_.advertise<std_msgs::Float32>("average_cpu_load", 1);
    m_timingMetricsPublisher = private_node_handle_.advertise<spencer_tracking_msgs::TrackingTimingMetrics>("tracking_timing_metrics", 10);

    // Set up circular buffer for benchmarking cycle times
    //m_lastCycleTimes.set_capacity(Params::get<int>("cycle_time_buffer_length", 50)); // = size of window for averaging
    m_lastCycleTimes.set_capacity(50); // = size of window for averaging

    ros::spin();
    return 0;
}


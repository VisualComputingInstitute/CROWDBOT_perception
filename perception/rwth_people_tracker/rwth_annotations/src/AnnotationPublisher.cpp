#include "ros/ros.h"
#include "ros/time.h"
#include "rwth_perception_people_msgs/AnnotatedFrame.h"
#include "rwth_perception_people_msgs/Annotation.h"
#include "rwth_perception_people_msgs/GroundPlane.h"
#include "rwth_perception_people_msgs/VisualOdometry.h"
#include "std_msgs/String.h"

#include <sensor_msgs/CameraInfo.h>

#include "Camera.h"
#include "frame_msgs/TrackedPersons.h"
#include "frame_msgs/TrackedPersons2d.h"

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

using namespace sensor_msgs;
using namespace rwth_perception_people_msgs;
using namespace message_filters;

typedef std::vector<std::string> stringvec;
typedef rwth_perception_people_msgs::Annotation annobox;
typedef rwth_perception_people_msgs::AnnotatedFrame annoframe;

namespace {

Vector<double> PlaneToWorld(const Camera &camera,
                            const Vector<double> &plane_in_camera) {
  Vector<double> pv(plane_in_camera(0), plane_in_camera(1), plane_in_camera(2));
  pv = camera.get_R() * pv;

  double d = plane_in_camera(3) - DotProduct(pv, camera.get_t());

  Vector<double> plane(4);
  plane(0) = pv(0) / pv.norm();
  plane(1) = pv(1) / pv.norm();
  plane(2) = pv(2) / pv.norm();
  plane(3) = d;

  return plane;
}

ros::Publisher pub_annotated_persons;
ros::Publisher pub_annotated_persons2d;
std::vector<annoframe> allAnnotatedFrames;

stringvec readFile(const char *filename) {
  std::ifstream filestr(filename, std::ifstream::in);

  stringvec outvec;

  std::string line;
  while (std::getline(filestr, line)) {
    outvec.push_back(line);
  }

  std::cout << filename << ": Parsed " << outvec.size() << " lines."
            << std::endl;

  return outvec;
}

annobox parseAnnoString(std::string annoString) {

  std::istringstream istr(annoString);
  int64_t frame, id;
  double topleft_x, topleft_y, width, height;
  istr >> frame >> id >> topleft_x >> topleft_y >> width >> height;

  // TODO: error handling?

  annobox a;
  a.frame = frame;
  a.id = id;
  a.tlx = topleft_x;
  a.tly = topleft_y;
  a.width = width;
  a.height = height;
  return a;
}

annoframe parseTimestampString(std::string timestampString) {
  int64_t frame, seq;
  std::string coordframe; // char coordframe[100];
  uint32_t seconds, nanoseconds;

  size_t pos = timestampString.find(".");
  if (pos == std::string::npos) {
    std::cout << "Malformed line in file. Could not find '.' delimiter."
              << std::endl;
    annoframe nope;
    return nope;
  }

  // Replace dot by whitespace
  timestampString[pos] = ' ';

  std::istringstream istr(timestampString);
  istr >> frame >> coordframe >> seq >> seconds >> nanoseconds;

  // TODO: error handling?
  annoframe annotatedFrame;
  annotatedFrame.frame = frame;
  annotatedFrame.timestamp.sec = seconds;
  annotatedFrame.timestamp.nsec = nanoseconds;

  return annotatedFrame;
}

Camera createCamera(Vector<double> &GP, const VisualOdometry::ConstPtr &vo,
                    const CameraInfoConstPtr &info) {
  // create camera from motion_matrix-, camera_info- and GP-topic
  //  * motion_matrix-topic need to have format [R|t] (+ 0 0 0 1 in last row)
  Matrix<double> motion_matrix(4, 4, (double *)(&vo->transformation_matrix[0]));
  Matrix<double> R(motion_matrix, 0, 2, 0, 2);
  Vector<double> t(motion_matrix(3, 0), motion_matrix(3, 1),
                   motion_matrix(3, 2));

  //  * K is read from camera_info-topic
  Matrix<double> K(3, 3, (double *)&info->K[0]);

  //  * GP is read from GP-topic [n1 n2 n3 d] and transfered to World
  //  coordinates
  Camera camera(K, R, t, GP);
  Vector<double> GP_world = PlaneToWorld(camera, GP);
  return Camera(K, R, t, GP_world);
}

Vector<double> projectPlaneToCam(Vector<double> p, Camera cam) {
  Vector<double> gpInCam(4, 0.0);

  Vector<double> pv;
  pv.pushBack(p(0));
  pv.pushBack(p(1));
  pv.pushBack(p(2));

  Vector<double> camPos = cam.get_t();

  Matrix<double> camRot = cam.get_R();

  pv = Transpose(camRot) * pv;
  camRot *= -1.0;
  Vector<double> t = Transpose(camRot) * camPos;

  double d = p(3) - (pv(0) * t(0) + pv(1) * t(1) + pv(2) * t(2));

  gpInCam(0) = pv(0);
  gpInCam(1) = pv(1);
  gpInCam(2) = pv(2);
  gpInCam(3) = d;

  return gpInCam;
}

Vector<double> fromCam2World(Vector<double> posInCamera, Camera cam) {
  Matrix<double> rotMat = cam.get_R();

  Vector<double> posCam = cam.get_t();

  Matrix<double> trMat(4, 4, 0.0);
  trMat(3, 3) = 1;
  trMat(0, 0) = rotMat(0, 0);
  trMat(0, 1) = rotMat(0, 1);
  trMat(0, 2) = rotMat(0, 2);
  trMat(1, 0) = rotMat(1, 0);
  trMat(1, 1) = rotMat(1, 1);
  trMat(1, 2) = rotMat(1, 2);
  trMat(2, 0) = rotMat(2, 0);
  trMat(2, 1) = rotMat(2, 1);
  trMat(2, 2) = rotMat(2, 2);

  trMat(3, 0) = posCam(0);
  trMat(3, 1) = posCam(1);
  trMat(3, 2) = posCam(2);

  Vector<double> transpoint = trMat * posInCamera;
  return transpoint;
}

// global var to keep track of which frame might be next
int currentIndex = 0;
int publishedBoxes = 0;

uint32_t lastSec = 0;

void callback(const CameraInfoConstPtr &info, const GroundPlane::ConstPtr &gp,
              const VisualOdometry::ConstPtr &vo) {

  ros::Time timestamp = info->header.stamp;

  if (lastSec != timestamp.sec) {
    lastSec = timestamp.sec;
    // std::cout << "Secs: " << lastSec << std::endl;
  }

  if (currentIndex >= allAnnotatedFrames.size()) {
    std::cout << std::endl;
    std::cout << "Annotation Callback: No further annotated frames available!"
              << std::endl;
    std::cout << "Published " << currentIndex + 1 << " frames and "
              << publishedBoxes << " boxes." << std::endl;

    // all available boxes published, shutdown
    pub_annotated_persons.shutdown();
    pub_annotated_persons2d.shutdown();
    ros::shutdown();
    return;
  }

  const annoframe &currentFrame = allAnnotatedFrames[currentIndex];

  if (timestamp < currentFrame.timestamp) {
    // Nothing to do here yet...
    return;
  }

  std::cout << "\rTimestamp diff: " << timestamp - currentFrame.timestamp
            << "       ";

  // Get camera from VO and GP
  Vector<double> GP(3, (double *)&gp->n[0]);
  GP.pushBack((double)gp->d);

  Camera camera = createCamera(GP, vo, info); // create Camera from main

  // initial cam to compute HOG-depth
  Matrix<double> camRot_i = Eye<double>(3);
  Vector<double> camPos_i(3, 0.0);
  Vector<double> gp_i = camera.get_GP();
  Matrix<double> camInt_i = camera.get_K();
  Vector<double> planeInCam_i = projectPlaneToCam(gp_i, camera);
  Camera camI(camInt_i, camRot_i, camPos_i, planeInCam_i);

  spencer_tracking_msgs::TrackedPersons annotatedPersons;
  annotatedPersons.header.stamp = currentFrame.timestamp;
  annotatedPersons.header.seq = currentIndex;
  annotatedPersons.header.frame_id =
      "odom"; //"optitrack" for eval, "odom" for vis

  spencer_tracking_msgs::TrackedPersons2d annotatedPersons2d;
  annotatedPersons2d.header.stamp = currentFrame.timestamp;
  annotatedPersons2d.header.seq = currentIndex;
  // annotatedPersons2d.header.frame_id = "rgbd_front_top_rgb_optical_frame";
  // //"optitrack" for eval, "odom" for vis
  annotatedPersons2d.header.frame_id =
      info->header.frame_id; //"optitrack" for eval, "odom" for vis

  auto &annos = currentFrame.annotations;

  // loop through anno-boxes
  for (auto &anno : annos) {
    // Convert from [x0 y0 x1 y1] to [x0 y0 w h] (Watch out: misleading name:
    // brx is actually width, bry is height FIXME)
    double anno_width = anno.width;
    double anno_height = anno.height;
    // std::cout << "ratio w/h: " << anno.brx/anno.bry << "(id: " << anno.id <<
    // ")" << std::endl; check bbox ratio and fix in a naive way (for boxes cut
    // off at the image boundaries)

    // check if bbox is cut left/right, in case let box grow in width -> often
    // minor differences + problems with height
    /*if(anno.tlx <1){
        //cut left
        anno.tlx -= anno_width
    }else if (anno.tlx+anno_width > info->width-1){
        //cut right

    }*/

    // check w/h ratio afterwards, if too big, let box grow in height
    if ((anno.tly + anno_height > info->height - 1) &&
        (anno_width / anno_height > 0.33)) {
      anno_height = 3 * anno_width;
    }
    Vector<double> bbox = {anno.tlx, anno.tly, anno_width, anno_height};
    Vector<double> pos3D_camera;
    double distance; // dummy, not used

    // Compute world position from bbox
    camI.bbToDetection(bbox, pos3D_camera, 1, distance); // 1: WorldScale
    Vector<double> posInCamCord(pos3D_camera(0), pos3D_camera(1),
                                pos3D_camera(2), 1);
    Vector<double> posInWorld = fromCam2World(posInCamCord, camera);

    // Init one tracked person (one 2D, one 3D)

    // 3D
    spencer_tracking_msgs::TrackedPerson annotatedPerson;
    annotatedPerson.track_id = anno.id;
    annotatedPerson.age = ros::Duration(0);
    annotatedPerson.is_occluded = false;
    annotatedPerson.is_matched = false;
    annotatedPerson.detection_id = 0;

    geometry_msgs::PoseWithCovariance pose;
    geometry_msgs::TwistWithCovariance twist;

    // Set pose (coordinate frame ([x y z 1]) to Spencer Tracked Persons: [z -x
    // -y]!)
    pose.pose.position.x = posInWorld[2];
    pose.pose.position.y = -posInWorld[0];
    pose.pose.position.z = -posInWorld[1];

    pose.covariance.fill(0.0);
    pose.covariance[0 * 6 + 0] = 0.1;
    pose.covariance[1 * 6 + 1] = 0.1;
    pose.covariance[2 * 6 + 2] = 0.1;
    pose.covariance[3 * 6 + 3] = 0.1;
    pose.covariance[4 * 6 + 4] = 0.1;
    pose.covariance[5 * 6 + 5] = 0.1;

    // Set twist (=velocities)
    twist.twist.linear.x = 0;
    twist.twist.linear.y = 0;
    twist.twist.linear.z = 0;
    twist.covariance.fill(0.0);

    // set pose and twist and add to tracked persons
    annotatedPerson.pose = pose;
    annotatedPerson.twist = twist;
    annotatedPersons.tracks.push_back(annotatedPerson);

    // 2D
    spencer_tracking_msgs::TrackedPerson2d annotatedPerson2d;
    annotatedPerson2d.track_id = anno.id;
    annotatedPerson2d.x = anno.tlx;
    annotatedPerson2d.y = anno.tly;
    annotatedPerson2d.w = anno.width;
    annotatedPerson2d.h = anno.height;
    annotatedPersons2d.boxes.push_back(annotatedPerson2d);

    ++publishedBoxes;
  }

  // Publish annotated persons
  pub_annotated_persons.publish(annotatedPersons);

  // Publish 2D annotations
  pub_annotated_persons2d.publish(annotatedPersons2d);

  // Increase index to go on with next annoframe next time...
  ++currentIndex;
}

// Connection callback that unsubscribes from the tracker if no one is
// subscribed.
void connectCallback(message_filters::Subscriber<CameraInfo> &sub_cam,
                     message_filters::Subscriber<GroundPlane> &sub_gp,
                     message_filters::Subscriber<VisualOdometry> &sub_vo) {
  if (!pub_annotated_persons.getNumSubscribers() &&
      !pub_annotated_persons2d.getNumSubscribers()) {
    ROS_DEBUG("Annotations: No subscribers. Unsubscribing.");
    sub_cam.unsubscribe();
    sub_gp.unsubscribe();
    sub_vo.unsubscribe();
  } else {
    ROS_DEBUG("Annotations: New subscribers. Subscribing.");
    sub_cam.subscribe();
    sub_gp.subscribe();
    sub_vo.subscribe();
  }
}

} // namespace

int main(int argc, char **argv) {

  ros::init(argc, argv, "annotationpublisher");
  ros::NodeHandle n;

  ///////////////////////////////////////////////////////////////////////////////////
  // Parameters
  int queue_size;
  string cam_ns;
  string topic_gp;
  string topic_vo;
  string boxes_path;
  string timestamps_path;
  string pub_topic_annotated_persons;
  string pub_topic_annotated_persons2d;

  ros::NodeHandle private_node_handle_("~");
  private_node_handle_.param("queue_size", queue_size, int(10));
  private_node_handle_.param("camera_namespace", cam_ns, string("/head_xtion"));
  private_node_handle_.param("ground_plane", topic_gp, string("/ground_plane"));
  private_node_handle_.param("visual_odometry", topic_vo,
                             string("/visual_odometry/motion_matrix"));
  private_node_handle_.param("boxes_path", boxes_path, string("boxes.txt"));
  private_node_handle_.param("timestamps_path", timestamps_path,
                             string("timestamps.txt"));

  string topic_camera_info = cam_ns + "/rgb/camera_info";
  ///////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////////////
  // Create a subscriber.
  // Set queue size to 1 because generating a queue here will only pile up
  // images and delay the output by the amount of queued images The immediate
  // unsubscribe is necessary to start without subscribing to any topic because
  // message_filters does not allow to do it another way.

  message_filters::Subscriber<CameraInfo> subscriber_camera_info(
      n, topic_camera_info.c_str(), 1);
  subscriber_camera_info.unsubscribe();
  message_filters::Subscriber<GroundPlane> subscriber_gp(n, topic_gp.c_str(),
                                                         1);
  subscriber_gp.unsubscribe();
  message_filters::Subscriber<VisualOdometry> subscriber_vo(n, topic_vo.c_str(),
                                                            1);
  subscriber_vo.unsubscribe();

  ros::SubscriberStatusCallback con_cb =
      boost::bind(&connectCallback, boost::ref(subscriber_camera_info),
                  boost::ref(subscriber_gp), boost::ref(subscriber_vo));
  ///////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////////////
  // Register Callback
  sync_policies::ApproximateTime<CameraInfo, GroundPlane, VisualOdometry>
      MySyncPolicy(
          queue_size); // The real queue size for synchronisation is set here.

  const sync_policies::ApproximateTime<CameraInfo, GroundPlane, VisualOdometry>
      MyConstSyncPolicy = MySyncPolicy;

  Synchronizer<
      sync_policies::ApproximateTime<CameraInfo, GroundPlane, VisualOdometry>>

      sync(MyConstSyncPolicy, subscriber_camera_info, subscriber_gp,
           subscriber_vo);

  sync.registerCallback(boost::bind(&callback, _1, _2, _3));
  ///////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////////////
  // Create Publishers
  private_node_handle_.param("annotated_persons", pub_topic_annotated_persons,
                             string("/spencer/perception/annotated_persons"));
  pub_annotated_persons = n.advertise<spencer_tracking_msgs::TrackedPersons>(
      pub_topic_annotated_persons, 10, con_cb, con_cb);

  private_node_handle_.param("annotated_persons2d",
                             pub_topic_annotated_persons2d,
                             string("/spencer/perception/annotated_persons2d"));
  pub_annotated_persons2d =
      n.advertise<spencer_tracking_msgs::TrackedPersons2d>(
          pub_topic_annotated_persons2d, 10, con_cb, con_cb);
  ///////////////////////////////////////////////////////////////////////////////////

  ///////////////////////////////////////////////////////////////////////////////////
  // Parse annotated frames and timestamps
  stringvec boxes = readFile(boxes_path.c_str());
  stringvec timestamps = readFile(timestamps_path.c_str());

  // Parse timestamps
  bool parseFirstString = false;
  std::map<int64_t, annoframe> frameToAnnotatedFrame;

  std::cout << "Parsing timestamps..." << std::endl;
  for (std::string timestampString : timestamps) {
    if (!parseFirstString) {
      parseFirstString = true;
      continue;
    }

    annoframe annotatedFrame = parseTimestampString(timestampString);
    int64_t frameID = annotatedFrame.frame;

    // Insert annotated frame into map
    bool firstOccurrence =
        frameToAnnotatedFrame.insert(std::make_pair(frameID, annotatedFrame))
            .second;
    if (!firstOccurrence) {
      std::cerr << "Frame " << frameID << " exists multiple times!"
                << std::endl;
    }
  }
  if (frameToAnnotatedFrame.empty()) {
    std::cout << "No timestamps parsed. Exiting." << std::endl;
    return 0;
  }

  // Parse annotation boxes and add them to their corresponding annoframes
  std::cout << "Parsing boxes..." << std::endl;
  for (std::string annoString : boxes) {
    annobox box = parseAnnoString(annoString);
    int64_t curFrameID = box.frame;
    auto &annoframe = frameToAnnotatedFrame[curFrameID];
    annoframe.annotations.push_back(box);
  }

  std::cout << "Parsing done." << std::endl;

  // Move annoframes from map to (global) vector
  // (for quicker access and easier indexing)
  for (auto &af : frameToAnnotatedFrame) {
    allAnnotatedFrames.push_back(af.second);
  }

  // TODO: assert that annoframes are sorted by time, ascending

  // Spin around
  ros::spin();
  return 0;
}

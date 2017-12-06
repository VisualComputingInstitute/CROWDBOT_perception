#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int8.h>
#include <fstream>
#include <string>
#include <iostream>
#include <message_filters/subscriber.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/time_synchronizer.h>
#include <rwth_perception_people_msgs/UpperBodyDetector.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "rwth_upper_body_skeleton_detector/GetUpperBodySkeleton.h"
#include "forest_utilities.h"
#include "bodypose_regressor_utilities.h"
#include <vector>
#include <string>
using namespace std;

unsigned counter = 0;
//intializing the forest with 3 trees, image_width = 640 and image_height = 480 
FOREST f(3,640,480);

void frames_to_file(rwth_upper_body_skeleton_detector::GetUpperBodySkeleton::Request  &req)
{
    ROS_INFO("Writing frame : %d", (int) counter);
    //counter++;
    size_t N;
    char filepath[200];
    std::ofstream myfile;
    
    sprintf(filepath,"data/images/image_%d.txt",counter);
    myfile.open (filepath);
    N = req.depth_image.size();
    for (unsigned row = 0; row < N; row++)
     {
	
        myfile << req.depth_image[row];
	myfile << "\n";
      }
    myfile.close();

    sprintf(filepath,"data/images/depths_%d.txt",counter);
    myfile.open (filepath);
    N = req.depths.size();
    for (unsigned row = 0; row < N; row++)
     {
	
        myfile << req.depths[row];
	myfile << "\n";
      }
    myfile.close();

    sprintf(filepath,"data/images/pixels_x_%d.txt",counter);
    myfile.open (filepath);
    N = req.pixels_x.size();
    for (unsigned row = 0; row < N; row++)
     {
	
        myfile << req.pixels_x[row];
	myfile << "\n";
      }
    myfile.close();
    
    sprintf(filepath,"data/images/pixels_y_%d.txt",counter);
    myfile.open (filepath);
    N = req.pixels_y.size();
    for (unsigned row = 0; row < N; row++)
     {
	
        myfile << req.pixels_y[row];
	myfile << "\n";
      }
    myfile.close();

}

void write_joints_to_file(const std::vector<std::vector<float> > joints)
{
    size_t N;
    char filepath[200];
    std::ofstream myfile;
    sprintf(filepath,"data/images/joints_%d.txt",counter);
    myfile.open (filepath);
    N = joints.size();
    for (unsigned row = 0; row < N; row++)
     {
	
        myfile << joints[row][0] << "," << joints[row][1] << joints[row][2];
	myfile << "\n";
      }
    myfile.close();
}

bool get_upper_body_skeleton(rwth_upper_body_skeleton_detector::GetUpperBodySkeleton::Request  &req,
                             rwth_upper_body_skeleton_detector::GetUpperBodySkeleton::Response &res)
{

    std::vector<std::vector<float> > max_scoring_joints(9,std::vector<float>(3));
    counter++;
    //ROS_INFO("Recieved upper bpdy skeleton request for frame %d", (int) counter);
    
    //frames_to_file(req);
    //ROS_INFO("Frame written"); 
     compute_upper_body_pose(req.depth_image.c_array(),
                             req.depths.c_array(), req.pixels_x.c_array(),
                             req.pixels_y.c_array(), f, max_scoring_joints);
    //write_joints_to_file(max_scoring_joints);
    size_t N = max_scoring_joints.size();
    //res.upper_body_skeleton_joint_positions.resize(N);
    unsigned jointcounter = 0;
    //ROS_INFO("upper_body_skeleton_computed_successfully with %d joints",(int)N);
    for (unsigned short i = 0 ;i < N ; i++)
	 {
		res.upper_body_skeleton_joint_positions[jointcounter] = max_scoring_joints[i][0];
		res.upper_body_skeleton_joint_positions[jointcounter + 1] = max_scoring_joints[i][1];
		res.upper_body_skeleton_joint_positions[jointcounter + 2] = max_scoring_joints[i][2];
                jointcounter = jointcounter + 3;
  		//std::cout << "\n" << max_scoring_joints[i][0] << "," << max_scoring_joints[i][1] << "," << max_scoring_joints[i][2] << "\n";
	
	}
    return true; 
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "upper_body_skeleton_detector_server");
  ros::NodeHandle n;
  string forest_path;
  
  ros::NodeHandle private_node_handle_("~");
  private_node_handle_.param("forest_path", forest_path, string(""));
  
  //loading the forest
  f.load_forest(forest_path); 

  ros::ServiceServer service = n.advertiseService("/rwth_upper_body_skeleton_detector/get_upper_body_skeleton", get_upper_body_skeleton);
  ROS_INFO("Starting upper_body_skeleton_server....");
  ros::spin();

  return 0;
}

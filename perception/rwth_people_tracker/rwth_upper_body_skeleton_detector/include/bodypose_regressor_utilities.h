#ifndef BODYPOSE_REGRESSOR_UTILITIES
#define BODYPOSE_REGRESSOR_UTILITIES
#include<vector>
#include <opencv2/opencv.hpp>
#include "forest_utilities.h"

//#####################################################################################
//Wrapper functions for preprocessing input when using ROS
//before applying regression forest to get
//upper body pose
//#####################################################################################

void remove_background(const cv::Mat &depth_image, \
                       std::vector<unsigned short> &bounding_box, \
                       std::vector<float> &depths,  \
                       std::vector<std::vector<float> > &pixel_locations,\
                       std::vector<float> &test_image,\
                       cv::Mat &tmp_depth_image
                      );

void compute_point_cloud(std::vector<float> &depths, unsigned short focal_length, \
                         unsigned short &width, unsigned short &height, \
                         std::vector<unsigned> &pixels_x, \
                         std::vector<unsigned> &pixels_y, \
                         std::vector<float> &point_cloud);

//void read_camera_parameters(const char *path);

void compute_upper_body_pose(float *depthimage,
                     float *depths,
                     unsigned *pixels_x,
                     unsigned *pixels_y,
                     FOREST &f,
                     std::vector<std::vector<float> > &max_scoring_joints);

void sample_joint_poistions(const std::vector<std::vector<std::vector<float> > > & joint_positions, \
                            const std::vector<std::vector<std::vector <float > > > &joint_positions_confidence, \
                            std::vector<std::vector<std::vector<float> > > & sampled_joint_postions, \
                            std::vector<std::vector<std::vector <float> > > & sampled_joint_positions_confidence);

void find_top_scoring_joint_postion(const std::vector<float> & joint_scores, \
                                    const std::vector<unsigned short> &joint_id, \
                                    std::vector<unsigned short> & top_scoring_joints,\
                                    unsigned size);




#endif // BODYPOSE_REGRESSOR_UTILITIES



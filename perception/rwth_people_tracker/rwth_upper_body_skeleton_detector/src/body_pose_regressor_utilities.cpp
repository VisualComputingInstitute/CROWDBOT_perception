#include "bodypose_regressor_utilities.h"
#include "forest_utilities.h"
#include <stdlib.h>
#include "mean_shift_utilities.h"
//#include "image_utilities.h"
#include <opencv2/opencv.hpp>
//#include <random>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include "datatypes.h"

unsigned short number_of_samples = 200;

struct joints
{
    float confidence;
    std::vector<float> position;
};


struct by_confidence
{
    bool operator ()(joints const &left, joints const &right) {
        return left.confidence > right.confidence;
    }
};

//##########################################################################################################
//////////////////////////Function to sample joint positions before applying mean shift clustering//////////
//##########################################################################################################

void sample_joint_poistions(const std::vector<std::vector<std::vector < float> > > &joint_positions, \
                            const std::vector<std::vector<std ::vector <float> > > &joint_positions_confidence, \
                            std::vector<std::vector<std::vector <float> > > & sampled_joint_postions, \
                            std::vector<std::vector<std::vector <float> > > & sampled_joint_positions_confidence)
{



    //size_t N = joint_positions_confidence.size();
    for (unsigned orientation = 0 ; orientation < 3 ; orientation++)
    {
       // std::vector<std::vector<float> > tmp_sampled_joint_positions;
       // std::vector<std::vector<float> > tmp_sampled_joint_positions_confidence;

        for (unsigned i = 0 ; i < 9 ; i++)
        {
            //std::vector<std::vector<float> > tmp_positions;
           // std::vector<float> tmp_weights;
           /* for (unsigned short j= 0 ; j < N ; j++ )
           // {
                if (joint_positions_confidence[j][i] > 0)
                        {
                            std :: vector<float> tmp;
                            tmp_weights.push_back(joint_positions_confidence[j][i]);
                            tmp.push_back(joint_positions[j][i*3+0]);
                            tmp.push_back(joint_positions[j][i*3+1]);
                            tmp.push_back(joint_positions[j][i*3+2]);
                            std::cout << j << tmp[0] << "," << tmp[1] << "," <<tmp[2] << "\n";
                            tmp_positions.push_back(tmp);
                        }
            }*/

            long total_votes = joint_positions_confidence[orientation][i].size();
            //std::random_device rd;
            //std::mt19937 gen(rd());
            //std::uniform_int_distribution<> dis(0,total_votes-1);
           // std::vector<float> tmp_joints_positions_confidence(total_votes);
          //  std::vector<float> tmp_joints_positions(total_votes * 3);
            std::vector<joints> tmp_joints(total_votes);
            for (unsigned j = 0 ; j < total_votes; j++)
            {
                //joints tmp;
                tmp_joints[j].confidence = joint_positions_confidence[orientation][i][j];
                tmp_joints[j].position.resize(3);
                tmp_joints[j].position[0] = joint_positions[orientation][i][j*3];
                tmp_joints[j].position[1] = joint_positions[orientation][i][j*3 + 1];
                tmp_joints[j].position[2] = joint_positions[orientation][i][j*3 + 2];

            }
            std::sort(tmp_joints.begin(),tmp_joints.end(), by_confidence());
            for(unsigned j = 0 ; j < number_of_samples ;j++)

            {
             //   unsigned short r = rand()%total_votes;
            //    std::cout << r << "\n";
            //    float tmp_confidence = joint_positions_confidence[orientation][i][r];
                sampled_joint_positions_confidence[orientation][i][j] = tmp_joints[j].confidence;
                float x,y,z;
               // x = joint_positions[orientation][i][r*3];
              //  y = joint_positions[orientation][i][r*3 +1];
              //  z = joint_positions[orientation][i][r*3 + 2];
               sampled_joint_postions[orientation][i][j*3 ]  = tmp_joints[j].position[0];
               sampled_joint_postions[orientation][i][j*3 + 1]  = tmp_joints[j].position[1];
               sampled_joint_postions[orientation][i][j*3 + 2]  = tmp_joints[j].position[2];


            }
          // tmp_sampled_joint_positions.push_back(tmp_joints_positions);
          // tmp_sampled_joint_positions_confidence.push_back(tmp_joints_positions_confidence);
        }
       // sampled_joint_postions.push_back(tmp_sampled_joint_positions);
       // sampled_joint_positions_confidence.push_back(tmp_sampled_joint_positions_confidence);
    }

}

//############################################################################
////////Function to remove background from the upper body template////////////
//############################################################################

void remove_background(const cv::Mat &depth_image, \
                       std::vector<float> &bounding_box, \
                       std::vector<float> &depths, \
                       std::vector<std::vector<float> > &pixel_locations, std::vector<float> &test_image,\
                       cv::Mat &tmp_depth_image)
{

    unsigned short start_col = bounding_box[0]-10;
    unsigned short start_row = bounding_box[1]-10;
    unsigned short width = start_col + bounding_box[2] + 10;
    float median_depth = bounding_box[4];
    unsigned short height = start_row + bounding_box[3] + 60;


    for (unsigned row = start_row+1; row <  height; row++)
    {
        for (unsigned col = start_col+1; col < width ; col++)
        {
              if (depth_image.at<float>(row,col) <= (median_depth + .6) && depth_image.at<float>(row,col) > 0)

                 {
                      float depth = depth_image.at<float>(row,col);
                      depths.push_back(depth);
                      tmp_depth_image.at<float>(row,col) = depth;
                      std ::vector<float> tmp;
                      tmp.push_back(row+1);
                      tmp.push_back(col+1);
                      pixel_locations.push_back(tmp);
                      unsigned long linearindex = 480*(tmp[1] - 1)+ tmp[0];
                      test_image[linearindex-1] = depth_image.at<float>(row,col);
              }
        }
     }
}

//############################################################################
/////////////////////////Function to compute point cloud /////////////////////
//############################################################################


void compute_point_cloud(std::vector<float> &depths, unsigned short focal_length, \
                         unsigned short &width, unsigned short &height, \
                         std::vector<unsigned> &pixels_x, std::vector<unsigned> &pixels_y, \
                         std::vector<float> &point_cloud)
{

    int  cy = height/2;
    int  cx = width/2;
    size_t N = depths.size();
    point_cloud.resize(N*3);
    float X,Y;
   // unsigned counter = 0;
    omp_set_num_threads(8);
    #pragma omp parallel for
    for (unsigned i = 0 ;i < N ; i++)
    {
        //int x =  pixels_y[i];
        //int y =  pixels_x[i];
        //float depth = depths[i];
        X = (pixels_y[i] - cy)* depths[i];
        X = X/focal_length;
        Y = (pixels_x[i] - cx)* depths[i];
        Y = Y/focal_length;
        point_cloud[i*3] = Y;
        point_cloud[i*3 + 1] = -X;
        point_cloud[i*3 + 2] = depths[i];
     //   counter = counter + 3;
    }

}

//########################################################################################
/////////////////Function to find the top scoring joints hypothesis///////////////////////
//########################################################################################
void find_top_scoring_joint_postion(const Joint *proposals, \
                                    unsigned *joint_id, \
                                    unsigned *top_scoring_joints, unsigned size)
{
    float max_scores[9];
    for (unsigned i = 0 ; i < 9 ; i++)
    {
        top_scoring_joints[i] = 0;
        max_scores[i] = 0 ;
    }

  //  size_t N = joint_scores.size();

    for (unsigned i = 0; i < size ; i++)
    {
        float tmp_score = proposals[i].joint_positions_confidence;
        unsigned id = joint_id[i];
        if (tmp_score > max_scores[id-1])
        {
            max_scores[id-1] = tmp_score;
            top_scoring_joints[id-1] = i;
        }
    }

}

//#######################################################################################
//////// Function removes background, computes point cloud and gets the pose ////////////
//#######################################################################################

void compute_upper_body_pose(float *depthimage,
                     float *depths,
                     unsigned *pixels_x,
                     unsigned *pixels_y,
                     FOREST &f,
                     std::vector<std::vector<float> > &max_scoring_joints)


{

//============================================================================
// removing background from bounding box provided by upper body detector
// Here background consists of nan values plus noise
//============================================================================

    //std::vector<float> depths;
    //std::vector<float> test_image;
    //std::vector<std::vector<float> > pixel_locations;
    //size_t N = image_height*image_width;
    //for(unsigned long i = 0 ;i < N ; i++)
    //    test_image.push_back(0);
    //remove_background(depthimage,depths,depths,pixel_locations,test_image,pixels_x);


//=======================================================================================
// computing the pointcloud  of upper body as required by the regression forest
//=======================================================================================

  /*  std::vector<float> point_cloud; //input
    //need to change the focal length if different depth camera is used

    unsigned focal_length = 525;
    unsigned short image_width = 640;
    unsigned short image_height = 480;

    //computing the point cloud
    //compute_point_cloud(depths,focal_length,image_width,image_height,pixels_x,pixels_y,point_cloud);*/




//================================================================================================
// doing preprossing on vectors before passing it to regression forest
//===============================================================================================

    size_t N = depths[0];

   // std::cout << "\nN : " << N << "\n";
   // std ::vector<std::vector<float> > bodypartslabels(N,std::vector<float>(14));//output
   // std::vector<std::vector<float> > pixel_body_orientation_vote(N,std::vector<float>(12));
   // std::vector<std::vector<long> > leaf_nodes_for_pixels(N,std::vector<long>(3));

    float bodypartslabels[N*14];
    float pixel_body_orientation_vote[N*12];
    float leaf_nodes_for_pixels[N*3];

    //omp_set_num_threads(8);
    //#pragma omp parallel for
    for (unsigned i = 0 ; i < N*14 ; i++)
    {
            bodypartslabels[i] = 0;
            if (i < N*12)
                pixel_body_orientation_vote[i] = 0;
            if (i < N*3)
                leaf_nodes_for_pixels[i] = 0;


    }

    //long total_joint_predictions = N*6*9/4;
    Joint JointProposals[12*1800];
    //JointProposals = (Joint *)malloc(3*total_joint_predictions * sizeof(Joint));

//=================================================================
////////// Applying the forest
//=================================================================

   // FOREST f(3,image_width,image_height);

   // f.load_forest("../Forest/");



    f.apply_forest(depths,pixels_x,pixels_y,depthimage,bodypartslabels,
                   pixel_body_orientation_vote,leaf_nodes_for_pixels,
                   JointProposals);

   //for debugging
   /* unsigned start_position = 1*(6*N)/4 ;
    for (unsigned i = start_position; i < start_position + 200; i++)
     {
        std::cout<< JointProposals[i].joint_position.x << "," << JointProposals[i].joint_position.y << "," << JointProposals[i].joint_position.z;
        std::cout <<"," << JointProposals[i].joint_positions_confidence <<"\n";
    }*/




//========================================================================
///////////////Sampling the propsed joint positions from forest/////////
///////////////              for speed      ///////////////////////////
//========================================================================

    //std::vector<std::vector<std::vector<float> > > sampled_joint_positions(3,std::vector<std::vector<float> >(9, std::vector<float>(600)));;
   /* std::vector<std::vector<std::vector<float> > > sampled_joint_positions_confidence(3,std::vector<std::vector<float> >(9, std::vector<float>(200)));;
    sample_joint_poistions(joint_positions,joint_positions_confidence,\
                           sampled_joint_positions,sampled_joint_positions_confidence);





   //for debugging
//    for(unsigned i = 0 ; i < 200; i=i+3)
 //   {
  //      std::vector<float> tmp;
   //     tmp.push_back(sampled_joint_positions[0][0][i]);
   //     tmp.push_back(sampled_joint_positions[0][0][i+1]);
   //     tmp.push_back(sampled_joint_positions[0][0][i+2]);
   //     max_scoring_joints.push_back(tmp);
   // }*/

//==================================================================================
/////////////////////Applying mean shift clustering/////////////////////////////////
/////////////////////on sampled joint positions ///////////////////////////////////
//==================================================================================

    std::vector<std::vector<std::vector<float> > > max_scoring_joints_per_orientation(12,std::vector<std::vector<float> >(9, std::vector<float>(3)));
    std::vector<std::vector<float> > max_scoring_joints_confidence_per_orientation(12,std::vector<float> (9));

    for (unsigned orientation = 0 ; orientation < 3 ; orientation++)
    {

        MEAN_SHIFT M(9,200,1800*orientation);
        Joint final_joint_proposals[1800];
        unsigned JointId[1800];
        unsigned size;
        float bandwidth = 0.05;
        unsigned max_scoring_joints_ids[9];


        M.apply_meanshift(final_joint_proposals,JointProposals,bandwidth,JointId,size);

        find_top_scoring_joint_postion(final_joint_proposals,\
                                       JointId,max_scoring_joints_ids,\
                                       size);
        for (unsigned i = 0; i < 9 ; i++)
         {
            unsigned id = max_scoring_joints_ids[i];
            max_scoring_joints_per_orientation[orientation][i][0] = final_joint_proposals[id].joint_position.x;
            max_scoring_joints_per_orientation[orientation][i][1] = final_joint_proposals[id].joint_position.y;
            max_scoring_joints_per_orientation[orientation][i][2] = final_joint_proposals[id].joint_position.z;
            max_scoring_joints_confidence_per_orientation[orientation][i] = final_joint_proposals[id].joint_positions_confidence;
         }
       }
        std::vector<float> score_per_orientation(12);
        for (unsigned orientation = 0 ; orientation < 3 ; orientation++)
        {
            score_per_orientation[orientation] = 0;
            for (unsigned joint = 0 ; joint < 9 ; joint++)
                score_per_orientation[orientation] = score_per_orientation[orientation] + max_scoring_joints_confidence_per_orientation[orientation][joint];
        }

        float max_score = 0.0;
        unsigned max_scoring_orientation;
        for (unsigned orientation = 0 ; orientation < 3 ; orientation++)
        {
            if(score_per_orientation[orientation] > max_score)
            {
                max_score = score_per_orientation[orientation];
                max_scoring_orientation = orientation;
            }
        }

        //std::cout << "\n Maximum Scoring orientation after voting : " ;
        for(unsigned joint = 0 ; joint < 9 ; joint++)
        {
            max_scoring_joints[joint][0] = max_scoring_joints_per_orientation[max_scoring_orientation][joint][0];
            max_scoring_joints[joint][1] = max_scoring_joints_per_orientation[max_scoring_orientation][joint][1];
            max_scoring_joints[joint][2] = max_scoring_joints_per_orientation[max_scoring_orientation][joint][2];
        }



}


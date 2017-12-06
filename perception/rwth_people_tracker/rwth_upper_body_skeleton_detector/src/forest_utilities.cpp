#include "forest_utilities.h"
#include "tree_utilities.h"
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <stdlib.h>
#include "datatypes.h"
#include "mean_shift_utilities.h"

struct orientations
{
    float confidence;
    unsigned index;
};


struct by_confidence
{
    bool operator ()(orientations const &left, orientations const &right) {
        return left.confidence > right.confidence;
    }
};

struct by_joints_confidence
{
    bool operator ()(Joint const &left, Joint const &right) {
        return left.joint_positions_confidence > right.joint_positions_confidence;
    }
};


//###################################################################
//////Constructor for intitializing forest parameters////////////////
//###################################################################
std::vector<long> tree::voteforjointcounter;
FOREST :: FOREST(size_t no_of_tree, unsigned width, unsigned height)
{
    this->image_height = height;
    this->image_width = width;
    this->total_trees = no_of_tree;
    for (size_t i = 0 ; i < this->total_trees ; i++)
    {
            tree tr(this->image_width,this->image_height,i);
            this->trees.push_back(tr);
    }
}

//###################################################################
/////////////////Loading the forest from file///////////////////////
//###################################################################

void FOREST ::load_forest(std::string path)
{
    tree::initialize_vote_count();
    std::cout << "loading forest....\n";
    for (size_t i = 0; i < this->total_trees ; i++)
    {
            std::cout << "loading tree : " << i+1 << "\n";
            this->trees[i].load_tree(path,i);
    }

}

//Function to apply the forest

//###################################################################
//============================INPUTS================================

//depths : depths of input pixels
//pixel_locations : (x,y) location of input pixels in
//test image
//point_cloud : (x,y,z) position of input pixels w.r.t
//camera
//test_image : image containing the person

///////////////////////////OUTPUTS////////////////////////////////

//body_parts_labels : bodypart each input pixel
//belongs to
//joint_positions : (x,y,z) positions of joints of
//interest w.r.t camera
//joint_positions_confidence : confidence for each
// proposed position

//###################################################################



void FOREST :: apply_forest(float *depths,\
                            unsigned *pixels_x,\
                            unsigned *pixels_y,\
                            //const std::vector<float>& point_cloud,
                            float *test_image,\
                            float *body_parts_labels,\
                            //float *joint_positions,
                           // float *joint_positions_confidence,
                            float *pixels_body_orientation_votes,\
                            float *leaf_nodes_for_pixels,\
                            Joint *JointProposals)
{

    //tree::re_initialize_vote_count();

    unsigned N = depths[0];
   /* float body_parts_labels[N*14];
    float pixels_body_orientation_votes[N*12];
    float leaf_nodes_for_pixels[N*3];

    //omp_set_num_threads(8);
    //#pragma omp parallel for
    for (unsigned i = 0 ; i < N*14 ; i++)
    {
            body_parts_labels[i] = 0;
            if (i < N*12)
                pixels_body_orientation_votes[i] = 0;
            if (i < N*3)
                leaf_nodes_for_pixels[i] = 0;


    }
     */

    for (size_t i = 0 ; i < this->total_trees ; i++)
    {
        //this->trees[i].total_pixels = N;
        this->trees[i].apply_tree(depths,pixels_x,pixels_y,test_image,\
                                  body_parts_labels,\
                                  pixels_body_orientation_votes,\
                                  leaf_nodes_for_pixels);
    }



     orientations body_orientation[12];
     for (unsigned i = 0; i < 12 ; i++)
     {
         body_orientation[i].confidence = 0;
         body_orientation[i].index = i;
         omp_set_num_threads(8);
         #pragma omp parallel for
         for (unsigned j = i; j < N ;j = j + 12)
             body_orientation[i].confidence = body_orientation[i].confidence + pixels_body_orientation_votes[j];
     }

     std::sort(body_orientation,body_orientation+12,by_confidence());

    //size_t N = depths.size();
   /* std::vector<float> body_orientation;*/
   // long total_joint_predictions = N*6*9/4;//(9*6*N*27)/4 ;
   // Joint *j;
   // j = (Joint *)malloc(total_joint_predictions * sizeof(Joint));
    //j.joint_positions = (float *)malloc(total_joint_predictions * sizeof(float));
    //j.joint_positions_confidence = (float*)malloc(((9*6*N*9)/4)*sizeof(float));
    unsigned jointcounter[9];


    long total_joint_predictions = N*6*9/2;
    Joint *tmpJointProposals;

   // std::cout << "\nMaximum Scoring Orientation before voting: " << o[0].index << "\n";
   for (unsigned orientation = 0; orientation < 3 ; orientation++)
   {

       tmpJointProposals = (Joint *)malloc(total_joint_predictions * sizeof(Joint));

      for( unsigned i = 0; i < 9; i++)
            jointcounter[i] = 0;

      // tree::re_initialize_vote_count();
       for (unsigned i = 0 ; i < 3 ; i++)
       {
           omp_set_num_threads(8);
    	   #pragma omp parallel for
           for (unsigned pixel_number = 1; pixel_number < N ; pixel_number++)
           {

                this->trees[i].get_pixel_vote(pixel_number,\
                                              pixels_x,\
                                              pixels_y,\
                                              depths,\
                                              leaf_nodes_for_pixels[(pixel_number-1)*3 + i],\
                                              tmpJointProposals,\
                                              body_orientation[orientation].index,\
                                              jointcounter);
           }
       }
       for (unsigned i = 0 ; i < 9 ;i++)
        {
          unsigned tmp = jointcounter[i];
          unsigned start = i*(6*N)/4 ;
          std::sort(tmpJointProposals + start,\
                    tmpJointProposals + start + tmp,\
                    by_joints_confidence());
          for(unsigned jointcounter = 0; jointcounter < 200 ; jointcounter++)
          {
              unsigned index = (orientation * 1800) + (i*200) + jointcounter;
              JointProposals[index].joint_position.x = tmpJointProposals[start + jointcounter].joint_position.x;
              JointProposals[index].joint_position.y = tmpJointProposals[start + jointcounter].joint_position.y;
              JointProposals[index].joint_position.z = tmpJointProposals[start + jointcounter].joint_position.z;
              JointProposals[index].joint_positions_confidence = tmpJointProposals[start + jointcounter].joint_positions_confidence;
          }
        }
       free(tmpJointProposals);
    }


}


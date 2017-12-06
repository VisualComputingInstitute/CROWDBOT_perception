#ifndef TREE_UTILITIES
#define TREE_UTILITIES

#include <vector>
#include <string>
#include "datatypes.h"

class tree
{
private :


    std::vector<float> leaf_nodes;
    std::vector<float> split_nodes;
    std::vector<float> offsets;
    std::vector<float> offsets_weights;
    std::vector<float> vote_length_thresholds;
    std::vector<float> leaf_nodes_orientations;

   // float *leaf_nodes;
   // float *split_nodes;
   // float *offsets;
   // float *offsets_weights;
   // float *leaf_nodes_orientations;

    size_t total_split_nodes;
    size_t total_leaf_nodes;
    size_t total_classes;
    //size_t offset_position;
    size_t total_pixels;
    unsigned short self_no;
    unsigned short image_width, image_height;

    //functions for loading tree model and hyperparameters  from files
    void load_leafnodes(const char * path);
    void load_leafnodes_orientations(const char *path);
    void load_splitnodes(const char * path);
    void load_offsets(const char * path);
    void load_offsetsweights(const char * path);
    void load_vote_length_thresholds(const char *path);

    //function to return pixel vote for each joint && pixel's bodypart label
    void get_pixel_hypothesis(unsigned int pixel_number,\
                              unsigned *pixels_x,\
                              unsigned *pixels_y,\
                              float *depths,\
                           /*   const std::vector<float>& point_cloud,\*/
                              float *test_image,\
                          /*    std::vector<std::vector<float> >& joint_positions,\*/
                              float *body_parts_labels,\
                         /*     std::vector<std::vector<float> >& joint_positions_confidence,\ */
                              float *pixels_body_orientation_votes,\
                              float *leaf_nodes_for_pixels);


    //function to get pixel vote



    //function to get the leaf node reached by the pixel
    int get_pixel_leaf_node(float *pixel,float *test_image,float depth);

    //function to compute feature response for a Pixel
    double feature_response(float*F, float *pixel,\
                            float *test_image,float &depth);

    //function to get weighted vote for the pixel
    void GetWeightedVote (float *PointCloud, float *Mode , float ModeWeight,\
                                   float threshold, float *Vote, float &Weight);


    static void increment_vote_counter(unsigned index)
    {
        voteforjointcounter[index] = voteforjointcounter[index] + 3;
    }


public:


    static std::vector<long> voteforjointcounter;
    static long get_vote_count(unsigned index){return voteforjointcounter[index];}
    static void initialize_vote_count()
    {
        for (unsigned i = 0 ; i < 9 ; i++)
            voteforjointcounter.push_back(0);

    }
    static void re_initialize_vote_count()
    {
        for (unsigned i = 0 ; i < 9 ; i++)
            voteforjointcounter[i] = 0;
    }

    tree (unsigned short width,unsigned short height,unsigned no)
    {   this->total_classes = 14;
        this->image_height = height ;
        this->image_width = width;
        this->self_no = no;
    }
    void load_tree(std::string, unsigned short number); //loading the tree
    void apply_tree(float *depths,\
                    unsigned *pixels_x,\
                    unsigned *pixels_y,\
                  /*  const std::vector<float>& point_cloud,\*/
                    float *test_image,\
                    float  *body_parts_labels,\
                 /*   std::vector<std::vector<float> >& joint_positions,\*/
                 /*   std::vector<std::vector<float> >& joint_positions_confidence,\*/
                    float *pixels_body_orientation_votes,\
                    float  *leaf_nodes_for_pixels);
    void get_pixel_vote(unsigned int pixel_number, \
                        unsigned *pixels_x, \
                        unsigned *pixels_y, \
                        float *depth, \
                        unsigned int leafnode, \
                        Joint *j, \
                        unsigned orientation, \
                        unsigned *jointscounter);

};

#endif // TREE_UTILITIES



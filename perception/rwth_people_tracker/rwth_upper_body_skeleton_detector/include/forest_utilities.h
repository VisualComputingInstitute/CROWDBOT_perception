#ifndef FOREST_UTILITIES
#define FOREST_UTILITIES

#include <string>
#include <vector>
#include "tree_utilities.h"
#include "datatypes.h"


class FOREST
{

private :
    size_t total_trees;
    std::vector<tree> trees;
    unsigned short image_width;
    unsigned short image_height;

public:
    static long get_vote_count(unsigned index) {return tree::get_vote_count(index);}
    FOREST(size_t no_of_tree, unsigned width, unsigned height); //Initializing the forest
    void load_forest(std::string path); //Loading the forest
    void apply_forest(float *depths,\
                      unsigned *pixels_x,\
                      unsigned *pixels_y,\
                      //const std::vector<float>& point_cloud,
                      float *test_image,\
                      float *body_parts_labels,\
                      //float *joint_positions,
                     // float *joint_positions_confidence,
                      float *pixels_body_orientation_votes,\
                      float *leaf_nodes_for_pixels,\
                      Joint *Proposals); //Applying the forest
};

#endif // FOREST_UTILITIES



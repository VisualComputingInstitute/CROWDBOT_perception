#include "tree_utilities.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

//###################################################################
////////////Function to load tree from file//////////////////////////
//###################################################################

void tree::load_tree(std::string path, unsigned short number)
{

    char fullpath[200];
    //loading the leaf nodes
    sprintf(fullpath,"%sLeafNodes_tree_%d",path.c_str(),number+1);
    load_leafnodes(fullpath);

    //loading lead nodes orientations
     fullpath[0] = '\0';
    sprintf(fullpath,"%sLeafNodesOrientations_tree_%d",path.c_str(),number+1);
    load_leafnodes_orientations(fullpath);

    //loading the split nodes
    fullpath[0] = '\0';
    sprintf(fullpath,"%sSplitNodes_tree_%d",path.c_str(),number+1);
    load_splitnodes(fullpath);

    //loading the offsets
    fullpath[0] = '\0';
    sprintf(fullpath,"%sModes_tree_%d",path.c_str(),number+1);
    load_offsets(fullpath);

    //loading the offsetsweights
    fullpath[0] = '\0';
    sprintf(fullpath,"%sModeWeights_tree_%d",path.c_str(),number+1);
    load_offsetsweights(fullpath);

    //loading the votelength thresholds
    fullpath[0] = '\0';
    sprintf(fullpath,"%sthresholds",path.c_str());
    load_vote_length_thresholds(fullpath);

}

//###################################################################
////////////////////////Loading leaf nodes///////////////////////////
//###################################################################

void tree ::load_leafnodes(const char *path)
{
    std::ifstream file(path,std::ios::in | std::ios::binary);
    if (!file)
    {
        std::cout << "Please provide a valid path\n";
        while (*path != '\0')
              std::cout << *path++ ;
        std::cout << "\n";					
         return;
    }
    else
    {
        std::cout << "....loading leafnodes ...\n";
        int size ;
        file.read((char *) &size , sizeof size);
        int tmp_data;
        this->leaf_nodes.resize(size);
        //this->leaf_nodes = (float *)malloc(size * sizeof(float));
        for (int i = 0 ; i < size ; i++)
        {
            file.read((char *) &tmp_data , sizeof tmp_data);
            this->leaf_nodes[i] = tmp_data;
        }

        //computing the total leaf nodes in the tree
        //unsigned long tmp = this->leaf_nodes.size();
        this->total_leaf_nodes = size/this->total_classes;
    }

}

//########################################################################
//////////////loading leaf nodes orientations////////////////////////////
//#######################################################################

void tree::load_leafnodes_orientations(const char *path)
{
    std::ifstream file(path,std::ios::in|std::ios::binary);
    if(!file)
    {
        std::cout << "file not found\n";
    }
    int size;
    file.read((char *) &size, sizeof size);
    int tmp_data;
    this->leaf_nodes_orientations.resize(size);
    //this->leaf_nodes_orientations = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        file.read((char *)&tmp_data,sizeof tmp_data);
        this->leaf_nodes_orientations[i] = tmp_data;
    }
}

//###################################################################
////////////////////Loading split nodes/////////////////////////////
//###################################################################

void tree ::load_splitnodes(const char *path)
{
    std::ifstream file(path,std::ios::in | std::ios::binary);
    if (!file)
    {
        std::cout << "Please provide a valid path";
         return;
    }
    else
    {
        std::cout << "....loading Split Nodes ....\n";
        int size ;
        file.read((char *) &size , sizeof size);
        int tmp_data;
        this->split_nodes.resize(size);
       //this->split_nodes = (float *)malloc(size * sizeof(float));
        for (int i = 0 ; i < size ; i++)
        {
            file.read((char *) &tmp_data , sizeof tmp_data);
            this->split_nodes[i] = tmp_data;
        }

        //computing the total split nodes in the tree
        //unsigned long tmp = this->split_nodes.size();
        this->total_split_nodes = size/5;
        }
}


//###################################################################
/////////////////////////Loading offsets////////////////////////////
//###################################################################

void tree ::load_offsets(const char *path)
{
    std::ifstream file(path,std::ios::in | std::ios::binary);
    if (!file)
    {
        std::cout << "Please provide a valid path";
        return;
    }
    else
    {
        std::cout << "....loading offsets....\n";
        int size;
        file.read((char *) &size , sizeof size);
        int tmp_data;
        this->offsets.resize(size);
        //this->offsets = (float *)malloc(size * sizeof(float));
        for (int i = 0 ; i < size ; i++)
        {
            file.read((char *) &tmp_data , sizeof tmp_data);
            this->offsets[i] = tmp_data;
        }
    }
}


//###################################################################
////////////////////Loading offsets weights//////////////////////////
//###################################################################

void tree :: load_offsetsweights(const char *path)
{
        std::ifstream file(path,std::ios::in | std::ios::binary);
        if (!file)
        {
            std::cout << "Please provide a valid path";
            return;
        }
        else
        {
            std::cout << "....loading offset weights....\n";
            int size ;
            file.read((char *) &size , sizeof size);
            int tmp_data;
            this->offsets_weights.resize(size);
          // this->offsets_weights = (float *)malloc(size * sizeof(float));
            for (int i = 0 ; i < size ; i++)
            {
                file.read((char *) &tmp_data , sizeof tmp_data);
                this->offsets_weights[i] = tmp_data;
            }
        }

}

//###################################################################
//////////////////Loading vote length thresholds/////////////////////
//###################################################################

void tree ::load_vote_length_thresholds(const char *path)
{
    std::ifstream file(path);
    std::string line;
    std::cout << "loading thresholds ...\n";
    while (std::getline(file, line))
    {
        this->vote_length_thresholds.push_back(::atof(line.c_str()));
    }

}

//###################################################################
////////////computing feature response for a pixel///////////////////
//###################################################################

double tree ::feature_response(float *F, float *pixel,\
                            float *test_image,float &depth)
{

    double response;
    long linearindex;
    int tmpx = round(((F[0]*1.5)/depth + pixel[0]));
    int tmpy = round(((F[1]*1.5)/depth + pixel[1]));


    //=================for debugging======================
    //std::cout << tmpx << ":" << tmpy << "\n";

    if ( (tmpx > this->image_height) || (tmpx < 1) )
        tmpx = 0;
    if ( (tmpy > this->image_width) || (tmpy < 1) )
        tmpy = 0;

    if ( (tmpx == 0 ) || (tmpy == 0 ))
        {
        //depth1 =  (unsigned long)( depths[i] *1000);
            response = abs(100000 -  (unsigned long)( depth *1000));

        }
   else
        {
            linearindex = this->image_height*(tmpy - 1)+ tmpx;
            double offset_depth = test_image[linearindex-1];

            if (abs(offset_depth) == 0)
                response = abs(100000 - (unsigned long)(depth*1000));
            else
                response = abs((unsigned long)(offset_depth*1000) -(unsigned long)( depth*1000));
        }
    return response;
}

//###################################################################
//////////computing vote from a pixel for a joint////////////////////
//###################################################################

void tree ::GetWeightedVote(float *PointCloud, \
                            float *Mode, float ModeWeight, \
                            float threshold, \
                            float *Vote, float &Weight)
{

    float distance = Mode[0]*Mode[0] + Mode[1]*Mode[1] + Mode[2]*Mode[2];
    distance = sqrt(distance);

    //debugging
    //std::cout << "\n";
    //std::cout << "Mode : " << Mode[0] << "," << Mode[1] << "," << Mode[2] << "\n";
    //std::cout << "PointCloud : " << PointCloud[0] << "," << PointCloud[1] << "," << PointCloud[2] <<"\n";
    //std::cout << "Distance : " << distance << "\n";
    //std::cout << "threshold : " << threshold <<"\n";

    Weight = 0;
    Vote[0] = 0;
    Vote[1] = 0;
    Vote[2] = 0;

   //std :: cout << "\nPoint :" << PointCloud[0] << "\t" << PointCloud[1] << "t" << PointCloud[2];
   //std :: cout << "\n";

    if (distance > threshold)
        return;
    else
    {
        Vote[0] = PointCloud[0] + Mode[0];
        Vote[1] = PointCloud[1] + Mode[1];
        Vote[2] = PointCloud[2] + Mode[2];
        Weight  = PointCloud[2]*PointCloud[2]*ModeWeight;
        if (Vote[0] + Vote[1] + Vote[2] == 0)
            Weight = 0;
        //std :: cout << "\nModeWeight:" << ModeWeight <<"\n";
        //std :: cout << "\nWeight:" << Weight <<"\n";
    }

}

//###################################################################
////////computing vote from a pixel for each joint of interest///////
//###################################################################

void tree::get_pixel_vote(unsigned int pixel_number, \
                         unsigned *pixels_x, \
                         unsigned *pixels_y, \
                         float *depth, \
                         unsigned int leafnode, \
                         Joint *j, \
                         unsigned orientation, unsigned *jointscounter)
{
    if (leafnode > this->total_leaf_nodes || leafnode < 0)
        return;

    float Modes[3];
    float ModeWeight;
    float threshold;
    int CounterForMode = orientation * 54;
    int CounterForModeWeight = orientation * 18;
    int CounterForVote = 0;

    float Weight;
    float Vote[3];


    float Point_3d[3];
    unsigned  width = 640;
    unsigned  height = 480;
    unsigned  cy = height/2;
    unsigned  cx = width/2;
    unsigned focal_length = 525;


    Point_3d[1] = -1*((int)(pixels_y[pixel_number] - cy)* depth[pixel_number])/focal_length;
    Point_3d[0] = ((int)(pixels_x[pixel_number] - cx)* depth[pixel_number])/focal_length;
    Point_3d[2] =  depth[pixel_number];

    //std :: cout << "\nLeafNode : " << leafnode <<"\n" ;



    for (int joint = 0 ; joint < 9 ; joint ++)
    {

       threshold = this->vote_length_thresholds[joint];
       long position = (joint * ((this->total_pixels*6)/4)) ;
     //  long position_conf = joint * ((this->total_pixels*6*9)/4);
       //computing the first vote
       Modes[0] = this->offsets[leafnode + CounterForMode*this->total_leaf_nodes]/100;
       Modes[1] = this->offsets[leafnode + (CounterForMode + 1)*this->total_leaf_nodes]/100;
       Modes[2] = this->offsets[leafnode + (CounterForMode + 2)*this->total_leaf_nodes]/100;
       ModeWeight = this->offsets_weights[leafnode + CounterForModeWeight*this->total_leaf_nodes]/100;
       GetWeightedVote(Point_3d,Modes,ModeWeight,threshold,Vote,Weight);
       //std::cout << Vote[0] << "," << Vote[1] << "," << Vote[2] << "\n";
       //storing the first vote
       //long position = pixel_number + 2*this->total_pixels*this->self_no;
       //joint_positions[position][3*joint] = Vote[0];
       //joint_positions[position][3*joint + 1] = Vote[1];
       //joint_positions[position][3*joint + 2] = Vote[2];
       //joint_positions_confidence[position][joint] = Weight;
       if (Weight > 0)
       {
          // if( joint == 0)
            //   std::cout << "Debug";

          // long position = get_vote_count(joint);
           long index = position + jointscounter[joint];//*3;
           //j.joint_positions[index] = Vote[0];
           //j.joint_positions[index + 1] = Vote[1];
           //j.joint_positions[index + 2] = Vote[2];

           j[index].joint_position.x = Vote[0];
           j[index].joint_position.y = Vote[1];
           j[index].joint_position.z = Vote[2];

          // if( j->joint_positions[0] > 1)
          //      std::cout << "Debug";

           //index = position_conf + jointscounter[joint]*1;
           //j.joint_positions_confidence[index] = Weight;
           j[index].joint_positions_confidence= Weight;
           jointscounter[joint] = jointscounter[joint] + 1;

          // increment_vote_counter(joint);
       }

       //computing the second vote
       Modes[0] = this->offsets[(3 + CounterForMode)*this->total_leaf_nodes + leafnode]/100;
       Modes[1] = this->offsets[(4 + CounterForMode)*this->total_leaf_nodes + leafnode]/100;
       Modes[2] = this->offsets[(5 + CounterForMode)*this->total_leaf_nodes + leafnode]/100;
       ModeWeight = this->offsets_weights[(CounterForModeWeight + 1 )*this->total_leaf_nodes + leafnode]/100;

       GetWeightedVote(Point_3d,Modes,ModeWeight,threshold,Vote,Weight);

       //storing the second vote
       //position = position + this->total_pixels;
      // joint_positions[position][3*joint] = Vote[0];
      // joint_positions[position][3*joint + 1] = Vote[1];
      // joint_positions[position][3*joint + 2] = Vote[2];
      // joint_positions_confidence[position][joint] = Weight;
       if (Weight > 0)
       {
           //if(joint == 0 )
            //   std::cout << "Debug";
           long index = position + jointscounter[joint];
          // j.joint_positions[index] = Vote[0];
          // j.joint_positions[index + 1] = Vote[1];
          // j.joint_positions[index + 2] = Vote[2];
          // if( j.joint_positions[0] > 1)
          //      std::cout << "Debug";
           j[index].joint_position.x = Vote[0];
           j[index].joint_position.y = Vote[1];
           j[index].joint_position.z = Vote[2];
           //index = position_conf + jointscounter[joint]*1;
          // j.joint_positions_confidence[index] = Weight;
           j[index].joint_positions_confidence = Weight;
           jointscounter[joint] = jointscounter[joint] + 1;
         //  increment_vote_counter(joint);
       }


       CounterForMode = CounterForMode + 6;
       CounterForModeWeight = CounterForModeWeight + 2;
       CounterForVote = CounterForVote + 3;

     }


}

//#############################jointscounter[joint]######################################
////push the pixel through the tree  until it reaches a leaf/////////
//###################################################################

int tree :: get_pixel_leaf_node(float *pixel,float *test_image,float depth)
{

    //=====================for debugging========================
    //std::cout << "\n" << pixel[0] << ":" << pixel[1] << ":" << depth <<"\n";

    int currentnode = 0;
    int type = 1;
    double response;
    float F[2];


    while (type == 1)
    {
        F[0] =  this->split_nodes[currentnode + 2*this->total_split_nodes]/100;
        F[1] =  this->split_nodes[currentnode + 3*this->total_split_nodes]/100;
        response = feature_response(F,pixel,test_image,depth);
        if (response <= (this->split_nodes[currentnode + 4*this->total_split_nodes]/100))
            currentnode  =(int)this->split_nodes[currentnode]/100;
        else
            currentnode  = (int)this->split_nodes[currentnode + this->total_split_nodes]/100;
        if (currentnode > 0)
            type = 1;
        else
            type = -1;
        //====================for debugging =================
        //std::cout << currentnode << "\n";

        currentnode = abs(currentnode) - 1;
    }


    return (currentnode + 1) ;


}

//#########################################################################
//################ computing overall response from a pixel i.e ############
//################  votes to all joints of interest     ###################
//###############      body part it belongs to         ####################
//#########################################################################

void tree :: get_pixel_hypothesis(unsigned int pixel_number,\
                          unsigned *pixels_x,\
                          unsigned *pixels_y,\
                          float *depths,\
                          /*   const std::vector<float>& point_cloud,\*/
                          float *test_image,\
                          /*    std::vector<std::vector<float> >& joint_positions,\*/
                          float *body_parts_labels,\
                         /*     std::vector<std::vector<float> >& joint_positions_confidence,\ */
                          float *pixels_body_orientation_votes,\
                          float *leaf_nodes_for_pixels)
{

    //getting the pixel with pixel_number
   // if (pixels_y[pixel_number] == 0 || pixels_x[pixel_number] == 0)
   //	{
   //		 leaf_nodes_for_pixels[pixel_number][this->self_no] = -1;
   //	         return;
   //	}

    float pixel[2];
    pixel[0] = pixels_y[pixel_number];
    pixel[1] = pixels_x[pixel_number];

    int LeafNode = get_pixel_leaf_node(pixel,test_image,depths[pixel_number]);

    //========for debugging
    //std::cout << "\n" << LeafNode << "\n";

    int index = LeafNode -1;
    leaf_nodes_for_pixels[((pixel_number-1)*3)+this->self_no] = index;
    for (unsigned i = 0; i < this->total_classes; i ++){
            float value = (this->leaf_nodes[index + this->total_leaf_nodes*i]/100)/3;
            //std::cout << value << "," ;
            body_parts_labels[((pixel_number-1)*this->total_classes) + i]=  body_parts_labels[((pixel_number-1)*this->total_classes) + i] + value;
            if (i < 12)
                {
                  value = (this->leaf_nodes_orientations[index + this->total_leaf_nodes*i]/100)/3;
                //  std::cout << value ;
                  pixels_body_orientation_votes[((pixel_number-1)*12) + i] = pixels_body_orientation_votes[((pixel_number-1)*12) + i] + value;
                }
           // std::cout << "\n";
            }
        //body_parts_labels[pixel_number + i*this->total_pixels] = this->leaf_nodes[index + this->total_leaf_nodes*i];
  //  get_pixel_vote(pixel_number,point_cloud,index,joint_positions,joint_positions_confidence);

}

//###################################################################
//////applying the tree model to all the pixels in test image////////
//###################################################################

void tree :: apply_tree(float *depths,\
                unsigned *pixels_x,\
                unsigned *pixels_y,\
                /*  const std::vector<float>& point_cloud,\*/
                float *test_image,\
                float  *body_parts_labels,\
                /*   std::vector<std::vector<float> >& joint_positions,\*/
                /*   std::vector<std::vector<float> >& joint_positions_confidence,\*/
                float *pixels_body_orientation_votes,\
                float  *leaf_nodes_for_pixels)
{


    //this->offset_position = this->self_no*2*this->total_pixels;
    this->total_pixels = depths[0];
    omp_set_num_threads(8);
    #pragma omp parallel for
    for (unsigned int i = 1; i < this->total_pixels ; i++)
    {
        get_pixel_hypothesis(i,pixels_x,pixels_y,depths,test_image,\
                             body_parts_labels,\
                             pixels_body_orientation_votes,leaf_nodes_for_pixels);
    }

}
























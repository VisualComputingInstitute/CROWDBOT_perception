#ifndef MEAN_SHIFT_UTILITIES
#define MEAN_SHIFT_UTILITIES

#include <vector>
#include <datatypes.h>

class MEAN_SHIFT
{

private:

   // unsigned long totalpoints;
    unsigned int totaljoints;
    unsigned total_samples_per_joint;
    unsigned mode_counter;
    unsigned tmp_peak_counter;
    unsigned nns_to_use;
    unsigned start_position;
    unsigned increment;
   // unsigned orientation_increment;
    void findpeaks(std ::vector<float> &peak,float & peakscore,\
                        std :: vector<unsigned int>& indices,\
                        std :: vector<unsigned int>& clustervotes,\
                        Joint *points,\
                        unsigned index,
                        //std :: vector <float> & point,
                        //std :: vector<float > &weights,
                        float bandwidth);
    void reshape(std :: vector <float> & points,\
                 std :: vector <std :: vector <float> > &Points);
    void meanshift(//std ::vector<float > & points,
                   //std :: vector <float >& weights,
                   Joint *input_joint_proposals,
                   float bandwidth,
                   //std ::vector<std :: vector<float> > &joint_positions,
                   //std :: vector <float> & joint_postions_confidence,
                   Joint *final_joint_proposals,
                   int start_index, unsigned *JointId);

public:

    MEAN_SHIFT(int totaljoints, unsigned samples_per_joint,unsigned increment);
    void apply_meanshift(//std ::vector<std ::vector <float> > &joint_positions,
                         //std ::vector<float> &joint_positions_confidence,
                         Joint  *final_joint_proposals,
                         //std ::vector<std :: vector<float> > & points,
                         //std :: vector <std :: vector<float> >& weights,
                         Joint  *input_joint_proposals,
                        // unsigned *jointcounter,
                         float bandwidth,unsigned *JointId,\
                         unsigned &size );


};
#endif // MEAN_SHIFT_UTILITIES



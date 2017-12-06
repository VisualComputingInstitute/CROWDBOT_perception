#ifndef DATATYPES
#define DATATYPES

struct position
{
    float x,y,z;
};

struct Joint
{
    position joint_position;
    float    joint_positions_confidence;
};
#endif // DATATYPES


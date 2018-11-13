#include "Globals.h"
#include <string>

////////////////////////////////////////
// World scale
////////////////////////////////////////
double  Globals::WORLD_SCALE;

////////////////////////////////////////
// height and width for precomputed Segmentations
////////////////////////////////////////
int Globals::dImHeight;
int Globals::dImWidth;

////////////////////////////////////////
// (Number of Frames / offset : deprecated) / currenFrame
////////////////////////////////////////
//int Globals::numberFrames;
//int Globals::nOffset;
int Globals::currentFrame;


/////////////////////////////////TRACKING PART/////////////////////////

// Detections
double Globals::dSameIdThresh;

// Kalman
int Globals::frameRate;
double Globals::dt;
double Globals::oldTimeForFPSUpdate;
Vector< double > Globals::dtVector;

// Others
double Globals::pedSizeWVis;

double Globals::pedSizeWCom;
double Globals::pedSizeHCom;

int Globals::history;

//Parameters for image-plane hypothesescom/
double  Globals::dObjHeight;
double  Globals::dObjHVar;
double  Globals::probHeight;

// Colorhistogram
double Globals::cutHeightBBOXforColor;
double Globals::cutWidthBBOXColor;
double Globals::posponeCenterBBOXColor;
int Globals::binSize;

// Visualisation
bool Globals::render_bbox3D;
bool Globals::render_bbox2D;
bool Globals::render_tracking_numbers;

//MDL parameters for trajectories
double  Globals::k1; // "counterweight": min. support for a hypothesis
double  Globals::k2 ; // rel. importance of #poconst ints vs. poconst double strength
double  Globals::k3; // overlap penalty
double  Globals::k4; // temp. decay for static objects

// Threshold for distinction between static/moving object
double  Globals::minvel;
double  Globals::dMaxPedVel; // This is in meter per second = ~ 5km/h

// Trajectory
double Globals::threshLengthTraj;

// Thresholds for accepted and displayed hypotheses
double  Globals::dTheta2;

// Time ant for temporal decay
double Globals:: dTau;

// Time horizon for event cone search
int  Globals::coneTimeHorizon;
double  Globals::maxHoleLen;
double  Globals::dHolePenalty;

/* Q - the system covariance */
double Globals::sysUncX;
double Globals::sysUncY;
double Globals::sysUncRot;
double Globals::sysUncVel;
double Globals::sysUncAcc;

/* P - the initial state covariance */
double Globals::initPX;
double Globals::initPY;
double Globals::initPVX;
double Globals::initPVY;

double Globals::kalmanObsMotionModelthresh;
double Globals::kalmanObsColorModelthresh;

int Globals::accepted_frames_without_det;

////////////////////////Evaluation////////////////////////
bool Globals::save_for_eval;
string Globals::save_path_tracking;
string Globals::save_path_img_info;
string Globals::save_path_cam_info;
string Globals::save_path_img;
string Globals::save_path_cam;



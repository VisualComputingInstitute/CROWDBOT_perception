/* 
 * File:   Globals.h
 * Author: dennis
 *
 * Created on May 15, 2009, 10:02 AM
 */

#ifndef _GLOBALS_DENNIS_H
#define	_GLOBALS_DENNIS_H

#include <string>
#include "Vector.h"
using namespace std;

class Globals {
public:

    ////////////////////////////////////////
    // World scale
    ////////////////////////////////////////
    static  double  WORLD_SCALE;

    ////////////////////////////////////////
    // height and width for precomputed Segmentations
    ////////////////////////////////////////
    static  int dImHeight;
    static  int dImWidth;

    ////////////////////////////////////////
    // Number of Frames / offset
    ////////////////////////////////////////
    static  int numberFrames;
    static int nOffset;

/////////////////////////////////TRACKING PART/////////////////////////

    // Detections
    static int frameRate;
    static double dt;
    static double oldTimeForFPSUpdate;
    static Vector< double > dtVector;

    // other
    static  double pedSizeWVis;

    static double pedSizeWCom;
    static double pedSizeHCom;

    static int history;

    static double dSameIdThresh;

    //Parameters for image-plane hypothesescom/
    static  double  dObjHeight;
    static  double  dObjHVar;
    static  double  probHeight;

    // Colorhistogram
    static  double cutHeightBBOXforColor;
    static  double cutWidthBBOXColor;
    static  double posponeCenterBBOXColor;
    static  int    binSize;

    // Visualisation
    static  bool render_bbox3D;
    static  bool render_bbox2D;
    static  bool render_tracking_numbers;

    //MDL parameters for trajectories
    static  double  k1; // "counterweight": min. support for a hypothesis
    static  double  k2 ; // rel. importance of #poconst ints vs. poconst double strength
    static  double  k3; // overlap penalty
    static  double  k4; // temp. decay for static objects

    // Threshold for distinction between static/moving object
    static  double  minvel;
    static  double  dMaxPedVel; // This is in meter per second = ~ 5km/h

    // Trajectory
    static  double threshLengthTraj;

    // Thresholds for accepted and displayed hypotheses
    static  double  dTheta2;

    // Time ant for temporal decay
    static  double  dTau;

    // Time horizon for event cone search
    static  int  coneTimeHorizon;
    static  double  maxHoleLen;
    static  double  dHolePenalty;

    /* Q - the system covariance */
    static  double sysUncX;
    static  double sysUncY;
    static  double sysUncRot;
    static  double sysUncVel;
    static  double sysUncAcc;

    /* P_init - the initial state covariance */
    static double initPX;
    static double initPY;
    static double initPVX;
    static double initPVY;


    static double kalmanObsMotionModelthresh;
    static double kalmanObsColorModelthresh;

    static int accepted_frames_without_det;


    /////////////////////////GP estimator//////////////////////
    static int nrInter_ransac;
    static int numberOfPoints_reconAsObstacle;

    ////////////////////////Evaluation////////////////////////
    static bool save_for_eval;
    static string save_path_tracking;
    static string save_path_img_info;
    static string save_path_cam_info;
    static string save_path_img;
    static string save_path_cam;
};

#endif	/* _GLOBALS_DENNIS_H */

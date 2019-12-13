/*
 * Copyright 2012 Dennis Mitzel
 *
 * Authors: Dennis Mitzel
 * Computer Vision Group RWTH Aachen.
 */

#ifndef _DENNIS_DETECTIONS_H
#define _DENNIS_DETECTIONS_H

#include "Math.h"
#include "Camera.h"
#include "CImageHeader.h"

#include <ros/ros.h>
#include "frame_msgs/DetectedPersons.h"

#ifndef cim_v
#include <QColor>
#include <QImage>
#endif

class Detections {
public:
  Detections(int x, const int flag);
  ~Detections();

  /* get information from other messages */
  int globalFrameToBufferFrame(int global_frame);
  int numberDetectionsAtFrame(int frame);

  void getPos3D(int frame, int detec, Vector<double> &pos);
  void getBBox(int frame, int detec, Vector<double> &bbox);
  double getScore(int frame, int detec);
  double getHeight(int frame, int detec);
  double getWarp(int frame, int detec);

  int getCategory(int frame, int detec);

  /* Method for adding online Detections */
#ifdef cim_v
  void addDetsOneFrame(const frame_msgs::DetectedPersons::ConstPtr &det,
                       int frame);
#else
  void addHOGdetOneFrame(const frame_msgs::DetectedPersons::ConstPtr &det,
                         int frame,
                         QImage &imageLeft,
                         Camera cam,
                         Matrix<double> &depth);
#endif
  int prepareDet(Vector<double> &detContent,
                 const frame_msgs::DetectedPersons::ConstPtr &det,
                 int i,
                 int frame,
                 bool leftDet,
                 Matrix<double> &covariance);

  /* Compute 3D Position out of BBox */
  void compute3DPosition(Vector<double> &detection, Camera cam);

  /* Compute 3D Position out of BBox */
#ifdef cim_v
  void computeColorHist(Volume<double> &colHist,
                        Vector<double> &bbox,
                        int nBins,
                        CImg<unsigned char> &imageLeft);
#else
  void computeColorHist(Volume<double> &colHist,
                        Vector<double> &bbox,
                        int nBins,
                        QImage &imageLeft);
#endif

  void getColorHist(int frame, int pos, Volume<double> &colHist);
  void getEmbVec(int frame, int pos, Vector<double> &embVecs);

  /* Compute the 3D uncertainty for a point */
  void compute3DCov(Vector<double> pos3d,
                    Matrix<double> &cov, Camera camL,
                    Camera camR);
  void get3Dcovmatrix(int frame, int pos, Matrix<double> &covariance);

  Vector<double> fromCamera2World(Vector<double> posInCamera, Camera cam);
  Vector<double> projectPlaneToCam(Vector<double> p, Camera cam);

  double get_mediandepth_inradius(Vector<double> &bbox,
                                  int radius,
                                  Matrix<double> &depthMap,
                                  double var,
                                  double pOnGp);

protected:
  Vector<Vector<Vector<double>>> detC;
  Vector<Vector<Matrix<double>>> cov3d;
  Vector<Vector<Volume<double>>> colHists;
  Vector<Vector<Vector<double>>> embed_vecs;
  int offSet;
  int img_num;
  int hypo_num;
  int center_x;
  int center_y;
  int scale;
  int categ;
  int bbox;
  int initscore;
  int score;
  int warp_loss;
  int dist;
  int height;
  int rot;
  int pos;
  int numberAllAccDetections;
  int numberOfFrames;
  int nrColinDetFile;
  int carOrient;
  int det_id;
  int buff_size;
};

#endif /* _DENNIS_DETECTIONS_H */

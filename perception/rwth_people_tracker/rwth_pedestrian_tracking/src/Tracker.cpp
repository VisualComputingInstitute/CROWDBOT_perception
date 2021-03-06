/*
 * Copyright 2012 Dennis Mitzel
 *
 * Authors: Dennis Mitzel
 * Computer Vision Group RWTH Aachen.
 */

#include "Tracker.h"

#include "ros/ros.h"
#include "ros/time.h"

// #define TIME_TRACKER

#ifdef TIME_TRACKER
#include <chrono>
#endif

using namespace std;


Tracker::Tracker()
{
    unsigned char color_array[] = {   204,     0,   255,
                                      255,     0,     0,
                                      0,   178,   255,
                                      255,     0,   191,
                                      255,   229,     0,
                                      0,   255,   102,
                                      89,   255,     0,
                                      128,     0,   255,
                                      242,     0,   255,
                                      242,   255,     0,
                                      255,     0,    77,
                                      51,     0,   255,
                                      0,   255,   140,
                                      0,   255,    25,
                                      204,   255,     0,
                                      255,   191,     0,
                                      89,     0,   255,
                                      0,   217,   255,
                                      0,    64,   255,
                                      255,   115,     0,
                                      255,     0,   115,
                                      166,     0,   255,
                                      13,     0,   255,
                                      0,    25,   255,
                                      0,   255,   217,
                                      0,   255,    64,
                                      255,    38,     0,
                                      255,     0,   153,
                                      0,   140,   255,
                                      255,    77,     0,
                                      255,   153,     0,
                                      0,   255,   179,
                                      0,   102,   255,
                                      255,     0,    38,
                                      13,   255,     0,
                                      166,   255,     0,
                                      0,   255,   255,
                                      128,   255,     0,
                                      255,     0,   230,
                                      51,   255,     0
                                  };

    possibleColors = Matrix<unsigned char>(3, 40, color_array);
    lastHypoID = -1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////                                               MONO                                                    /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


double rand_doubleRange(double a, double b)
{
    return ((b-a)*((double)rand()/RAND_MAX))+a;
}

void Tracker::check_termination(Camera &cam, Vector<Hypo>& HyposAll)
{
    Vector<double> gp = cam.get_GP();

    if (HyposAll.getSize() != 0)
    {
        //*********************************************************
        // Define the "exit zones"
        //*********************************************************
        Vector<double> aux(3, 1);
        aux(0) = 1;
        aux(1) = Globals::dImHeight*(5.0/5.0);
        Vector<double> vXTL = AncillaryMethods::backprojectGP(aux, cam, gp);
        aux(0) = 1;
        aux(1) = Globals::dImHeight*(1.0/5.0);
        Vector<double> vXBL = AncillaryMethods::backprojectGP(aux, cam, gp);
        aux(0) = Globals::dImWidth - 1;
        aux(1) = Globals::dImHeight*(5.0/5.0);
        Vector<double> vXTR = AncillaryMethods::backprojectGP(aux, cam, gp);
        aux(0) = Globals::dImWidth - 1;
        aux(1) = Globals::dImHeight*(1.0/5.0);
        Vector<double> vXBR = AncillaryMethods::backprojectGP(aux, cam, gp);

        // Calculate exit planes, 4th value in normal just keeps time stamp
        Vector<double> Gp = cam.get_GPN();
        Vector<double> nR4d;
        Vector<double> nL4d;
        Vector<double> nC4d;
        double dL4d;
        double dR4d;
        double dC4d;

        vXBL -= vXTL;
        nL4d = cross(Gp, vXBL);
        nL4d *= 1.0 / nL4d.norm();
        dL4d = -(DotProduct(nL4d, vXTL));

        vXBR -= vXTR;
        nR4d = cross(Gp, vXBR);
        nR4d *= -1.0;
        nR4d *= 1.0 / nR4d.norm();
        dR4d = -(DotProduct(nR4d, vXTR));

        // Principal plane of left camera as 'behind camera' exit zone

        Matrix<double> KRt = cam.get_KRt();
        Vector<double> KRtT = cam.get_KRtT();

        nC4d = KRt.getRow(2);
        dC4d = -KRtT(2) * Globals::WORLD_SCALE;

        int nEndTolerance = 0;
        double zonesizeL = 0.7;
        double zonesizeR = 0.7;
        double zonesizeC = 1;
        double zonesizeC2 = 1.5;

        Vector<Vector <double> > TrajPts;

        Vector<int> hyposToRemove;

        for (int i = 0; i < HyposAll.getSize(); i++)
        {

            HyposAll(i).getTrajPts(TrajPts);

            Vector<double> dvec1;
            Vector<double> dvec2;
            Vector<double> dvec3;

            int j = TrajPts.getSize()-1;
            dvec1.pushBack(TrajPts(j)(0) * nL4d(0) + TrajPts(j)(1)*nL4d(1) + TrajPts(j)(2)*nL4d(2) + dL4d);
            dvec2.pushBack(TrajPts(j)(0) * nR4d(0) + TrajPts(j)(1)*nR4d(1) + TrajPts(j)(2)*nR4d(2) + dR4d);
            dvec3.pushBack(TrajPts(j)(0) * nC4d(0) + TrajPts(j)(1)*nC4d(1) + TrajPts(j)(2)*nC4d(2) + dC4d);

            Vector<int> IdxVec1;
            Vector<int> IdxVec2;
            Vector<int> IdxVec3;
            Vector<int> IdxVec4;

            for (int j = 0; j < dvec1.getSize(); j++)
            {
                if (dvec1(j) < zonesizeL) IdxVec1.pushBack(j);
                if (dvec2(j) < zonesizeR) IdxVec2.pushBack(j);
                if (dvec3(j) < zonesizeC) IdxVec3.pushBack(j);
                if (dvec3(j) < zonesizeC2) IdxVec4.pushBack(j);
            }

            if(IdxVec1.getSize() > nEndTolerance)
            {
                ROS_DEBUG("HYPO entered left EXIT ZONE");
                hyposToRemove.pushBack(i);
            }
            else if(IdxVec2.getSize() > nEndTolerance)
            {
                ROS_DEBUG("HYPO entered right EXIT ZONE");
                hyposToRemove.pushBack(i);
            }else if (IdxVec3.getSize() > nEndTolerance || (IdxVec4.getSize() > 2))
            {
                ROS_DEBUG("HYPO entered entered behind camera EXIT ZONE");
                hyposToRemove.pushBack(i);
            }



        }
        Vector<Hypo> remainedHypos;
        for(int i = 0; i < HyposAll.getSize() ; i++)
        {
            if(hyposToRemove.findV(i) < 0)
            {
                remainedHypos.pushBack(HyposAll(i));
            }
        }

        HyposAll = remainedHypos;
    }
}

#ifndef cim_v
void Tracker::process_tracking_oneFrame(Vector<Hypo>& HyposAll, Detections& allDet, int frame,
                                        Vector<Vector<double> > foundDetInFrame, QImage& im, Camera cam, Matrix<double> depthMap)
{

    char imageSavePath[200];

    #ifdef TIME_TRACKER
    auto t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    allDet.addHOGdetOneFrame(foundDetInFrame, frame, im, cam, depthMap);

    #ifdef TIME_TRACKER
    auto t1_debug = std::chrono::high_resolution_clock::now();
    #endif

    //*****************************************************************************************************
    HyposMDL.clearContent();
    process_frame( allDet , cam, frame, HyposMDL, HyposAll, HypoEnded, hypoStack, possibleColors, assignedBBoxCol, hypoLastSelForVis);

    #ifdef TIME_TRACKER
    auto t2_debug = std::chrono::high_resolution_clock::now();

    auto dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    auto dt12_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t2_debug - t1_debug);

    std::cout << "Tracker.cpp summary, adding detection time: "<< dt01_debug.count() << "ms, ";
    std::cout << "process frame time: "<< dt12_debug.count() << "ms\n";
    #endif

    //***************************************************************************************
    // Visualization part 3D
    //***************************************************************************************
    Vector<Vector<double> > vvHypoTrajPts;
    Vector<double> vX(3);
    Vector<double> vDir;
    Vector<FrameInlier> Idx;

    Vector<double> bbox;
    Vector<int> colors;
    Visualization vis;

    int number_of_frames_hypo_visualized_without_inlier = 7;


    Vector<Vector<double> > hyposToWrite(HyposMDL.getSize());

    if(Globals::render_bbox3D)
    {

        for(int i = 0; i < HyposMDL.getSize(); i++)
        {

            //***************************************************************************************
            // Render only if the last occurence of an inlier is not far away
            //***************************************************************************************
            Vector<FrameInlier> inlier;
            HyposMDL(i).getIdx(inlier);

            hyposToWrite(i).setSize(5);

            bbox.clearContent();
            HyposMDL(i).getTrajPts(vvHypoTrajPts);
            Matrix<double> allP;
            HyposMDL(i).getXProj(allP);
            vX = allP.getRow(allP.y_size()-1);
            vX(2) = vX(1);
            vX(1) = vX(3);
            vX.resize(3);
            hyposToWrite(i)(0) = HyposMDL(i).getHypoID();
            hyposToWrite(i)(1) = vX(0);
            hyposToWrite(i)(2) = vX(1);
            hyposToWrite(i)(3) = vX(2);
            hyposToWrite(i)(4) = HyposMDL(i).getHeight();

            Vector<double> b(3);
            vvHypoTrajPts.clearContent();
            vvHypoTrajPts.setSize(allP.y_size());
            for(int j = 0; j < vvHypoTrajPts.getSize(); j++)
            {
                b(0) = allP(0,j);
                b(1) = allP(3,j);
                b(2) = allP(1,j);
                vvHypoTrajPts(j) = b;
            }

            HyposMDL(i).getDir(vDir);
            HyposMDL(i).getIdx(Idx);


            vis.render_hypos(cam, frame, assignedBBoxCol, hypoLastSelForVis, possibleColors, im, HyposMDL(i).getSpeed(), Globals::minvel, Globals::pedSizeWVis,
                             Globals::pedSizeWVis, HyposMDL(i).getHeight(), HyposMDL(i).getHypoID(), vvHypoTrajPts, Globals::WORLD_SCALE, vX, vDir, bbox, colors);


        }

        int nrDet = allDet.numberDetectionsAtFrame(frame);
        Vector<double> bbox;
        for(int i = 0; i < nrDet; i++)
        {
            allDet.getBBox(frame, i, bbox);
            vis.render_bbox_2D(bbox, im, 0, 255, 0, 1);
        }

    }

    AncillaryMethods::exportBBOX(HyposMDL, cam, frame, *aStream);
    //    AncillaryMethods::write_vec_vec_to_disk(hyposToWrite,imageSavePath);

}
#else
void Tracker::process_tracking_oneFrame(Vector<Hypo>& HyposAll, Detections& allDet, int frame,
                                        const frame_msgs::DetectedPersons::ConstPtr &foundDetInFrame/*, CImg<unsigned char>& im, Camera &cam, Matrix<double> &depthMap*/)
{
    #ifdef TIME_TRACKER
    auto t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    allDet.addDetsOneFrame(foundDetInFrame, frame/*, im, cam, depthMap*/);

    #ifdef TIME_TRACKER
    auto t1_debug = std::chrono::high_resolution_clock::now();
    #endif

    //*****************************************************************************************************
    HyposMDL.clearContent();
    process_frame( allDet , /*cam,*/ frame, HyposAll);

    #ifdef TIME_TRACKER
    auto t2_debug = std::chrono::high_resolution_clock::now();

    auto dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    auto dt12_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t2_debug - t1_debug);

    std::cout << "Tracker.cpp summary, adding detection time: "<< dt01_debug.count() << "ms, ";
    std::cout << "process frame time: "<< dt12_debug.count() << "ms\n";
    #endif
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////                                               STEREO                                                    ///////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Tracker::process_frame(Detections& det, /*Camera &cam,*/ int t,  Vector< Hypo >& HyposAll)
{


    MDL mdl;
    Vector < FrameInlier > currIdx;
    Vector < FrameInlier > oldIdx;
    Vector < FrameInlier > overlap;
    Vector <int> vRemoveHypos;
    int hypoIdNew;
    Vector<int> HypoIdx;
    double bestFraction = Globals::dSameIdThresh;
    double reIDThresh = Globals::reIdThresh_HypoLevel;
    int bestNumMatches = 0;
    double bestEmbDist = 0;
    int numObsCurr;
    int numObsOld;
    int numOverlap;
    double equalFraction;
    int bTerminated = 0;
    Matrix<double> Q;
    Vector<double> m;
    Vector< Hypo > HypoNew;
    Vector< Hypo > HypoExtended;


    double normfct = 0.3;
    //*****************************************************************
    // Define frame range for finding new Hypos
    //*****************************************************************

    int LTPmax = t;
    int LTPmin = max(LTPmax-Globals::history, 0); //Globals::nOffset

    //******************************************************************
    // Extend Trajectories Hypos
    //******************************************************************
    #ifdef TIME_TRACKER
    auto t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    Vector<int> extendUsedDet;
    extend_trajectories(HyposAll,  det, LTPmax, LTPmin, normfct, HypoExtended, extendUsedDet/*, cam*/);
    ROS_DEBUG("\33[36;40;1m Extended %i trajectories\33[0m", HypoExtended.getSize());

    #ifdef TIME_TRACKER
    auto t1_debug = std::chrono::high_resolution_clock::now();

    auto dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);

    std::cout << "Tracker.cpp, extend_trajectories: "<< dt01_debug.count() << "ms";
    std::cout << "  (extended " << HypoExtended.getSize() << " trajectories)\n";
    #endif


    //******************************************************************
    // Find new Hypos
    //******************************************************************
    #ifdef TIME_TRACKER
    t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    make_new_hypos(LTPmax, LTPmin, det, HypoNew, normfct, extendUsedDet);
    ROS_DEBUG("\33[31;40;1m     Created %i new Trajectories \33[0m", HypoNew.getSize());

    #ifdef TIME_TRACKER
    t1_debug = std::chrono::high_resolution_clock::now();

    dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);

    std::cout << "Tracker.cpp, make_new_hypos: "<< dt01_debug.count() << "ms";
    std::cout << " (created " << HypoNew.getSize() << " new trajectories)\n";
    #endif

    HyposAll.clearContent();
//    HyposAll.append(HypoEnded);
    HyposAll.append(HypoExtended);
    HyposAll.append(HypoNew);
    HypoNew.clearContent();
    HypoExtended.clearContent();

    //******************************************************************
    // Prepare Hypos
    //******************************************************************
    #ifdef TIME_TRACKER
    t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    prepare_hypos(HyposAll);

    Vector <Hypo> temp;

    for (int i = 0; i < HyposAll.getSize(); i++)
    {
        if (HyposAll(i).getScoreMDL() > Globals::dTheta2)
        {
            temp.pushBack(HyposAll(i));
        }
    }

    HyposAll = temp;

    #ifdef TIME_TRACKER
    t1_debug = std::chrono::high_resolution_clock::now();
    dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    std::cout << "Tracker.cpp, prepare_hypos: "<< dt01_debug.count() << "ms\n";
    #endif

    //******************************************************************
    // Build the MDL Matrix
    //******************************************************************
    #ifdef TIME_TRACKER
    t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    mdl.build_mdl_matrix(Q, HyposAll, LTPmax, normfct);

    #ifdef TIME_TRACKER
    t1_debug = std::chrono::high_resolution_clock::now();
    dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    std::cout << "Tracker.cpp, build_mdl_matrix: "<< dt01_debug.count() << "ms\n";
    #endif

    //******************************************************************
    //  Solve MDL Greedy
    //******************************************************************
    #ifdef TIME_TRACKER
    t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    //    mdl.solve_mdl_exactly(Q, m, HyposMDL, HypoIdx, HyposAll);

    mdl.solve_mdl_greedy(Q, m, HyposMDL, HypoIdx, HyposAll);

    #ifdef TIME_TRACKER
    t1_debug = std::chrono::high_resolution_clock::now();
    dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    std::cout << "Tracker.cpp, solve_mdl_greedy: "<< dt01_debug.count() << "ms\n";
    #endif
    //******************************************************************
    //  Fix IDs
    //******************************************************************

    #ifdef TIME_TRACKER
    t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    for(int i = 0; i < HyposMDL.getSize(); i++)
    {
        // skip terminated hypotheses
        if (HyposMDL(i).isTerminated()) continue;
        bTerminated = 0;
        // if the parent is known from trajectory extension
        if (!Globals::changeID_onthefly && HyposMDL(i).getParentID() > 0)
        {
            HyposMDL(i).setHypoID(HyposMDL(i).getParentID());
            ROS_DEBUG("Continuing extended trajectory %d (%f - %f) \n", HyposMDL(i).getHypoID(), (1 - Globals::k2)*HyposMDL(i).getNW(), Globals::k2*HyposMDL(i).getScoreW());
            for (int j = 0; j < hypoStack.getSize(); j++)
            {
                if (hypoStack(j).getHypoID() == HyposMDL(i).getHypoID())
                {
                    hypoStack(j) = HyposMDL(i);
                    break;
                }
            }
        }
        else
        {
            //if the parent is unknown - determine overlap with existing hypothesis OR check reID embedding vector
            hypoIdNew = -1;
            bestNumMatches = 0;
            bestEmbDist = 999.0;
            int lengthOldStack = hypoStack.getSize();
            HyposMDL(i).getIdx(currIdx);
            numObsCurr = AncillaryMethods::getSizeIdx(currIdx);

            // check the hypothesis against the stack of previous ones (try to
            // find the trajectory ID in the past)
            for(int j = 0; j < lengthOldStack; j++)
            {
                hypoStack(j).getIdx(oldIdx);
                AncillaryMethods::intersectIdx(currIdx, oldIdx, overlap);
                numObsOld = AncillaryMethods::getSizeIdx(oldIdx);
                numOverlap = AncillaryMethods::getSizeIdx(overlap);

                Vector<double> embDistVec = hypoStack(j).getEmb_vec() - HyposMDL(i).getEmb_vec();
                double embDist = embDistVec.norm();

                if(overlap.getSize() > 0)
                {
                    equalFraction = double(numOverlap) / min(double(numObsCurr), double(numObsOld));
                }
                else
                {
                    equalFraction = 0.0;
                }

                // if embVectors are available (size of Dist vector >0), use reID emb; otherwise use the physical overlap
                if( (equalFraction > bestFraction && numOverlap > bestNumMatches && embDistVec.getSize()<=0 ) || (embDist < reIDThresh && embDist < bestEmbDist && embDistVec.getSize()>0) )
                {
                    hypoIdNew = hypoStack(j).getHypoID();
                    bestNumMatches = numOverlap;
                    bestEmbDist = embDist;
                }
            }

            if (hypoIdNew > -1)
            {
                //ID was found - set ID of current trajectory, and update the stack
                HyposMDL(i).setHypoID(hypoIdNew);
                if(bTerminated == 0)
                {
                    //hypoStack(hypoIdNew) = (HyposMDL(i));
                    for (int j = 0; j < hypoStack.getSize(); j++)
                    {
                        if (hypoStack(j).getHypoID() == hypoIdNew)
                        {
                            hypoStack(j) = HyposMDL(i);
                            break;
                        }
                    }
                }else
                {
                    vRemoveHypos.pushBack(i);
                }

                ROS_DEBUG("Replacing trajectory %d with new hypos (%f - %f) \n", HyposMDL(i).getHypoID(), (1 - Globals::k2)*HyposMDL(i).getNW(), Globals::k2*HyposMDL(i).getScoreW());
            }
            else
            {
                // ID was not found - create a new ID for the current trajectory
                // and add it to the stack
                lastHypoID +=1;
                HyposMDL(i).setHypoID(lastHypoID);
                hypoStack.pushBack(HyposMDL(i));
                ROS_DEBUG("Creating new Trajectory %i (%f - %f) ", HyposMDL(i).getHypoID(), (1 - Globals::k2)*HyposMDL(i).getNW(), Globals::k2*HyposMDL(i).getScoreW());
            }
        }
    }

    #ifdef TIME_TRACKER
    t1_debug = std::chrono::high_resolution_clock::now();
    dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    std::cout << "Tracker.cpp, Fix IDs: "<< dt01_debug.count() << "ms\n";
    #endif

    //******************************************************************
    // in case, "FixID" made new duplicate IDs, remove them
    //******************************************************************
    #ifdef TIME_TRACKER
    t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    HypoIdx = remove_ID_duplicates(HyposMDL, HypoIdx);

    #ifdef TIME_TRACKER
    t1_debug = std::chrono::high_resolution_clock::now();
    dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    std::cout << "Tracker.cpp, remove_ID_duplicates: "<< dt01_debug.count() << "ms\n";
    #endif

    //******************************************************************
    // Print HypoMDL results
    //******************************************************************

    #ifdef TIME_TRACKER
    t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    for(int i = 0; i < HyposMDL.getSize(); i++)
    {
        if(HyposMDL(i).isMoving())
        {
            ROS_DEBUG("\33[1;32;40;1m Score of Hypo %i is = %f (pedestrian, moving, speed = %f )\33[0m", HyposMDL(i).getHypoID(), HyposMDL(i).getScoreMDL(), HyposMDL(i).getSpeed());
        }
        else
        {
            ROS_DEBUG("\33[1;32;40;1m Score of Hypo %i is = %f (pedestrian, static) \33[0m", HyposMDL(i).getHypoID(), HyposMDL(i).getScoreMDL());
        }
    }

    for(int i = 0; i < HypoIdx.getSize(); i++)
    {
        if(!HyposMDL(i).isTerminated())
        {
            HyposAll(HypoIdx(i)).setLastSelected(t);
            HyposAll(HypoIdx(i)).setHypoID(HyposMDL(i).getHypoID());
            HyposAll(HypoIdx(i)).setParentID(HyposMDL(i).getParentID());
        }
    }

    #ifdef TIME_TRACKER
    t1_debug = std::chrono::high_resolution_clock::now();
    dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    std::cout << "Tracker.cpp, print HypoMDL results: "<< dt01_debug.count() << "ms\n";
    #endif

    // we dont check termination (exit zones, eg, at image border) anymore, to continue (or reidentify) targets, even if they have left the image
    //check_termination(cam, HyposAll);

    //******************************************************************
    // Only propagate non-terminated hypotheses to the next frame
    //******************************************************************
    #ifdef TIME_TRACKER
    t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    unsigned int nr =  HyposAll.getSize();
    temp.clearContent();
    for(unsigned int i = 0; i < nr; i++)
    {
        // termination is not used anymore (see above)
        //if((!HyposAll(i).isTerminated()) && ((t - HyposAll(i).getLastSelected()) < Globals::coneTimeHorizon))
        if((t - HyposAll(i).getLastSelected()) < Globals::coneTimeHorizon)
        {
            temp.pushBack(HyposAll(i));
            //std::cout << "prop hypoID " << HyposAll(i).getHypoID() << "(last selected: " << HyposAll(i).getLastSelected() << "-- now: " << t << ")" << std::endl;
        }/*else{
            std::cout << "not propagated hypoID " << HyposAll(i).getHypoID() << " to next frame, as it is too old (last selected: " << HyposAll(i).getLastSelected() << "-- now: " << t << ")" << std::endl;
        }*/
    }
    HyposAll = temp;

    // update hypoStack to exclude way too old tracks from reID
    Vector<Hypo> newHypoStack;
    newHypoStack.clearContent();
    for (int j = 0; j < hypoStack.getSize(); j++)
    {
        if((t - hypoStack(j).getLastSelected()) < Globals::coneTimeHorizon*5)
        {
            newHypoStack.pushBack(hypoStack(j));
            //std::cout << "kept hypoID " << hypoStack(j).getHypoID() << " in stack for reID" << std::endl;
            //std::cout << "last selected: " << (t - hypoStack(j).getLastSelected()) << " frames ago" << std::endl;
        }
        /*else{
            std::cout << "deleted hypoID " << hypoStack(j).getHypoID() << " from stack, as it is too old (no reID of this hypo anymore)." << std::endl;
            std::cout << "last selected: " << (t - hypoStack(j).getLastSelected()) << " frames ago" << std::endl;
        }*/
    }
    hypoStack = newHypoStack;

    #ifdef TIME_TRACKER
    t1_debug = std::chrono::high_resolution_clock::now();
    dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    std::cout << "Tracker.cpp, propagate hypotheses: "<< dt01_debug.count() << "ms";
    std::cout << " (" << hypoStack.getSize() << " hypotheses)\n";
    #endif

    // test for robust ID_map

    #ifdef TIME_TRACKER
    t0_debug = std::chrono::high_resolution_clock::now();
    #endif

    if (Globals::changeID_onthefly)
    {
        bool already_in_set = false;
        id_map.clear();
        for (int j = 0; j < hypoStack.getSize(); j++)
        {
            //already in a set? continue!
            already_in_set = false;
            std::map<int, std::set<int>>::iterator it = id_map.begin();
            while(it != id_map.end())
            {
                //std::cout<<it->first<<" :: "<<it->second<<std::endl;
                if(it->second.find(hypoStack(j).getHypoID())!=it->second.end()){
                    already_in_set = true;
                    break;
                }
                it++;
            }
            if(already_in_set){
                continue;
            } else{
                id_map[hypoStack(j).getHypoID()].insert(hypoStack(j).getHypoID());
            }
            for (int k = j+1; k < hypoStack.getSize(); k++)
            {
                //already in a set? continue!
                already_in_set = false;
                std::map<int, std::set<int>>::iterator it = id_map.begin();
                while(it != id_map.end())
                {
                    if(it->second.find(hypoStack(k).getHypoID())!=it->second.end()){
                        already_in_set = true;
                        break;
                    }
                    it++;
                }
                if(already_in_set){
                    continue;
                } else{
                    id_map[hypoStack(j).getHypoID()].insert(hypoStack(j).getHypoID());
                }
                Vector<double> embDistVec = hypoStack(j).getEmb_vec() - hypoStack(k).getEmb_vec();
                double embDist = embDistVec.norm();
                if(embDist<=Globals::reIdThresh_HypoLevel && embDistVec.getSize()>0){
                    id_map[hypoStack(j).getHypoID()].insert(hypoStack(k).getHypoID());
                    // CHANGE ID ON-THE-FLY
                    if(Globals::changeID_onthefly) hypoStack(k).setHypoID(hypoStack(j).getHypoID());
                }
                //std::cout << "kept hypoID " << hypoStack(j).getHypoID() << " in stack for reID" << std::endl;
                //std::cout << "last selected: " << (t - hypoStack(j).getLastSelected()) << " frames ago" << std::endl;
            }
        }
        /*std::map<int, std::set<int>>::iterator it = id_map.begin();
        while(it != id_map.end())
        {
            std::cout<<it->first<<" :: {";
            std::set<int>::iterator it2 = it->second.begin();
            while(it2 != it->second.end()){
                std::cout<< *it2 << ", ";
                it2++;
            }
            std::cout<< "} "<<std::endl;
            it++;
        }*/
    }

    #ifdef TIME_TRACKER
    t1_debug = std::chrono::high_resolution_clock::now();
    dt01_debug = std::chrono::duration_cast<std::chrono::milliseconds>(t1_debug - t0_debug);
    std::cout << "Tracker.cpp, test for robust_ID map: "<< dt01_debug.count() << "ms\n";
    #endif
}

void Tracker::prepare_hypos(Vector<Hypo>& vHypos)
{

    int nrHyposOld  = vHypos.getSize();
    int nrHyposOldOut = nrHyposOld;

    double k1 = Globals::k1;
    double k2 = Globals::k2;

    double nw;
    double scoreW;
    double scoreMDL;
    Vector<int> vTrajT;
    Vector<FrameInlier> Idx;

    //**********************************************************************
    // Prepare the hypothesis scores
    //**********************************************************************

    for (int i = 0; i < nrHyposOld; i++)
    {
        if (vHypos(i).getSpeed() > Globals::minvel)
        {
            vHypos(i).setMoving(true);
        }else
        {
            vHypos(i).setMoving(false);
        }
    }

    for(int i = 0; i < nrHyposOld; i++)
    {


        nw = vHypos(i).getNW();
        scoreW = vHypos(i).getScoreW();
        scoreMDL = -k1 + ((1-k2)*nw + k2*scoreW);
        vHypos(i).setScoreMDL(scoreMDL);

        vHypos(i).getTrajT(vTrajT);
        vHypos(i).getIdx(Idx);

        int nrInlier = 0;

        for(int j = 0; j < Idx.getSize(); j++)
        {
            nrInlier += Idx(j).getNumberInlier();
        }

        if(/*vHypos(i).isMoving() &&*/ nrInlier < Globals::threshLengthTraj)
        {
            vHypos(i).setScoreMDL(-1.0);
        }
    }

    ROS_DEBUG("Filtering out low-scoring hypothesis...");
    ROS_DEBUG("  compacted hypothesis set from %i to %i", nrHyposOldOut, vHypos.getSize());


    //*************************************************************************
    // Remove duplicates
    //*************************************************************************
    remove_duplicates(vHypos);
}

void Tracker::remove_duplicates(Vector<Hypo>& hypos)
{
    //strict_remove = 1: always remove without looking at isMoving or inlierIntersection
    Vector<int> duplicats(hypos.getSize(), -1);
    Vector<FrameInlier> idx1;
    Vector<FrameInlier> idx2;
    Vector<FrameInlier> intersection;

    int nrHyposOld = hypos.getSize();


    //*******************************************************************
    // Find Duplicates
    //*******************************************************************

    for(int i = 0; i < hypos.getSize(); i++)
    {
        if(duplicats(i)==-1)
        {
            for(int j = i+1; j < hypos.getSize(); j++)
            {
                if(duplicats(j)==-1 &&  hypos(i).isMoving() == hypos(j).isMoving() )
                {
                    hypos(i).getIdx(idx1);
                    hypos(j).getIdx(idx2);
                    AncillaryMethods::intersectIdx(idx1, idx2, intersection);

                    if( AncillaryMethods::getSizeIdx(idx1)*0.89 < AncillaryMethods::getSizeIdx(intersection) || AncillaryMethods::getSizeIdx(idx2)*0.89 < AncillaryMethods::getSizeIdx(intersection) )
                    {
                        if(hypos(i).getScoreMDL() >= hypos(j).getScoreMDL())
                        {
                            duplicats(j) = 1;
                        }
                        else
                        {
                            duplicats(i) = 1;
                        }
                    }
                }
            }
        }
    }

    //***********************************************************************
    // Remove Duplicates
    //***********************************************************************

    Vector<Hypo> copyH;
    for(int i = 0; i < hypos.getSize(); i++)
    {
        if(duplicats(i)==-1)
        {
            copyH.pushBack(hypos(i));
        }
    }

    hypos = copyH;
    ROS_DEBUG("Removing duplicate hypos...");
    ROS_DEBUG("  compacted hypothesis set from %i to %i", nrHyposOld, hypos.getSize());
}

Vector<int> Tracker::remove_ID_duplicates(Vector<Hypo>& hypos, Vector<int>& hypoIdx)
{
    Vector<int> duplicats(hypos.getSize(), -1);
    Vector<int> new_hypoIdx;

    //*******************************************************************
    // Find Duplicate IDs
    //*******************************************************************

    for(int i = 0; i < hypos.getSize(); i++)
    {
        if(duplicats(i)==-1)
        {
            for(int j = i+1; j < hypos.getSize(); j++)
            {
                if(duplicats(j)==-1 && hypos(i).getHypoID()==hypos(j).getHypoID())
                {
                    if(hypos(i).getScoreMDL() >= hypos(j).getScoreMDL())
                    {
                        duplicats(j) = 1;
                    }
                    else
                    {
                        duplicats(i) = 1;
                    }

                }
            }
        }
    }

    //***********************************************************************
    // Remove Duplicate IDs (remove the one with lower score)
    //***********************************************************************
    Vector<Hypo> copyH;
    for(int i = 0; i < hypos.getSize(); i++)
    {
        if(duplicats(i)==-1)
        {
            copyH.pushBack(hypos(i));
            if(hypoIdx.getSize()>i) new_hypoIdx.pushBack(hypoIdx(i));
        }
    }

    // result
    hypos = copyH;
    return new_hypoIdx;
}


void getCurrentSmoothDirection(Matrix<double> &pts, int smoothing_window, double &dir, double &vel)
{

    Matrix<double> smoothed = AncillaryMethods::smoothTrajMatrix(pts, smoothing_window);
    smoothed = AncillaryMethods::smoothTrajMatrix(smoothed, smoothing_window);

    Vector<double> xs = smoothed.getRow(1);
    Vector<double> xe = smoothed.getRow(min(smoothed.y_size()-1, smoothing_window));

    xe -= xs;
    xe *= 1.0/(double)(min(smoothed.y_size(), smoothing_window));

    dir = atan2(xe(1),xe(0));
    vel = sqrtf(xe(0)*xe(0)+xe(1)*xe(1));

}


void Tracker::extend_trajectories(Vector< Hypo >& vHypos,  Detections& det, int t, int /*LTPmin*/,
                                  double normfct, Vector< Hypo >& HypoExtended, Vector<int>& extendUsedDet/*,
                                  Camera &*/ /*cam*/)
{
    //std::cout << "EXTEND..." << std::endl;
    //*************************************************************************
    // Create trajectory hypotheses by extending the previously existing
    //*************************************************************************

    extendUsedDet.clearContent();
    Vector<int> extendedUseTemp;

    // not used here, yet
    //Vector<double> Lmax(3, Globals::pedSizeWCom);
    //Lmax(2) = Globals::pedSizeHCom;

    //double dLastT;
    //int nrFramesWithInlier;
    //Vector<double> vMeanPoint3D(3, 0.0);

    //Matrix<double> Rot4D;

    int numberHypos = vHypos.getSize();
    Hypo* auxHypo;
    Vector< Hypo > newHypos;

    Matrix<double> mAllXnewUp;
    Vector<FrameInlier> vvIdxUp;
    Vector<double>  vVXUp;
    Vector<double>  vVYUp;

    Matrix<double> mNewAllX;
    Vector<FrameInlier> vvNewIdx;
    Vector<double>  vNewVX;
    Vector<double>  vNewVY;

    Matrix<double> mCurrAllX;
    Vector<FrameInlier> vvCurrIdx;
    Vector<double>  vCurrVX;
    Vector<double>  vCurrVY;

    Volume<double> volHMean;
    //int pointPos;
    //Vector<int> inlier;
    int n_HypoId;
    //Vector<double> pos3DStl;
    Vector<double> bbox(4,0.0);


    int timeHorizon = Globals::coneTimeHorizon;

    //*************************************
    // Kalman
    //*************************************

    Vector<Matrix<double> > stateCovMatsNew;
    Vector<Volume<double> > colHistsNew;

    Vector<Matrix<double> > stateCovMatsOld;
    Vector<Volume<double> > colHistsOld;
    Vector<double> embVecOld;
    for (int i = 0; i < numberHypos; i++)
    {
        auxHypo = &(vHypos(i));
        mAllXnewUp.set_size(0,0,0.0);

        //**********************************************************
        // if a hypothesis was useless for too long, abandon it
        //***********************************************************

        if(t - auxHypo->getLastSelected() > timeHorizon)
        {
            ROS_DEBUG("  DEBUG: Hypothesis %i too old ==> dropped.", i);
            continue;
        }
        auxHypo->getIdx(vvCurrIdx);
        auxHypo->getVX(vCurrVX);
        auxHypo->getVY(vCurrVY);
        auxHypo->getXProj(mCurrAllX);
        //auxHypo->getRot4D(Rot4D);
        n_HypoId = auxHypo->getHypoID();

        // ---commented out some stuff here regarding (unused) positions and bboxes, which cannot be used anymore due to detection buffer (or save them in inlier) and 3d tracking---
        //nrFramesWithInlier = vvCurrIdx.getSize();
        //dLastT = vvCurrIdx(nrFramesWithInlier-1).getFrame();
        //vMeanPoint3D.fill(0.0);
        //inlier = vvCurrIdx(nrFramesWithInlier-1).getInlier();

        // BBOX not used anymore! 3d Tracking, cam_info unknown!
        /*Vector<double> bbox(4,0.0);
        Vector<double> bbox_sum(4,0.0);
        for(int j = 0; j < inlier.getSize(); j++)
        {
            det.getPos3D(dLastT, inlier(j), pos3DStl);
            vMeanPoint3D(0) += pos3DStl(0);
            vMeanPoint3D(1) += pos3DStl(1);
            vMeanPoint3D(2) += pos3DStl(2);
            det.getBBox(dLastT, inlier(j), bbox);
            bbox_sum += bbox;

        }
        vMeanPoint3D *=(1.0/(inlier.getSize()));
        bbox_sum *=(1.0/(inlier.getSize()));
        bbox = bbox_sum;*/

        // not used?
        /*double minValue = 100000.0;

        for(int j = 0; j < det.numberDetectionsAtFrame(dLastT); j++)
        {
            det.getPos3D(dLastT, j, pos3DStl);

            pos3DStl -= vMeanPoint3D;

            if (minValue > pos3DStl.norm())
            {
                minValue = pos3DStl.norm();
                pointPos = j;
            }
        }*/
        // ---end of comment section---

        Hypo newHypo;

        Matrix<double> mXProj;
        auxHypo->getXProj(mXProj);

        Vector<double> xInit;

        // commented out to allow for track with length 1
        //Vector<Vector<double> > vvTrajPTS;
        //auxHypo->getTrajPts(vvTrajPTS);
        //if(vvTrajPTS.getSize() < 2) continue;
        //Vector<double> vX;
        //auxHypo->getVX(vX);

        xInit.setSize(4, 0.0);

        xInit(0) = mXProj(0, mXProj.y_size()-1);
        xInit(1) = mXProj(1, mXProj.y_size()-1);
        xInit(2) = vCurrVX(vCurrVX.getSize()-1);
        xInit(3) = vCurrVY(vCurrVY.getSize()-1);

        //double dir, vel;
        //getCurrentSmoothDirection(mXProj, 12, dir, vel);
        //xInit(2) = rotation(rotation.getSize()-1);
        //xInit(3) = vCurrV(vCurrV.getSize()-1);
        //xInit(2) = dir;
        //xInit(3) = vel*Globals::frameRate;
        //printf("dir: %.4f, vel: %.4f\n",dir,vel*Globals::frameRate);
        //if(vel < Globals::minvel*3) xInit(3) = 0;

        auxHypo->getColHists(colHistsOld);
        auxHypo->getStateCovMats(stateCovMatsOld);
        embVecOld = auxHypo->getEmb_vec();

        EKalman kalman;
        kalman.init(xInit, stateCovMatsOld(stateCovMatsOld.getSize()-1), Globals::dt);
        kalman.runKalmanUp(det, t-1, t, mAllXnewUp, vvIdxUp, vVXUp, vVYUp, volHMean,
                           stateCovMatsNew, colHistsOld(colHistsOld.getSize()-1), embVecOld,
                           colHistsNew, mXProj(3, mXProj.y_size()-1), bbox);

        // calcualte new Size for allX, V, R, W ...

        // FIX ME: if det used for extension is excluded to make new hypos is only based on #inlier>0
        // + the number of inlier taken here is in first frame of traj, which should be always >0
        // diss says: this decision is based on association prob., available here?
        if(vvIdxUp.getSize() > 0)
            extendedUseTemp = vvIdxUp(0).getInlier();

        if(extendedUseTemp.getSize() > 0)
            extendUsedDet.append(extendedUseTemp);

        int newSize = mXProj.y_size() + 1;

        mNewAllX.set_size(4, newSize);
        vNewVX.setSize(newSize);
        vNewVY.setSize(newSize);

        // get the part of the original trajectory up to, but excluding, the last detection
        // and add the newly appened part


        for(int j = 0; j < mXProj.y_size(); j++)
        {
            vNewVX(j) = vCurrVX(j);
            vNewVY(j) = vCurrVY(j);
        }

        for(int j = mXProj.y_size(); j < newSize; j++)
        {
            vNewVX(j) = vVXUp(j -  (mXProj.y_size()));
            vNewVY(j) = vVYUp(j -  (mXProj.y_size()));
        }


        for(int j = 0; j < vvCurrIdx.getSize(); j++)
        {
            vvNewIdx.pushBack(vvCurrIdx(j));
        }

        vvNewIdx.append(vvIdxUp);

        stateCovMatsOld.append(stateCovMatsNew);
        colHistsOld.append(colHistsNew);

        for(int j = 0; j < 4; j++)
        {
            for(int k = 0; k < mXProj.y_size(); k++)
            {
                mNewAllX(j,k) = mCurrAllX(j,k);
            }
        }

        for(int j = 0; j < 4; j++)
        {
            for(int k = mXProj.y_size(); k < newSize ; k++)
            {
                mNewAllX(j,k) = mAllXnewUp(j,k - (mXProj.y_size()));
            }
        }
        //printf("mNewAllX after KU (extend): \n");
        //mNewAllX.Show();

        // FIXME (smooth trajectories)
        //Matrix<double> pp;
        //mNewAllX.cutPart(0,1,0,mNewAllX.y_size()-1,pp);
        //Matrix<double> smoothed = AncillaryMethods::smoothTrajMatrix(pp, 12);
        //smoothed = AncillaryMethods::smoothTrajMatrix(smoothed, 12);
        //mNewAllX.insert(pp,0,0);


        // REDUCING TRAJECTORIE LENGTH (stef: "what is done here exactly?", FIXME? + both this and smooth traj. missing in pureTracker!!)

        int max_length = 20; //Globals::frameRate*0.5; //framerate too unstable (4-29), changing to fixed value 20

//        if(mNewAllX.y_size()>max_length)
        if(false)
        {
            Matrix<double> cutMat;
            mNewAllX.cutPart(0,mNewAllX.x_size()-1, mNewAllX.y_size()-max_length, mNewAllX.y_size()-1, cutMat);
            mNewAllX = cutMat;
//            vNewR.swap();
//            vNewR.resize(max_length);
//            vNewR.swap();
            vNewVX.resize_from_end(max_length);

//            vNewV.swap();
//            vNewV.resize(max_length);
//            vNewV.swap();
            vNewVY.resize_from_end(max_length);

            int first_frame = mNewAllX(2, 0);

            int counter = 0;

            for(int j=vvNewIdx.getSize()-1; j >=0; j--)
            {
                if(vvNewIdx(j).getFrame()>=first_frame){
                    counter += 1;
                    continue;
                }
                else{
                    break;
                }
            }

//            vvNewIdx.swap();
//            vvNewIdx.resize(counter);
//            vvNewIdx.swap();
            vvNewIdx.resize_from_end(counter);

        }


        compute_hypo_entries(mNewAllX, vNewVX, vNewVY, vvNewIdx, det, newHypo, normfct, t);
        newHypo.setParentID(n_HypoId);
        newHypo.setStateCovMats(stateCovMatsOld);
        newHypo.setColHists(colHistsOld);
        if (embVecOld.getSize()>0){
            newHypo.setEmb_vec(embVecOld);
        }
        ros::Time creationTimeOld;
        auxHypo->getCreationTime(creationTimeOld);
        newHypo.setCreationTime(creationTimeOld);

        int lastSelectedOld = auxHypo->getLastSelected();
        newHypo.setLastSelected(lastSelectedOld);

        if (newHypo.getCategory() != -1)
        {
            newHypos.pushBack(newHypo);
        }
        vvNewIdx.clearContent();
        vvIdxUp.clearContent();
    }

    HypoExtended.clearContent();

    for(int i = 0; i < newHypos.getSize(); i++)
    {
        HypoExtended.pushBack(newHypos(i));
    }
    //std::cout << "...EXTEND DONE." << std::endl;
}

void Tracker::make_new_hypos(int endFrame, int tmin, Detections& det, Vector< Hypo >& hypos,  double normfct, Vector<int>& extendUsedDet)
{
    //std::cout << "MAKE NEW..." << std::endl;
    Vector<double> xInit;
    Matrix<double> PInit;
    Vector<double> pos3d;
    Matrix<double> cov3d;

    PInit.set_size(4,4, 0.0);

    PInit(0,0) = Globals::initPX; // 0.5; //0.4
    PInit(1,1) = Globals::initPY; // 0.5; //1.2
    PInit(2,2) = Globals::initPVX; // 1.0; //0.2
    PInit(3,3) = Globals::initPVY; //1.0; //0.2


    Matrix<double> mAllXnewDown;
    Volume<double> volHMean;
    Vector<FrameInlier> vvIdxDown;
    Vector<double>  vvXDown;
    Vector<double>  vvYDown;

    Vector<Matrix<double> > stateCovMats;
    Vector<Volume<double> > colHists;
    Vector<Vector<double> > embVecs;
    Vector<double> currEmbVec;

    //double v = 0.5;//Globals::dMaxPedVel/3.0;
    //double r = M_PI; // Assume as Prior that the persons are front orientated.

    // if we can get velocity out of a detection, set here (otherwise best guess to assume no velocity)
    double vx_init = 0.0;
    double vy_init = 0.0;

    hypos.clearContent();

//    Vector<double> bbox;

    int nrOfDet = det.numberDetectionsAtFrame(endFrame);

    for(int j = 0; j < nrOfDet; j++)
    {
        if(extendUsedDet.findV(j) >= 0)
            continue;

//        det.getBBox(endFrame, j, bbox);
        det.getPos3D(endFrame, j, pos3d);
        det.getEmbVec(endFrame, j, currEmbVec);
        det.get3Dcovmatrix(endFrame, j, cov3d);
        xInit.setSize(4);
        xInit(0) = pos3d(0);
        xInit(1) = pos3d(1);
        xInit(2) = vx_init;
        xInit(3) = vy_init;
        PInit(0,0) = cov3d(0,0); // initP by detectionP
        PInit(1,1) = cov3d(1,1);
        PInit(0,1) = cov3d(0,1);
        PInit(1,0) = cov3d(1,0);

        EKalman kalman;
        kalman.init(xInit, PInit, Globals::dt);
        kalman.runKalmanDown(det, endFrame, j, tmin, mAllXnewDown, vvIdxDown, vvXDown, vvYDown, volHMean, stateCovMats, colHists, embVecs);
        //*********************************
        // swap data to make it consistent
        //*********************************
        vvXDown.swap();
        vvYDown.swap();
        vvIdxDown.sortV();
        mAllXnewDown.swap();
        AncillaryMethods::swapVectorMatrix(stateCovMats);
        AncillaryMethods::swapVectorVolume(colHists);

        //*********************************
        // now run kf back upwards in time to get more precise tracks
        //*********************************
        //printf("mAllXnewDown after KD: \n");
        //mAllXnewDown.Show();
        //printf("vRDown: \n");
        //vRDown.show();
        //printf("vVDown: \n");
        //vVDown.show();
        if (mAllXnewDown.total_size() != 0){
            xInit(0) = mAllXnewDown(0,0);
            xInit(1) = mAllXnewDown(1,0);
            xInit(2) = vvXDown(0);
            xInit(3) = vvYDown(0);
            Volume<double> colHistsInit = colHists(0);
            Vector<double> bboxInit(4,0.0);
            Vector<double> embVecInit = embVecs(embVecs.getSize()-1);

            EKalman kalmanBi;
            kalmanBi.init(xInit, stateCovMats(0), Globals::dt);
            //start where above KalmanDown has stopped (at mAllXnewDown(2,0)-1)
            kalmanBi.runKalmanUp(det, /*tmin-1*/mAllXnewDown(2,0), endFrame, mAllXnewDown, vvIdxDown, vvXDown, vvYDown, volHMean,
                                  stateCovMats, colHistsInit, embVecInit, colHists, xInit(1), bboxInit);
            //*********************************
            // swap data to make it consistent
            //*********************************
            //vRDown.swap();
            //vVDown.swap();
            //vvIdxDown.sortV();
            //mAllXnewDown.swap();

            // SMOOTHING
            //
            //        // FIXME
            //Matrix<double> pp;
            //mAllXnewDown.cutPart(0,1,0,mAllXnewDown.y_size()-1,pp);
            //Matrix<double> smoothed = AncillaryMethods::smoothTrajMatrix(pp, 12);
            //smoothed = AncillaryMethods::smoothTrajMatrix(smoothed, 12);
            //mAllXnewDown.insert(pp,0,0);

            //AncillaryMethods::swapVectorMatrix(stateCovMats);
            //AncillaryMethods::swapVectorVolume(colHists);
            //printf("mAllXnewDown after KU: \n");
            //mAllXnewDown.Show();
            //printf("swapped\n");
            Hypo hypo;
            hypo.setStateCovMats(stateCovMats);
            hypo.setColHists(colHists);
            if(embVecInit.getSize()>0){
                hypo.setEmb_vec(embVecInit);
            }

            compute_hypo_entries(mAllXnewDown, vvXDown, vvYDown, vvIdxDown, det, hypo, normfct, endFrame);
            hypo.setParentID(-1);

            if (hypo.getCategory() != -1)
            {
                hypo.setLastSelected(endFrame);
                hypos.pushBack(hypo);
            }
        }
        vvIdxDown.clearContent();
    }
    //std::cout << "...NEW DONE." << std::endl;
}

void Tracker::compute_hypo_entries(Matrix<double>& allX,  Vector<double>& vX, Vector<double>& vY, Vector <FrameInlier >& Idx, Detections& det, Hypo& hypo, double normfct, int frame)
{
    //std::cout << "CHE..." << std::endl;
    // ***********************************************************************
    //   Init
    // ***********************************************************************

    double maxvel = Globals::dMaxPedVel;
    double holePen = Globals::dHolePenalty;
    int maxHoleLen = Globals::maxHoleLen;
    double tau = Globals::dTau;
    int numberInlier = 0;
    if (Idx.getSize() == 0) {return;}

    for( int i = 0; i < Idx.getSize(); i++)
    {
        numberInlier =  numberInlier + Idx(i).getNumberInlier();
    }

    Vector<double>  Lmax(3);

    Lmax(0) = Globals::pedSizeWCom;
    Lmax(1) = Globals::pedSizeWCom;
    Lmax(2) = Globals::pedSizeHCom;

    int nFrames = 10; // number of Frames for further extropolation of Traj.

    // num inliers shouldn't be a KO for a hypo (if we allow for tracklength=1 and/or allow for further extrapolation)
    if ( true || numberInlier > 0 && Idx.getSize() > 1)
    {
        hypo.setVY(vY);
        hypo.setVX(vX);
        hypo.setXProj(allX);
        hypo.setHypoID(-1);

        // ***********************************************************************
        //   Rotation 4D
        // ***********************************************************************
        // Warnin: outdated (and not used?)
        /*Matrix<double> rot4D(3, vY.getSize(), 0.0);
        for(int i = 0; i < V.getSize(); i++)
        {
            rot4D(0,i) = cos(R(i));
            rot4D(1,i) = sin(R(i));
        }
        hypo.setRot4D(rot4D);*/

        // ***********************************************************************
        //   Weight, ScoreW, Height
        // ***********************************************************************
        double fct;
        double sumW = 0.0;
        double heightValue = 0.0;
        int currentFrame;
        Vector<int> inlier;
        Vector<double> weights;
        double sumFct = 0;

        for( int i = 0; i < Idx.getSize(); i++)
        {
            currentFrame = Idx(i).getFrame();

            inlier = Idx(i).getInlier();
            weights = Idx(i).getWeight();
            heightValue = Idx(i).getHeight();
            for( int j = 0; j < inlier.getSize(); j++)
            {
                fct = exp(-(frame-currentFrame)/tau);
                weights(j) =  weights(j)*fct;
                sumW += weights(j);
                //heightValue += Idx(j).getHeight(); //det.getHeight(currentFrame, inlier(j))*fct;
                sumFct += fct;

                // we don't need category right now (if we need it safe it in inlier detection, just like height, due to the limit of the Detection buffer)
                //hypo.setCategory(det.getCategory(currentFrame, inlier(j)));

            }

            Idx(i).clearWeights();
            Idx(i).setAllWeightsCoincident(weights);
        }

        hypo.setIdx(Idx);
        hypo.setScoreW(sumW);
        hypo.setHeight(heightValue);//hypo.setHeight(heightValue/sumFct);

        // ***********************************************************************
        // Holes // Assume that Idx(frames) are sorted "ascend"
        // ***********************************************************************

        int maxHole = 0;
        int aux;
        int totalHoles = 0;
        for( int i = 1; i < Idx.getSize(); i++)
        {
            //            cout <<" IDX " <<Idx(i).getFrame() << endl;
            aux = Idx(i).getFrame() - Idx(i-1).getFrame();
            totalHoles += aux - 1;
            if(aux > maxHole) maxHole = aux;
        }

        totalHoles += (frame - Idx(Idx.getSize()-1).getFrame());
        maxHole = max(maxHole, frame - Idx(Idx.getSize()-1).getFrame());

        double nw = numberInlier*normfct;
        nw = nw - (holePen*totalHoles*normfct);
        hypo.setNW(nw);


        // ***********************************************************************
        // Hypo Start and Hypo End. // Assume that Idx(frames) are sorted "ascend"
        // ***********************************************************************

        Vector<double>  point(3, 0.0);
        Vector<double>  point4D(4,0.0);
        Vector<double>  oldX;
        Vector<double>  pos3DStl;

        if(maxHole < maxHoleLen)
        {
            int nrFrWithInl = Idx.getSize();

            // average position of all inlier detections at start
            //inlier = Idx(0).getInlier();
            //for( int i = 0; i < inlier.getSize(); i++)
            //{
            //    det.getPos3D(Idx(0).getFrame(), inlier(i), pos3DStl);
            //    point(0) += pos3DStl(0);
            //    point(1) += pos3DStl(1);
            //    point(2) += pos3DStl(2);
            //}

            //Extend the point by the fourth coordinate - the frame number
            //point *=(1.0/double(inlier.getSize()));
            point = Idx(0).getPos3D();
            oldX = point;
            point4D(0) = point(0);
            point4D(1) = point(1);
            point4D(2) = point(2);
            point4D(3) = Idx(0).getFrame();

            hypo.setStart(point4D);
            point.fill(0.0);

            // average position of all inlier detections at end
            /*inlier = Idx(nrFrWithInl-1).getInlier();
            for( int i = 0; i < inlier.getSize(); i++)
            {
                det.getPos3D(Idx(nrFrWithInl-1).getFrame(), inlier(i), pos3DStl);
                point(0) += pos3DStl(0);
                point(1) += pos3DStl(1);
                point(2) += pos3DStl(2);
            }*/

            point = Idx(nrFrWithInl-1).getPos3D();
            //point *=(1.0/double(inlier.getSize()));

            //Extend the point by the fourth coordinate - the frame number
            point4D(0) = point(0);
            point4D(1) = point(1);
            point4D(2) = point(2);
            point4D(3) = Idx(nrFrWithInl-1).getFrame();

            hypo.setEnd(point4D);
//            hypo.setX4D(point4D);
//            hypo.setX(point);


            if(Idx(0).getFrame() <= Idx(nrFrWithInl-1).getFrame()) // Changed to <= 22.10.09
            {

                Vector<double>  first;
                Vector<double>  moved;
                Vector<double>  lengthL(2);
                Vector<double>  avgSpeed(2, 0.0);
                Vector<double>  main4D;
                Vector<double>  up4D(3, 0.0);
                up4D(2) = 1.0;
                Vector<double>  ort4D(3);
                Vector<double>  main3D(3, 0.0);

                // ***********************************************************************
                // First get last and first inlier position / Moved ?
                // ***********************************************************************

                int firstInlPos = -1;
                int lastInlPos = -1;

                int i = 0;

                while(Idx(0).getFrame() != allX(2,i)) i++;
                assert(i < allX.y_size());
                firstInlPos = i;

                i = allX.y_size()-1;

                while(Idx(nrFrWithInl-1).getFrame() != allX(2,i)) i--;
                assert(i >= 0);
                lastInlPos = i;

                Vector<double> assistant(4, 0.0);
                int c = 0;


                //std::cout << "allX: " << std::endl;
                //allX.Show();
                for(int j = 1; j < Globals::history+1; j++)
                {
                    if(allX.y_size()-(j+1)<0)
                        break;
                    allX.getRow((max(0, allX.y_size()-(j+1))), first);
                    allX.getRow((max(0, allX.y_size()-j)), moved);
                    assistant += moved - first;
                    //printf("assistant:\n");
                    //assistant.show();
                    //assistant *= Globals::frameRateVector(j-1);
                    //printf("assistant*=frameRateVector(.):\n");
                    //assistant.show();
                    //std::cout << "dtVector at " << (j-1) << std::endl;
                    avgSpeed(0) += assistant(0)*1.0/Globals::dtVector(j-1);
                    avgSpeed(1) += assistant(1)*1.0/Globals::dtVector(j-1);
                    c++;
                }

                if(c > 0)
                {
                    assistant *= (1.0)/double(c);
                    avgSpeed *= (1.0)/double(c);
                }

                assistant.resize(3);
                moved = assistant;

                //lengthL(0) = moved(0);
                //lengthL(1) = moved(1);
                lengthL(0) = avgSpeed(0);
                lengthL(1) = avgSpeed(1);

                // do not exclude not moving hypos (why would we?), especially to allow for tracks with length 1
                if (true || lengthL.norm() != 0)
                {

                    // recover the motion speed
                    //printf("moved_vec:\n");
                    //moved.show();
                    //printf("lengthL_vec:\n");
                    //lengthL.show();
                    //printf("set speed: (lengthL.norm()/moved(2)) = %.4f/%.4f  === %.4f\n",lengthL.norm(),moved(2),(lengthL.norm()/moved(2)));
                    hypo.setSpeed(min(lengthL.norm(),Globals::dMaxPedVel));
                    //printf("speed: %d\n",lengthL.norm()*Globals::frameRate);
                    hypo.setMoving(true);

                    main4D = moved;
                    main4D(2) = 0.0;
                    // if there is no main direction (especially for track with length 1), set to (arbitrary) 1,0,0-vector, otherwise normalize
                    if(main4D.norm() != 0){
                        main4D *=(1.0/main4D.norm());
                    }
                    else{
                        main4D(0) = 1.0;
                    }

                    //printf("main4d\n");
                    //main4D.show();

                    // also compute rectangles etc. if main dir is 0 (especially to handle tracks with length 1)
                    if(true || main4D.norm() != 0)
                    {
                        // ***********************************************************************
                        // Prepare the spacetime object trajectory projected into 3D
                        // ***********************************************************************

                        ort4D = cross(main4D, up4D); // cross product
                        main3D(0) = main4D(0);
                        main3D(1) = main4D(1);

                        //printf("main3d (=hypoDir):\n");
                        //main3D.show();

                        hypo.setDir(main3D);
//                        hypo.setOri4D(main4D);
//                        hypo.setSize(Lmax);

                        Vector<double>  xFirst(3);
                        Vector<double>  xLast(3);
                        Matrix<double> startRect(3, 4);
                        Matrix<double> endRect(3, 4);
//                        Matrix<double> points(3,4, 0.0);
                        allX.getRow(firstInlPos, xFirst);
                        allX.getRow(lastInlPos, xLast);

                        xFirst.resize(3);
                        xLast.resize(3);


                        AncillaryMethods::compute_rectangle(main4D, ort4D, Lmax, xFirst, startRect);
                        hypo.setStartRect(startRect);
                        //printf("startRect:\n");
                        //startRect.Show();

                        AncillaryMethods::compute_rectangle(main4D, ort4D, Lmax, xLast, endRect);
                        hypo.setEndRect(endRect);
                        //printf("endRect:\n");
                        //endRect.Show();


//                        points(0,0) = endRect(0,0);
//                        points(2,0) = endRect(1,0);
//                        points(0,1) = endRect(0,1);
//                        points(2,1) = endRect(1,1);
//                        points(0,2) = endRect(0,2);
//                        points(2,2) = endRect(1,2);
//                        points(0,3) = endRect(0,3);
//                        points(2,3) = endRect(1,3);

//                        hypo.setPoints(points);

                        // ***********************************************************************
                        // Prepare BBOX for each trajectory point.
                        // ***********************************************************************

                        //******** Necessary for linear interpol. of missing trajrect. **********
                        Matrix<double> mOldRect(3,4);
                        Matrix<double> mD;
                        Matrix<double> mRj;
                        Vector<double> vXj;
                        Vector<double> copyX3D;
                        // ***********************************************************************

                        int startFrame = Idx(0).getFrame();
                        int endFrame = Idx(nrFrWithInl-1).getFrame();

                        int nMissing = 0;
                        int LoopCounter = 0;
                        int countAllMissing = 0;

                        int nrFrBetweenStartEnd = endFrame - startFrame + 1; // include the start as well the end frame

                        Vector<double> x3D(3);
                        Vector<Matrix <double> > TrajRect;
                        Vector<Vector < double > > TrajPts;
                        Vector <int> TrajT;
                        Matrix<double> rect(3,4);

                        mOldRect = startRect;
                        int cp_nrFrBetweenStartEnd = nrFrBetweenStartEnd;

                        for(int i = 0; i < cp_nrFrBetweenStartEnd; i++)
                        {
                            //printf("allX:\n");
                            //allX.Show();
                            allX.getRow(firstInlPos + i, point);

                            if (point(2) == startFrame + i + LoopCounter)
                            {

                                x3D(0) = point(0);
                                x3D(1) = point(3);
                                x3D(2) = point(1);

                                point.resize(3);

                                AncillaryMethods::compute_rectangle(main4D, ort4D, Lmax, point, rect);
                                //printf("rect inbetween:\n");
                                //rect.Show();

                                //---------------------------------------------------------
                                // Fill in missing trajectory bboxes by linear interpolation
                                //---------------------------------------------------------

                                for(int j = 0; j < nMissing; j++)
                                {

                                    mD = rect-mOldRect;

                                    mD *= (1.0/(nMissing + 1));
                                    mD *=(j+1);
                                    mRj = mOldRect+mD;

                                    copyX3D = x3D - oldX;
                                    copyX3D *=(j+1);
                                    copyX3D *= (1.0/(nMissing + 1));
                                    vXj = oldX + copyX3D;

                                    TrajRect.pushBack(mRj);
                                    TrajPts.pushBack(vXj);
                                    TrajT.pushBack(startFrame + i + countAllMissing);

                                    countAllMissing +=1;
                                }

                                TrajRect.pushBack(rect);

                                TrajPts.pushBack(x3D);
                                TrajT.pushBack(startFrame + i + countAllMissing);

                                oldX = x3D;
                                mOldRect = rect;
                                nMissing = 0;
                            }
                            else
                            {
                                LoopCounter += 1;
                                nMissing += 1;
                                i = i - 1;
                                cp_nrFrBetweenStartEnd = cp_nrFrBetweenStartEnd - 1;
                            }
                        }

                        //************************************************************************
                        // Predict up to nFrames extra frames
                        //************************************************************************

                        int posT = 0;
                        int len = min(frame-max(maxHoleLen, 10), endFrame - 1);

                        if(len >= 0 && len >= startFrame)
                        {
                            if(endFrame - len >= 0)
                            {
                                posT = len - startFrame;
                            }
                            else
                            {
                                posT = nrFrBetweenStartEnd - 1;
                            }
                        }

                        // Init for the prediction procedure

                        Vector<double> x1;
                        Vector<double> x2;

                        Matrix<double> rect1;
                        Matrix<double> rect2;
                        int tlen;
                        Vector<double> v;
                        Matrix<double> vrect;
                        double vnorm;

                        // I have to do this distinction since at the begining the
                        // trajectories contain only 2 Points.

                        x1 = TrajPts(posT);
                        x2 = TrajPts(TrajPts.getSize() - 1);
                        rect1 = TrajRect(posT);
                        rect2 = TrajRect(TrajRect.getSize() - 1);
                        tlen = TrajPts.getSize() - 1 - posT;
                        v = x2;
                        v -= x1;
                        v *=(1.0/tlen);
                        vrect = rect2;
                        vrect -= rect1;
                        vrect *=(1.0 / tlen);

                        vnorm = v.norm();

                        if (vnorm > maxvel)
                        {
                            v *= (maxvel/vnorm);
                            vrect *=(maxvel/vnorm);
                            //cout << "WARNING(compute_hypo): Extrapolation exceeds maxvel" << endl;
                        }

                        for(int ff = 0; ff < vrect.y_size(); ff++)
                        {
                            vrect(2, ff) = 1.0;
                        }
                        //                        vrect.fillColumn(1.0, 2);

                        Matrix<double> copyVrect;
                        Vector<double> copyV;

                        // Change ... 22.02.2010 delete +1 by i <
                        for (int i = endFrame; i < min(endFrame + nFrames, frame); i++)
                        {
                            mRj = rect2;
                            copyVrect = vrect;

                            copyVrect *=(i - endFrame);
                            mRj += (copyVrect);

                            copyV = v;
                            copyV *= (i - endFrame);

                            vXj = x2;
                            vXj +=copyV;

                            TrajRect.pushBack(mRj);
                            TrajPts.pushBack(vXj);
                            TrajT.pushBack(startFrame + nrFrBetweenStartEnd + (i - endFrame));

                        }


                        hypo.setTrajRect(TrajRect);
                        hypo.setTrajPts(TrajPts);
                        hypo.setTrajT(TrajT);

                        // ***********************************************************************
                        // Precompute a spacetime bbox for the hypothesis.
                        // ***********************************************************************

                        hypo.getStartRect(startRect);
                        hypo.getEndRect(endRect);

                        Vector<double> start0;
                        Vector<double> start1;
                        Vector<double> end0;
                        Vector<double> end1;

                        start0 = startRect.getColumn(0);
                        start1 = startRect.getColumn(1);
                        end0 = endRect.getColumn(0);
                        end1 = endRect.getColumn(1);

                        double xmin = min(start0.minim().first, end0.minim().first);
                        double xmax = max(start0.maxim().first, end0.maxim().first);
                        double ymin = min(start1.minim().first, end1.minim().first);
                        double ymax = max(start1.maxim().first, end1.maxim().first);

                        hypo.getStart(start0);
                        hypo.getEnd(end0);
                        Matrix<double> bbox(3,2);
                        bbox(0,0) = xmin;
                        bbox(1,0) = ymin;
                        bbox(2,0) = start0(3);
                        bbox(0,1) = xmax;
                        bbox(1,1) = ymax;
                        bbox(2,1) = end0(3);

                        hypo.setBBox4D(bbox);

                    }
                    else
                    {
                        hypo.setCategory(-1);
                        //std::cout << "no main dir" << std::endl;
                        ROS_DEBUG("hypo has no main direction => reject!");
                    }
                }
                else
                {
                    hypo.setSpeed(0);
                    hypo.setCategory(-1);
                    //std::cout << "not moving" << std::endl;
                    ROS_DEBUG("Hypo %i is not moving => reject!", hypo.getHypoID());
                }
            }
            else
            {
                //std::cout << "only single frame" << std::endl;
                hypo.setCategory(-1);
                ROS_DEBUG(" Hypo contains only single frame => reject! ");
            }
        }
        else
        {
            //std::cout << "large holes" << std::endl;
            hypo.setCategory(-1);
            ROS_DEBUG("Hypo %i had large holes : MaxHoleLength - %i => reject", hypo.getHypoID(), maxHole);
        }
    }else
    {
        //std::cout << "only one inlier" << std::endl;
        hypo.setCategory(-1);
        ROS_DEBUG("Size of Idx is 1, so no hypo can be computed");
    }
    //std::cout << "...CHE end!" << std::endl;
}

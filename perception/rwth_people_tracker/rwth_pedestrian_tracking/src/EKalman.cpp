/*
 * Copyright 2012 Dennis Mitzel
 *
 * Authors: Dennis Mitzel
 * Computer Vision Group RWTH Aachen.
 */

#include "EKalman.h"
#include "AncillaryMethods.h"

string kalmanResTemp = "icp_%04d_%05d.txt";

void initrand()
{
    srand((unsigned)(time(0)));
}

double randdouble()
{
    return rand()/(double(RAND_MAX)+1);
}

EKalman::EKalman()
{
    m_R.set_size(2, 2, 0.0);
    //m_dt = 1.0 / Globals::frameRate;
    m_dt = Globals::dt;
    m_height = 1.7;
}

Vector<double> EKalman::non_lin_state_equation(Vector<double> x, double dt)
{
    int size = x.getSize();
    Vector<double> res(size, 0.0);

    // Constant velocity model for pedestrians
    res(0) = x(0) + x(3)*cos(x(2))*dt;
    res(1) = x(1) + x(3)*sin(x(2))*dt;
    res(2) = x(2);
    res(3) = x(3);

    return res;
}

Matrix<double> EKalman::makeQ(Vector<double> /*x*/, double dt)
{
    Matrix<double> Q;

    Q.set_size(4,4, 0.0);
    /*Q(0,0) = Globals::sysUncX*Globals::sysUncX;
    Q(1,1) = Globals::sysUncY*Globals::sysUncY;
    Q(2,2) = Globals::sysUncRot*Globals::sysUncRot;
    Q(3,3) = Globals::sysUncVel*Globals::sysUncVel;*/
    Q(0,0) = (dt*dt*dt)/3;
    Q(1,1) = (dt*dt*dt)/3;
    Q(2,2) = dt;
    Q(3,3) = dt;
    Q(0,2) = (dt*dt)/2;
    Q(2,0) = (dt*dt)/2;
    Q(1,3) = (dt*dt)/2;
    Q(3,1) = (dt*dt)/2;

    double q_l = 0.035;//0.267;
    Q *= q_l;

    return Q;
}

Matrix<double> EKalman::makeF(Vector<double> x, double dt)
{
    Matrix<double> F;

    F.set_size(4,4, 0.0);

    /*F(0,0) = 1.0;
    F(1,1) = 1.0;
    F(2,0) = -sin(x(2))*x(3)*dt;
    F(3,0) = cos(x(2))*dt;
    F(2,1) = cos(x(2))*x(3)*dt;
    F(3,1) = sin(x(2))*dt;
    F(2,2) = 1.0;
    F(3,3) = 1.0;*/

    F(0,0) = 1.0;
    F(1,1) = 1.0;
    F(2,0) = dt;
    F(3,1) = dt;
    F(2,2) = 1.0;
    F(3,3) = 1.0;

    return F;
}

Matrix<double> EKalman::makeW()
{
    // Jac for predict step
    Matrix<double> W = Eye<double>(4);
    return W;
}

Matrix<double> EKalman::makeH()
{
    // state-to-observation space mapping
    Matrix<double> H;

    H.set_size(4,2,0.0);
    H(0,0) = 1.0;
    H(1,1) = 1.0;
    //H(2,2) = 1.0;
    //H(3,3) = 1.0;

    return H;
}

Matrix<double> EKalman::makeR(){
    return m_R;
}


void EKalman::saveData(int i)
{
    Vector<double> pos;
    pos.pushBack(m_xpost(0));
    pos.pushBack(m_xpost(1));
    pos.pushBack(i);
    pos.pushBack(m_yPos(1));

    m_allXnew.pushBack(pos);
    m_vX.pushBack(min(Globals::dMaxPedVel, m_xpost(2)));
    m_vY.pushBack(min(Globals::dMaxPedVel, m_xpost(3)));
    m_CovMats.pushBack(m_Ppost);
    m_colHists.pushBack(m_colHist);
}


bool checkPointInsideElCylinder(double x, double y, double z, double centerX, double centerZ, double height, double ax, double bz)
{
    if(((pow(x-centerX, 2.0)/pow(ax, 2.0)) + (pow(z - centerZ, 2.0)/pow(bz, 2.0))) <= 1)
    {
        if(y > -1.0 && y < height-1.1)
        {
            return true;
        }
    }

    return false;
}

bool EKalman::findObservation(Detections& det, int frame)
{

    Vector<double> succPoint;
    if(m_Up)
        frame = frame + 1;
    else
        frame = frame - 1;

    Volume<double> obsCol;
    FrameInlier inlier(frame);

    Vector<int> inl;
    Vector<double> weights;
    double colScore = 1.0;
    double weight;

    Vector<int> allInlierInOneFrame;
    Vector<double> weightOfAllInliersInOneFrame;

//    Matrix<double> covCopy;

    for(int i = 0; i < det.numberDetectionsAtFrame(frame); i++)
    {
        Matrix<double> devObs;
        Vector<double> currBbox;
        det.getPos3D(frame, i, succPoint);
        det.getColorHist(frame, i, obsCol);
        det.get3Dcovmatrix(frame, i, devObs);
        det.getBBox(frame, i, currBbox);

        // TODO: app/reid score here? (for now no color hist computation)
        //colScore = Math::hist_bhatta(obsCol, m_colHist);

        Matrix<double> covariance(2,2, 0.0);

        //FIXED: not take sqrt again
        //covariance(0,0) = sqrt(devObs(0,0));
        //covariance(1,1) = sqrt(devObs(2,2));
        covariance(0,0) = devObs(0,0);
        covariance(1,1) = devObs(1,1);

        //double covariance_det = covariance(0,0)*covariance(1,1)-covariance(1,0)*covariance(0,1);

        covariance.inv();
//        covCopy = covariance;
        Vector<double> p1(2,0.0);
        Vector<double> p2(2,0.0);

        p1(0) = m_xprio(0);
        p1(1) = m_xprio(1);

        p2(0) = succPoint(0);
        p2(1) = succPoint(1);

        Vector<double> pDiff(2,0.0);
        pDiff = p1;
        pDiff -= p2;

        covariance *= pDiff;
        covariance.Transpose();
        covariance *= pDiff;
        
        //39.47841 magic constant: (2*pi)^2
        //double denom = std::sqrt(39.47841 * std::abs(covariance_det));

        //weight = (1.0/denom) * exp(-0.5*covariance(0,0));
        weight = exp(-0.5*covariance(0,0));


        // IMAGE BASED
        // !! deprecated: only 3d tracking now! bbox is not properly set (0.0), do not use!
        //Vector<double> rectInter;
        //AncillaryMethods::IntersetRect(m_bbox, currBbox, rectInter);
        //double iou = rectInter(2)*rectInter(3)/(m_bbox(2)*m_bbox(3)+currBbox(2)*currBbox(3)-rectInter(2)*rectInter(3));

        //if(iou>0.2 /*&& colScore > Globals::kalmanObsColorModelthresh*/)
        //{
        //    allInlierInOneFrame.pushBack(i);
        //    weightOfAllInliersInOneFrame.pushBack(iou*colScore);
        //}

        //3D POSITION BASED
        //if(pDiff.norm() < Globals::kalmanObsMotionModelthresh /*&& colScore > Globals::kalmanObsColorModelthresh*/)
        //std::cout << "pDiff.norm(): " << pDiff.norm() << std::endl;
        //std::cout << "weight: " << weight << std::endl;
        //std::cout << "colScore: " << colScore << std::endl;
        if(weight > Globals::kalmanObsMotionModelthresh /*&& colScore > Globals::kalmanObsColorModelthresh*/)
        {
            allInlierInOneFrame.pushBack(i);
            //weightOfAllInliersInOneFrame.pushBack(weight*colScore);
            weightOfAllInliersInOneFrame.pushBack(weight);
        }
    }

    // find the inlier with the maximum weight
    if(allInlierInOneFrame.getSize()>0)
    {
        pair<double, int> maxPosValue = weightOfAllInliersInOneFrame.maxim();
        int pos = maxPosValue.second;

        inlier.addInlier(allInlierInOneFrame(pos));
        inlier.addWeight(weightOfAllInliersInOneFrame(pos));
    }

    m_measurement_found = false;
    if(inlier.getNumberInlier() > 0)
    {

        m_measurement_found = true;
        Matrix<double> covMatrix;
        inl = inlier.getInlier();
        weights = inlier.getWeight();

        // Update the color histogram
        Volume<double> newColHist;

        det.getColorHist(frame, inl(0), newColHist);

        m_colHist *= 0.4;
        newColHist *= 0.6;
        m_colHist += newColHist;
        m_height = 0.0;

        det.get3Dcovmatrix(frame, inl(0), covMatrix);

        Vector<double> inlierPos;
        m_yPos.setSize(3,0.0);
        for(int i = 0; i < allInlierInOneFrame.getSize(); i++)
        {
            det.getPos3D(frame, allInlierInOneFrame(i), inlierPos);
            m_height += det.getHeight(frame, allInlierInOneFrame(i));
            m_yPos += inlierPos;
        }

        m_height *= 1.0/(double) allInlierInOneFrame.getSize();
        m_yPos *= 1.0/(double) allInlierInOneFrame.getSize();

        FrameInlier newInlier(frame);
        newInlier.setHeight(m_height);
        newInlier.setPos3D(m_yPos);
        newInlier.addInlier(inl(0));
        newInlier.addWeight(weights(0)*det.getScore(frame, inl(0)));
        m_Idx.pushBack(newInlier);

        m_R.set_size(2,2, 0.0);
        m_R(0,0) = covMatrix(0,0);
        m_R(1,1) = covMatrix(1,1);
    }

    return m_measurement_found;
}

Vector<double> EKalman::makeMeasurement()
{
    Vector<double> measurement;

    double xo = m_yPos(0);
    double yo = m_yPos(1);
    //std::cout << "m_yPos:" << std::endl;
    //m_yPos.show();
    //double xp = m_xprio(0);
    //double yp = m_xprio(1);

    //double dirp = m_xprio(2);
    //double velp = m_xprio(3);

    //double diffx = (xo - (xp - m_dt*velp*cos(dirp)));
    //double diffy = (yo - (yp - m_dt*velp*sin(dirp)));

    measurement.setSize(2, 0.0);
    measurement(0) = xo;
    measurement(1) = yo;
    //measurement(2) = atan2(diffy, diffx);
    //measurement(3) = min(double(Globals::dMaxPedVel),  sqrt(diffx*diffx + diffy*diffy)*m_dt);
//    measurement(3) = Globals::minvel;

    return measurement;
}

void EKalman::runKalmanDown(Detections& det, int frame, int pointPos, int t, Matrix<double>& allXnew, Vector<FrameInlier>& Idx,
                            Vector<double>& vX, Vector<double>& vY, Volume<double>& hMean, Vector<Matrix<double> >& stateCovMats,
                            Vector<Volume<double> >& colHists)
{
    //std::cout << "Kalman down..." << std::endl;
    m_Up = false;

    det.getColorHist(frame, pointPos, m_colHist);
    det.getBBox(frame, pointPos, m_bbox);

    Matrix<double> copyInitStateUnc = m_Ppost;
    Vector<double> copyInitState = m_xpost;

    //////////////////////////////////////////////////////////////////////

    Matrix<double> covMatrix;
    det.get3Dcovmatrix(frame, pointPos, covMatrix);
    Vector<double> startingDet;
    det.getPos3D(frame, pointPos, startingDet);

    m_R.fill(0.0);
    m_R(0,0) = covMatrix(0,0);
    m_R(1,1) = covMatrix(1,1);
    //m_R.Show();

    //m_R(2,2) = 0.2*0.2;
    //m_R(3,3) = 0.2*0.2;

    int tLastSupport = 0;

    m_yPos.pushBack(m_xpost(0));
    m_yPos.pushBack(startingDet(1));
    m_yPos.pushBack(m_xpost(1));

    // be sure to save the first frame (+corresponding inliers) to allow for tracks with length 1
    saveData(frame);
    m_measurement_found = findObservation(det, frame+1);

    for(int i = frame; i > t; i--)
    {

        //take framerate from past
        //printf("i: %d, frame: %d, t: %d\n", i,frame,t);
        //printf("1: take element frame+1-i (%d)\n", frame+1-i);
        //printf("currentFrameRateVector:\n");
        //Globals::frameRateVector.show();
        //m_dt = 1.0 / Globals::frameRateVector(frame+1-i);
        //std::cout << "access dt vector at " << frame-i << std::endl;
        m_dt = Globals::dtVector(frame-i);


        predict();
        m_measurement_found = findObservation(det, i);

        if(m_measurement_found)
        {

            m_measurement = makeMeasurement();
            update();

            if(i == frame)
            {
                m_Ppost = copyInitStateUnc;
                m_xpost = copyInitState;
            }

            saveData(i-1);
            tLastSupport = 0;
            pointPos = m_Idx(m_Idx.getSize()-1).getInlier()(0);
        }
        else
        {
            tLastSupport += 1;
            // should check for last support here to allow for "hole-less" backwards search (accepted_frames_without_det=0)
            if(tLastSupport > Globals::accepted_frames_without_det)
                break;
            update();
            saveData(i-1);
        }
    }

    if(m_vX.getSize() > 1) {
        // As this is kalman-DOWN, we start with an unassociated detection (InitState)
        // The position is fine, but we fix the velocities by copying the one from the successor state
        m_vX(0) = m_vX(1);
        m_vY(0) = m_vY(1);
    }

    //Vel = m_vY;
    // turn rotations 180 degree/+pi (so "flip") in order to adjust the
    // rotations from the "backwards in time parse" to "forward in time"
    //turnRotations(R, m_vX);
    turnVelocities(vX, m_vX);
    turnVelocities(vY, m_vY);
    Idx = m_Idx;

    allXnew = Matrix<double>(m_allXnew);

    hMean = m_colHist;
    colHists = m_colHists;
    stateCovMats = m_CovMats;
    //ROS_INFO("...KD done.\n");
}

void EKalman::turnRotations(Vector<double> &dest,Vector<double> &src)
{
    int srclen = src.getSize();
    dest.setSize(srclen);

    for(int i = 0; i < srclen; i++) {
        double newrot = src(i) + M_PI;
        if(newrot > M_PI) newrot -= 2*M_PI;
        dest(i) = newrot;
    }
}

void EKalman::turnVelocities(Vector<double> &dest,Vector<double> &src)
{
    int srclen = src.getSize();
    dest.setSize(srclen);

    // turning velocities by simply negating them
    for(int i = 0; i < srclen; i++) {
        dest(i) = -src(i);
    }
}

void EKalman::runKalmanUp(Detections& det, int frame, int t, Matrix<double>& allXnew, Vector<FrameInlier>& Idx,
                          Vector<double>& vX, Vector<double>& vY, Volume<double>& hMean,  Vector<Matrix<double> >& stateCovMats, Volume<double>& colHistInit,
                          Vector<Volume<double> >& colHists, double yPosOfStartingPoint, Vector<double>& bbox)
{
    //ROS_INFO("Kalman up...\n");
    //std::cout << "Kalman up..." << std::endl;
    m_bbox = bbox;
    m_Up = true;
    int tLastSupport = 0;
    m_colHist = colHistInit;

    m_yPos.pushBack(m_xpost(0));
    m_yPos.pushBack(yPosOfStartingPoint);
    m_yPos.pushBack(m_xpost(1));

    // if allXnew from previous KalmanDown are given (not empty) be sure to save the first frame (+corresponding inliers) to allow for tracks with length 1
    if(allXnew.total_size()>0){
        saveData(frame);
        m_measurement_found = findObservation(det, frame-1);
    }

    for(int i = frame; i < t; i++)
    {
        if(tLastSupport > Globals::maxHoleLen) break;

        // m_dt = 1.0 / Globals::frameRateVector(t-1-i);
        m_dt = Globals::dtVector(t-1-i);

        predict();

        m_measurement_found = findObservation(det, i);

        if(m_measurement_found)
        {
            m_measurement = makeMeasurement();
            update();
            saveData(i+1);
            tLastSupport = 0;
        }
        else
        {
            update();
            saveData(i+1);
            tLastSupport += 1;
        }
    }

    vX = m_vX;
    vY = m_vY;

    Idx = m_Idx;

    allXnew = Matrix<double>(m_allXnew);

    hMean = m_colHist;
    colHists = m_colHists;
    stateCovMats = m_CovMats;
    //ROS_INFO("...KU done.\n");
}

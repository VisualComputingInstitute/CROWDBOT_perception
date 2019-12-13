/*
 * Copyright 2012 Dennis Mitzel
 *
 * Authors: Dennis Mitzel
 * Computer Vision Group RWTH Aachen.
 */

#include "Detections.h"
#include "AncillaryMethods.h"

using namespace std;

// ++++++++++++++++++++++++ constructor ++++++++++++++++++++++++++++++++++++++
Detections::Detections(int x, const int flag)
{
//    sizeOfDetVec = x;
    if (x == 22) {
        offSet = 0;
    } else {
        offSet = 1;
    }

    img_num = 0;
    hypo_num = 1;
    center_x = 2;
    center_y = 3;
    scale = 4;
    categ = 5;
    bbox = 5 + offSet; //consists of 4 num
    initscore = 9 + offSet;
    score = 10 + offSet;
    dist = 11 + offSet;
    height = 12 + offSet;
    rot = 19 + offSet; //consists of 3 num
    warp_loss = 13 + offSet; // FIXME this code need so much refactoring

    if (flag == 1) {
        pos = 13 + offSet; //consists of 3 num
    } else {
        pos = 16 + offSet; //consists of 3 num
    }

    buff_size = Globals::history+1;

    colHists.setSize(buff_size);
    cov3d.setSize(buff_size);
    detC.setSize(buff_size);
    embed_vecs.setSize(buff_size);
}



// ++++++++++++++++++++++++ Implementation ++++++++++++++++++++++++++++++++++++++

int Detections::globalFrameToBufferFrame(int global_frame)
{
    if(Globals::currentFrame > buff_size-1){
        return max(0, ((buff_size - 1) - (Globals::currentFrame - global_frame)));
    }
    else{
        return global_frame;
    }
}


int Detections::numberDetectionsAtFrame(int frame)
{
    frame = globalFrameToBufferFrame(frame);
    if(frame >= detC.getSize())
    {
        return 0;
    }
    else
    {
        return detC(frame).getSize();
    }

}

double Detections::getHeight(int frame, int detec) {
    frame = globalFrameToBufferFrame(frame);
    assert(detC.getSize() > frame);
    assert(detC(frame).getSize() > detec);

    return detC(frame)(detec)(height);
}

double Detections::getWarp(int frame, int detec) {
    frame = globalFrameToBufferFrame(frame);
    assert(detC.getSize() > frame);
    assert(detC(frame).getSize() > detec);

    return detC(frame)(detec)(warp_loss);
}

double Detections::getScore( int frame,  int detec) {
    frame = globalFrameToBufferFrame(frame);
    assert(detC.getSize() > frame);
    assert(detC(frame).getSize() > detec);

    return detC(frame)(detec)(score);
}

void Detections::getPos3D( int frame,  int detec, Vector<double>& v_pos)
{
    frame = globalFrameToBufferFrame(frame);
    v_pos.setSize(3);
    assert(detC.getSize() > frame);
    assert(detC(frame).getSize() > detec);

    v_pos(0) = (detC(frame)(detec)(pos));
    v_pos(1) = (detC(frame)(detec)(pos+1));
    v_pos(2) = (detC(frame)(detec)(pos+2));
}

void Detections::getBBox( int frame,  int detec, Vector<double>& v_bbox)
{
    frame = globalFrameToBufferFrame(frame);
    v_bbox.clearContent();
    assert(detC.getSize() > frame);
    assert(detC(frame).getSize() > detec);

    v_bbox.pushBack(detC(frame)(detec)(bbox));
    v_bbox.pushBack(detC(frame)(detec)(bbox+1));
    v_bbox.pushBack(detC(frame)(detec)(bbox+2));
    v_bbox.pushBack(detC(frame)(detec)(bbox+3));
}

Detections::~Detections()
{
    detC.clearContent();
    colHists.clearContent();
    cov3d.clearContent();
    embed_vecs.clearContent();
}

int Detections::prepareDet(Vector<double> &detContent,
                           const frame_msgs::DetectedPersons::ConstPtr & det,
                           int i,
                           int frame,
                           bool leftDet,
                           Matrix<double>& covariance)
{

    detContent.setSize(24, 0.0);

    frame_msgs::DetectedPerson currentDetection = det->detections[i];

    detContent(score) = currentDetection.confidence;
    detContent(scale) = 1;
    detContent(img_num) = frame;
    detContent(hypo_num) = i;
    detContent(bbox) = currentDetection.bbox_x;
    detContent(bbox + 1) = currentDetection.bbox_y;
    detContent(bbox + 2) = currentDetection.bbox_w;
    detContent(bbox + 3) = currentDetection.bbox_h;
    if(leftDet)
    {
        detContent(22) = 0;
    }else
    {
        detContent(22) = 1;
    }

    detContent(height) = currentDetection.height;
    detContent(warp_loss) = currentDetection.warp_loss;
    detContent(pos) = currentDetection.pose.pose.position.x;
    detContent(pos+1) = currentDetection.pose.pose.position.y;
    detContent(pos+2) = currentDetection.pose.pose.position.z;

    if(detContent(0) < 0) return 0;

    return 1;
}

Vector<double> Detections::projectPlaneToCam(Vector<double> p, Camera cam)
{
    Vector<double> gpInCam(4, 0.0);

    Vector<double> pv;
    pv.pushBack(p(0));
    pv.pushBack(p(1));
    pv.pushBack(p(2));

    Vector<double> camPos = cam.get_t();

    Matrix<double> camRot = cam.get_R();

    pv = Transpose(camRot)*pv;
    camRot *= -1.0;
    Vector<double> t = Transpose(camRot)*camPos;

    double d = p(3) - (pv(0)*t(0) + pv(1)*t(1) + pv(2)*t(2));

    gpInCam(0) = pv(0);
    gpInCam(1) = pv(1);
    gpInCam(2) = pv(2);
    gpInCam(3) = d;

    return gpInCam;
}

Vector<double> Detections::fromCamera2World(Vector<double> posInCamera, Camera cam)
{
    Matrix<double> rotMat = cam.get_R();

    Vector<double> posCam = cam.get_t();

    Matrix<double> trMat(4,4,0.0);
    trMat(3,3) = 1;
    trMat(0,0) = rotMat(0,0);
    trMat(0,1) = rotMat(0,1);
    trMat(0,2) = rotMat(0,2);
    trMat(1,0) = rotMat(1,0);
    trMat(1,1) = rotMat(1,1);
    trMat(1,2) = rotMat(1,2);
    trMat(2,0) = rotMat(2,0);
    trMat(2,1) = rotMat(2,1);
    trMat(2,2) = rotMat(2,2);

    posCam *= Globals::WORLD_SCALE;

    trMat(3,0) = posCam(0);
    trMat(3,1) = posCam(1);
    trMat(3,2) = posCam(2);
Vector<double> transpoint = trMat*posInCamera;
    return transpoint;

}

double Detections::get_mediandepth_inradius(Vector<double>& bbox,
                                            int radius,
                                            Matrix<double>& depthMap,
                                            double var,
                                            double pOnGp)
{

    int middleX = max(radius,
                min((int)floor((bbox(0) + bbox(2)/2.0)),
                Globals::dImWidth-(radius + 1)));

    int middleY = max(min((int)floor((bbox(1) + bbox(3)/2.0)),
                Globals::dImHeight-(radius + 1)),
                radius);

    Vector<double> depthInRadius(2*radius*2*radius);
    int co = 0;
    for(int i = middleX - radius; i < middleX + radius; i++)
    {
        for(int j = middleY - radius; j < middleY + radius; j++)
        {
            depthInRadius(co) = (depthMap(i,j));
            co++;
        }
    }

    depthInRadius.sortV();
    double medianDepth = depthInRadius(floor(depthInRadius.getSize()/2.0));
    //     double meanDepth = depthInRadius(0);

    if(fabs(pOnGp-medianDepth) > var)
    {
        return -1;
    }
    else
    {
        return medianDepth;
    }
}

#ifdef cim_v
void Detections::addDetsOneFrame(const frame_msgs::DetectedPersons::ConstPtr & det, int frame)
{
    // FIXME find a different approach for determing a 3D cov.

    Matrix<double> covariance(3,3,0.0);

    Volume<double> colhist;
    Vector<double> pos3d(3);
    Vector<double> v_bbox(4);
    Vector<double> emb_vec;


    Vector<double>detContent;

    int buffer_idx = min(frame, buff_size-1);

    if (frame > buff_size-1){
        detC.swap();
        colHists.swap();
        cov3d.swap();
        embed_vecs.swap();
        detC.resize(buff_size-1);
        colHists.resize(buff_size-1);
        cov3d.resize(buff_size-1);
        embed_vecs.resize(buff_size-1);
        detC.swap();
        colHists.swap();
        cov3d.swap();
        embed_vecs.swap();
        detC.resize(buff_size);
        colHists.resize(buff_size);
        cov3d.resize(buff_size);
        embed_vecs.resize(buff_size);
    }

    for ( int i = 0; i < det->detections.size(); i++)
    {
        if(prepareDet(detContent, det, i, frame, true, /*cam,*/ covariance))
        {
            pos3d(0) = (detContent(pos));
            pos3d(1) = (detContent(pos+1));
            pos3d(2) = (detContent(pos+2));
            //FIXME: why pos3d here?

            v_bbox(0) = (detContent(bbox));
            v_bbox(1) = (detContent(bbox+1));
            v_bbox(2) = (detContent(bbox+2));
            v_bbox(3) = (detContent(bbox+3));

            //TODO: delte color hists completely here and save elsewhere (e.g. in detection message... also not color hist but reid-emb?)

            // set embedding vector
            emb_vec.clearContent();
            for(int evi=0; evi<det->detections[i].embed_vector.size(); evi++){
                emb_vec.pushBack(det->detections[i].embed_vector.at(evi));
            }

            // covariance from detection message
            covariance(0,0) = det->detections[i].pose.covariance[0];
            covariance(1,1) = det->detections[i].pose.covariance[7];
            covariance(2,2) = det->detections[i].pose.covariance[14];
            covariance(0,1) = det->detections[i].pose.covariance[6];
            covariance(0,2) = det->detections[i].pose.covariance[12];
            covariance(1,0) = det->detections[i].pose.covariance[1];
            covariance(1,2) = det->detections[i].pose.covariance[13];
            covariance(2,0) = det->detections[i].pose.covariance[2];
            covariance(2,1) = det->detections[i].pose.covariance[8];

            detC(buffer_idx).pushBack(detContent);
            colHists(buffer_idx).pushBack(colhist);
            cov3d(buffer_idx).pushBack(covariance);
            embed_vecs(buffer_idx).pushBack(emb_vec);

        }
    }
}
#else
void Detections::addHOGdetOneFrame(Vector<Vector <double> >& det,
                                   int frame,
                                   QImage& imageLeft,
                                   Camera cam,
                                   Matrix<double>& depthMap)
{

    // FIXME find a different approach for determing a 3D cov.

    Matrix<double> covariance(3,3,0.0);

    Volume<double> colhist;
    Vector<double> pos3d;
    Vector<double> v_bbox;

    Vector<double>detContent;

    for ( int i = 0; i < det.getSize(); i++)
    {
        if(prepareDet(detContent, det, i, frame, true, cam, depthMap, covariance))
        {
            pos3d.clearContent();
            pos3d.pushBack(detContent(pos));
            pos3d.pushBack(detContent(pos+1));
            pos3d.pushBack(detContent(pos+2));

            v_bbox.clearContent();
            v_bbox.pushBack(detContent(bbox));
            v_bbox.pushBack(detContent(bbox+1));
            v_bbox.pushBack(detContent(bbox+2));
            v_bbox.pushBack(detContent(bbox+3));

            computeColorHist(colhist, v_bbox, Globals::binSize, imageLeft);

            detC(frame).pushBack(detContent);
            colHists(frame).pushBack(colhist);

            covariance(0,0) = sqrtf(covariance(0,0));
            covariance(2,2) = sqrtf(covariance(2,2));
            cov3d(frame).pushBack(covariance);
        }
    }
}
#endif

int Detections::getCategory(int frame, int detec)
{
    frame = globalFrameToBufferFrame(frame);
    return detC(frame)(detec)(categ);
}


void Detections::compute3DPosition(Vector<double>& detection, Camera cam)
{
    Vector<double> pos3D;
    Vector<double> v_bbox(4);
    double distance;

    v_bbox(0) = (detection(bbox));
    v_bbox(1) = (detection(bbox+1));
    v_bbox(2) = (detection(bbox+2));
    v_bbox(3) = (detection(bbox+3));

    double f_height = cam.bbToDetection(v_bbox, pos3D, Globals::WORLD_SCALE, distance);
    detection(dist) = distance;

    detection(pos) = pos3D(0);
    detection(pos+1) = pos3D(1);
    detection(pos+2) = pos3D(2);
    detection(height) =  f_height;

    //**********************************************************************************
    // Test Hard decision, if height doent reach some threshold dont use this detection
    //**********************************************************************************

    if(exp(-((Globals::dObjHeight - f_height)*(Globals::dObjHeight - f_height)) /
           (2 * Globals::dObjHVar * Globals::dObjHVar))  < Globals::probHeight || distance < 0)
    {
        ROS_DEBUG("Height Test Failed! %f", f_height);
        detection(0) *= -1;
    }
}

#ifdef cim_v
void Detections::computeColorHist(Volume<double>& colHist,
                                  Vector<double>& bbox,
                                  int nBins,
                                  CImg<unsigned char>& m_image)
{
    colHist.setSize(nBins, nBins, nBins, 0.0);

    Matrix<double> mNorm;
    Vector<double> vNorm;

    int r = 0;
    int g = 0;
    int b = 0;

    double x = bbox[0];
    double y = bbox[1];
    double w = bbox[2];
    double h = bbox[3];

    double binSize = 256.0 / double(nBins);
    double weight;

    double a = 1.0/(w * Globals::cutWidthBBOXColor * w * Globals::cutWidthBBOXColor * 0.6);
    double c = 1.0/(h * Globals::cutHeightBBOXforColor * h * Globals::cutHeightBBOXforColor * 0.6);

    //*********************************************************
    // Parameter for the eliptic shape
    //*********************************************************
    double newHeight = floor(h - (h * Globals::cutHeightBBOXforColor));
    double newWidth = floor(w - (w * Globals::cutWidthBBOXColor));
    int centerEliX = floor(x + (w / 2.0));
    int centerEliY = floor(y + (newHeight / 2.0) + newHeight*Globals::posponeCenterBBOXColor);

    for(int i = x; i < x + w; i++)
    {
        for(int j = y; j < y+h; j++)
        {
            if(Math::evalElipse(newWidth, newHeight, centerEliX, centerEliY, i, j)
               && i < Globals::dImWidth && j < Globals::dImHeight
               && i > -1
               && j > -1)
            {
                r = m_image(i,j,0,0);
                g = m_image(i,j,0,1);
                b = m_image(i,j,0,2);

                r = floor(double(r)/binSize);
                g = floor(double(g)/binSize);
                b = floor(double(b)/binSize);

                weight = exp(-(a*(i-centerEliX)*(i-centerEliX) + c*(j - centerEliY)*(j - centerEliY)));
                colHist(r, g, b) += weight;

            }
        }
    }

    colHist.sumAlongAxisZ(mNorm);
    mNorm.sumAlongAxisX(vNorm);
    double number = vNorm.sum();
    colHist *= (1.0/number);
}
#else
void Detections::computeColorHist(Volume<double>& colHist, Vector<double>& bbox, int nBins, QImage& m_image) {

    colHist.setSize(nBins, nBins, nBins, 0.0);

    Matrix<double> mNorm;
    Vector<double> vNorm;

    int r = 0;
    int g = 0;
    int b = 0;

    QColor color(r, g, b);
    QRgb rgb = color.rgb();

    double x = bbox[0];
    double y = bbox[1];
    double w = bbox[2];
    double h = bbox[3];

    double binSize = 256.0 / double(nBins);
    double weight;

    double a = 1.0/(w * Globals::cutWidthBBOXColor * w * Globals::cutWidthBBOXColor * 0.6);
    double c = 1.0/(h * Globals::cutHeightBBOXforColor * h * Globals::cutHeightBBOXforColor * 0.6);

    //*********************************************************
    // Parameter for the eliptic shape
    //*********************************************************
    double newHeight = floor(h - (h * Globals::cutHeightBBOXforColor));
    double newWidth = floor(w - (w * Globals::cutWidthBBOXColor));
    int centerEliX = floor(x + (w / 2.0));
    int centerEliY = floor(y + (newHeight / 2.0) + newHeight * Globals::posponeCenterBBOXColor);

    for(int i = x; i < x + w; i++)
    {
        for(int j = y; j < y+h; j++)
        {
            if(Math::evalElipse(newWidth, newHeight, centerEliX, centerEliY, i, j)
               && i < Globals::dImWidth
               && j < Globals::dImHeight
               && i > -1
               && j > -1)
            {
                rgb = m_image.pixel(i,j);
                r = qRed(rgb);
                g = qGreen(rgb);
                b = qBlue(rgb);

                r = floor(double(r)/binSize);
                g = floor(double(g)/binSize);
                b = floor(double(b)/binSize);

                weight = exp(-(a*(i-centerEliX)*(i-centerEliX) + c*(j - centerEliY)*(j - centerEliY)));
                colHist(r, g, b) += weight;

            }
        }
    }

    colHist.sumAlongAxisZ(mNorm);
    mNorm.sumAlongAxisX(vNorm);
    double number = vNorm.sum();
    colHist *= (1.0/number);
}
#endif

void Detections::getColorHist(int frame, int pos, Volume<double>& colHist)
{
    frame = globalFrameToBufferFrame(frame);
    colHist = colHists(frame)(pos);
}

void Detections::getEmbVec(int frame, int pos, Vector<double>& embVecs)
{
    frame = globalFrameToBufferFrame(frame);
    embVecs = embed_vecs(frame)(pos);
}

void Detections::compute3DCov(Vector<double> pos3d, Matrix<double>& cov, Camera camL, Camera camR)
{

    Matrix<double> covL;
    Matrix<double> covR;

    Matrix<double> c2d(2,2,0.0);
    c2d(0,0) = 0.5;
    c2d(1,1) = 0.5;

    c2d.inv();
    //****************************************************//
    // Computed as follows                                //
    // C = inv(F1' * inv(c2d) * F1 + F2' * inv(c2d) * F2);//
    //****************************************************//

    pos3d(0) *= (1.0 / Globals::WORLD_SCALE);
    pos3d(1) *= (1.0 / Globals::WORLD_SCALE);
    pos3d(2) *= (1.0 / Globals::WORLD_SCALE);

    camL.jacFor3DCov(pos3d, covL);
    camR.jacFor3DCov(pos3d, covR);

    Matrix<double> covLT = Transpose(covL);
    Matrix<double> covRT = Transpose(covR);

    covLT *= c2d;
    covLT *= covL;

    covRT *= c2d;
    covRT *= covR;

    covLT += covRT;
    covLT.inv();

    cov = covLT;
    cov *= (Globals::WORLD_SCALE)*(Globals::WORLD_SCALE);
}

void Detections::get3Dcovmatrix(int frame, int pos, Matrix<double>& covariance)
{
    frame = globalFrameToBufferFrame(frame);
    covariance = cov3d(frame)(pos);
}

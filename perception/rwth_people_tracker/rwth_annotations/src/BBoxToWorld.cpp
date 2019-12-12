// ROS includes.
#include "ros/ros.h"
#include "ros/time.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "Matrix.h"
#include "Vector.h"
#include "Camera.h"
#include "AncillaryMethods.h"

#include <sensor_msgs/CameraInfo.h>

#include "rwth_perception_people_msgs/GroundPlane.h"
#include "rwth_perception_people_msgs/VisualOdometry.h"
#include "rwth_perception_people_msgs/AnnotatedFrame.h"
#include "rwth_perception_people_msgs/Annotation.h"

#include "spencer_tracking_msgs/TrackedPersons.h"

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;
using namespace rwth_perception_people_msgs;


//Helper methods

Camera createCamera(Vector<double>& GP,
                    const VisualOdometry::ConstPtr &vo,
                    const CameraInfoConstPtr &info) {
    // create camera from motion_matrix-, camera_info- and GP-topic
    //  * motion_matrix-topic need to have format [R|t] (+ 0 0 0 1 in last row)
    Matrix<double> motion_matrix(4,4, (double*) (&vo->transformation_matrix[0]));
    Matrix<double> R(motion_matrix, 0,2,0,2);
    Vector<double> t(motion_matrix(3,0), motion_matrix(3,1), motion_matrix(3,2));

    //  * K is read from camera_info-topic
    Matrix<double> K(3,3, (double*)&info->K[0]);

    //  * GP is read from GP-topic [n1 n2 n3 d] and transfered to World coordinates
    Camera camera(K, R, t, GP);
    Vector<double> GP_world = AncillaryMethods::PlaneToWorld(camera, GP); // make sure AncillaryMethods are imported
    return Camera(K, R, t, GP_world);
}

Vector<double> projectPlaneToCam(Vector<double> p, Camera cam)
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

Vector<double> fromCam2World(Vector<double> posInCamera, Camera cam)
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


void callback(const CameraInfoConstPtr &info,
              const GroundPlane::ConstPtr &gp,
              const VisualOdometry::ConstPtr &vo,
              const AnnotatedFrame::ConstPtr &af)
{
    // ...

    // Get camera from VO and GP
    Vector<double> GP(3, (double*) &gp->n[0]);
    GP.pushBack((double) gp->d);

    Camera camera = createCamera(GP, vo, info); //create Camera from main

    //initial cam to compute HOG-depth
    Matrix<double> camRot_i = Eye<double>(3);
    Vector<double> camPos_i(3, 0.0);
    Vector<double> gp_i = camera.get_GP();
    Matrix<double> camInt_i = camera.get_K();
    Vector<double> planeInCam_i = projectPlaneToCam(gp_i, camera);
    Camera camI(camInt_i, camRot_i, camPos_i, planeInCam_i);

    auto& annos = af->annotations;

    //loop through anno-boxes
    for(auto& anno : annos)
    {
        // Convert from [x0 y0 x1 y1] to [x0 y0 w h]
        Vector<double> bbox = {anno.tlx, anno.tly, anno.brx - anno.tlx, anno.bry - anno.tly};
        Vector<double> pos3D_camera;
        double distance; // dummy, not used

        // compute world position from bbox
        camI.bbToDetection(bbox, pos3D_camera, 1, distance); // 1: WorldScale
        Vector<double> posInCamCord(pos3D_camera(0), pos3D_camera(1), pos3D_camera(2), 1);
        Vector<double> posInWorld = fromCam2World(posInCamCord, camera);

        // posInWorld then in our coordinate frame ([x y z 1]), for Spencer Tracked Persons: [z -x -y]!!
        // create TrackedPersons-Message + publish

    }



    //...
}

int main(int argc, char **argv)
{
    // ...

    // approximate timing stuff with CameraInfo, GroundPlane, VisualOdometry
    sync_policies::ApproximateTime<CameraInfo, GroundPlane, VisualOdometry> MySyncPolicy(10); //queue size

    const sync_policies::ApproximateTime<CameraInfo, GroundPlane, VisualOdometry> MyConstSyncPolicy = MySyncPolicy;

    Synchronizer< sync_policies::ApproximateTime<CameraInfo, GroundPlane, VisualOdometry> >
            sync(MyConstSyncPolicy, subscriber_camera_info, subscriber_gp, subscriber_vo);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3));
 
    // ...

    ros::spin();
    return 0;
}

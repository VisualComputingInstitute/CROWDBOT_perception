/*
 * Copyright (C) by
 *   MetraLabs GmbH (MLAB), GERMANY
 *  and
 *   Neuroinformatics and Cognitive Robotics Labs (NICR) at TU Ilmenau, GERMANY
 * All rights reserved.
 *
 * Redistribution and modification of this code is strictly prohibited.
 *
 * IN NO EVENT SHALL "MLAB" OR "NICR" BE LIABLE TO ANY PARTY FOR DIRECT,
 * INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
 * THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF "MLAB" OR
 * "NICR" HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * "MLAB" AND "NICR" SPECIFICALLY DISCLAIM ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND "MLAB" AND "NICR" HAVE NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS OR MODIFICATIONS.
 */

/**
 * @file PanoramaCameraModel.h
 *    Model for Panorama cameras
 *    Cylindrical Equidistant Projection
 *
 * @author Michael Volkhardt
 * @date   2011/09/22
 */

#ifndef _MIRA_PANORAMACAMERAMODEL_H__
#define _MIRA_PANORAMACAMERAMODEL_H__


#include <cmath>
//#include <geometry/Point.h>
//#include <cameraparameters/PanoramaCameraIntrinsic.h>
#include "PanoramaCameraIntrinsic.h"
#include "Vector.h"


using namespace std;

namespace mira {
namespace camera {
namespace PanoramaCameraModel {





//////////////////////////////////////////////////////////////////////////////

/**
 * projects image point to world ray
 * @param iImgPoint image point to project
 * @param oRay resulting normalized world ray (length to unit sphere)
 * @param iParam intrinsic panorama camera model parameters
 */
inline void projectPixelTo3dRay(Vector<double> const& iImgPoint, Vector<double>& oRay,
                                const PanoramaCameraIntrinsic& iParam) {

//	float phi = ((iParam.maxPhi - iParam.minPhi) * iImgPoint.x() / iParam.width)
//			+ iParam.minPhi;
    float phi = ((iParam.maxPhi - iParam.minPhi) * iImgPoint(0) / iParam.width)
            + iParam.minPhi;
//	float height = ((iParam.maxHeight - iParam.minHeight) * iImgPoint.y()
//			/ iParam.height) + iParam.minHeight;
    float height = ((iParam.maxHeight - iParam.minHeight) * iImgPoint(1)
            / iParam.height) + iParam.minHeight;


//	oRay = Point3f(std::cos(phi), std::sin(phi), height);
    //oRay = Vector<double>(-std::sin(phi), height, std::cos(phi));
    oRay = Vector<double>(std::sin(phi), height, std::cos(phi));
    oRay = oRay * (1.0/oRay.norm());  // do normalization
}

/**
 * projects camera ray to image pixel
 * @param iRay world ray
 * @param oImgPoint resulting image pixel
 * @param iParam intrinsic panorama camera model parameters
 */
inline void project3dRayToPixel(Vector<double> const& iRay, Vector<double>& oImgPoint,
                                const PanoramaCameraIntrinsic& iParam) {

//	float phi = std::atan2(iRay.y(), iRay.x());
//	float height = iRay.z() / std::hypot(iRay.y(), iRay.x());
//	float phi = std::atan2(-iRay.x(), iRay.z());
//	float height = iRay.y() / std::hypot(-iRay.x(), iRay.z());
    //float phi = std::atan2(-iRay(0), iRay(2));
    float phi = std::atan2(iRay(0), iRay(2));
    float height = iRay(1) / std::sqrt(iRay(0)*iRay(0) + iRay(2)*iRay(2)); // hypot is for c++11, here we simply use sqrt(x^2, y^2)
//	oImgPoint.x() = -1 * (iParam.minPhi - phi) / (iParam.maxPhi - iParam.minPhi)
//			* iParam.width;
//	oImgPoint.y() = (height - iParam.minHeight)
//			/ (iParam.maxHeight - iParam.minHeight) * iParam.height;
    oImgPoint(0) = (phi - iParam.minPhi) / (iParam.maxPhi - iParam.minPhi)
            * iParam.width;
    oImgPoint(1) = (height - iParam.minHeight)
            / (iParam.maxHeight - iParam.minHeight) * iParam.height;
}

//////////////////////////////////////////////////////////////////////////////

}
}
} // namespace

#endif /* _MIRA_PANORAMACAMERAMODEL_H__ */
